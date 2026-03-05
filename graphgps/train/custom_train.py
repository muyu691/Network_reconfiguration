import logging
import time

import numpy as np
import torch
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch

# Subtoken loss only used for ogbg-code2 dataset (not included in clean version)
# from graphgps.loss.subtoken_prediction_loss import subtoken_cross_entropy
from graphgps.utils import cfg_to_dict, flatten_dict, make_wandb_name

# ── Phase 4 Injection: PINN Physical Conservation Loss ────────────────────────────────
# When cfg.model.type == 'topology_gnn', use this function to replace compute_loss(pred, true).
# Reason: Standard compute_loss only takes (pred, true), but conservation loss also requires
#         batch-level topology info like batch.edge_index_new, batch.non_centroid_mask, etc.
from graphgps.loss.flow_conservation_loss import compute_pinn_loss

from graphgps.metric_wrapper import wmape
from torchmetrics.functional import mean_absolute_error


def subtoken_cross_entropy(pred, true):
    """Stub function for ogbg-code2 dataset (not supported in clean version)."""
    raise NotImplementedError("ogbg-code2 dataset and subtoken prediction loss not available in this clean version.")


def _compute_loss(pred, true, batch):
    """
    Unified loss dispatch function: automatically selects loss function according to cfg.model.type.

    - 'topology_gnn' → compute_pinn_loss(pred, batch)
      Uses physical conservation penalty (Phase 3), requires batch context.
    - Other models → compute_loss(pred, true)
      Uses standard GraphGym loss (backward compatible with all previous experiments).

    Args:
        pred  : Model output tensor
        true  : Ground Truth tensor
        batch : Current batch's PyG Batch object

    Returns:
        loss       : scalar, total loss for backpropagation
        pred_score : predicted score tensor, passed to logger for metric recording
    """
    if cfg.model.type == 'topology_gnn':
        # PINN composite loss: L_sup (in normalized space) + λ × L_cons (dimensionless conservation penalty)
        return compute_pinn_loss(pred, batch)
    else:
        # Standard GraphGym loss (L1/MSE etc., controlled by cfg.model.loss_fun)
        return compute_loss(pred, true)


def train_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    for iter, batch in enumerate(loader):
        batch.split = 'train'
        batch.to(torch.device(cfg.accelerator))
        pred, true = model(batch)
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            # Phase 4: Dispatch via _compute_loss, topology models automatically use PINN loss
            loss, pred_score = _compute_loss(pred, true, batch)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        loss.backward()
        # Parameters update after accumulating gradients for given num. batches.
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.optim.clip_grad_norm_value)
            optimizer.step()
            optimizer.zero_grad()
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name)
        time_start = time.time()


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.accelerator))
        if cfg.gnn.head == 'inductive_edge':
            pred, true, extra_stats = model(batch)
        else:
            pred, true = model(batch)
            extra_stats = {}
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            # Phase 4: Also use PINN loss in validation/test phase to ensure metric consistency
            loss, pred_score = _compute_loss(pred, true, batch)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name,
                            **extra_stats)
        time_start = time.time()


def _compute_new_edge_mask(batch, device):
    """
    Vectorized computation of new-edge boolean mask for the batched new graph.

    Uses hash-key encoding + torch.isin for memory-safe set membership test.
    Space complexity: O(E_old + E_new) instead of O(V^2).

    Returns:
        is_new_edge : [E_new_total], bool, True = edge only in G' (added edge)
    """
    total_nodes = batch.num_nodes
    old_keys = batch.edge_index_old[0] * total_nodes + batch.edge_index_old[1]
    new_keys = batch.edge_index_new[0] * total_nodes + batch.edge_index_new[1]

    is_new_edge = ~torch.isin(new_keys, old_keys)
    return is_new_edge


@torch.no_grad()
def detailed_test_evaluation(loader, model, split='test'):
    """
    Comprehensive test-phase evaluation. Three analyses in one pass:

      1. Per-graph WMAPE  → 25%, 50%, 75%, 95% percentiles
      2. New-edge vs Old-edge  → WMAPE, MAE for each subset
      3. Inference timing → average time per graph (GPU-aware)

    All WMAPE / MAE are computed in **denormalized** (real flow) space when
    cfg.dataset.flow_mean / flow_std are available, otherwise in normalized space.

    Args:
        loader : DataLoader for the target split (typically test)
        model  : trained model (already in eval mode by caller)
        split  : split name string for logging
    """
    model.eval()
    device = torch.device(cfg.accelerator)

    flow_mean = getattr(cfg.dataset, 'flow_mean', 0.0)
    flow_std = getattr(cfg.dataset, 'flow_std', 1.0)
    use_denorm = not (flow_mean == 0.0 and flow_std == 1.0)

    per_graph_wmapes = []

    all_preds_new = []
    all_trues_new = []
    all_preds_old = []
    all_trues_old = []

    total_time_ms = 0.0
    total_graphs = 0

    use_cuda = (device.type == 'cuda')

    for batch in loader:
        batch.to(device)
        num_graphs_in_batch = batch.num_graphs
        total_graphs += num_graphs_in_batch

        # ── Timed forward pass ──────────────────────────────────────────
        # Compatible with models returning 2-tuple (pred, true) or
        # 3-tuple (pred, true, extra_stats) when cfg.gnn.head == 'inductive_edge'.
        if use_cuda:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            out = model(batch)
            end_evt.record()
            torch.cuda.synchronize()
            batch_time_ms = start_evt.elapsed_time(end_evt)
        else:
            t0 = time.perf_counter()
            out = model(batch)
            batch_time_ms = (time.perf_counter() - t0) * 1000.0

        if isinstance(out, tuple) and len(out) == 3:
            pred, true, _ = out
        else:
            pred, true = out
        total_time_ms += batch_time_ms

        pred_cpu = pred.detach().cpu().float()
        true_cpu = true.detach().cpu().float()

        # Denormalize: real_flow = scaled * std + mean
        if use_denorm:
            pred_real = pred_cpu * flow_std + flow_mean
            true_real = true_cpu * flow_std + flow_mean
        else:
            pred_real = pred_cpu
            true_real = true_cpu

        # ── Per-graph WMAPE ─────────────────────────────────────────────
        edge_idx = getattr(batch, 'edge_index_new', batch.edge_index)
        edge_batch = batch.batch[edge_idx[0]].detach().cpu()
        for g in range(num_graphs_in_batch):
            mask_g = (edge_batch == g)
            p_g = pred_real[mask_g]
            t_g = true_real[mask_g]
            if t_g.numel() > 0:
                w = wmape(p_g, t_g).item()
                per_graph_wmapes.append(w)

        # ── New / Old edge separation ───────────────────────────────────
        if hasattr(batch, 'new_edge_mask'):
            is_new_edge = batch.new_edge_mask.bool().detach().cpu()
        elif hasattr(batch, 'edge_index_new') and hasattr(batch, 'edge_index_old'):
            is_new_edge = _compute_new_edge_mask(batch, device).detach().cpu()
        else:
            is_new_edge = torch.zeros(pred_real.shape[0], dtype=torch.bool)

        if is_new_edge.any():
            all_preds_new.append(pred_real[is_new_edge])
            all_trues_new.append(true_real[is_new_edge])
        if (~is_new_edge).any():
            all_preds_old.append(pred_real[~is_new_edge])
            all_trues_old.append(true_real[~is_new_edge])

    # ════════════════════════════════════════════════════════════════════
    # Aggregate & Log
    # ════════════════════════════════════════════════════════════════════
    space_tag = "denormalized (veh/hr)" if use_denorm else "normalized"

    # ── 1. WMAPE Percentiles ────────────────────────────────────────────
    if per_graph_wmapes:
        w_tensor = torch.tensor(per_graph_wmapes, dtype=torch.float64)
        quantiles = [0.25, 0.50, 0.75, 0.95]
        q_values = torch.quantile(w_tensor, torch.tensor(quantiles, dtype=torch.float64))
        logging.info(
            f"\n{'='*70}\n"
            f"  [{split.upper()}] Per-Graph WMAPE Distribution ({len(per_graph_wmapes)} graphs, {space_tag})\n"
            f"{'='*70}\n"
            f"    Mean   WMAPE : {w_tensor.mean():.6f}\n"
            f"    Std    WMAPE : {w_tensor.std():.6f}\n"
            f"    Min    WMAPE : {w_tensor.min():.6f}\n"
            f"    25%  percentile : {q_values[0]:.6f}\n"
            f"    50%  percentile : {q_values[1]:.6f} (median)\n"
            f"    75%  percentile : {q_values[2]:.6f}\n"
            f"    95%  percentile : {q_values[3]:.6f}\n"
            f"    Max    WMAPE : {w_tensor.max():.6f}\n"
            f"{'='*70}"
        )

    # ── 2. New-Edge vs Old-Edge Metrics ─────────────────────────────────
    def _edge_metrics(preds_list, trues_list, label):
        if not preds_list:
            logging.info(f"    {label}: No edges found in test set.")
            return
        p = torch.cat(preds_list)
        t = torch.cat(trues_list)
        w = wmape(p, t).item()
        m = mean_absolute_error(p.view(-1), t.view(-1)).item()
        logging.info(f"    {label}: count={p.shape[0]:>7d}  WMAPE={w:.6f}  MAE={m:.4f}")

    logging.info(
        f"\n{'='*70}\n"
        f"  [{split.upper()}] New-Edge vs Old-Edge Metrics ({space_tag})\n"
        f"{'='*70}"
    )
    _edge_metrics(all_preds_old, all_trues_old, "Old edges (retained)")
    _edge_metrics(all_preds_new, all_trues_new, "New edges (added)   ")
    # Combined
    all_p = all_preds_old + all_preds_new
    all_t = all_trues_old + all_trues_new
    _edge_metrics(all_p, all_t, "All edges (combined) ")
    logging.info(f"{'='*70}")

    # ── 3. Inference Timing ─────────────────────────────────────────────
    avg_time_per_graph = total_time_ms / max(total_graphs, 1)
    logging.info(
        f"\n{'='*70}\n"
        f"  [{split.upper()}] Inference Timing\n"
        f"{'='*70}\n"
        f"    Device               : {device}\n"
        f"    Total graphs         : {total_graphs}\n"
        f"    Total inference time : {total_time_ms:.2f} ms\n"
        f"    Avg time per graph   : {avg_time_per_graph:.4f} ms\n"
        f"{'='*70}"
    )

    return {
        'wmape_percentiles': {
            'p25': float(q_values[0]) if per_graph_wmapes else None,
            'p50': float(q_values[1]) if per_graph_wmapes else None,
            'p75': float(q_values[2]) if per_graph_wmapes else None,
            'p95': float(q_values[3]) if per_graph_wmapes else None,
        },
        'wmape_mean': float(w_tensor.mean()) if per_graph_wmapes else None,
        'avg_time_per_graph_ms': avg_time_per_graph,
        'total_graphs': total_graphs,
    }


@register_train('custom')
def custom_train(loggers, loaders, model, optimizer, scheduler):
    """
    Customized training pipeline.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)

    if cfg.wandb.use:
        try:
            import wandb
        except:
            raise ImportError('WandB is not installed.')
        if cfg.wandb.name == '':
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                         name=wandb_name)
        run.config.update(cfg_to_dict(cfg))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        start_time = time.perf_counter()
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler,
                    cfg.optim.batch_accumulation)
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i - 1])
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        # Checkpoint with regular frequency (if enabled).
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best \
                and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                # Select again based on val perf of `cfg.metric_best`.
                m = cfg.metric_best
                best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                                     cfg.metric_agg)()
                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    # Note: For some datasets it is too expensive to compute
                    # the main metric on the training set.
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                if cfg.wandb.use:
                    bstats = {"best/epoch": best_epoch}
                    for i, s in enumerate(['train', 'val', 'test']):
                        bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
                        if m in perf[i][best_epoch]:
                            bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                            run.summary[f"best_{s}_perf"] = \
                                perf[i][best_epoch][m]
                        for x in ['hits@1', 'hits@3', 'hits@10', 'mrr']:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                    run.log(bstats, step=cur_epoch)
                    run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                    run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)
            # Checkpoint the best epoch params (if enabled).
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                    best_epoch == cur_epoch:
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                    clean_ckpt()
            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
            )
            if hasattr(model, 'trf_layers'):
                # Log SAN's gamma parameter values if they are trainable.
                for li, gtl in enumerate(model.trf_layers):
                    if torch.is_tensor(gtl.attention.gamma) and \
                            gtl.attention.gamma.requires_grad:
                        logging.info(f"    {gtl.__class__.__name__} {li}: "
                                     f"gamma={gtl.attention.gamma.item()}")
    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")

    # ── Detailed test evaluation (WMAPE percentiles, new/old edge, timing) ──
    if len(loaders) >= 3:
        logging.info("\n" + "=" * 70)
        logging.info("  Running detailed test evaluation ...")
        logging.info("=" * 70)
        detailed_test_evaluation(loaders[2], model, split='test')

    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    # close wandb
    if cfg.wandb.use:
        run.finish()
        run = None

    logging.info('Task done, results saved in %s', cfg.run_dir)


@register_train('inference-only')
def inference_only(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference only.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    num_splits = len(loggers)
    split_names = ['train', 'val', 'test']
    perf = [[] for _ in range(num_splits)]
    cur_epoch = 0
    start_time = time.perf_counter()

    for i in range(0, num_splits):
        eval_epoch(loggers[i], loaders[i], model,
                   split=split_names[i])
        perf[i].append(loggers[i].write_epoch(cur_epoch))

    best_epoch = 0
    best_train = best_val = best_test = ""
    if cfg.metric_best != 'auto':
        # Select again based on val perf of `cfg.metric_best`.
        m = cfg.metric_best
        if m in perf[0][best_epoch]:
            best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
        else:
            # Note: For some datasets it is too expensive to compute
            # the main metric on the training set.
            best_train = f"train_{m}: {0:.4f}"
        best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
        best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

    logging.info(
        f"> Inference | "
        f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
        f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
        f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
    )
    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')

    # ── Detailed test evaluation (WMAPE percentiles, new/old edge, timing) ──
    if len(loaders) >= 3:
        logging.info("\n" + "=" * 70)
        logging.info("  Running detailed test evaluation ...")
        logging.info("=" * 70)
        detailed_test_evaluation(loaders[2], model, split='test')

    for logger in loggers:
        logger.close()


@register_train('PCQM4Mv2-inference')
def ogblsc_inference(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference on OGB-LSC PCQM4Mv2.

    Args:
        loggers: Unused, exists just for API compatibility
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    from ogb.lsc import PCQM4Mv2Evaluator
    evaluator = PCQM4Mv2Evaluator()

    num_splits = 3
    split_names = ['valid', 'test-dev', 'test-challenge']
    assert len(loaders) == num_splits, "Expecting 3 particular splits."

    # Check PCQM4Mv2 prediction targets.
    logging.info(f"0 ({split_names[0]}): {len(loaders[0].dataset)}")
    assert(all([not torch.isnan(d.y)[0] for d in loaders[0].dataset]))
    logging.info(f"1 ({split_names[1]}): {len(loaders[1].dataset)}")
    assert(all([torch.isnan(d.y)[0] for d in loaders[1].dataset]))
    logging.info(f"2 ({split_names[2]}): {len(loaders[2].dataset)}")
    assert(all([torch.isnan(d.y)[0] for d in loaders[2].dataset]))

    model.eval()
    for i in range(num_splits):
        all_true = []
        all_pred = []
        for batch in loaders[i]:
            batch.to(torch.device(cfg.accelerator))
            pred, true = model(batch)
            all_true.append(true.detach().to('cpu', non_blocking=True))
            all_pred.append(pred.detach().to('cpu', non_blocking=True))
        all_true, all_pred = torch.cat(all_true), torch.cat(all_pred)

        if i == 0:
            input_dict = {'y_pred': all_pred.squeeze(),
                          'y_true': all_true.squeeze()}
            result_dict = evaluator.eval(input_dict)
            logging.info(f"{split_names[i]}: MAE = {result_dict['mae']}")  # Get MAE.
        else:
            input_dict = {'y_pred': all_pred.squeeze()}
            evaluator.save_test_submission(input_dict=input_dict,
                                           dir_path=cfg.run_dir,
                                           mode=split_names[i])


@ register_train('log-attn-weights')
def log_attn_weights(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to inference on the test set and log the attention
    weights in Transformer modules.

    Args:
        loggers: Unused, exists just for API compatibility
        loaders: List of loaders
        model (torch.nn.Module): GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    import os.path as osp
    from torch_geometric.loader.dataloader import DataLoader
    from graphgps.utils import unbatch, unbatch_edge_index

    start_time = time.perf_counter()

    # The last loader is a test set.
    l = loaders[-1]
    # To get a random sample, create a new loader that shuffles the test set.
    loader = DataLoader(l.dataset, batch_size=l.batch_size,
                        shuffle=True, num_workers=0)

    output = []
    # batch = next(iter(loader))  # Run one random batch.
    for b_index, batch in enumerate(loader):
        bsize = batch.batch.max().item() + 1  # Batch size.
        if len(output) >= 128:
            break
        print(f">> Batch {b_index}:")

        X_orig = unbatch(batch.x.cpu(), batch.batch.cpu())
        batch.to(torch.device(cfg.accelerator))
        model.eval()
        model(batch)

        # Unbatch to individual graphs.
        X = unbatch(batch.x.cpu(), batch.batch.cpu())
        edge_indices = unbatch_edge_index(batch.edge_index.cpu(),
                                          batch.batch.cpu())
        graphs = []
        for i in range(bsize):
            graphs.append({'num_nodes': len(X[i]),
                           'x_orig': X_orig[i],
                           'x_final': X[i],
                           'edge_index': edge_indices[i],
                           'attn_weights': []  # List with attn weights in layers from 0 to L-1.
                           })

        # Iterate through GPS layers and pull out stored attn weights.
        for l_i, (name, module) in enumerate(model.model.layers.named_children()):
            if hasattr(module, 'attn_weights'):
                print(l_i, name, module.attn_weights.shape)
                for g_i in range(bsize):
                    # Clip to the number of nodes in this graph.
                    # num_nodes = graphs[g_i]['num_nodes']
                    # aw = module.attn_weights[g_i, :num_nodes, :num_nodes]
                    aw = module.attn_weights[g_i]
                    graphs[g_i]['attn_weights'].append(aw.cpu())
        output += graphs

    logging.info(
        f"[*] Collected a total of {len(output)} graphs and their "
        f"attention weights for {len(output[0]['attn_weights'])} layers.")

    # Save the graphs and their attention stats.
    save_file = osp.join(cfg.run_dir, 'graph_attn_stats.pt')
    logging.info(f"Saving to file: {save_file}")
    torch.save(output, save_file)

    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
