import logging

import torch
from torch_geometric.utils import subgraph
from torch_geometric.graphgym.config import cfg
from tqdm import tqdm


class MaskEdgeFeatureTransform:
    """
    Zero-out specific columns in edge_attr_old and edge_attr_new for
    input feature importance ablation.

    edge_attr column layout (from build_network_pairs_dataset.py):
      col 0 : capacity   (vehicles/hour, normalized)
      col 1 : speed      (km/h, normalized) — proxy for FFT since FFT = length/speed
      col 2 : length     (km, normalized)

    When mask_capacity=True, col 0 is zeroed in both edge_attr_old and edge_attr_new.
    When mask_fft=True,      col 1 is zeroed (masking speed ≈ removing FFT information).
    """

    def __init__(self, mask_capacity: bool = False, mask_fft: bool = False):
        self.mask_capacity = mask_capacity
        self.mask_fft = mask_fft

    def __call__(self, data):
        cols_to_mask = []
        if self.mask_capacity:
            cols_to_mask.append(0)
        if self.mask_fft:
            cols_to_mask.append(1)

        if not cols_to_mask:
            return data

        for attr_name in ('edge_attr_old', 'edge_attr_new'):
            attr = getattr(data, attr_name, None)
            if attr is not None:
                attr = attr.clone()
                for col in cols_to_mask:
                    if col < attr.size(1):
                        attr[:, col] = 0.0
                setattr(data, attr_name, attr)

        return data

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"mask_capacity={self.mask_capacity}, mask_fft={self.mask_fft})")


def pre_transform_in_memory(dataset, transform_func, show_progress=False):
    """Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return dataset

    data_list = [transform_func(dataset.get(i))
                 for i in tqdm(range(len(dataset)),
                               disable=not show_progress,
                               mininterval=10,
                               miniters=len(dataset)//20)]
    data_list = list(filter(None, data_list))

    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)


def typecast_x(data, type_str):
    if type_str == 'float':
        data.x = data.x.float()
    elif type_str == 'long':
        data.x = data.x.long()
    else:
        raise ValueError(f"Unexpected type '{type_str}'.")
    return data


def concat_x_and_pos(data):
    data.x = torch.cat((data.x, data.pos), 1)
    return data


def clip_graphs_to_size(data, size_limit=5000):
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    if N <= size_limit:
        return data
    else:
        logging.info(f'  ...clip to {size_limit} a graph of size: {N}')
        if hasattr(data, 'edge_attr'):
            edge_attr = data.edge_attr
        else:
            edge_attr = None
        edge_index, edge_attr = subgraph(list(range(size_limit)),
                                         data.edge_index, edge_attr)
        if hasattr(data, 'x'):
            data.x = data.x[:size_limit]
            data.num_nodes = size_limit
        else:
            data.num_nodes = size_limit
        if hasattr(data, 'node_is_attributed'):  # for ogbg-code2 dataset
            data.node_is_attributed = data.node_is_attributed[:size_limit]
            data.node_dfs_order = data.node_dfs_order[:size_limit]
            data.node_depth = data.node_depth[:size_limit]
        data.edge_index = edge_index
        if hasattr(data, 'edge_attr'):
            data.edge_attr = edge_attr
        return data
