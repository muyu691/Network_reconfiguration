"""
Phase 4：网络对拓扑数据集加载器
================================

加载由 Phase 1（build_network_pairs_dataset.py）生成的 (G, G') 网络对数据集。

期望的目录结构（cfg.dataset.dir 指向的路径）：
  <dataset_dir>/
  ├── train_dataset.pt      ← list of PyG Data (训练集)
  ├── val_dataset.pt        ← list of PyG Data (验证集)
  ├── test_dataset.pt       ← list of PyG Data (测试集)
  └── scalers/
      ├── attr_scaler.pkl   ← 边物理属性 StandardScaler
      └── flow_scaler.pkl   ← 流量 StandardScaler（用于反归一化）

每个 Data 对象的字段（严格对应 Phase 1 输出）：
  x               : [24, 1]       全 1 节点占位符
  edge_index_old  : [2, E_old]    旧图连接关系（0-indexed）
  edge_attr_old   : [E_old, 3]    旧图物理属性（归一化）
  flow_old        : [E_old, 1]    旧图历史流量（归一化）
  edge_index_new  : [2, E_new]    新图连接关系（0-indexed）
  edge_attr_new   : [E_new, 3]    新图物理属性（归一化）
  y               : [E_new, 1]    新图均衡流量（归一化，GT）
  non_centroid_mask: [24]         bool，标记非质心节点（用于守恒损失）
  mutation_type   : str           变异类型（仅用于调试分析）

批处理兼容性说明：
  PyG 对含 'index' 后缀的字段自动施加节点偏移（__inc__ 机制），
  因此 edge_index_old 和 edge_index_new 在 DataLoader 批处理后
  节点索引依然全局正确，无需手动处理。
  non_centroid_mask [24] 会被 PyG 沿 dim=0 拼接为 [24 × batch_size]，
  与批处理后的 batch.num_nodes 完全对齐。
"""

import logging
import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset


class NetworkPairsTopologyDataset(InMemoryDataset):
    """
    (G, G') 网络对拓扑数据集。

    直接加载由 Phase 1 生成的 .pt 文件（Data 对象列表），
    使用 PyG 的 InMemoryDataset.collate() 进行内存内索引，
    与 PyG DataLoader 无缝对接。

    Args:
        root  : dataset_dir，指向含有 train/val/test_dataset.pt 的目录
        split : 'train' | 'val' | 'test'
    """

    # Phase 1 生成的三个分割文件名（固定，与 build_network_pairs_dataset.py 保持一致）
    _SPLIT_FILES = {
        'train': 'train_dataset.pt',
        'val':   'val_dataset.pt',
        'test':  'test_dataset.pt',
    }

    def _download(self):
        pass

    def _process(self):
        pass

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ) -> None:
        assert split in self._SPLIT_FILES, \
            f"split 必须为 'train'/'val'/'test'，收到：'{split}'"
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)

        # ── 加载对应分割的 Data 列表并整理为内存格式 ────────────────
        pt_path = osp.join(self.processed_dir, self._SPLIT_FILES[split])
        if not osp.exists(pt_path):
            raise FileNotFoundError(
                f"找不到数据文件：{pt_path}\n"
                f"请先运行 Phase 1：\n"
                f"  cd create_sioux_data\n"
                f"  python build_network_pairs_dataset.py "
                f"--output_dir processed_data/pyg_dataset"
            )

        data_list = torch.load(pt_path, weights_only=False)
        logging.info(
            f"[NetworkPairsTopologyDataset] 加载 {split} 集：{len(data_list)} 个样本 "
            f"← {pt_path}"
        )
        self.data, self.slices = self.collate(data_list)

    # ── PyG InMemoryDataset 接口实现 ────────────────────────────────

    @property
    def processed_dir(self) -> str:
        """直接将 root 目录视为已处理数据目录，不再创建子目录。"""
        return self.root

    @property
    def raw_file_names(self):
        """不需要原始文件（数据由 Phase 1 脚本生成）。"""
        return []

    @property
    def processed_file_names(self):
        """声明三个分割文件，供 PyG 校验是否存在。"""
        return list(self._SPLIT_FILES.values())

    def download(self):
        """数据不可自动下载，引导用户手动运行 Phase 1。"""
        raise FileNotFoundError(
            "NetworkPairs 数据集不支持自动下载。\n"
            "请先运行 Phase 1 数据生成脚本：\n"
            "  cd create_sioux_data\n"
            "  python build_network_pairs_dataset.py\n"
            "生成的文件将保存在 processed_data/pyg_dataset/ 目录中。"
        )

    def process(self):
        """数据已由 Phase 1 脚本预处理完毕，此处不需要额外处理。"""
        pass

    def __repr__(self) -> str:
        return (
            f"NetworkPairsTopologyDataset("
            f"split={self.split}, "
            f"num_graphs={len(self)})"
        )
