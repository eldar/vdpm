import numpy as np
import torch
from typing import List
from torch import Tensor


class TensorList:
    def __init__(self, tensor_list: List[Tensor] | Tensor, cumsum):
        self._len = len(tensor_list)
        if isinstance(tensor_list, List):
            tensor_list = torch.cat(tensor_list, dim=0)
        self._data = tensor_list
        self._cumsum = cumsum

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        start_idx = self._cumsum[idx]
        end_idx = self._cumsum[idx+1]
        return self._data[start_idx:end_idx]

    def cumsum(self):
        return self._cumsum


def compute_cumsum(tensors: List[Tensor]):
    seq_lens = torch.tensor([0] + [p.shape[0] for p in tensors], dtype=torch.int64)
    return torch.cumsum(seq_lens, dim=0)


def make_tensorlist(tensor_list: List[Tensor]):
    return TensorList(tensor_list, compute_cumsum(tensor_list))


def compute_cumsum_np(tensors: List[np.ndarray]):
    seq_lens = np.array([0] + [p.shape[0] for p in tensors], dtype=np.int64)
    return np.cumsum(seq_lens, axis=0)

