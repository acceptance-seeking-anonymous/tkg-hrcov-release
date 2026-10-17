# Pure-torch replacement for torch_scatter.scatter(reduce='sum') along dim 0 with a 1-D index,
# the only form used by CognTKE (models/layers.py). Same semantics as torch_scatter (sum, zeros for empty rows).
import torch
__version__ = "shim-index_add"
def scatter(src, index, dim=0, out=None, dim_size=None, reduce='sum'):
    assert dim == 0 and out is None and reduce in ('sum', 'add') and index.dim() == 1, (dim, reduce, index.shape)
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() else 0
    res = torch.zeros((dim_size,) + tuple(src.shape[1:]), dtype=src.dtype, device=src.device)
    return res.index_add_(0, index, src)
