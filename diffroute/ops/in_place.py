from torch.autograd import Function
import torch

class _WriteSlice(Function):
    """
    Assemble the final output buffer without torch.cat:
        target[row_start:row_end] = src
        returns target  (so loss(target) backprops to src)
    """
    @staticmethod
    def forward(ctx, target, src, row_start: int, row_end: int):
        ctx.mark_dirty(target)
        ctx.slice = (row_start, row_end)
        target[:, row_start:row_end].copy_(src)  # in-place write
        return target

    @staticmethod
    def backward(ctx, grad_target):
        rs, re = ctx.slice
        grad_target_in = grad_target        # pass-through
        grad_src       = grad_target[:,rs:re] # slice
        return grad_target_in, grad_src, None, None

def write_slice(target, src, row_start, row_end):
    return _WriteSlice.apply(target, src, row_start, row_end)

class _IndexAddInplace(Function):
    """
    In-place index_add:
        target.index_add_(dim, index, src)
        returns target (so loss(target) backprops to both target and src)

    Notes:
    - index must be 1D Long tensor (same device as target/src).
    - This is first-order differentiable. Higher-order grads through this in-place op
      are generally not supported/reliable.
    """
    @staticmethod
    def forward(ctx, target: torch.Tensor, index: torch.Tensor, src: torch.Tensor, dim: int):
        if index.dtype != torch.long:
            raise TypeError(f"index must be torch.long, got {index.dtype}")
        if index.dim() != 1:
            raise ValueError(f"index must be 1D, got shape {tuple(index.shape)}")
        if dim < 0:
            dim = target.dim() + dim

        ctx.mark_dirty(target)
        ctx.dim = dim
        ctx.save_for_backward(index)

        # in-place scatter-add
        target.index_add_(dim, index, src)
        return target

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (index,) = ctx.saved_tensors
        dim = ctx.dim

        # output depends on input target with coefficient 1 everywhere
        grad_target = grad_out

        # each src element contributed to target at position index[k]
        grad_src = grad_out.index_select(dim, index)

        # no grad for index (integer) nor dim (python int)
        return grad_target, None, grad_src, None

def index_add_inplace(target: torch.Tensor, index: torch.Tensor, src: torch.Tensor, dim: int = 0):
    return _IndexAddInplace.apply(target, index, src, dim)