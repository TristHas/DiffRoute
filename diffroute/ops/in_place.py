from torch.autograd import Function

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