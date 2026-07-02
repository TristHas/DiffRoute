import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _down_tri_relu_flip_norm_fwd_kernel(
    x_ptr,
    out_ptr,
    denom_ptr,
    n_rows: tl.int32,
    n_time: tl.int32,
    n_out: tl.int32,
    FACTOR: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    out_cols = tl.arange(0, BLOCK_OUT)[None, :]
    row_mask = rows < n_rows
    out_mask = out_cols < n_out

    pad = FACTOR - 1
    kernel_size = 2 * FACTOR - 1
    acc = tl.zeros((BLOCK_ROWS, BLOCK_OUT), dtype=tl.float32)

    for u in range(0, kernel_size):
        t = out_cols * FACTOR + u - pad
        valid = row_mask & out_mask & (t >= 0) & (t < n_time)
        x = tl.load(x_ptr + rows * n_time + t, mask=valid, other=0.0)
        x = tl.maximum(x, 0.0)
        dist = tl.abs(u - pad).to(tl.float32)
        weight = (FACTOR - dist) / FACTOR
        acc += x * weight

    denom = tl.sum(acc, axis=1)[:, None]
    tl.store(denom_ptr + rows, denom, mask=row_mask)

    flipped_cols = n_out - 1 - out_cols
    out = acc / denom
    tl.store(
        out_ptr + rows * n_out + flipped_cols,
        out,
        mask=row_mask & out_mask,
    )


@triton.jit
def _down_tri_relu_bwd_kernel(
    grad_down_ptr,
    x_ptr,
    grad_x_ptr,
    n_rows: tl.int32,
    n_time: tl.int32,
    n_out: tl.int32,
    FACTOR: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    t_offsets = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)[None, :]

    row_mask = row_offsets < n_rows
    t_mask = t_offsets < n_time
    mask = row_mask & t_mask

    pad = FACTOR - 1
    kernel_size = 2 * FACTOR - 1
    j0 = (t_offsets + pad) // FACTOR

    acc = tl.zeros((BLOCK_ROWS, BLOCK_T), dtype=tl.float32)
    for delta in range(2):
        j = j0 - delta
        u = t_offsets - j * FACTOR + pad
        valid = mask & (j >= 0) & (j < n_out) & (u >= 0) & (u < kernel_size)
        dist = tl.abs(u - pad).to(tl.float32)
        weight = (FACTOR - dist) / FACTOR
        grad_down = tl.load(
            grad_down_ptr + row_offsets * n_out + j,
            mask=valid,
            other=0.0,
        )
        acc += grad_down * weight

    x = tl.load(x_ptr + row_offsets * n_time + t_offsets, mask=mask, other=0.0)
    acc = tl.where(x > 0.0, acc, 0.0)
    tl.store(grad_x_ptr + row_offsets * n_time + t_offsets, acc, mask=mask)


class _DownTriReluFlipNormalize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, tri_kernel: torch.Tensor, factor: int):
        if x.is_cuda and x.ndim == 2:
            x = x.contiguous()
            n_rows, n_time = x.shape
            n_out = (n_time - 1) // factor + 1
            out = torch.empty((n_rows, n_out), dtype=x.dtype, device=x.device)
            denom = torch.empty((n_rows, 1), dtype=x.dtype, device=x.device)

            block_rows = 8
            block_out = triton.next_power_of_2(n_out)
            grid = (triton.cdiv(n_rows, block_rows),)
            with torch.cuda.device(x.device):
                _down_tri_relu_flip_norm_fwd_kernel[grid](
                    x,
                    out,
                    denom,
                    n_rows,
                    n_time,
                    n_out,
                    FACTOR=factor,
                    BLOCK_ROWS=block_rows,
                    BLOCK_OUT=block_out,
                    num_warps=8,
                )
        else:
            relu_x = torch.relu(x)
            down = F.conv1d(
                relu_x.unsqueeze(1),
                tri_kernel,
                stride=factor,
                padding=factor - 1,
            ).squeeze(1)
            z = down.flip(-1)
            denom = z.sum(-1, keepdim=True)
            out = z / denom
        ctx.save_for_backward(x, out, denom)
        ctx.factor = factor
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, out, denom = ctx.saved_tensors
        factor = ctx.factor
        grad_z = (grad_out - (grad_out * out).sum(-1, keepdim=True)) / denom
        grad_down = grad_z.flip(-1).contiguous()
        grad_x = torch.empty_like(x)

        n_rows, n_time = x.shape
        n_out = grad_down.shape[-1]
        block_rows = 8
        block_t = 128
        grid = (triton.cdiv(n_rows, block_rows), triton.cdiv(n_time, block_t))
        with torch.cuda.device(x.device):
            _down_tri_relu_bwd_kernel[grid](
                grad_down,
                x,
                grad_x,
                n_rows,
                n_time,
                n_out,
                FACTOR=factor,
                BLOCK_ROWS=block_rows,
                BLOCK_T=block_t,
                num_warps=8,
            )
        return grad_x, None, None

class SubResolutionSampler(nn.Module):
    """
    """
    def __init__(self, dt: float, out_mode: str = "avg"):
        super().__init__()
        inv = 1.0 / float(dt)
        factor = int(round(inv))
        self.factor = factor
        self.dt = float(dt)
        self.out_mode = out_mode

        if factor > 1:
            u = torch.arange(-(factor - 1), factor, dtype=torch.float32)  # [2*factor-1]
            tri = (factor - u.abs()) / factor
            self.register_buffer("tri_kernel", tri.view(1, 1, -1), persistent=False)
        else:
            self.register_buffer("tri_kernel", torch.ones(1, 1, 1), persistent=False)

        if out_mode == "avg":
            self.phi_k = self.down_tri
            self.phi_inv = self.down_pool
        elif out_mode == "sample":
            self.phi_k = self.down_pool
            self.phi_inv = self.down_sample
        else:
            raise NotImplementedError(f"Unknown out_mode: {out_mode}")

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        if self.factor == 1: return x
        x_nct = x.unsqueeze(1)
        y = F.interpolate(x_nct, scale_factor=self.factor, mode="nearest") * self.dt
        return y.squeeze(1)

    def down_pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.factor == 1: return x
        x_nct = x.unsqueeze(1)
        y = F.avg_pool1d(x_nct, kernel_size=self.factor, stride=self.factor) * self.factor
        return y.squeeze(1)

    def down_sample(self, x: torch.Tensor) -> torch.Tensor:
        if self.factor == 1: return x
        return x[..., self.factor - 1 :: self.factor] * self.factor

    def down_tri(self, kernel: torch.Tensor) -> torch.Tensor:
        """
        """
        if self.factor == 1: return kernel
        return F.conv1d(kernel.unsqueeze(1), 
                        self.tri_kernel, 
                        stride=self.factor, 
                        padding=self.factor - 1).squeeze(1) 

    def kernel_postprocess(self, kernel: torch.Tensor) -> torch.Tensor:
        if self.factor > 1 and self.out_mode == "avg" and kernel.is_cuda:
            return _DownTriReluFlipNormalize.apply(kernel, self.tri_kernel, self.factor)
        kernel = torch.relu(kernel)
        kernel = self.phi_k(kernel).flip(-1)
        return kernel / kernel.sum(-1, keepdims=True)
