import torch
import torch.nn as nn
import torch.nn.functional as F

class SubResolutionSampler():
    def __init__(self, dt, out_mode="avg"):
        self.factor = int(1/dt)
        self.dt = dt
        self.out_mode = out_mode
        
        if out_mode == "avg":
            self.phi_k = self.down_tri
            self.phi_inv = self.down_pool
        elif out_mode == "sample":
            self.phi_k = self.down_pool
            self.phi_inv = self.down_sample
        else:
            raise NotImplemetedError

        if dt == 1: 
            self.phi_k = lambda x:x
            self.phi_inv = lambda x:x
            
    def conv(self, x, w):
        return F.conv1d(x, w, padding=w.shape[-1]-1)[..., :x.shape[-1]].squeeze(0)

    def phi(self, x):
        return x.repeat_interleave(self.factor, -1) * self.dt

    def down_pool(self, x):
        return F.avg_pool1d(x, kernel_size=self.factor, stride=self.factor)

    def down_sample(self, x):
        return x[..., self.factor-1::self.factor]

    def down_tri(self, kernel):
        u = torch.arange(-self.factor+1, self.factor, dtype=kernel.dtype, device=kernel.device)
        tri_weights = (self.factor - u.abs()).float()  # shape [2*factor-1]
        tri_weights = tri_weights / (self.factor**2)
        pad = self.factor - 1

        kernel = kernel.unsqueeze(1)
        kernel_padded = F.pad(kernel, (pad, pad))
        kernel_conv = F.conv1d(kernel_padded, tri_weights.view(1, 1, -1))
        kernel_d = kernel_conv[..., pad::self.factor]
        kernel_d = kernel_d.squeeze(1)#.view(c,t)
        return kernel_d        
        
class SubResolutionSamplerNew(nn.Module):
    def __init__(self, dt, out_mode="avg"):
        super(SubResolutionSampler, self).__init__()
        self.factor = int(1 / dt)
        self.dt = dt
        self.out_mode = out_mode

        if out_mode == "avg":
            self.phi_k = self.down_tri
            self.phi_inv = self.down_pool
        elif out_mode == "sample":
            self.phi_k = self.down_pool
            self.phi_inv = self.down_sample
        else:
            raise NotImplementedError

        if dt == 1:
            self.phi_k = lambda x: x
            self.phi_inv = lambda x: x

        # Precompute the triangular kernel weights only once for "avg" mode.
        if out_mode == "avg":
            # Create a symmetric range: [-factor+1, ..., factor-1]
            u = torch.arange(-self.factor + 1, self.factor, dtype=torch.float32)
            tri_weights = (self.factor - u.abs()) / (self.factor ** 2)
            self.register_buffer("tri_weights", tri_weights.view(1, 1, -1))
            self.pad = self.factor - 1  # symmetric padding amount
            self.crop = (self.pad + self.factor - 1) // self.factor

    def conv(self, x, w):
        return F.conv1d(x, w, padding=w.shape[-1] - 1)[..., :x.shape[-1]].squeeze(0)

    def phi(self, x):
        return x.repeat_interleave(self.factor, dim=-1) * self.dt

    def down_pool(self, x):
        return F.avg_pool1d(x, kernel_size=self.factor, stride=self.factor)

    def down_sample(self, x):
        return x[..., self.factor - 1 :: self.factor]

    def down_tri(self, kernel):
        conv_out = F.conv1d(kernel.unsqueeze(1), self.tri_weights, 
                            stride=self.factor, padding=self.pad)
        return conv_out[..., self.crop:].squeeze(1)

def test_subresolution_sampler(B=16, T=36, dt=0.25):
    kernel = torch.randn(B, T)
    sampler_avg = SubResolutionSampler(dt, out_mode="avg")
    sampler_avg_baseline = SubResolutionSamplerOld(dt, out_mode="avg")
    out = sampler_avg.down_tri(kernel)
    out_base = sampler_avg.down_tri(kernel)
    assert torch.equal(out, out_base)
    print("Test passed")
