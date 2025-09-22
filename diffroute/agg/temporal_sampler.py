import torch
import torch.nn as nn
import torch.nn.functional as F

class SubResolutionSampler(nn.Module):
    """
    """
    def __init__(self, dt: float, out_mode: str = "avg"):
        super().__init__()
        inv = 1.0 / float(dt)
        factor = int(round(inv))
        self.factor = factor
        self.dt = float(dt)

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