import torch
import torch.nn as nn

class CustomDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in range [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If probability is 0 or model is in eval mode, dropout is inactive
        if self.p == 0 or not self.training:
            return x
        
        # Create a binary mask of the same shape as input and proper scaling
        mask = (torch.rand(x.shape, device=x.device) > self.p).float()
        
        mask = mask / (1.0 - self.p)
        
        return x * mask