import torch
import torch.nn as nn


class IoULoss(nn.Module):
    def __init__(self, eps=1e-6, reduction="mean"):
        super().__init__()
        self._eps = eps

        if reduction not in ("none", "mean", "sum"):
            raise ValueError("bad reduction type")
        self._mode = reduction

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # unpack center format
        ax, ay, aw, ah = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bx, by, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

        # convert to corners
        a_x1 = ax - aw * 0.5
        a_y1 = ay - ah * 0.5
        a_x2 = ax + aw * 0.5
        a_y2 = ay + ah * 0.5

        b_x1 = bx - bw * 0.5
        b_y1 = by - bh * 0.5
        b_x2 = bx + bw * 0.5
        b_y2 = by + bh * 0.5

        # intersection box
        xx1 = torch.maximum(a_x1, b_x1)
        yy1 = torch.maximum(a_y1, b_y1)
        xx2 = torch.minimum(a_x2, b_x2)
        yy2 = torch.minimum(a_y2, b_y2)

        w = xx2 - xx1
        h = yy2 - yy1

        w = torch.clamp(w, min=0)
        h = torch.clamp(h, min=0)

        inter = w * h

        # areas
        area_a = aw * ah
        area_b = bw * bh

        # union
        denom = area_a + area_b - inter

        iou_val = inter / (denom + self._eps)
        out = 1.0 - iou_val

        if self._mode == "none":
            return out
        if self._mode == "sum":
            return torch.sum(out)
        return torch.mean(out)