import torch
from torch import Tensor


def f(
    masks: Tensor,
    labels: Tensor,
    scores: Tensor,
    threshold: float,
    retained_labels: Tensor = None,
) -> Tensor:
    n, h, w = masks.shape
    c = scores > threshold

    if retained_labels is not None:
        l = torch.zeros_like(c)
        for retained_label in retained_labels:
            l |= labels == retained_label
        c &= l

    masks = masks[c[..., None, None].expand(n, h, w)].reshape(-1, h, w)

    return masks


def g(masks: Tensor, policy: str):
    m, h, w = masks.shape

    if m > 0:
        if policy == "aggregate":
            mask = masks.sum(dim=-3, dtype=torch.bool)
        elif policy == "biggest":
            mask = masks[masks.sum(dim=[-2, -1]).argmax()]
        elif policy in ["left", "right", "top", "bottom", "center"]:
            x, y = torch.zeros(m, device=masks.device), torch.zeros(m, device=masks.device)
            for i in range(m):
                y[i], x[i] = torch.argwhere(masks[i]).float().mean(dim=0)
            if policy == "left":
                i = x.argmin()
            elif policy == "right":
                i = x.argmax()
            elif policy == "top":
                i = y.argmin()
            elif policy == "bottom":
                i = y.argmax()
            elif policy == "center":
                d = torch.stack([x - w / 2, y - h / 2], dim=-1).norm(p=2, dim=-1)
                i = d.argmin()
            else:
                raise ValueError(f"Unknown policy: {policy}")
            mask = masks[i]
        else:
            raise ValueError(f"Unknown policy: {policy}")
    else:
        mask = torch.zeros(h, w, dtype=torch.bool)

    return mask
