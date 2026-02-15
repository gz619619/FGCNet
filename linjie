import torch
import torch.nn.functional as F


def aff_to_adj(
    last_layer_data_src,
    fuzzy_center=None,
    fuzzy_sigma=None,
    beta=0.3,             
    eps=1e-6,
    device='cuda:2'
):
    x = F.normalize(last_layer_data_src, dim=-1)

    if x.shape[0] != last_layer_data_src.shape[0]:
        x = x.transpose(0, 1)
    N, D = x.shape

    if fuzzy_center is None:
        fuzzy_center = torch.zeros((1, D), device=x.device)
    if fuzzy_sigma is None:
        fuzzy_sigma = torch.ones((1, D), device=x.device)

    x1 = x.unsqueeze(1)   # [N, 1, D]
    x2 = x.unsqueeze(0)   # [1, N, D]
    dist = (x1 - x2) ** 2
    sim = torch.exp(-dist / (2 * (fuzzy_sigma ** 2 + eps)))
    adj = sim.mean(dim=-1)  # [N, N]

    mean = adj.mean(dim=1, keepdim=True)
    std = adj.std(dim=1, keepdim=True)
    tau = mean + beta * std

    mask = (adj >= tau).float()
    adj = adj * mask

    rowsum = adj.sum(dim=1, keepdim=True)
    D_inv_sqrt = torch.rsqrt(rowsum + eps)
    adj_normalized = adj * D_inv_sqrt * D_inv_sqrt
    adj_normalized += torch.eye(N, device=device)

    return adj_normalized.to(device)
