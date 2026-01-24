import torch


def sample_action_sequences(
    mean: torch.Tensor,
    std: torch.Tensor,
    num_samples: int,
    clamp_low: float = -1.0,
    clamp_high: float = 1.0
) -> torch.Tensor:
    """Sample Gaussian action sequences with batch support.

    Args:
        mean (Tensor[B, T, A]): Per-timestep action mean for each batch element.
        std (Tensor[B, T, A]): Per-timestep action std (already clipped by caller).
        num_samples (int): Number of trajectories to sample (N) per batch element.
        clamp_low (float): Minimum action value.
        clamp_high (float): Maximum action value.

    Returns:
        Tensor[B, N, T, A]: Sampled action trajectories.
    """
    B, T, A = mean.shape
    # Sample noise: [B, N, T, A]
    eps = torch.randn(B, num_samples, T, A, device=mean.device, dtype=mean.dtype)
    # mean/std: [B, T, A] -> [B, 1, T, A] for broadcasting
    actions = mean.unsqueeze(1) + eps * std.unsqueeze(1)  # float32[B, N, T, A]
    return actions.clamp(clamp_low, clamp_high)
