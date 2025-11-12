import torch


def sample_action_sequences(mean: torch.Tensor, std: torch.Tensor, num_samples: int, clamp_low: float = -1.0, clamp_high: float = 1.0) -> torch.Tensor:
    """Sample Gaussian action sequences.

    Args:
        mean (Tensor[T,A]): Per-timestep action mean.
        std (Tensor[T,A]): Per-timestep action std (already clipped by caller).
        num_samples (int): Number of trajectories to sample (N).
        clamp_low (float): Minimum action value.
        clamp_high (float): Maximum action value.

    Returns:
        Tensor[N,T,A]: Sampled action trajectories.
    """
    T, A = mean.shape
    eps = torch.randn(num_samples, T, A, device=mean.device, dtype=mean.dtype)
    actions = mean.unsqueeze(0) + eps * std.unsqueeze(0)
    return actions.clamp(clamp_low, clamp_high)
