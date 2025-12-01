"""PyTorch utilities for model-based RL.

Contains utilities for freezing model parameters during gradient computation
and soft-updating target networks.
"""

import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Union, List


@contextmanager
def frozen_params(modules: Union[nn.Module, List[nn.Module]]):
    """
    Context manager to temporarily freeze parameters of one or more modules.
    
    This allows gradients to flow *through* the module(s) (to inputs) but not
    *to* the module's weights. Useful for model-based policy optimization where
    we backprop through dynamics/reward but don't update their weights.
    
    HOW IT WORKS:
    - Setting requires_grad=False during FORWARD pass means no gradient paths
      are recorded for those parameters in the computational graph.
    - Gradients can still flow THROUGH the frozen module's outputs to upstream
      tensors (like inputs) that had requires_grad=True during forward.
    - The backward() call can happen AFTER exiting the context - it uses the
      graph that was built during forward, not the current requires_grad state.
    
    Args:
        modules: A single nn.Module or a list of nn.Modules to freeze.
        
    Example:
        >>> # Policy optimization through frozen dynamics
        >>> action = policy(z)  # action has gradients
        >>> with frozen_params([dynamics, reward]):
        ...     next_z = dynamics(z, action)  # graph built with frozen params
        ...     r = reward(z, action)
        >>> # After context: params restored, but graph already built
        >>> loss = -r.mean()
        >>> loss.backward()  # gradients flow to policy via action, NOT to dynamics/reward
    """
    # Normalize to list
    if isinstance(modules, nn.Module):
        modules = [modules]
    
    # Save original requires_grad states for all parameters
    original_states = {}
    for module in modules:
        for p in module.parameters():
            original_states[p] = p.requires_grad
    
    # Set all to False
    for p in original_states:
        p.requires_grad = False
        
    try:
        yield
    finally:
        # Restore original states
        for p, state in original_states.items():
            p.requires_grad = state


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """
    Soft update target network parameters towards source network parameters.
    
    Performs Polyak averaging: target = (1 - tau) * target + tau * source
    
    Args:
        target: Target network to update.
        source: Source network to copy from.
        tau: Interpolation factor in [0, 1]. tau=1 means hard copy.
    """
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.lerp_(param.data, tau)
