"""Per-episode diagnostics: registration-based suite of analysis modules.

Add new diagnostics by subclassing EpisodeAnalyzer and calling
EpisodeDiagnostics.add_analyzer().  The suite merges all results into a
single dict that can be fed directly to Logger.log().
"""
from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch

from common.logger import get_logger

log = get_logger(__name__)


class EpisodeAnalyzer(ABC):
    """Base class for per-episode analysis modules.

    Each analyzer receives the full flat-state observation sequence of one
    episode and returns a dict of scalars and/or PIL.Image objects for
    wandb logging.
    """

    @abstractmethod
    def analyze(
        self,
        obs_seq: torch.Tensor,  # float32[T, D]
        include_images: bool = True,
    ) -> dict:
        """Analyze a sequence of observations from one episode.

        Args:
            obs_seq (Tensor[T, D]): Flat state observations, float32.
                Must be 2-D; dict/image obs should be filtered out by caller.
            include_images: When False, skip expensive image generation and
                return scalar metrics only.

        Returns:
            dict: Scalar metrics (float/int) and optionally PIL.Image values.
        """
        raise NotImplementedError


class ObservationCoverageAnalyzer(EpisodeAnalyzer):
    """Pairwise-MSE coverage analysis for proprioceptive episode observations.

    Scalars:
        obs_total_variation (float): Sum of consecutive-step MSEs.
            Measures how far the agent moved through observation space.
        obs_mean_pairwise_mse (float): Mean MSE over all unique observation
            pairs.  Measures overall episode diversity.

    Images (when include_images=True):
        obs_coverage_heatmap (PIL.Image): Lower-triangular heatmap of pairwise
            MSE.  Green = similar (low MSE), red = different (high).
            Rows and columns both represent time-steps.
            Downsampled to cfg.episode_diag_max_resolution if T exceeds it.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_pairwise_mse(self, obs_seq: torch.Tensor) -> torch.Tensor:  # float32[T, T]
        """Compute symmetric pairwise MSE matrix.

        Args:
            obs_seq (Tensor[T, D]): Episode observations, float32.

        Returns:
            Tensor[T, T]: Pairwise mean-squared-error, float32, on CPU.
        """
        # Normalize each dimension by its mean absolute value so all dims
        # contribute equally to the MSE (prevents high-magnitude dims from dominating)
        obs = obs_seq.detach().float()
        scale = obs.abs().mean(dim=0).clamp(min=1e-8)  # float32[D]
        obs_normed = obs / scale                        # float32[T, D]
        dists = torch.cdist(obs_normed, obs_normed)     # float32[T, T]
        mse = (dists ** 2) / obs_seq.shape[1]           # float32[T, T]
        return mse.cpu()

    def _downsample_matrix(
        self,
        mat: torch.Tensor,  # float32[T, T]
        target: int,
    ) -> torch.Tensor:      # float32[target, target]
        """Average-pool a square matrix down to [target, target].

        Args:
            mat (Tensor[T, T]): Input matrix.
            target: Target side length after downsampling.

        Returns:
            Tensor[target, target]: Downsampled matrix.
        """
        T = mat.shape[0]
        if T <= target:
            return mat
        kernel = T // target
        # avg_pool2d requires [N, C, H, W]
        mat_4d = mat.unsqueeze(0).unsqueeze(0)                       # [1, 1, T, T]
        pooled = torch.nn.functional.avg_pool2d(mat_4d, kernel_size=kernel)
        return pooled.squeeze(0).squeeze(0)                          # [target, target]

    def _make_heatmap(self, mse_matrix: torch.Tensor, T: int):  # → PIL.Image
        """Render lower-triangular pairwise MSE as a heatmap PNG.

        Green (low MSE) = similar observations; red (high MSE) = different.
        Upper triangle is masked (symmetric matrix — no new information).

        Args:
            mse_matrix (Tensor[T, T]): Symmetric pairwise MSE matrix.
            T: Original episode length (used in title only).

        Returns:
            PIL.Image: Rendered heatmap.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage

        max_res = self.cfg.episode_diag_max_resolution
        mat = self._downsample_matrix(mse_matrix, max_res)   # [R, R]
        R = mat.shape[0]
        mat_np = mat.numpy()                                  # [R, R]

        # Mask upper triangle and diagonal — symmetric, no new information
        upper_mask = np.triu(np.ones((R, R), dtype=bool), k=0)
        # Log scale: clamp to [1, ∞) so log10 ≥ 0, then use data-driven vmax
        mat_log = np.log10(np.clip(mat_np, a_min=1.0, a_max=None))
        mat_masked = np.ma.array(mat_log, mask=upper_mask)

        # Spread colormap across observed range with vmin=0
        vmax = float(np.nanmax(mat_log[~upper_mask])) if (~upper_mask).any() else 1.0
        vmax = max(vmax, 0.01)  # avoid degenerate range

        fig, ax = plt.subplots(figsize=(5, 5))
        # turbo: many distinct colors (blue→cyan→green→yellow→red) for high MSE resolution
        im = ax.imshow(
            mat_masked, aspect='auto', cmap='turbo',
            origin='upper', interpolation='nearest',
            vmin=0.0, vmax=vmax,
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='log₁₀(MSE)')
        ax.set_xlabel('Step j')
        ax.set_ylabel('Step i')
        ax.set_title(f'Obs coverage — pairwise MSE  (T={T})')
        fig.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return PILImage.open(buf).copy()   # .copy() detaches from the BytesIO buffer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        obs_seq: torch.Tensor,  # float32[T, D]
        include_images: bool = True,
    ) -> dict:
        """Run coverage analysis on one episode.

        Args:
            obs_seq (Tensor[T, D]): Episode state observations, float32.
            include_images: When True, also generate the pairwise MSE heatmap.

        Returns:
            dict with keys:
                obs_total_variation (float)
                obs_mean_pairwise_mse (float)
                obs_coverage_heatmap (PIL.Image, only when include_images=True)
        """
        assert obs_seq.ndim == 2, f"Expected obs_seq [T, D], got {obs_seq.shape}"
        T, D = obs_seq.shape
        assert T > 1, f"Need at least 2 observations, got T={T}"

        mse_matrix = self._compute_pairwise_mse(obs_seq)   # float32[T, T]

        # Sum of consecutive-step MSEs — path length in observation space
        consec_mse = mse_matrix.diagonal(offset=1)          # float32[T-1]
        total_variation = float(consec_mse.sum().item())

        # Mean over all unique pairs (lower triangle, k < 0)
        num_pairs = T * (T - 1) // 2
        lower_sum = float(torch.tril(mse_matrix, diagonal=-1).sum().item())
        mean_pairwise_mse = lower_sum / max(num_pairs, 1)

        result: dict = {
            'obs_total_variation': total_variation,
            'obs_mean_pairwise_mse': mean_pairwise_mse,
        }

        if include_images:
            try:
                result['obs_coverage_heatmap'] = self._make_heatmap(mse_matrix, T)
            except Exception as exc:
                log.warning('obs_coverage_heatmap generation failed: %s', exc)

        return result


class ValueTrajectoryAnalyzer(EpisodeAnalyzer):
    """Per-timestep V-function analysis along an episode trajectory.

    Encodes raw observations through the world model encoder, evaluates all
    V-heads, and produces a line plot of mean/std/min/max value over time.

    Scalars:
        v_traj_mean (float): Mean value across all heads and timesteps.
        v_traj_std  (float): Mean per-timestep std across V-heads.

    Images (when include_images=True):
        v_trajectory_plot (PIL.Image): Line plot of V(t) with mean ± std band
            and min/max lines.
    """

    def __init__(self, cfg, world_model):
        self.cfg = cfg
        self.world_model = world_model

    @torch.no_grad()
    def analyze(
        self,
        obs_seq: torch.Tensor,  # float32[T, D]
        include_images: bool = True,
    ) -> dict:
        """Encode observations and evaluate V-heads along the trajectory.

        Args:
            obs_seq (Tensor[T, D]): Episode state observations, float32.
            include_images: When True, generate the V-trajectory line plot.

        Returns:
            dict with scalar summaries and optionally a PIL.Image plot.
        """
        assert obs_seq.ndim == 2, f"Expected obs_seq [T, D], got {obs_seq.shape}"
        T = obs_seq.shape[0]

        device = next(self.world_model.parameters()).device
        obs_gpu = obs_seq.to(device)  # float32[T, D]

        z = self.world_model.encode(obs_gpu)              # float32[T, L]
        # Per-head values: float32[num_q, T, 1]
        v_all = self.world_model.V(z, return_type='all_values')
        v_all = v_all.squeeze(-1).cpu()                    # float32[num_q, T]

        v_mean = v_all.mean(dim=0)                         # float32[T]
        v_std = v_all.std(dim=0)                           # float32[T]
        v_min = v_all.min(dim=0).values                    # float32[T]
        v_max = v_all.max(dim=0).values                    # float32[T]

        result: dict = {
            'v_traj_mean': float(v_mean.mean().item()),
            'v_traj_std': float(v_std.mean().item()),
        }

        if include_images:
            v_all_np = v_all.numpy()
            v_mean_np = v_mean.numpy()
            v_std_np = v_std.numpy()
            try:
                result['v_trajectory_plot'] = self._make_plot(
                    v_all_np, v_mean_np, T,
                )
            except Exception as exc:
                log.warning('v_trajectory_plot generation failed: %s', exc)
            try:
                result['v_std_trajectory_plot'] = self._make_std_plot(
                    v_std_np, T, num_q=v_all.shape[0],
                )
            except Exception as exc:
                log.warning('v_std_trajectory_plot generation failed: %s', exc)

        return result

    @staticmethod
    def _make_plot(
        v_all: np.ndarray,    # [num_q, T]
        v_mean: np.ndarray,   # [T]
        T: int,
    ):
        """Render per-head V lines with the mean and std bands overlaid.

        Args:
            v_all: Per-head values, shape [num_q, T].
            v_mean: Mean across heads per timestep, shape [T].
            T: Episode length.

        Returns:
            PIL.Image: Rendered plot.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage

        num_q = v_all.shape[0]
        v_std = v_all.std(axis=0)  # [T]
        steps = np.arange(T)
        fig, ax = plt.subplots(figsize=(7, 4))

        # Individual heads in light, distinct colors
        cmap = plt.get_cmap('tab10')
        for i in range(num_q):
            ax.plot(steps, v_all[i], color=cmap(i % 10), alpha=0.4,
                    linewidth=0.7, label=f'head {i}')

        # Mean as a bold black line on top
        ax.plot(steps, v_mean, color='black', linewidth=2.0, label='mean')
        # mean ± 1σ and mean ± 3σ bands
        ax.plot(steps, v_mean + v_std, color='black', linewidth=1.0,
                linestyle='--', alpha=0.6, label='mean±1σ')
        ax.plot(steps, v_mean - v_std, color='black', linewidth=1.0,
                linestyle='--', alpha=0.6)
        ax.plot(steps, v_mean + 3 * v_std, color='black', linewidth=0.8,
                linestyle=':', alpha=0.4, label='mean±3σ')
        ax.plot(steps, v_mean - 3 * v_std, color='black', linewidth=0.8,
                linestyle=':', alpha=0.4)

        ax.set_xlabel('Timestep')
        ax.set_ylabel('V')
        ax.set_title(f'V-head trajectory  (T={T}, heads={num_q})')
        ax.legend(loc='best', fontsize='x-small', ncol=2)
        fig.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return PILImage.open(buf).copy()


    @staticmethod
    def _make_std_plot(
        v_std: np.ndarray,   # [T]
        T: int,
        num_q: int,
    ):
        """Render inter-head V standard deviation over episode timesteps.

        Args:
            v_std: Per-timestep std across V-heads, shape [T].
            T: Episode length.
            num_q: Number of V-heads (for title annotation).

        Returns:
            PIL.Image: Rendered plot.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage

        steps = np.arange(T)
        fig, ax = plt.subplots(figsize=(7, 4))

        ax.plot(steps, v_std, color='tab:red', linewidth=1.5)
        ax.fill_between(steps, 0, v_std, color='tab:red', alpha=0.15)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('V std (across heads)')
        ax.set_title(f'V-head disagreement  (T={T}, heads={num_q})')
        ax.set_ylim(bottom=0)
        fig.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return PILImage.open(buf).copy()


class EpisodeDiagnostics:
    """Runs a configurable suite of per-episode analyses.

    Usage::

        diag = EpisodeDiagnostics(cfg)
        results = diag.analyze(obs_seq)                        # scalars + images
        scalars = diag.analyze(obs_seq, include_images=False)  # scalars only

    Add new diagnostics at any time::

        diag.add_analyzer(MyCustomAnalyzer(cfg))

    The result dict is suitable for direct ingestion by Logger.log() — scalars
    log as time-series charts, PIL.Image values log as wandb image panels.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._analyzers: list[EpisodeAnalyzer] = [
            ObservationCoverageAnalyzer(cfg),
        ]

    def add_analyzer(self, analyzer: EpisodeAnalyzer) -> None:
        """Append an additional analyzer to the suite.

        Args:
            analyzer: EpisodeAnalyzer instance to register.
        """
        self._analyzers.append(analyzer)

    def analyze(
        self,
        obs_seq: torch.Tensor,  # float32[T, D]
        include_images: bool = True,
    ) -> dict:
        """Run all registered analyzers on one episode observation sequence.

        Only handles flat 2-D state tensors.  Caller must pre-filter dict or
        image observations (returns empty dict for non-2D input).

        Args:
            obs_seq (Tensor[T, D]): Episode state observations, float32.
            include_images: When False, no images are generated (faster path
                suitable for dense training-episode logging).

        Returns:
            dict: Merged results keyed by metric name.  Values are float/int
                scalars or PIL.Image objects for wandb.
        """
        if obs_seq.ndim != 2:
            log.debug(
                'EpisodeDiagnostics.analyze skipped: expected 2-D obs_seq, got %s',
                tuple(obs_seq.shape),
            )
            return {}

        results: dict = {}
        for analyzer in self._analyzers:
            try:
                results.update(analyzer.analyze(obs_seq, include_images=include_images))
            except Exception as exc:
                log.warning('Analyzer %s failed: %s', type(analyzer).__name__, exc)
        # Prefix all keys so they appear under episode_diagnostics/ in wandb
        return {f'episode_diagnostics/{k}': v for k, v in results.items()}
