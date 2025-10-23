#!/usr/bin/env bash
# set -euo pipefail

# Choose Python (prefer the tdmpc2 conda env directly; no conda activate needed)
PYTHON="/space/thomasevers/conda-envs/tdmpc2/bin/python"
if [[ ! -x "$PYTHON" ]]; then
	PYTHON="python"
fi

RUN_NAME="compile-fullupdate"

# Resolve repository root from this script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Timestamped output directory under nsys_reports/
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/nsys_reports/$TIMESTAMP-$RUN_NAME"
mkdir -p "$OUT_DIR"


# cuda, cuda-hw, cudnn cublas cublas-verbose cusparse-verbose, cudla, cudla-verbose, cusolver, opengl, cusolver-verbose, opengl-annotations openacc, openmp, osrt mpi, nvvideo, nvtx, dx11,dx11-annotations dx12,dx12-annotations tegra-accelerators, ucx, openxr, oshmem, openxr-annotations, python-gil, gds(beta) wddm, vulkan, vulkan-annotations, none
# --gpu-metrics-devices=all 
# --sample=none \
# Run Nsight Systems profiling; outputs go into $OUT_DIR
export MUJOCO_GL=egl
export EGL_PLATFORM=surfaceless
# export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

CUDA_VISIBLE_DEVICES=0 nohup nsys profile \
	--trace=cuda,nvtx,cudnn,cublas,cudla \
	--delay=800 \
	--duration=20 \
	--sample=none \
	--force-overwrite=true	-o "$OUT_DIR/profile" \
	"$PYTHON" -u tdmpc2/train.py \
		task=reacher-easy \
		obs=state \
		compile=True \
		compile_type=reduce-overhead \
		nvtx_profiler=true \
		enable_wandb=true \
		steps=2000000 \
		model_size=5\
	> "$OUT_DIR/profile_$RUN_NAME.log" 2>&1 &

echo "Nsight profiling started. Reports: $OUT_DIR"
