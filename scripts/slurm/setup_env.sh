#!/usr/bin/env bash
# One-time environment setup on the cluster (needs internet).
# Installs uv if missing, a uv-managed Python 3.12 (the system python3 on
# Ubuntu 22.04 is 3.10, too old for the pinned numpy/scipy) and the pinned
# requirements into the venv for this machine's architecture: <repo>/.venv on
# x86_64, <repo>/.venv-<arch> otherwise (job.sbatch picks the same one). Safe
# to re-run after requirements change.
#
#   bash scripts/slurm/setup_env.sh                     # x86_64: login node
#   srun -p a40q --gres=gpu:1 -c 8 --propagate=NONE \
#       bash scripts/slurm/setup_env.sh                 # aarch64: on an a40q node
#
# On a GPU node the final check also runs a conv on the GPU, which catches a
# torch build without kernels for that GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."

arch="$(uname -m)"
venv=.venv
[ "$arch" = x86_64 ] || venv=".venv-$arch"

# ~/.local/bin is shared with nodes of the other architecture, so a uv found
# there may not run here; non-x86_64 copies go to ~/.local/bin/<arch>.
uv_ok() { command -v uv >/dev/null && uv --version >/dev/null 2>&1; }
uv_dir="$HOME/.local/bin"
[ "$arch" = x86_64 ] || uv_dir="$HOME/.local/bin/$arch"
uv_ok || export PATH="$uv_dir:$PATH"
if ! uv_ok; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 UV_INSTALL_DIR="$uv_dir" sh
fi

# Use the system CA store: uv's bundled roots reject the cluster's TLS
# certificates ("invalid peer certificate: UnknownIssuer"), curl's don't.
export UV_SYSTEM_CERTS=1

uv python install 3.12
mkdir -p runs/slurm  # sbatch --output dir must exist at submit time
[ -d "$venv" ] || uv venv --python 3.12 "$venv"
# Default PyPI torch wheel = CUDA 13 build; needs NVIDIA driver >= 580
# (a100q and a40q: 615.71, checked 2026-10-05).
uv pip install --python "$venv/bin/python" -r requirements.txt
uv pip install --python "$venv/bin/python" -e . --no-deps

"$venv/bin/python" -c "
import platform, torch, ultralytics
print(platform.machine(), '| torch', torch.__version__, 'cuda', torch.version.cuda,
      '| ultralytics', ultralytics.__version__, '| archs', torch.cuda.get_arch_list())
if torch.cuda.is_available():
    torch.nn.Conv2d(3, 8, 3).cuda()(torch.randn(2, 3, 64, 64, device='cuda')).sum().item()
    print('GPU check passed:', torch.cuda.get_device_name())
"
