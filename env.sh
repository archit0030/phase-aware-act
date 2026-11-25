# usage: source env.sh
# activates the venv created by setup.sh and puts the repo + detr/ on the path
AR523_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$AR523_DIR/.venv}"
source "$VENV/bin/activate"
export PYTHONPATH="$AR523_DIR:$AR523_DIR/detr"
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl          # headless rendering; use osmesa if EGL is not available
