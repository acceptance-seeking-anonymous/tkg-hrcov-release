SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src}"   # repository source tree
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
# phase2_forward_holdout / fh_gpu_wait.sh -- sourced by the stage scripts.  Sets FH_GPU (physical card index) once a card is
# allowed under the GPU-sharing rule (2026-10-14):
#   GPU 1 (shared by other tracks): only if > 35 GB free at dispatch; this track's process is capped at 30 GB.
#   GPU 0: only once nvidia-smi shows NO G8 process (cmdline containing g8_ / cogntke) on it (and > 35 GB free).
# Checks every FH_WAIT_S seconds (default 1800 = 30 min); every check is logged to $L/gpu_wait.log.
fh_gpu_wait() {
  FH_GPU=""
  while [ -z "$FH_GPU" ]; do
    F1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${GPU:-0} | tr -d ' ')
    F0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${GPU:-0} | tr -d ' ')
    U0=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i ${GPU:-0} | tr -d ' ')
    NG8=0
    for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | tr -d ' ' | grep "^$U0," | cut -d, -f2); do
      ps -o args= -p "$p" 2>/dev/null | grep -qiE 'g8_|cogntke' && NG8=$((NG8 + 1))
    done
    echo "[gpu-wait] $(date -u +%FT%TZ) gpu0_free=${F0}MiB g8_procs_on_gpu0=${NG8} gpu1_free=${F1}MiB" >> "$L/gpu_wait.log"
    if [ "${F1:-0}" -gt 35840 ]; then FH_GPU=1
    elif [ "$NG8" -eq 0 ] && [ "${F0:-0}" -gt 35840 ]; then FH_GPU=0
    else sleep "${FH_WAIT_S:-1800}"; fi
  done
  echo "[gpu-wait] $(date -u +%FT%TZ) -> physical GPU $FH_GPU (cap ${FH_GPU_CAP_GB:-30} GB)" >> "$L/gpu_wait.log"
}
