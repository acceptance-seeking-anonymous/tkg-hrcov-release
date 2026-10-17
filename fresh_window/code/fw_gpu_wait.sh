# phase2_fresh_window / fw_gpu_wait.sh -- sourced by the stage scripts. v79 rule (lead 2026-10-16): serverA GPU 0 ONLY.
# Sets FH_GPU=0 once GPU 0 has > 35 GB free (this track's process is capped at FH_GPU_CAP_GB = 30 GB). Every check logs
# GPU 0 free memory, utilisation and all compute processes on it to $L/gpu_wait.log. GPU 1 is never selected.
fh_gpu_wait() {
  FH_GPU=""
  while [ -z "$FH_GPU" ]; do
    F0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')
    U0=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ' ')
    ID0=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0 | tr -d ' ')
    PR=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | tr -d ' ' | grep "^$ID0," | cut -d, -f2,3 | tr '\n' ' ')
    echo "[gpu-wait] $(date -u +%FT%TZ) gpu0_free=${F0}MiB gpu0_util=${U0}% gpu0_procs=[${PR}] load=$(cut -d' ' -f1-3 /proc/loadavg)" >> "$L/gpu_wait.log"
    if [ "${F0:-0}" -gt 35840 ]; then FH_GPU=0; else sleep "${FH_WAIT_S:-600}"; fi
  done
  echo "[gpu-wait] $(date -u +%FT%TZ) -> physical GPU $FH_GPU (cap ${FH_GPU_CAP_GB:-30} GB)" >> "$L/gpu_wait.log"
}
