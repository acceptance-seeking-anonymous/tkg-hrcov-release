# fresh_window/fw_env.sh -- source before running the fresh-window scripts from a localized copy (see README.md).
#   H   harness root (the scripts' hard-coded root, /home/anon/cs_scratch/open_ideas_harness in this release, is rewritten to $H)
#   FW  fresh-window track directory (data, stage, logs, CountTRuCoLa runs); default $H/tracks/phase2_fresh_window
#   B   external-comparator directory with the CountTRuCoLa clone (repos/) and its venv (env/ct); default $H/tracks/v78_external
# If WORK_DIR is set (configs/paths.env of the repository), it is the default for H.
export H="${H:-${WORK_DIR:?set H (or WORK_DIR) to the harness root}}"
export FW="${FW:-$H/tracks/phase2_fresh_window}"
export B="${B:-$H/tracks/v78_external}"
echo "fresh window: H=$H FW=$FW B=$B"
