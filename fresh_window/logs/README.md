# fresh_window_logs_v2 -- logs supporting the fresh-window chronology

This bundle holds the logs and small outputs that support `fresh_window_chronology.md/.json` and `boundary_reconciliation.md`.
Everything was copied read-only from the compute hosts after the experiment; nothing was recomputed. The bundle contains
no data split, no dump or ranking file, and no model.

## Layout

| path | content |
|---|---|
| `track/` | serverA `tracks/phase2_fresh_window/`: `logs/` (every stage log, including `stage1_chain.log` and `stage2_test.log` with per-step completion times), `freeze/` (the host freeze manifest and the rule documents installed for the freeze, anonymised), `GDELTFW/run_stage1|run_stage2/` and `GDELTFW/work/**/*.json` (step summaries), `ct/C0|C1|C2/` (CountTRuCoLa run records, stage logs, memory records, dump metadata, and the Stage-1 results CSV written by the released code), `data/GDELTFW/` (build summary, missing-export list, `stat.txt`, `timestamp2id.txt`; no split files) |
| `jobs/<job_id>/` | the 12 serverA job directories of the fresh window: `cmd.sh` (exact command), `nohup.out`, and the outputs harvested to `h/` or `out/`. `jobs/job_status.tsv` gives each job's end time, state and exit code. |
| `scans/<host>_<job_id>/` | the uninspected-check scans of serverA, serverB and serverC (`scan.json`, `scan.err`, `scan_mode.txt`, `cmd.sh`) |
| `inventory/` | `track_files.tsv` (mtime UTC, bytes, path) for every file of the fresh-window track except the raw GDELT downloads; `job_dir_files.tsv` for the 12 job directories |
| `orchestration_commands.tsv` | UTC time, kind, intent and command prefix of every orchestration command and job submission of the fresh window, from the command log of the analysis workstation |
| `boundary_sources/` | GDELT25X records used for the boundary reconciliation: the serverB build job (wrapper with the four date arguments, builder, summary, log, missing-export list, directory listing), the serverA snapshot's `timestamp2id.txt`/`stat.txt`/checksums, and the code-release copies of the missing-export list and `GDELT25X.sha256` |
| `track_extra/` | `grep num_times` of the staged TiRGN `src/rrgcn.py` (TiRGN's time-encoder input size) |

## Anonymisation (applied to every file)

- User names and home paths are replaced by `[REDACTED:~ANON]` (e.g. `/home/[REDACTED:~ANON]`).
- Host addresses are replaced by the labels `serverA`, `serverB`, `serverC`; any other IPv4 address by `[REDACTED:IP]`. The host name by `[REDACTED:HOST]`, GPU UUIDs by `GPU-[REDACTED]`.
- Job-runner directory names are shortened to `.../cs_scratch/jobs/<job_id>`, and runner-specific identifiers are replaced by `[REDACTED:PLATFORM]`.
- The runner's own status files (`_submitted`, `_status.json`, `_pgid`, `_staged`, `job.sh`) are not included. Their end time, state and exit code are in `jobs/job_status.tsv`.
- Job ids are kept.
- Because of these replacements, the copies of the rule documents in `track/freeze/` are not byte-identical to the registered files. The canonical, hashed versions are in the code release (`fresh_window_code_v1/registration/`), and the SHA-256 manifest refers to those.

## Times

All times are UTC. Some third-party log lines (CountTRuCoLa / Ray) carry serverA local time, UTC+8; the chronology converts
them.

`SHA256SUMS` lists every file of this bundle except itself (`sha256sum -c SHA256SUMS`).

## Version 2 (anonymised for the public release)

This is version 2 of the bundle. Every file of version 1 is here under the same path; `fresh_window/ADDITIONS.tsv` of the code
release gives, per file, the SHA-256 in version 1 and in version 2 and the number of replacements. Changes against version 1:

- Account names of other users of the shared servers are replaced by neutral tokens (`user1`, `user2`, ...); any remaining home
  directory of another account by `/home/otheruser`. This bundle's own account keeps the token of version 1.
- Process listings: in `ps` output, the lines of other accounts and of the system are replaced by one marker line per listing
  (only this study's own processes are kept); the GPU process rows of `nvidia-smi` tables, which show a PID and a process name
  but not the owner, are replaced by one marker row.
- Bare server numbers are replaced by the labels `serverA`, `serverB`, `serverC`, and analysis-platform identifiers by neutral
  text, as in the rest of the release.
- `SHA256SUMS` is recomputed for version 2.

**Rule documents.** The copies of the rule documents under `track/freeze/` are for reading only: they are anonymised and do not
match the registered hashes. The canonical copies are in `registration/` of the code release (identical copies in
`fresh_window/registration/`); their hashes are verified by `registration/verify_registration_hashes.py`.
