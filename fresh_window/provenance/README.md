# fresh_window/provenance/ -- PROVENANCE ONLY, NOT RUNNABLE

`steps/` holds the orchestration and local-analysis cells of the fresh-window comparison in the order they ran, including the
failed attempts (see `../FILES.md`, section Provenance). They were executed in the authors' analysis environment and call its
job and file-store interfaces (remote job submission, artifact store), which are not part of this repository; they cannot be run
as they stand and are kept as the record of what was executed.

Release changes: the account name was replaced by `anon`, server addresses and bare host numbers by the labels
serverA/serverB/serverC, and analysis-platform identifiers by neutral text (the tool name by "the authors' analysis
environment", the job-store path segment by `.jobs`, account-store organisation and project ids by `<id>`, language-model
identifiers by `<LLM model id>`) (fresh-window rules, `../verify_fresh_window.py`); nothing else was changed. Internal job
and store identifiers are kept because the cells refer to them.

The runnable path is the plain-shell order of commands in `../FILES.md` ("Order of commands to reproduce"), run from a localized
copy made with `../fw_localize.sh` (see `../README.md`).
