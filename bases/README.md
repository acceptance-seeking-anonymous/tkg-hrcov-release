# bases/

`fetch_upstream.sh` clones TiRGN, CognTKE and TGB at the pinned commits into `$BASES_DIR`, applies the patches, checks every Python file of the patched trees against the MD5 lists, and places the trees where the scripts expect them (one TiRGN copy per track, as in the original runs).

| file | content |
|---|---|
| `patches/tirgn_stage.patch` | TiRGN as used for tkgl-polecat, tkgl-smallpedia (pilot 2, main test, reviewer-requested analyses) and GDELT25X: `dgl_shim.py` (pure-PyTorch implementation of the DGL graph operations TiRGN calls; DGL has no build for torch 2.8 / CUDA 12.8), small changes in `rgcn/` and `src/rrgcn.py` (shim import, CPU fallback), `src/main.py` (shim import, `torch.load(..., weights_only=False)`, score-dump hooks switched on by `DUMP_SCORES` / `DUMP_PATH`, and `DUMP_SPLIT=train|valid` to dump those splits from a saved checkpoint), `src/score_dump.py` (full-score dump: float32 log-score rows plus gold score and raw rank). The TGB 2.0 training loops themselves are in `src/pilot_smallpedia/sp_tirgn.py`, `src/hardening/hd_tirgn.py` and `src/forward_holdout/fh_tirgn_train.py`, which import the patched tree |
| `patches/tirgn_gdelt_dump.patch` | the same base changes plus the helpers used to dump ICEWS18 / GDELT scores (`add_score_dump.py`, `add_split_dump.py`, `compat_patch.py`, `src/get_history_fast.py`, a longer `src/score_dump.py`) |
| `patches/*.md5` | MD5 of every `.py` file in the patched trees (equal to the trees used in the paper) |

Which code trains each base:

| base | dataset(s) | code |
|---|---|---|
| RE-GCN, LogCL (pilot configurations) | tkgl-polecat, tkgl-smallpedia, ICEWS18, GDELT | `src/harness/base_dump.py` (re-implementations in `src/harness/regcn.py`, `logcl.py`) |
| RE-GCN, LogCL (tkgl-icews, window training) | tkgl-icews | `src/pilot_icews/pi_base.py` |
| RE-GCN / LogCL re-tuned | tkgl-smallpedia | `src/hardening/hd_base_train.py` |
| RE-GCN, TGB 2.0 configuration | tkgl-smallpedia | `src/presub/g3_tgb_regcn.py` + TGB example script (fetched), dump `g3_dump.py` |
| TiRGN | tkgl-smallpedia | `src/pilot_smallpedia/sp_tirgn.py` (seeds: `src/v76/m6_tirgn.sh`; validation-trained: `src/v76/sp_tirgn_valfit.py`) |
| TiRGN | tkgl-polecat | `src/hardening/hd_tirgn.py`, `hd_tirgn_replay.py` (second seed: `src/c4/c4_chain.sh`) |
| TiRGN | GDELT25X | `src/forward_holdout/fh_tirgn_train.py`, `fh_tirgn_rows.py` |
| TiRGN | ICEWS18, GDELT | checkpoints from before this study, dumped with the `tirgn_gdelt_dump` tree (`data/README.md`) |
| CognTKE | tkgl-polecat | `src/g8_cogntke/g8_train.py`, `g8_dump.py` (`g8_setup.sh`, `g8_train.sh`, `g8_chain.sh`) |
