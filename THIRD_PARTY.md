# Third-party code and data

No third-party source code is included in this repository. Code without a licence file is fetched from its authors' repositories at the commits used in the paper (`bases/fetch_upstream.sh`; the two external methods of the standalone and external comparisons and the 2026 GDELT window: `bases/fetch_external_methods.sh`); our changes are distributed as patches.

| component | source | commit / version | licence (as found upstream) | how it is used |
|---|---|---|---|---|
| TiRGN | https://github.com/Liyyy2122/TiRGN | `4b74c98b24b200a138e22b7e69f948f0f8d0232a` | no licence file | fetched; `bases/patches/tirgn_stage.patch` (pure-PyTorch replacement of the DGL calls, CPU / `torch.load` compatibility, score-dump hooks) and `tirgn_gdelt_dump.patch` (ICEWS18 / GDELT dumps); the patched trees are checked against `bases/patches/*.md5` |
| CognTKE | https://github.com/WeiChen3690/CognTKE | `4f30966ec2c4775bad0a9addee6e22bdd1118452` | no licence file | fetched, used unmodified (empty diff); driven by `src/g8_cogntke/` |
| TGB | https://github.com/shenyangHuang/TGB | `3b83e5cfb2010743fd4f7a80a01b0e19dfcf78cb` | MIT | fetched; RE-GCN example scripts (G3, M6) and the repository copy of the RecB predictor (`modules/recurrencybaseline_predictor.py`) |
| EdgeReMIND | https://github.com/BryantPollard/EdgeReMIND | `41c0db2df31bf4ab60e6e4fec8176932314e3ec8` | MIT (as reported by GitHub, 2026-10-16) | fetched by `bases/fetch_external_methods.sh`, used unmodified (released code and configuration, seed 1337); driven by `src/v78_external/` (external learned-history comparison) |
| CountTRuCoLa | https://github.com/JuliaGast/counttrucola_submission | `5d9aedf4106926d13befa141e0855d684f3f5f9d` | no licence file detected (GitHub, 2026-10-16) | fetched by `bases/fetch_external_methods.sh`, used unmodified; driven by `src/v78_external/` (external comparison; the label-adapted run adds `ct_adapted_main.py` beside the released code in a copy) and by `fresh_window/code_ct/` (2026 GDELT window, released default configuration) |
| py-tgb | PyPI | 2.3.0 | MIT | installed from `requirements.txt`; datasets, shipped negatives, official `Evaluator` |
| RecB (authors' repository) | https://github.com/nec-research/recurrency_baseline_tkg | fix commits `ff23bc26819cc382083e44af0e2647a57226ee85`, `b6f34ab7651adb589381b0b6dabe621be5a3c216`, `086d010526b054ded98d39886a7e30657ba7324d` | licence file present, not identified by GitHub (NOASSERTION) | not used directly; `src/hardening/hd_recb.py` is our re-implementation (faithful = TGB copy, clean = the authors' 2025 fix), checked per query against the TGB script |
| RE-GCN, LogCL | https://github.com/Lee-zix/RE-GCN, https://github.com/WeiChen3690/LogCL | `bc64211e…`, `0c0f5536…` | no licence file | not used; `src/harness/regcn.py` and `logcl.py` are re-implementations in PyTorch written for this study |
| LightGBM | PyPI | 4.6.0 | MIT | LambdaMART ranker (`src/v76/v76_lmart.py`) |
| torch_scatter | PyPI wheel index of PyG | 2.1.2 (torch 2.8.0, CUDA 12.8) | MIT | imported by the CognTKE code; installed by `src/g8_cogntke/g8_setup.sh` (a pure-PyTorch fallback is in `src/g8_cogntke/shim/`) |
| loguru | PyPI | latest at install | MIT | imported by the CognTKE code |

## Data

No data are redistributed here; the scripts download them from the original sources. Check each source's terms before use.

| dataset | source | used for |
|---|---|---|
| tkgl-polecat, tkgl-smallpedia, tkgl-icews | TGB 2.0, through the py-tgb 2.3.0 download table (`data/download_tgb.py`) | main experiments |
| ICEWS18, GDELT | RE-GCN release (data folders of https://github.com/Lee-zix/RE-GCN) | G2, development history |
| GDELT 2.0 event exports, CAMEO codes | http://data.gdeltproject.org/gdeltv2/ (`data/build_gdelt25x.py`) | GDELT25X forward holdout |
| GDELT 2.0 event exports, CAMEO codes | http://data.gdeltproject.org/gdeltv2/ (`fresh_window/01_window_build/`, same builder, window 2025-08-01 .. 2026-04-30) | 2026 GDELT window (GDELTFW), `fresh_window/fresh_window_data_card.md` |
