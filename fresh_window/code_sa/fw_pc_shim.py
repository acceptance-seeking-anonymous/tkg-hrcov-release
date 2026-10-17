"""phase2_fresh_window / fw_pc_shim.py -- the 'common' module v78_lists / v78_pools import for GDELTFW: fh_common of the
fresh-window pipeline (registers GDELTFW into oih_common.DATASETS; Splits, DS) plus RS_CODE = that code dir, so rs_features
(Hist, DyadReader) is the byte-identical copy used by the HRcov features."""
import os
from fh_common import *          # noqa: F401,F403
from fh_common import CODE, DS, Splits, H   # noqa: F401
RS_CODE = CODE
TRACKS = f"{H}/tracks"
