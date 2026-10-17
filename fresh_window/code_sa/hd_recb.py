"""phase2_fresh_window / code_sa/hd_recb.py -- shim so that v76_recb_xf_select.py (`import hd_recb as HR`) uses the RecB
implementation of the fresh-window pipeline (fw code/fh_recb.py = presub hd_recb.py + --only_valid)."""
import os, sys
sys.path.insert(0, os.environ.get("FW_CODE", "/home/anon/cs_scratch/open_ideas_harness/tracks/phase2_fresh_window/code"))
from fh_recb import LAMS, ALPHAS, I_TRDEF_L, I_TRDEF_A, load_quads, split_queries   # noqa: E402,F401
