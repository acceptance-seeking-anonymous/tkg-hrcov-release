#!/usr/bin/env python3
"""make_table9.py -- alias of make_fresh_window_table.py, kept under the name used in the review: the fresh-window table
\\label{tab:fresh} is Table 9 in Manuscript A v7.9 (in the v8.0 text it is Table 10, and Table 9 is the ladder table
tab:ladder).  Same arguments and defaults as make_fresh_window_table.py."""
import os, runpy, sys
_target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_fresh_window_table.py")
sys.argv[0] = _target
runpy.run_path(_target, run_name="__main__")
