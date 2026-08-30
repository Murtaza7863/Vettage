#!/usr/bin/env python3
"""Download WildFake training zips (never DALLE/coco). Resume-friendly."""
from prepare_gen_v2 import TRAIN_ZIPS, download_modelscope
import os, sys

raw = "data/raw/wildfake"
keys = sys.argv[1:] or list(TRAIN_ZIPS)
for key in keys:
    rel = TRAIN_ZIPS[key]
    dest = os.path.join(raw, os.path.basename(rel))
    download_modelscope(rel, dest)
print("done", keys)
