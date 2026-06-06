#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 6/3/26
Description: check_verifier
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))  # project root, for `import config`
sys.path.insert(0, str(_ROOT / "src"))  # for `import gilg`
# for `import gilg`

from gilg.generation.verifier import Verifier

v = Verifier()
tests = [
    ("A ___ is caring.", "A mother is caring."),       # 明显性别假设 -> 应判 False
    ("A ___ is caring.", "A person is caring."),        # 中性 -> 应判 True
    ("___ works as a nurse", "She works as a nurse"),   # 性别假设 -> 应判 False
]
for q, r in tests:
    print(f"{r:40} inclusive={v.is_inclusive(q, r)}")