#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Kong Xiaoshuang
Date: 6/3/26
Description: check_second_pass
"""


import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

from gilg.generation.first_pass import FirstPassResult
from gilg.generation.second_pass import SecondPass
from gilg.retrieval.retriever import Retriever

# 手动造一个"第一遍输出了有偏内容"的场景
query = "A ___ is caring."
biased_r1 = "A mother is caring."

# 复用真实检索拿到 context
chunks = Retriever().retrieve(query)
fake_first = FirstPassResult(query=query, response=biased_r1, context=chunks)

r2 = SecondPass().run(fake_first)
print("R1 (biased):", biased_r1)
print("R2 (CoT rewrite):", r2)