# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from .utils import parse_spmw_module


def instantiate_for_hls(module, top_name):
    parsed = parse_spmw_module(module, top_name)
