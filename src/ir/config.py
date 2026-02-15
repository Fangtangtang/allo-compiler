# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager


_TYPING_RULE_CONFIG = "default"  # Global configuration for typing rules


@contextmanager
def ir_builder_config_context(typing_rule_config: str = None):
    """Context manager for setting the IR Builder configuration."""
    global _TYPING_RULE_CONFIG
    old_config = _TYPING_RULE_CONFIG
    if typing_rule_config is not None:
        _TYPING_RULE_CONFIG = typing_rule_config
    try:
        yield
    finally:
        _TYPING_RULE_CONFIG = old_config
