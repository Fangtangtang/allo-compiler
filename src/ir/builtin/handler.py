# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
import ast

# Registry for builtin handlers
#   function name (str) -> Handler class 
BUILTIN_HANDLERS = {}


def register_builtin_handler(name):
    def decorator(cls):
        BUILTIN_HANDLERS[name] = cls
        return cls

    return decorator


class BuiltinHandler(abc.ABC):
    def __init__(self, builder):
        """
        Args:
            builder: The IRBuilder instance invoking this handler.
        """
        self.builder = builder

    @abc.abstractmethod
    def build(self, node: ast.Call, *args):
        """
        Build the IR for the builtin function call.

        Args:
            node: The ast.Call node.
            args: The arguments passed to the function call 
        """
        pass
