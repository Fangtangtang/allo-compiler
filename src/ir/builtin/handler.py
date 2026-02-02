# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
import ast

# Registry for builtin handlers
# Key: function name (str)
# Value: Handler class
BUILTIN_HANDLERS = {}


def register_builtin_handler(name):
    def decorator(cls):
        BUILTIN_HANDLERS[name] = cls
        return cls

    return decorator


class BuiltinHandler(abc.ABC):
    def __init__(self, builder):
        """
        Initialize the handler with the IRBuilder instance.

        Args:
            builder: The IRBuilder instance invoking this handler.
        """
        self.builder = builder
        self.ctx = builder.ctx

    @abc.abstractmethod
    def build(self, node: ast.Call, *args):
        """
        Build the IR for the builtin function call.

        Args:
            node: The ast.Call node.
            args: The arguments passed to the function call (already visited/evaluated if needed,
                  but usually we might want raw nodes or visited values depending on design.
                  In IRBuilder.visit_Call, usually args are visited before call.
                  Let's assume args are the MLIR values or Mock objects corresponding to the arguments).
        """
        pass
