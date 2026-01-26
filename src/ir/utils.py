# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
import textwrap
from collections.abc import Callable
from types import FunctionType as PyFunctionType
from allo.ir.types import AlloType, Struct
from allo.memory import Memory


def parse_ast(src, verbose=False) -> ast.Module:
    if isinstance(src, str):
        starting_line_no = 1
    else:
        src, starting_line_no = inspect.getsourcelines(src)
        src = [textwrap.fill(line, tabsize=4, width=9999) for line in src]
        src = textwrap.dedent("\n".join(src))
    tree = ast.parse(src)
    for child in ast.walk(tree):
        if hasattr(child, "lineno"):
            child.lineno += starting_line_no - 1
        if hasattr(child, "end_lineno"):
            child.end_lineno += starting_line_no - 1
    if verbose:
        print(src)
        try:
            import astpretty

            astpretty.pprint(tree, indent=2, show_offsets=False)
        except ImportError:
            print(ast.dump(tree))
    return tree


class SymbolTable:
    def __init__(self):
        self.functions = (
            {}
        )  # function name -> function instance (instantiated from templates) node
        self.variables = {}  # variable name -> variable node
        # ----- tools -----
        self.symbol_mangler = {}  # template name -> instance args -> instance name

    def name_mangling(self, name: str, args: list) -> str:
        """
        Name mangling for instantiated functions.

        Args:
            name: The name of the function.
            args: The arguments to instantiate the function. If the last argument is a string, it is used as the user defined suffix of the instantiated function.

        Returns:
            The name of the instantiated function.
        """
        if isinstance(args[-1], str):
            suffix = args.pop()
            return "_" + name + "_" + suffix
        key = tuple(args)
        func_dict = self.symbol_mangler.setdefault(name, {})
        if key not in func_dict:
            func_dict[key] = (
                "_" + name + "_" + "_".join(map(str, args)) + "_" + str(len(func_dict))
            )
        return func_dict[key]


def get_global_vars(func):
    def _get_global_vars(_func, skip={"get_global_vars", "process"}, stop={"<module>"}):
        if isinstance(_func, Callable):
            # Discussions: https://github.com/taichi-dev/taichi/issues/282
            global_vars = _func.__globals__.copy()
        else:
            global_vars = {}

        # Get back to outer scopes
        # Mainly used to get the annotation definitions (shape and type),
        # which are probably not defined in __globals__
        frame = inspect.currentframe().f_back
        while frame:
            if frame.f_code.co_name in skip:
                frame = frame.f_back
                continue
            # collect allowed types
            for name, var in frame.f_locals.items():
                # FIXME: find a better way to collect required symbols
                if isinstance(
                    var, (int, float, AlloType, Memory, list)
                ) or inspect.isfunction(var):
                    global_vars[name] = var
            # boundary
            if frame.f_code.co_name in stop:
                break
            frame = frame.f_back

        if isinstance(_func, Callable):
            freevar_names = _func.__code__.co_freevars
            closure = _func.__closure__
            if closure:
                freevar_values = [x.cell_contents for x in closure]
                for name, value in zip(freevar_names, freevar_values):
                    global_vars[name] = value
        return global_vars

    all_globals = {}
    worklist = [func]
    visited_funcs = set()

    while worklist:
        f = worklist.pop()
        if f in visited_funcs:
            continue
        visited_funcs.add(f)

        gv = _get_global_vars(f)
        for name, val in gv.items():
            if name not in all_globals:
                all_globals[name] = val
                # import functions from other files
                if isinstance(val, PyFunctionType):
                    worklist.append(val)

    return all_globals


class Scope:
    def __init__(self):
        self.consts = {}
        self.vars = {}


class ASTResolver:
    """Provides helper methods to resolve AST nodes."""

    # pylint: disable=too-many-return-statements
    @staticmethod
    def resolve(node, scope):
        """resolve a given AST node to a Python object.

        This is only intended to check if a given AST node resolves to a symbol
        under some namespaces, e.g. the ``a.b.c.foo`` pattern, but not meant for
        more complicated expressions like ``(a + b).foo``.

        Args:
            node (Union[ast.Attribute, ast.Name]): an AST node to be resolved.
            scope (Dict[str, Any]): Maps from symbol names to objects, for
                example, globals()

        Returns:
            object: The actual Python object that ``node`` resolves to.
        """
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.BinOp):
            # pylint: disable=eval-used
            return eval(compile(ast.Expression(node), "", "eval"), scope)

        if isinstance(node, ast.Call):
            # Handle function/constructor calls like Memory(resource="URAM")
            func_obj = ASTResolver.resolve(node.func, scope)
            if func_obj is None:
                return None
            # Resolve positional arguments
            args = []
            for arg in node.args:
                if isinstance(arg, ast.Constant):
                    args.append(arg.value)
                else:
                    resolved = ASTResolver.resolve(arg, scope)
                    if resolved is None:
                        return None
                    args.append(resolved)
            # Resolve keyword arguments
            kwargs = {}
            for kw in node.keywords:
                if isinstance(kw.value, ast.Constant):
                    kwargs[kw.arg] = kw.value.value
                else:
                    resolved = ASTResolver.resolve(kw.value, scope)
                    if resolved is None:
                        return None
                    kwargs[kw.arg] = resolved
            try:
                return func_obj(*args, **kwargs)
            # pylint: disable=broad-exception-caught
            except Exception:
                return None

        if isinstance(node, ast.List):
            values = [ASTResolver.resolve(v, scope) for v in node.elts]
            return values

        if isinstance(node, ast.Dict):
            # Resolve dictionary literals to struct types
            keys = [k.value if isinstance(k, ast.Constant) else None for k in node.keys]
            # If any key is not a string constant, this isn't a valid struct type
            if any(not isinstance(k, str) for k in keys):
                return None
            values = [ASTResolver.resolve(v, scope) for v in node.values]
            # If any value type couldn't be resolved, return None
            if any(v is None for v in values):
                return None
            return Struct(dict(zip(keys, values)))

        if isinstance(node, ast.Name):
            return scope.get(node.id)

        if not isinstance(node, ast.Attribute):
            return None

        v = node.value
        chain = [node.attr]
        while isinstance(v, ast.Attribute):
            chain.append(v.attr)
            v = v.value
        if not isinstance(v, ast.Name):
            # Example cases that fall under this branch:
            #
            # x[i].attr: ast.Subscript
            # (a + b).attr: ast.BinOp
            # ...
            return None
        chain.append(v.id)

        for attr in reversed(chain):
            try:
                if isinstance(scope, dict):
                    scope = scope[attr]
                else:
                    scope = getattr(scope, attr)
            except (KeyError, AttributeError):
                return None
        # The name ``scope`` here could be a bit confusing
        return scope

    @staticmethod
    def resolve_slice(node, ctx):
        if isinstance(node, (ast.ExtSlice, ast.Tuple)):
            return list(ASTResolver.resolve_slice(s, ctx) for s in node.dims)
        if isinstance(node, ast.Slice):
            return tuple(
                (
                    ASTResolver.resolve_constant(node.lower, ctx),
                    ASTResolver.resolve_constant(node.upper, ctx),
                    ASTResolver.resolve_constant(node.step, ctx),
                )
            )
        if isinstance(node, ast.Index):
            return ASTResolver.resolve_constant(node.value, ctx)
        return None

    @staticmethod
    def resolve_constant(node, ctx):
        if node is None:
            return None
        try:
            # pylint: disable=eval-used
            return eval(compile(ast.Expression(node), "", "eval"), ctx.global_vars)
        # pylint: disable=broad-exception-caught
        except Exception:
            return None

    @staticmethod
    def resolve_param_types(node, global_vars):
        if isinstance(node, ast.Tuple):
            return list(
                ASTResolver.resolve_param_type(n, global_vars) for n in node.elts
            )
        if isinstance(node, ast.Name):
            return [ASTResolver.resolve_param_type(node, global_vars)]
        raise RuntimeError(f"Unsupported node type: {type(node)}")

    @staticmethod
    def resolve_param_type(node, global_vars):
        """Resolve a single parameter type, handling both symbols and constructor calls."""
        if isinstance(node, ast.Call):
            # Handle type constructor calls like Fixed(16, 10)
            func_obj = ASTResolver.resolve(node.func, global_vars)
            if func_obj is None:
                # Try to resolve the function name as a string
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    func_obj = global_vars.get(func_name)
                elif isinstance(node.func, ast.Attribute):
                    # Handle cases like types.Fixed
                    if (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "types"
                    ):
                        func_name = node.func.attr
                        func_obj = global_vars.get(func_name)
                    else:
                        return None
                else:
                    return None

            if func_obj is None:
                return None

            # Get the arguments
            args = []
            for arg in node.args:
                if isinstance(arg, ast.Constant):
                    args.append(arg.value)
                else:
                    # Try to resolve the argument
                    resolved_arg = ASTResolver.resolve_constant(
                        arg, type("Context", (), {"global_vars": global_vars})()
                    )
                    if resolved_arg is None:
                        return None
                    args.append(resolved_arg)

            # Construct the type instance using the resolved function object
            try:
                return func_obj(*args)
            # pylint: disable=broad-exception-caught
            except Exception:
                return None
        else:
            # Handle simple symbol resolution
            return ASTResolver.resolve(node, global_vars)
