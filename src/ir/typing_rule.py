# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
import types as python_types
from allo.ir.types import (
    AlloType,
    Index,
    Float,
    Int,
    UInt,
    Fixed,
    UFixed,
    int32,
    bool as allo_bool,
)


class TypingRule:
    """Type inference rule for a set of operations."""

    def __init__(self, inf_rules):
        """
        Parameters
        ----------
        inf_rules : a dictionary or a collection of dictionaries
            The inference rules for the operation class
            Each item should be (input types, lambda function)
        """
        # Check argument types
        if isinstance(inf_rules, dict):
            inf_rules = [inf_rules]
        elif not isinstance(inf_rules, tuple):
            inf_rules = list(inf_rules)
        elif not isinstance(inf_rules, list):
            raise TypeError(
                f"inf_rules must be a dict or a collection of dict, not {type(inf_rules)}"
            )

        # Inference rules
        self.inf_rules = {}
        # a dictionary of the form:
        # { input types (tuple) : inference function (lambda func) }
        # merge the collection of inference rules into a single dictionary
        for rule_set in inf_rules:
            for itype, inf_rule in rule_set.items():
                # check itype type
                if not isinstance(itype, tuple):
                    raise TypeError(f"itype must be a tuple, not {type(itype)}")
                for t in itype:
                    if not isinstance(t, type):
                        raise TypeError(
                            f"itype must be a tuple of Class, not {type(t)}"
                        )
                # check inf_rule type
                if not isinstance(inf_rule, python_types.LambdaType):
                    raise TypeError(
                        f"inf_rule must be a lambda function, not {type(inf_rule)}"
                    )
                # sort the input types
                itype = tuple(itype)
                # check if the input types are already in the dictionary
                if itype in self.inf_rules:
                    raise RuntimeError(
                        f"Duplicate inference rule for input types {itype}"
                    )
                # add the rule to the dictionary
                self.inf_rules[itype] = inf_rule

    def __call__(self, *args):
        """Call the inference rule with the given input types.

        It automatically finds the typing rule based on the input types.
        If no rule is found, it will raise an error.

        Parameters
        ----------
        args : list of input types

        Returns
        -------
        Type
            The inferred output type
        """
        itype_classes = [type(t) for t in args]
        itype_classes = tuple(itype_classes)
        if itype_classes not in self.inf_rules:
            raise RuntimeError(
                f"Typing rule is not defined with input types {itype_classes}"
            )
        rule = self.inf_rules[itype_classes]
        res_type = rule(*args)
        return res_type


def cpp_style_binary_arith_rule():
    int_rules = {
        (Int, Int): lambda t1, t2: (
            (
                Int(max(t1.bits, t2.bits)),
                Int(max(t1.bits, t2.bits)),
                Int(max(t1.bits, t2.bits)),
            )
            if all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
            else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
        ),
        (Int, UInt): lambda t1, t2: (
            (UInt(t2.bits), UInt(t2.bits), UInt(t2.bits))
            if t2.bits >= t1.bits and all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
            else (
                (Int(t1.bits), Int(t1.bits), Int(t1.bits))
                if all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
                else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
            )
        ),
        (Int, Index): lambda t1, t2: (
            (
                Int(max(t1.bits, t2.bits)),
                Int(max(t1.bits, t2.bits)),
                Int(max(t1.bits, t2.bits)),
            )
            if t1.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
        ),
        (Int, Float): lambda t1, t2: (
            (t2, t2, t2)
            if t1.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
        ),
        # python native value
        (Int, int): lambda t1, v2: (t1, t1, t1),
        (int, Int): lambda v1, t2: (t2, t2, t2),
        (Int, float): lambda t1, v2: (Float(64), Float(64), Float(64)),
        (float, Int): lambda v1, t2: (Float(64), Float(64), Float(64)),
    }
    uint_rules = {
        (UInt, Int): lambda t1, t2: (
            (UInt(t1.bits), UInt(t1.bits), UInt(t1.bits))
            if t1.bits >= t2.bits and all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
            else (
                (Int(t2.bits), Int(t2.bits), Int(t2.bits))
                if all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
                else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
            )
        ),
        (UInt, UInt): lambda t1, t2: (
            (
                UInt(max(t1.bits, t2.bits)),
                UInt(max(t1.bits, t2.bits)),
                UInt(max(t1.bits, t2.bits)),
            )
            if all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
            else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
        ),
        (UInt, Index): lambda t1, t2: (
            (UInt(t1.bits), UInt(t1.bits), UInt(t1.bits))
            if t1.bits >= 32 and t1.bits in {8, 16, 32, 64}
            else (
                (Index(), Index(), Index())
                if t1.bits in {8, 16, 32, 64}
                else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
            )
        ),
        (UInt, Float): lambda t1, t2: (
            (t2, t2, t2)
            if t1.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
        ),
        # python native value
        (UInt, int): lambda t1, v2: (t1, t1, t1),
        (int, UInt): lambda v1, t2: (t2, t2, t2),
        (UInt, float): lambda t1, v2: (Float(64), Float(64), Float(64)),
        (float, UInt): lambda v1, t2: (Float(64), Float(64), Float(64)),
    }
    index_rules = {
        (Index, Int): lambda t1, t2: (
            (
                Int(max(t1.bits, t2.bits)),
                Int(max(t1.bits, t2.bits)),
                Int(max(t1.bits, t2.bits)),
            )
            if t2.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
        ),
        (Index, UInt): lambda t1, t2: (
            (UInt(t2.bits), UInt(t2.bits), UInt(t2.bits))
            if t2.bits >= 32 and t2.bits in {8, 16, 32, 64}
            else (
                (Index(), Index(), Index())
                if t2.bits in {8, 16, 32, 64}
                else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
            )
        ),
        (Index, Index): lambda t1, t2: (t1, t2, t1),
        (Index, Float): lambda t1, t2: (t2, t2, t2),
        # python native value
        (Index, int): lambda t1, v2: (t1, t1, t1),
        (int, Index): lambda v1, t2: (t2, t2, t2),
    }
    float_rules = {
        (Float, Int): lambda t1, t2: (
            (t1, t1, t1)
            if t2.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
        ),
        (Float, UInt): lambda t1, t2: (
            (t1, t1, t1)
            if t2.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary arithmetic rule")
        ),
        (Float, Index): lambda t1, t2: (t1, t1, t1),
        (Float, Float): lambda t1, t2: (
            (t1, t1, t1) if t1.bits >= t2.bits else (t2, t2, t2)
        ),
        # python native value
        (Float, int): lambda t1, v2: (t1, t1, t1),
        (int, Float): lambda v1, t2: (t2, t2, t2),
        (Float, float): lambda t1, v2: (t1, t1, t1),
        (float, Float): lambda v1, t2: (t2, t2, t2),
    }
    return TypingRule(
        [int_rules, uint_rules, index_rules, float_rules],
    )


def cpp_style_intrin_rule():
    unaryrules = {
        (Float,): lambda t: (t, t),
        (Int,): lambda t: ((int32, int32) if t.bits < 32 else (t, t)),
        (UInt,): lambda t: (t, t),
        (Index,): lambda t: (t, t),
        (Fixed,): lambda t: (t, t),
        (UFixed,): lambda t: (t, t),
    }
    return TypingRule([unaryrules])


def cpp_style_comparison_rule():
    # [NOTE]: the return type is always bool (currently using i1)
    int_rules = {
        (Int, Int): lambda t1, t2: (
            (allo_bool, Int(max(t1.bits, t2.bits)), Int(max(t1.bits, t2.bits)))
            if all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
            else TypeError(f"{t1}, {t2} fail binary comparison rule")
        ),
        (Int, UInt): lambda t1, t2: (
            (allo_bool, UInt(t2.bits), UInt(t2.bits))
            if t2.bits >= t1.bits and all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
            else (
                (allo_bool, Int(t1.bits), Int(t1.bits))
                if all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
                else TypeError(f"{t1}, {t2} fail binary comparison rule")
            )
        ),
        (Int, Index): lambda t1, t2: (
            (allo_bool, Int(max(t1.bits, t2.bits)), Int(max(t1.bits, t2.bits)))
            if t1.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary comparison rule")
        ),
        (Int, Float): lambda t1, t2: (
            (allo_bool, t2, t2)
            if t1.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary comparison rule")
        ),
        # python native value
        (Int, int): lambda t1, v2: (allo_bool, t1, t1),
        (int, Int): lambda v1, t2: (allo_bool, t2, t2),
        (Int, float): lambda t1, v2: (allo_bool, Float(64), Float(64)),
        (float, Int): lambda v1, t2: (allo_bool, Float(64), Float(64)),
    }
    uint_rules = {
        (UInt, Int): lambda t1, t2: (
            (allo_bool, UInt(t1.bits), UInt(t1.bits))
            if t1.bits >= t2.bits and all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
            else (
                (allo_bool, Int(t2.bits), Int(t2.bits))
                if all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
                else TypeError(f"{t1}, {t2} fail binary comparison rule")
            )
        ),
        (UInt, UInt): lambda t1, t2: (
            (allo_bool, UInt(max(t1.bits, t2.bits)), UInt(max(t1.bits, t2.bits)))
            if all(t.bits in {8, 16, 32, 64} for t in (t1, t2))
            else TypeError(f"{t1}, {t2} fail binary comparison rule")
        ),
        (UInt, Index): lambda t1, t2: (
            (allo_bool, UInt(t1.bits), UInt(t1.bits))
            if t1.bits >= 32 and t1.bits in {8, 16, 32, 64}
            else (
                (allo_bool, Index(), Index())
                if t1.bits in {8, 16, 32, 64}
                else TypeError(f"{t1}, {t2} fail binary comparison rule")
            )
        ),
        (UInt, Float): lambda t1, t2: (
            (allo_bool, t2, t2)
            if t1.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary comparison rule")
        ),
        # python native value
        (UInt, int): lambda t1, v2: (allo_bool, t1, t1),
        (int, UInt): lambda v1, t2: (allo_bool, t2, t2),
        (UInt, float): lambda t1, v2: (allo_bool, Float(64), Float(64)),
        (float, UInt): lambda v1, t2: (allo_bool, Float(64), Float(64)),
    }
    index_rules = {
        (Index, Int): lambda t1, t2: (
            (allo_bool, Int(max(t1.bits, t2.bits)), Int(max(t1.bits, t2.bits)))
            if t2.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary comparison rule")
        ),
        (Index, UInt): lambda t1, t2: (
            (allo_bool, UInt(t2.bits), UInt(t2.bits))
            if t2.bits >= 32 and t2.bits in {8, 16, 32, 64}
            else (
                (allo_bool, Index(), Index())
                if t2.bits in {8, 16, 32, 64}
                else TypeError(f"{t1}, {t2} fail binary comparison rule")
            )
        ),
        (Index, Index): lambda t1, t2: (allo_bool, t1, t2),
        (Index, Float): lambda t1, t2: (allo_bool, t2, t2),
        # python native value
        (Index, int): lambda t1, v2: (allo_bool, t1, t1),
        (int, Index): lambda v1, t2: (allo_bool, t2, t2),
    }
    float_rules = {
        (Float, Int): lambda t1, t2: (
            (allo_bool, t1, t1)
            if t2.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary comparison rule")
        ),
        (Float, UInt): lambda t1, t2: (
            (allo_bool, t1, t1)
            if t2.bits in {8, 16, 32, 64}
            else TypeError(f"{t1}, {t2} fail binary comparison rule")
        ),
        (Float, Index): lambda t1, t2: (allo_bool, t1, t1),
        (Float, Float): lambda t1, t2: (
            (allo_bool, t1, t1) if t1.bits >= t2.bits else (allo_bool, t2, t2)
        ),
        # python native value
        (Float, int): lambda t1, v2: (allo_bool, t1, t1),
        (int, Float): lambda v1, t2: (allo_bool, t2, t2),
        (Float, float): lambda t1, v2: (allo_bool, t1, t1),
        (float, Float): lambda v1, t2: (allo_bool, t2, t2),
    }
    bool_rules = {
        (UInt, bool): lambda t1, v2: (
            (allo_bool, t1, t1)
            if t1.bits == 8
            else TypeError(f"{t1}, {v2} fail binary comparison rule")
        ),
        (bool, UInt): lambda v1, t2: (
            (allo_bool, t2, t2)
            if t2.bits == 8
            else TypeError(f"{v1}, {t2} fail binary comparison rule")
        ),
    }
    return TypingRule(
        [int_rules, uint_rules, index_rules, float_rules, bool_rules],
    )


def cpp_style_bool_op_rule():
    cpp_style_bool = allo_bool

    def rule(*args):
        # Check if any operand is unsupported
        for t in args:
            print(t)
            if isinstance(t, bool) or t == cpp_style_bool:
                continue
            raise TypeError(f"Type {t} not supported in boolean operation")
        return tuple([cpp_style_bool] * (len(args) + 1))

    return rule


CPP_STYLE_BINARY_ARITH_RULE = cpp_style_binary_arith_rule()
CPP_STYLE_COMPARISON_RULE = cpp_style_comparison_rule()
CPP_STYLE_INTRIN_RULE = cpp_style_intrin_rule()
CPP_STYLE_BOOL_OP_RULE = cpp_style_bool_op_rule()

cpp_style_registry = {
    ast.Add: CPP_STYLE_BINARY_ARITH_RULE,
    ast.Sub: CPP_STYLE_BINARY_ARITH_RULE,
    ast.Mult: CPP_STYLE_BINARY_ARITH_RULE,
    ast.Div: CPP_STYLE_BINARY_ARITH_RULE,
    ast.FloorDiv: CPP_STYLE_BINARY_ARITH_RULE,
    ast.Mod: CPP_STYLE_BINARY_ARITH_RULE,
    ast.Eq: CPP_STYLE_COMPARISON_RULE,
    ast.NotEq: CPP_STYLE_COMPARISON_RULE,
    ast.Lt: CPP_STYLE_COMPARISON_RULE,
    ast.LtE: CPP_STYLE_COMPARISON_RULE,
    ast.Gt: CPP_STYLE_COMPARISON_RULE,
    ast.GtE: CPP_STYLE_COMPARISON_RULE,
    ast.USub: CPP_STYLE_INTRIN_RULE,
    ast.UAdd: CPP_STYLE_INTRIN_RULE,
    ast.Invert: CPP_STYLE_INTRIN_RULE,
    ast.And: CPP_STYLE_BOOL_OP_RULE,
    ast.Or: CPP_STYLE_BOOL_OP_RULE,
}
