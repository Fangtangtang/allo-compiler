from allo.ir.types import int32
from src.ir.builtin import BuiltinHandler, register_custom_handler
from src.main import process
from allo.spmw import kernel


@register_custom_handler("bypass")
class Bypass(BuiltinHandler):
    def build(self, node, *args):
        print("lalala")
        return None

    @staticmethod
    def infer(*args):
        assert args[0] == int32 and args[1] == int32
        return int32, int32


def test_custom_handler():

    @kernel
    def bypass(a: int32) -> int32:
        b: int32 = a
        Bypass(a, b)
        return b

    s = process(bypass)
    print(s(1))


if __name__ == "__main__":
    test_custom_handler()
