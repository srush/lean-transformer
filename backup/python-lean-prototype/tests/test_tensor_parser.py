import unittest

from tensor_parser import ParseError, parse_source


class TensorParserTests(unittest.TestCase):
    def test_triu_shape(self):
        program = parse_source(
            """
def triu(n: int):
    rows = arange(n).unsqueeze()
    columns = arange(n)
    return where(rows <= columns, 1, 0)
"""
        )
        self.assertEqual(program.body.dtype, "num")
        self.assertEqual(program.body.shape, ("n", "n"))

    def test_all_requested_arithmetic_nodes(self):
        program = parse_source(
            """
def arithmetic(n: int, a: Tensor["n", "n"], b: Tensor["n", "n"]):
    mixed = (a + b) * (a - b) / 2
    return mixed @ b
"""
        )
        self.assertEqual(program.body.shape, ("n", "n"))
        for operator in (".add", ".sub", ".mul", ".div", ".matmul"):
            self.assertIn(operator, program.body.lean)

    def test_reassignment_is_rejected(self):
        with self.assertRaisesRegex(ParseError, "reassignment"):
            parse_source(
                """
def bad(n: int):
    x = arange(n)
    x = x + 1
    return x
"""
            )

    def test_unknown_calls_are_rejected(self):
        with self.assertRaisesRegex(ParseError, "supported calls"):
            parse_source(
                """
def bad(n: int):
    return sum(arange(n))
"""
            )


if __name__ == "__main__":
    unittest.main()

