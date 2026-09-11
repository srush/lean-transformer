#!/usr/bin/env python3
"""Translate a tiny, mutation-free Python tensor subset into Lean AST data.

This frontend uses only the Python standard library. Lean independently checks
the emitted AST's dtype and symbolic output shape before executing it.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias


Dim: TypeAlias = int | str
Shape: TypeAlias = tuple[Dim, ...]


class ParseError(ValueError):
    pass


def lean_string(value: str) -> str:
    return json.dumps(value)


def lean_dim(dim: Dim) -> str:
    if isinstance(dim, int):
        return f".lit {dim}"
    return f".param {lean_string(dim)}"


def lean_shape(shape: Shape) -> str:
    return "[" + ", ".join(lean_dim(dim) for dim in shape) + "]"


def broadcast_dim(left: Dim, right: Dim) -> Dim:
    if left == right:
        return left
    if left == 1:
        return right
    if right == 1:
        return left
    raise ParseError(f"cannot broadcast dimensions {left!r} and {right!r}")


def broadcast_shapes(left: Shape, right: Shape) -> Shape:
    output: list[Dim] = []
    left_rev = list(reversed(left))
    right_rev = list(reversed(right))
    for index in range(max(len(left_rev), len(right_rev))):
        if index >= len(left_rev):
            output.append(right_rev[index])
        elif index >= len(right_rev):
            output.append(left_rev[index])
        else:
            output.append(broadcast_dim(left_rev[index], right_rev[index]))
    return tuple(reversed(output))


@dataclass(frozen=True)
class TypedExpr:
    lean: str
    dtype: str
    shape: Shape


@dataclass(frozen=True)
class ParsedProgram:
    name: str
    dim_params: tuple[str, ...]
    tensor_params: tuple[tuple[str, Shape], ...]
    body: TypedExpr

    def lean_program(self) -> str:
        dims = "[" + ", ".join(lean_string(x) for x in self.dim_params) + "]"
        tensors = "[" + ", ".join(
            f"({lean_string(name)}, {lean_shape(shape)})"
            for name, shape in self.tensor_params
        ) + "]"
        return (
            f"def {self.name}Program : Program := {{\n"
            f"  dimParams := {dims}\n"
            f"  tensorParams := {tensors}\n"
            f"  body := {self.body.lean}\n"
            "}"
        )


class FunctionParser:
    def __init__(self, function: ast.FunctionDef):
        self.function = function
        self.dim_params: set[str] = set()
        self.tensor_params: dict[str, Shape] = {}
        self.bindings: dict[str, TypedExpr] = {}

    def error(self, node: ast.AST, message: str) -> ParseError:
        return ParseError(f"line {getattr(node, 'lineno', '?')}: {message}")

    def parse_annotation(self, arg: ast.arg) -> None:
        annotation = arg.annotation
        if isinstance(annotation, ast.Name) and annotation.id == "int":
            self.dim_params.add(arg.arg)
            return
        if (
            isinstance(annotation, ast.Subscript)
            and isinstance(annotation.value, ast.Name)
            and annotation.value.id == "Tensor"
        ):
            raw_dims = (
                annotation.slice.elts
                if isinstance(annotation.slice, ast.Tuple)
                else [annotation.slice]
            )
            shape: list[Dim] = []
            for raw in raw_dims:
                if isinstance(raw, ast.Constant) and isinstance(raw.value, int):
                    if raw.value < 0:
                        raise self.error(raw, "tensor dimensions must be nonnegative")
                    shape.append(raw.value)
                elif isinstance(raw, ast.Constant) and isinstance(raw.value, str):
                    shape.append(raw.value)
                else:
                    raise self.error(raw, "Tensor dimensions must be integers or strings")
            self.tensor_params[arg.arg] = tuple(shape)
            return
        raise self.error(arg, "parameters require int or Tensor[...] annotations")

    def parse_dim(self, node: ast.AST) -> Dim:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            if node.value < 0:
                raise self.error(node, "dimensions must be nonnegative")
            return node.value
        if isinstance(node, ast.Name) and node.id in self.dim_params:
            return node.id
        raise self.error(node, "arange expects an integer literal or int parameter")

    def require_num(self, node: ast.AST, value: TypedExpr) -> None:
        if value.dtype != "num":
            raise self.error(node, "expected a numeric tensor")

    def parse_expr(self, node: ast.AST) -> TypedExpr:
        if isinstance(node, ast.Name):
            if node.id in self.bindings:
                return self.bindings[node.id]
            if node.id in self.tensor_params:
                return TypedExpr(f".input {lean_string(node.id)}", "num", self.tensor_params[node.id])
            raise self.error(node, f"unknown value {node.id!r}")

        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return TypedExpr(f".scalar {node.value}", "num", ())

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = self.parse_expr(node.operand)
            self.require_num(node, value)
            zero = TypedExpr(".scalar 0", "num", ())
            shape = broadcast_shapes(zero.shape, value.shape)
            return TypedExpr(f".binary .sub ({zero.lean}) ({value.lean})", "num", shape)

        if isinstance(node, ast.BinOp):
            left = self.parse_expr(node.left)
            right = self.parse_expr(node.right)
            self.require_num(node.left, left)
            self.require_num(node.right, right)
            operators = {
                ast.Add: "add",
                ast.Sub: "sub",
                ast.Mult: "mul",
                ast.Div: "div",
                ast.MatMult: "matmul",
            }
            op = operators.get(type(node.op))
            if op is None:
                raise self.error(node, "only +, -, *, /, and @ are supported")
            if op == "matmul":
                if len(left.shape) != 2 or len(right.shape) != 2:
                    raise self.error(node, "minimal @ requires two rank-2 tensors")
                if left.shape[1] != right.shape[0]:
                    raise self.error(node, "matmul inner dimensions do not match")
                shape = (left.shape[0], right.shape[1])
            else:
                shape = broadcast_shapes(left.shape, right.shape)
            return TypedExpr(
                f".binary .{op} ({left.lean}) ({right.lean})", "num", shape
            )

        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise self.error(node, "chained comparisons are not supported")
            left = self.parse_expr(node.left)
            right = self.parse_expr(node.comparators[0])
            self.require_num(node.left, left)
            self.require_num(node.comparators[0], right)
            operators = {
                ast.Eq: "eq",
                ast.NotEq: "ne",
                ast.Lt: "lt",
                ast.LtE: "le",
                ast.Gt: "gt",
                ast.GtE: "ge",
            }
            op = operators.get(type(node.ops[0]))
            if op is None:
                raise self.error(node, "unsupported comparison")
            shape = broadcast_shapes(left.shape, right.shape)
            return TypedExpr(
                f".compare .{op} ({left.lean}) ({right.lean})", "bool", shape
            )

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if node.keywords:
                    raise self.error(node, "keyword arguments are not supported")
                if name == "arange" and len(node.args) == 1:
                    dim = self.parse_dim(node.args[0])
                    return TypedExpr(f".arange ({lean_dim(dim)})", "num", (dim,))
                if name == "where" and len(node.args) == 3:
                    condition, if_true, if_false = map(self.parse_expr, node.args)
                    if condition.dtype != "bool":
                        raise self.error(node.args[0], "where condition must be boolean")
                    self.require_num(node.args[1], if_true)
                    self.require_num(node.args[2], if_false)
                    branch_shape = broadcast_shapes(if_true.shape, if_false.shape)
                    shape = broadcast_shapes(condition.shape, branch_shape)
                    return TypedExpr(
                        f".where_ ({condition.lean}) ({if_true.lean}) ({if_false.lean})",
                        "num",
                        shape,
                    )
                if name == "unsqueeze" and len(node.args) == 1:
                    value = self.parse_expr(node.args[0])
                    return TypedExpr(f".unsqueeze ({value.lean})", value.dtype, value.shape + (1,))
                raise self.error(node, "supported calls are arange, where, and unsqueeze")

            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "unsqueeze"
                and not node.args
                and not node.keywords
            ):
                value = self.parse_expr(node.func.value)
                return TypedExpr(f".unsqueeze ({value.lean})", value.dtype, value.shape + (1,))

            raise self.error(node, "unsupported method or function call")

        raise self.error(node, f"unsupported expression: {type(node).__name__}")

    def parse(self) -> ParsedProgram:
        if self.function.decorator_list:
            raise self.error(self.function, "decorators are not supported")
        if self.function.args.vararg or self.function.args.kwarg:
            raise self.error(self.function, "variadic parameters are not supported")
        if self.function.args.posonlyargs or self.function.args.kwonlyargs:
            raise self.error(self.function, "positional-only and keyword-only parameters are not supported")

        for argument in self.function.args.args:
            self.parse_annotation(argument)

        known_dims = self.dim_params
        for _, shape in self.tensor_params.items():
            for dim in shape:
                if isinstance(dim, str) and dim not in known_dims:
                    raise self.error(
                        self.function,
                        f"tensor shape references unknown int parameter {dim!r}",
                    )

        returned: TypedExpr | None = None
        for index, statement in enumerate(self.function.body):
            if isinstance(statement, ast.Assign):
                if returned is not None:
                    raise self.error(statement, "statements after return are not supported")
                if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                    raise self.error(statement, "assignments require one simple name")
                name = statement.targets[0].id
                if name in self.bindings or name in self.tensor_params or name in self.dim_params:
                    raise self.error(statement, f"reassignment of {name!r} is forbidden")
                self.bindings[name] = self.parse_expr(statement.value)
            elif isinstance(statement, ast.Return):
                if index != len(self.function.body) - 1:
                    raise self.error(statement, "return must be the final statement")
                if statement.value is None:
                    raise self.error(statement, "return requires a value")
                returned = self.parse_expr(statement.value)
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                # Permit a function docstring.
                continue
            else:
                raise self.error(statement, "only immutable assignments and a final return are supported")

        if returned is None:
            raise self.error(self.function, "function requires a final return")

        return ParsedProgram(
            name=self.function.name,
            dim_params=tuple(arg.arg for arg in self.function.args.args if arg.arg in self.dim_params),
            tensor_params=tuple(
                (arg.arg, self.tensor_params[arg.arg])
                for arg in self.function.args.args
                if arg.arg in self.tensor_params
            ),
            body=returned,
        )


def parse_source(source: str, filename: str = "<input>") -> ParsedProgram:
    try:
        module = ast.parse(source, filename=filename)
    except SyntaxError as error:
        raise ParseError(str(error)) from error
    definitions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    if len(module.body) != 1 or len(definitions) != 1:
        raise ParseError("source must contain exactly one function definition")
    return FunctionParser(definitions[0]).parse()


def generate_lean(program: ParsedProgram, source_path: str) -> str:
    dtype = f".{program.body.dtype}"
    return f'''import TensorPuzzles.AstWidget

namespace TensorPuzzles.Generated

open TensorPuzzles

/- Generated from {source_path}. The theorem below makes Lean independently
   check the frontend's claimed dtype and symbolic output shape. -/

{program.lean_program()}

def {program.name}ClaimedShape : SymShape := {lean_shape(program.body.shape)}

theorem {program.name}_shape_checked :
    inferProgramChecked {program.name}Program
      ({dtype}, {program.name}ClaimedShape) = true := by
  decide

theorem {program.name}_inferred :
    inferProgram {program.name}Program =
      .ok ({dtype}, {program.name}ClaimedShape) := by
  tensor_ast {program.name}Program
  exact inferProgramChecked_sound {program.name}_shape_checked

end TensorPuzzles.Generated
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.read_text(encoding="utf-8")
    try:
        program = parse_source(source, str(args.source))
    except ParseError as error:
        parser.error(str(error))
    output = generate_lean(program, str(args.source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(f"generated {args.output}: {program.body.dtype} tensor {program.body.shape}")


if __name__ == "__main__":
    main()
