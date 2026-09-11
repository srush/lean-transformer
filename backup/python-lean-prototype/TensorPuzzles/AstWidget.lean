import Lean.Elab.Eval
import Lean.Elab.Tactic
import Lean.Widget
import TensorPuzzles.TensorLang

namespace TensorPuzzles

open Lean

private def astNode
    (kind label : String) (children : Array Json := #[]) : Json :=
  Json.mkObj [
    ("kind", kind),
    ("label", label),
    ("children", Json.arr children)
  ]

private def Dim.astJson : Dim → Json
  | .lit value => astNode "dimension" s!"dimension {value}"
  | .param name => astNode "dimension" s!"dimension {name}"

private def BinOp.symbol : BinOp → String
  | .add => "+"
  | .sub => "−"
  | .mul => "×"
  | .div => "÷"
  | .matmul => "@  matmul"

private def CmpOp.symbol : CmpOp → String
  | .eq => "=="
  | .ne => "!="
  | .lt => "<"
  | .le => "≤"
  | .gt => ">"
  | .ge => "≥"

def Expr.astJson : Expr → Json
  | .input name => astNode "input" s!"input {name}"
  | .scalar value => astNode "scalar" s!"scalar {value}"
  | .arange size => astNode "constructor" "arange" #[size.astJson]
  | .binary op left right =>
      astNode "operator" op.symbol #[left.astJson, right.astJson]
  | .compare op left right =>
      astNode "comparison" s!"compare {op.symbol}" #[left.astJson, right.astJson]
  | .where_ condition ifTrue ifFalse =>
      astNode "where" "where" #[
        astNode "role" "condition" #[condition.astJson],
        astNode "role" "then" #[ifTrue.astJson],
        astNode "role" "else" #[ifFalse.astJson]
      ]
  | .unsqueeze value =>
      astNode "shape" "unsqueeze  (append dimension 1)" #[value.astJson]

private def stringArrayJson (values : List String) : Json :=
  Json.arr (values.map toJson).toArray

private def tensorParamJson (param : String × SymShape) : Json :=
  Json.mkObj [
    ("name", param.1),
    ("shape", Json.arr (param.2.map Dim.astJson).toArray)
  ]

def Program.astWidgetProps (program : Program) : Json :=
  Json.mkObj [
    ("title", "Tensor expression AST"),
    ("dimensions", stringArrayJson program.dimParams),
    ("tensorParams", Json.arr (program.tensorParams.map tensorParamJson).toArray),
    ("root", program.body.astJson)
  ]

@[widget_module]
def tensorAstWidget : Lean.Widget.Module where
  javascript := "
import * as React from 'react';

const e = React.createElement;

const colors = {
  where: 'var(--vscode-symbolIcon-functionForeground)',
  comparison: 'var(--vscode-symbolIcon-operatorForeground)',
  operator: 'var(--vscode-symbolIcon-operatorForeground)',
  constructor: 'var(--vscode-symbolIcon-constructorForeground)',
  shape: 'var(--vscode-symbolIcon-structForeground)',
  input: 'var(--vscode-symbolIcon-variableForeground)',
  scalar: 'var(--vscode-symbolIcon-numberForeground)',
  dimension: 'var(--vscode-descriptionForeground)',
  role: 'var(--vscode-descriptionForeground)'
};

function AstNode({ node, depth = 0 }) {
  const children = node.children || [];
  const labelStyle = {
    color: colors[node.kind] || 'var(--vscode-foreground)',
    fontFamily: 'var(--vscode-editor-font-family)',
    fontWeight: depth === 0 ? 600 : 400
  };

  if (children.length === 0) {
    return e('div', {
      style: { marginLeft: '1.1em', padding: '0.15em 0' }
    }, e('span', { style: labelStyle }, node.label));
  }

  return e('details', { open: depth < 3, style: { marginLeft: depth ? '0.75em' : 0 } }, [
    e('summary', { key: 'summary', style: { cursor: 'pointer', padding: '0.15em 0' } },
      e('span', { style: labelStyle }, node.label)),
    e('div', {
      key: 'children',
      style: {
        borderLeft: '1px solid var(--vscode-widget-border)',
        marginLeft: '0.38em',
        paddingLeft: '0.35em'
      }
    }, children.map((child, index) =>
      e(AstNode, { node: child, depth: depth + 1, key: index })))
  ]);
}

function Shape({ shape }) {
  return e('span', {}, ['[', shape.map((dim, index) =>
    e(React.Fragment, { key: index }, [index ? ', ' : '', dim.label.replace('dimension ', '')])), ']']);
}

export default function TensorAst({ title, dimensions = [], tensorParams = [], root }) {
  const signature = [];
  if (dimensions.length) {
    signature.push(e('div', { key: 'dims' }, `dimensions: ${dimensions.join(', ')}`));
  }
  if (tensorParams.length) {
    signature.push(e('div', { key: 'tensors' }, [
      'tensors: ',
      tensorParams.map((param, index) => e(React.Fragment, { key: param.name }, [
        index ? ', ' : '',
        e('span', { style: { fontFamily: 'var(--vscode-editor-font-family)' } }, param.name),
        ': ',
        e(Shape, { shape: param.shape })
      ]))
    ]));
  }

  return e('div', {
    style: {
      border: '1px solid var(--vscode-widget-border)',
      borderRadius: '4px',
      padding: '0.65em 0.8em',
      marginTop: '0.4em'
    }
  }, [
    e('div', { key: 'title', style: { fontWeight: 600, marginBottom: '0.35em' } }, title),
    signature.length ? e('div', {
      key: 'signature',
      style: { color: 'var(--vscode-descriptionForeground)', marginBottom: '0.5em' }
    }, signature) : null,
    e(AstNode, { key: 'root', node: root })
  ]);
}
"

private def saveAstWidget (program : Program) (stx : Syntax) : CoreM Unit :=
  Lean.Widget.savePanelWidgetInfo
    tensorAstWidget.javascriptHash
    (pure program.astWidgetProps)
    stx

syntax (name := tensorAstCommand) "#tensor_ast " term : command

open Lean.Elab in
@[command_elab tensorAstCommand]
unsafe def elabTensorAstCommand : Command.CommandElab
  | stx@`(#tensor_ast $program:term) => Command.liftTermElabM do
      let value ← Term.evalTerm Program (mkConst ``Program) program
      saveAstWidget value stx
  | _ => throwUnsupportedSyntax

syntax (name := tensorAstTactic) "tensor_ast " term : tactic

open Lean.Elab in
@[tactic tensorAstTactic]
unsafe def elabTensorAstTactic : Tactic.Tactic := fun stx => do
  match stx with
  | `(tactic| tensor_ast $program:term) =>
      let expression ← Tactic.elabTermEnsuringType program (mkConst ``Program)
      let expression ← instantiateMVars expression
      if expression.hasFVar then
        throwError "tensor_ast expects a closed Program expression"
      if expression.hasMVar then
        throwError "tensor_ast program contains unresolved metavariables"
      let value ← Meta.evalExpr Program (mkConst ``Program) expression
      saveAstWidget value stx
  | _ => throwUnsupportedSyntax

end TensorPuzzles
