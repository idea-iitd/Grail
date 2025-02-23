import astunparse
import ast
from typing import Any

class _FunctionLineVisitor(ast.NodeVisitor):
  """Visitor that finds the last line number of a function with a given name."""

  def __init__(self, target_function_name: str) -> None:
    self._target_function_name: str = target_function_name
    self._function_end_line: int | None = None

  def visit_FunctionDef(self, node: Any) -> None:  # pylint: disable=invalid-name
    """Collects the end line number of the target function."""
    if node.name == self._target_function_name:
      self._function_end_line = node.end_lineno
    self.generic_visit(node)

  @property
  def function_end_line(self) -> int:
    """Line number of the final line of function `target_function_name`."""
    assert self._function_end_line is not None  # Check internal correctness.
    return self._function_end_line

def _trim_function_body(generated_code: str) -> str:
  """Extracts the body of the generated function, trimming anything after it."""
  if not generated_code:
    return ''
  code = f'def fake_function_header():\n{generated_code}'
  tree = None
  # We keep trying and deleting code from the end until the parser succeeds.
  while tree is None:
    try:
      tree = ast.parse(code)
    except SyntaxError as e:
      # code = '\n'.join(code.splitlines()[:e.lineno - 1])
      pass
  if not code:
    # Nothing could be saved from `generated_code`
    return ''

  visitor = _FunctionLineVisitor('fake_function_header')
  visitor.visit(tree)
  body_lines = code.splitlines()[1:visitor.function_end_line]
  return '\n'.join(body_lines) + '\n\n'

def _trim_function_body(generated_code: str) -> str:
  """Extracts the body of the generated function, trimming anything after it."""
  if not generated_code:
    return ''
  code = f'def fake_function_header():\n{generated_code}'
  tree = None
  # We keep trying and deleting code from the end until the parser succeeds.
  while tree is None:
    try:
      tree = ast.parse(code)
    except SyntaxError as e:
      # code = '\n'.join(code.splitlines()[:e.lineno - 1])
      pass
  if not code:
    # Nothing could be saved from `generated_code`
    return ''

  visitor = _FunctionLineVisitor('fake_function_header')
  visitor.visit(tree)
  body_lines = code.splitlines()[1:visitor.function_end_line]
  return '\n'.join(body_lines) + '\n\n'

def _trim_function_body1(generated_code: str) -> str:

    # Find the index of the substring ```
    index = generated_code.find('```')

    # If the substring is found, remove everything from that point onwards
    if index != -1:
      generated_code = generated_code[:index]
    
    if generated_code.lstrip()=="return 0.0":
        return _trim_function_body(generated_code)
    
    tree = ast.parse(generated_code)

    print('gen code: ', generated_code)
    if not isinstance(tree, ast.Module) or not tree.body:
        raise ValueError("Invalid function string")

    function_def = tree.body[0]

    print(function_def)
    if not isinstance(function_def, ast.FunctionDef):
        raise ValueError("Invalid function string")

    name = function_def.name
    args = ', '.join(arg.arg for arg in function_def.args.args)

    # Remove docstring from the body
    if function_def.body and isinstance(function_def.body[0], ast.Expr) and isinstance(function_def.body[0].value, ast.Str):
        docstring_len = len(astunparse.unparse(function_def.body[0]))
        body = astunparse.unparse(function_def.body[1:])
    else:
        docstring_len = 0
        body = astunparse.unparse(function_def.body)

    return_type = None

    if function_def.returns is not None:
        return_type = astunparse.unparse(function_def.returns)

    docstring = ast.get_docstring(function_def)
    return str(body)


# Specify the file path
file_path = 'temp.py'

# Open the file and read it as a string
with open(file_path, 'r') as file:
    file_content = file.read()

# Now, file_content contains the entire file as a string
_trim_function_body1(file_content)
