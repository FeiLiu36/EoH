# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tools for manipulating Python code.

It implements 2 classes representing unities of code:
- Function, containing all the information we need about functions: name, args,
  body and optionally a return type and a docstring.
- Program, which contains a code preface (which could be imports, global
  variables and classes, ...) and a list of Functions.

RZ: The frequently used word 'call' in this file refers to the name of the function. For example the following:
def add_five(param) -> int:
    return param + 5
The 'call' of the function is 'add_file'.
"""
from __future__ import annotations

import ast
from collections.abc import Iterator, MutableSet, Sequence
import dataclasses
import io
import tokenize

from absl import logging


@dataclasses.dataclass
class Function:
    """A parsed Python function."""

    name: str
    args: str
    body: str
    return_type: str | None = None
    docstring: str | None = None
    score: int | None = None  # RZ: add this to record the score of the function.
    global_sample_nums: int | None = None  # RZ: record the order of the current program in the sequence of samples taken.
    sample_time: float | None = None  # RZ: add this
    evaluate_time: float | None = None  # RZ: add this

    def __str__(self) -> str:
        return_type = f' -> {self.return_type}' if self.return_type else ''

        function = f'def {self.name}({self.args}){return_type}:\n'
        if self.docstring:
            # self.docstring is already indented on every line except the first one.
            # Here, we assume the indentation is always two spaces. => RZ: two spaces is ugly, change to four
            # RZ: two spaces is ugly, change to four spaces.
            new_line = '\n' if self.body else ''
            # function += f'  """{self.docstring}"""{new_line}'  # two spaces indentation
            function += f'    """{self.docstring}"""{new_line}'
        # self.body is already indented.
        function += self.body + '\n\n'
        return function

    def __setattr__(self, name: str, value: str) -> None:
        # Ensure there aren't leading & trailing new lines in `body`.
        if name == 'body':
            value = value.strip('\n')
        # Ensure there aren't leading & trailing quotes in `docstring``.
        if name == 'docstring' and value is not None:
            if '"""' in value:
                value = value.strip()
                value = value.replace('"""', '')
        super().__setattr__(name, value)


@dataclasses.dataclass(frozen=True)
class Program:
    """A parsed Python program."""

    # `preface` is everything from the beginning of the code till the first
    # function is found.
    preface: str
    functions: list[Function]

    def __str__(self) -> str:
        program = f'{self.preface}\n' if self.preface else ''
        program += '\n'.join([str(f) for f in self.functions])
        return program

    def find_function_index(self, function_name: str) -> int:
        """Returns the index of input function name."""
        function_names = [f.name for f in self.functions]
        count = function_names.count(function_name)
        if count == 0:
            raise ValueError(
                f'function {function_name} does not exist in program:\n{str(self)}'
            )
        if count > 1:
            raise ValueError(
                f'function {function_name} exists more than once in program:\n'
                f'{str(self)}'
            )
        index = function_names.index(function_name)
        return index

    def get_function(self, function_name: str) -> Function:
        index = self.find_function_index(function_name)
        return self.functions[index]


class ProgramVisitor(ast.NodeVisitor):
    """Parses code to collect all required information to produce a `Program`.

    Note that we do not store function decorators.
    """

    def __init__(self, sourcecode: str):
        self._codelines: list[str] = sourcecode.splitlines()
        self._preface: str = ''
        self._functions: list[Function] = []
        self._current_function: str | None = None

    def visit_FunctionDef(self,  # pylint: disable=invalid-name
                          node: ast.FunctionDef) -> None:
        """Collects all information about the function being parsed."""
        if node.col_offset == 0:  # We only care about first level functions.
            self._current_function = node.name

            # -------------------------------------------------------------------------
            # RZ: BUGS!!! 这里问题很大 没有考虑到一些极端的情况 例如解析到的第一个函数就有装饰器
            # 如果解析的第一个函数有装饰器 那么装饰器也会被保留在前面的代码片段中
            # 但在算法的流程中 这个装饰器是无效的 后续评估会报错
            # Here is an issue: some cases were not taken into consideration.
            # For instance, if the first parsed function has decorators,
            # these decorators will also be retained in the preceding code snippet.
            # However, in the algorithm's flow, these decorators are invalid,
            # leading to subsequent evaluations reporting errors.
            # -------------------------------------------------------------------------
            # if not self._functions:
            #     self._preface = '\n'.join(self._codelines[:node.lineno - 1])
            # -------------------------------------------------------------------------

            # TODO Fix bugs 判断下如果函数 is decorated, 找到装饰器的最小行号, 并只保留其上面的部分作为 preface
            if not self._functions:
                has_decorators = bool(node.decorator_list)
                if has_decorators:
                    # find the minimum line number and retain the code above
                    decorator_start_line = min(decorator.lineno for decorator in node.decorator_list)
                    self._preface = '\n'.join(self._codelines[:decorator_start_line - 1])
                else:
                    self._preface = '\n'.join(self._codelines[:node.lineno - 1])

            function_end_line = node.end_lineno
            body_start_line = node.body[0].lineno - 1
            # Extract the docstring.
            docstring = None
            if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                docstring = f'  """{ast.literal_eval(ast.unparse(node.body[0]))}"""'
                if len(node.body) > 1:
                    body_start_line = node.body[1].lineno - 1
                else:
                    body_start_line = function_end_line

            self._functions.append(Function(
                name=node.name,
                args=ast.unparse(node.args),  # RZ: ast.unparse() failed in Python3.8. Let us try Python3.9
                return_type=ast.unparse(node.returns) if node.returns else None,
                docstring=docstring,
                body='\n'.join(self._codelines[body_start_line:function_end_line]),
            ))
        self.generic_visit(node)

    def return_program(self) -> Program:
        return Program(preface=self._preface, functions=self._functions)


def text_to_program(text: str) -> Program:
    """Returns Program object by parsing input text using Python AST.
    """
    try:
        # We assume that the program is composed of some preface (e.g. imports,
        # classes, assignments, ...) followed by a sequence of functions.
        tree = ast.parse(text)
        visitor = ProgramVisitor(text)
        visitor.visit(tree)
        return visitor.return_program()
    except Exception as e:
        logging.warning('Failed parsing %s', text)
        raise e


def text_to_function(text: str) -> Function:
    """Returns Function object by parsing input text using Python AST."""
    program = text_to_program(text)
    if len(program.functions) != 1:
        raise ValueError(f'Only one function expected, got {len(program.functions)}'
                         f':\n{program.functions}')
    return program.functions[0]


def _tokenize(code: str) -> Iterator[tokenize.TokenInfo]:
    """Transforms `code` into Python tokens."""
    code_bytes = code.encode()
    code_io = io.BytesIO(code_bytes)
    return tokenize.tokenize(code_io.readline)


def _untokenize(tokens: Sequence[tokenize.TokenInfo]) -> str:
    """Transforms a list of Python tokens into code."""
    code_bytes = tokenize.untokenize(tokens)
    return code_bytes.decode()


def _yield_token_and_is_call(code: str) -> Iterator[tuple[tokenize.TokenInfo, bool]]:
    """Yields each token with a bool indicating whether it is a function call.
    """
    try:
        tokens = _tokenize(code)
        prev_token = None
        is_attribute_access = False
        for token in tokens:
            if (prev_token and  # If the previous token exists and
                    prev_token.type == tokenize.NAME and  # it is a Python identifier
                    token.type == tokenize.OP and  # and the current token is a delimiter
                    token.string == '('):  # and in particular it is '('.
                yield prev_token, not is_attribute_access
                is_attribute_access = False
            else:
                if prev_token:
                    is_attribute_access = (
                            prev_token.type == tokenize.OP and prev_token.string == '.'
                    )
                    yield prev_token, False
            prev_token = token
        if prev_token:
            yield prev_token, False
    except Exception as e:
        logging.warning('Failed parsing %s', code)
        raise e


def rename_function_calls(code: str, source_name: str, target_name: str) -> str:
    """Renames function calls from `source_name` to `target_name`.
    """
    if source_name not in code:
        return code
    modified_tokens = []
    for token, is_call in _yield_token_and_is_call(code):
        if is_call and token.string == source_name:
            # Replace the function name token
            modified_token = tokenize.TokenInfo(
                type=token.type,
                string=target_name,
                start=token.start,
                end=token.end,
                line=token.line
            )
            modified_tokens.append(modified_token)
        else:
            modified_tokens.append(token)
    return _untokenize(modified_tokens)


def get_functions_called(code: str) -> MutableSet[str]:
    """Returns the set of all functions called in `code`.
    """
    return set(token.string for token, is_call in
               _yield_token_and_is_call(code) if is_call)


def yield_decorated(code: str, module: str, name: str) -> Iterator[str]:
    """Yields names of functions decorated with `@module.name` in `code`.
    """
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                attribute = None
                if isinstance(decorator, ast.Attribute):
                    attribute = decorator
                elif isinstance(decorator, ast.Call):
                    attribute = decorator.func
                if (attribute is not None
                        and attribute.value.id == module
                        and attribute.attr == name):
                    yield node.name
