from __future__ import annotations

import ast
import io
import tokenize
from collections.abc import Iterator, MutableSet
from typing import Sequence, Tuple, List


class ModifyCode:
    @classmethod
    def add_import_package_statement(
            cls,
            program: str,
            package_name: str,
            as_name: str | None = None,
            *,
            check_imported: bool = True
    ) -> str:
        """Add 'import package_name as as_name' in the program code.
        Args:
            program       : The program in string.
            package_name  : The name of the package to be imported.
            as_name       : The alias of the imported package. Such as 'np' to 'numpy'.
            check_imported: Check if 'import {package_name} as {as_name}' statement has already existed,
                            this function returns the original program if it exists.
        """
        tree = ast.parse(program)
        if check_imported:
            # check if 'import package_name' code exists
            package_imported = False
            for node in tree.body:
                if isinstance(node, ast.Import) and any(alias.name == package_name for alias in node.names):
                    package_imported = True
                    break

            if package_imported:
                return ast.unparse(tree)

        # add 'import package_name' to the top of the program
        import_node = ast.Import(names=[ast.alias(name=package_name, asname=as_name)])
        tree.body.insert(0, import_node)
        program = ast.unparse(tree)
        return program

    @classmethod
    def add_numpy_random_seed_to_func(cls, program: str, func_name: str, seed: int = 2024) -> str:
        tree = ast.parse(program)

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                node.body = [ast.parse(f'np.random.seed({seed})').body[0]] + node.body

        modified_code = ast.unparse(tree)
        return modified_code

    @classmethod
    def replace_div_with_protected_div(
            cls,
            code_str: str,
            delta: float = 1e-5,
            numba_accelerate: bool = False,
            return_div_func_name: bool = False
    ) -> str | Tuple[str, str]:
        protected_div_str = f'''
def _protected_div(x, y, delta={delta}):
    return x / (y + delta)
        '''
        tree = ast.parse(code_str)
        transformer = _CustomDivisionTransformer('_protected_div')
        modified_tree = transformer.visit(tree)
        modified_code = ast.unparse(modified_tree)
        modified_code = '\n'.join([modified_code, '', '', protected_div_str])
        if numba_accelerate:
            modified_code = cls.add_numba_decorator(modified_code, '_protected_div')

        if return_div_func_name:
            return modified_code, '_protected_div'
        return modified_code

    @classmethod
    def add_np_random_seed_below_numpy_import(cls, program: str, seed: int = 2024) -> str:
        """Add 'import numpy as np' statement (if needed) to the program and insert 'np.random.seed(seed)' under it.
        Args:
            program: program you want to add.
            seed   : seed number.
        Returns:
            modified_code: program with 'np.random.seed(...)'.
        """
        program = cls.add_import_package_statement(program, 'numpy', 'np')
        tree = ast.parse(program)

        # find 'import numpy as np'
        found_numpy_import = False

        # find 'import numpy as np' statement
        for node in tree.body:
            if isinstance(node, ast.Import) and any(alias.name == 'numpy' and alias.asname == 'np' for alias in node.names):
                found_numpy_import = True
                # insert new node
                node_idx = tree.body.index(node)
                seed_node = ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Attribute(
                                value=ast.Name(id='np', ctx=ast.Load()),
                                attr='random',
                                ctx=ast.Load()
                            ),
                            attr='seed',
                            ctx=ast.Load()
                        ),
                        args=[ast.Num(n=seed)],
                        keywords=[]
                    )
                )
                tree.body.insert(node_idx + 1, seed_node)

        if not found_numpy_import:
            raise ValueError("No 'import numpy as np' found in the code.")

        modified_code = ast.unparse(tree)
        return modified_code

    @classmethod
    def add_numba_decorator(cls, program: str, function_name: str | List[str]) -> str:
        """
        This function aims to accelerate the evaluation of the searched code. This is achieved by decorating '@numba.jit()'
        to the function_to_evolve or other functions in the specification that can be speed up using numba.
        However, it should be noted that not all numpy functions support numba acceleration: such as np.piecewise().
        So use this function wisely. Haha!

        Example input program:
        ----------------------------------------------------------
            def func(a: np.ndarray):
                return a * 2
        ----------------------------------------------------------
        Example output program:
        ----------------------------------------------------------
            import numba

            numba.jit()
            def func(a: np.ndarray):
                return a * 2
        ----------------------------------------------------------
        """
        if isinstance(function_name, str):
            return _add_numba_decorator(program, function_name)
        for f_name in function_name:
            program = _add_numba_decorator(program, f_name)
        return program

    @classmethod
    def rename_function_calls(cls, code: str, source_name: str, target_name: str) -> str:
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

    @classmethod
    def get_functions_called(cls, code: str) -> MutableSet[str]:
        """Returns the set of all functions called in `code`.
        """
        return set(token.string for token, is_call in
                   _yield_token_and_is_call(code) if is_call)

    @classmethod
    def yield_decorated(cls, code: str, module: str, name: str) -> Iterator[str]:
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
        raise e


def _add_numba_decorator(
        program: str,
        function_name: str
) -> str:
    # parse to syntax tree
    tree = ast.parse(program)

    # check if 'import numba' already exists
    numba_imported = False
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == 'numba' for alias in node.names):
            numba_imported = True
            break

    # add 'import numba' to the top of the program
    if not numba_imported:
        import_node = ast.Import(names=[ast.alias(name='numba', asname=None)])
        tree.body.insert(0, import_node)

    # traverse the tree, and find the function_to_run
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # the '@numba.jit()' decorator instance
            decorator = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='numba', ctx=ast.Load()),
                    attr='jit',
                    ctx=ast.Load()
                ),
                args=[],  # args do not have argument name
                keywords=[ast.keyword(arg='nopython', value=ast.NameConstant(value=True))]
                # keywords have argument name
            )
            # add the decorator to the decorator_list of the node
            node.decorator_list.append(decorator)

    # turn the tree to string and return
    modified_program = ast.unparse(tree)
    return modified_program


class _CustomDivisionTransformer(ast.NodeTransformer):
    def __init__(self, custom_divide_func_name: str):
        super().__init__()
        self._custom_div_func = custom_divide_func_name

    def visit_BinOp(self, node):
        self.generic_visit(node)  # recur visit child nodes
        if isinstance(node.op, ast.Div):
            # self-defined node
            custom_divide_call = ast.Call(
                func=ast.Name(id=self._custom_div_func, ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[]
            )
            return custom_divide_call
        return node


if __name__ == '__main__':
    code_string = "result = 10 / 2 / a"

    # 调用函数并输出结果
    modified_code = ModifyCode.replace_div_with_protected_div(code_string)
    print(modified_code)
