# This file is implemented by RZ.
# This file aims to accelerate the original evaluate logic using 'numba' package.
# You should install numba package in your Python environment or the later evaluation will fail.
from __future__ import annotations

import ast
from typing import Sequence, Tuple


def add_import_package_statement(program: str, package_name: str, as_name=None, *, check_imported=True) -> str:
    """Add 'import package_name as as_name' in the program code.
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


def add_numba_decorator(
        program: str,
        function_name: str | Sequence[str],
) -> str:
    """
    This function aims to accelerate the evaluation of the searched code. This is achieved by decorating '@numba.jit()'
    to the function_to_evolve or other functions in the specification that can be speed up using numba.
    However, it should be noted that not all numpy functions support numba acceleration: such as np.piecewise().
    So use this function wisely. Hahaha!

    Example input program:
        def func(a: np.ndarray):
            return a * 2
    Example output program
        import numba

        numba.jit()
        def func(a: np.ndarray):
            return a * 2
    """
    if isinstance(function_name, str):
        return _add_numba_decorator(program, function_name)
    for f_name in function_name:
        program = _add_numba_decorator(program, f_name)
    return program


def add_np_random_seed_below_numpy_import(program: str, seed: int = 2024) -> str:
    """Add 'import numpy as np' statement (if needed) to the program and insert 'np.random.seed(seed)' under it.
    Args:
        program: program you want to add.
        seed: seed number.
    Returns:
        modified_code: program with 'np.random.seed(...)'.
    """
    program = add_import_package_statement(program, 'numpy', 'np')
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


def replace_div_with_protected_div(code_str: str, delta=1e-5, numba_accelerate=False) -> Tuple[str, str]:
    # protected_div_str = f'_protected_div = lambda x, y, delta={delta}: x / (y + delta) if y == 0 else x / y'
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
        modified_code = add_numba_decorator(modified_code, '_protected_div')
    return modified_code, '_protected_div'


def add_numpy_random_seed_to_func(program: str, func_name: str, seed: int = 2024) -> str:
    tree = ast.parse(program)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            node.body = [ast.parse(f'np.random.seed({seed})').body[0]] + node.body

    modified_code = ast.unparse(tree)
    return modified_code



if __name__ == '__main__':
    code = '''
import numpy as np

def hello():
    return 'xxx'


def func(a, b, c):
    return a + b - c
    '''

    output = add_numba_decorator(program=code, function_name='func')
    print(output)
