# Implemented by RZ.
# This file aims to accelerate the original evaluate logic using 'numba' package.
# You should install numba package in your Python environment or the later evaluation will fail.

import ast


def add_numba_decorator(
        program: str,
        function_to_evolve: str,
) -> str:
    """
    This function aims to accelerate the evaluation of the searched code. This is achieved by decorating '@numba.jit()'
    to the function_to_evolve. However, it should be noted that not all numpy functions support numba acceleration:
    such as np.piecewise(). So use this function wisely. Hahaha!

    Example input program:
        def func(a: np.ndarray):
            return a * 2
    Example output program
        import numba

        numba.jit()
        def func(a: np.ndarray):
            return a * 2
    """
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
        if isinstance(node, ast.FunctionDef) and node.name == function_to_evolve:
            # the @numba.jit() decorator instance
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


if __name__ == '__main__':
    code = '''
import numpy as np
import numba

def func1():
    return 3

def func():
    return 5
    '''
    res = add_numba_decorator(code, 'func')
    print(res)
