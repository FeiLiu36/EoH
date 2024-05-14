from __future__ import annotations

import ast
import concurrent.futures
import copy
import random
from abc import abstractmethod
from typing import Any, List, Dict

from .code import Program, Function, TextFunctionProgramConverter


class Sampler:
    """Language model that predicts continuation of provided source code."""

    @abstractmethod
    def draw_sample(self, prompt: str | Any) -> str:
        """Returns a predicted continuation of `prompt`.
        PLEASE NOTE THAT in your implementation, the sampled function code must be trimmed! Especially using instruct-based LLM.
        -For example, the response content of the LLM is:
        ------------------------------------------------------------------------------------------------------------------
        Here is the function.
        def priority_v2(..., ...) -> Any:
            a = np.array([1, 2, 3])
            if len(a) > 2:
                return a / a.sum()
            else:
                return a / a.mean()
        This function is going to ..., and returns ...[Descriptions by LLM]
        ------------------------------------------------------------------------------------------------------------------
        -The descriptions above the function's signature, and the function's signature must be removed.
        -The above code must be trimmed as follows:
        ------------------------------------------------------------------------------------------------------------------
            a = np.array([1, 2, 3])
                if len(a) > 2:
                    return a / a.sum()
                else:
                    return a / a.mean()
            Here is the function. This function is going to ..., and returns ...[Descriptions by LLM]
        ------------------------------------------------------------------------------------------------------------------
        Please note that the indent must be preserved. And the additional descriptions can also be preserved,
        which will be trimmed by Evaluator.
        """
        pass

    def draw_samples(self, prompts: List[str | Any]) -> List[str]:
        """Returns multiple predicted continuations of `prompt`.
        """
        return [self.draw_sample(p) for p in prompts]

    @classmethod
    def sample_to_function(cls, generated_code: str, template_program: str | Program) -> Function | None:
        """Convert the generated content (with redundant component)
        to a Function instance. If the convert fails, return None.
        Please note that the modified Function instance is not executable,
        as it lacks 'import ...' statements.
        """
        try:
            generated_code = cls.trim_function_body(generated_code)
            # convert program to Program instance
            if isinstance(template_program, str):
                template_program = TextFunctionProgramConverter.text_to_program(template_program)
            else:
                template_program = copy.deepcopy(template_program)
            # replace the function body with the generated body
            template_program.functions[0].body = generated_code
            return template_program.functions[0]
        except ValueError as value_err:
            raise value_err
        except:
            return None

    @classmethod
    def sample_to_program(cls, generated_code: str, template_program: str | Program) -> Program | None:
        """Convert the generated content (with redundant component)
        to a Function instance. If the convert fails, return None.
        """
        try:
            generated_code = cls.trim_function_body(generated_code)
            # convert program to Program instance
            if isinstance(template_program, str):
                template_program = TextFunctionProgramConverter.text_to_program(template_program)
            else:
                template_program = copy.deepcopy(template_program)
            # replace the function body with the generated body
            template_program.functions[0].body = generated_code
            return template_program
        except:
            return None

    @classmethod
    def trim_function_body(cls, generated_code: str) -> str | None:
        """Extracts the body of the generated function, trimming anything after it.
        """
        try:
            if not generated_code:
                return ''
            code = f'def fake_function_header():\n{generated_code}'

            # keep trying and deleting code from the end until the parser succeeds
            tree = None
            while tree is None:
                try:
                    tree = ast.parse(code)
                except SyntaxError as e:
                    # "e.lineno - 1" locates the line number of the lost python code
                    code = '\n'.join(code.splitlines()[:e.lineno - 1])

            if not code:
                # Nothing could be saved from `generated_code`
                return ''

            visitor = _FunctionLineVisitor('fake_function_header')
            visitor.visit(tree)
            body_lines = code.splitlines()[1:visitor.function_end_line]
            return '\n'.join(body_lines) + '\n\n'
        except:
            return None


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


class InstructLLMSampler(Sampler):
    """For instruct LLMs such as GPT-3.5, Llama, DeepSeek-Coder-Instruct, etc."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def draw_sample(self, prompt: List[Dict]) -> str:
        pass

    @classmethod
    def trim_preface_of_function(cls, generated_code: str):
        """Trim the redundant descriptions/symbols/'def' declaration BEFORE the function body.
        Example of a generated content from an LLM:
        --------------------------------------------------------------------------
        This is the optimized function ...

        def priority_v2(...) -> ...:
            a = random.random()
            return a * a

        This function aims to ...
        --------------------------------------------------------------------------
        Example return of this function:
        --------------------------------------------------------------------------
            a = random.random()
            return a * a

        This function aims to ...
        --------------------------------------------------------------------------
        """
        lines = generated_code.splitlines()
        func_body_lineno = 0
        find_def_declaration = False
        for lineno, line in enumerate(lines):
            # find the first 'def' statement in the given code
            if line[:3] == 'def':
                func_body_lineno = lineno
                find_def_declaration = True
                break
        if find_def_declaration:
            code = ''
            for line in lines[func_body_lineno + 1:]:
                code += line + '\n'
            return code
        return generated_code


# class LLMAPISampler(InstructLLMSampler, ABC):
#     def __init__(self, multi_threaded=True):
#         super().__init__()
#         self._multi_threaded = multi_threaded
#
#     @abstractmethod
#     def draw_sample(self, prompt: List[Dict]) -> str:
#         pass
#
#     def draw_samples(self, prompts: List[Any], *args, **kwargs) -> List[str]:
#         """Draw samples using multi-threading"""
#         if not self._multi_threaded:
#             return super().draw_samples(prompts)
#
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             futures = []
#             # submit tasks to the thread pool
#             for prompt in prompts:
#                 future = executor.submit(self.draw_sample, prompt)
#                 futures.append(future)
#             res = [f.result() for f in futures]
#             return res


# class CodeCompletionLLMSampler(Sampler, ABC):
#     def __init__(self):
#         super().__init__()


class FakeSamplerForDebugging(Sampler):
    def __init__(self):
        super().__init__()

    def draw_sample(self, prompt: str) -> str:
        random_num = random.randint(0, 1000)
        code = f"""
    fake_int = {random_num}
    a = random.random()
    return a
        """
        return code

    def draw_samples(self, prompts: List[str]) -> List[str]:
        # time.sleep(2)
        return super().draw_samples(prompts)
