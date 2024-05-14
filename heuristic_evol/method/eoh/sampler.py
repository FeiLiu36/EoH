from __future__ import annotations

import re
from typing import Tuple

from .prompt import EoHPrompt
from ...heuristic.sample import InstructLLMSampler, Function, Program


class EoHSampler:
    def __init__(self, sampler: InstructLLMSampler, template_program: str | Program):
        self._sampler = sampler
        self._template_program = template_program

    def get_thought_and_program(self, prompt: str) -> Tuple[str, Program]:
        prompt = EoHPrompt.create_instruct_prompt(prompt)
        response = self._sampler.draw_sample(prompt)
        thought = self.__class__.trim_thought_from_response(response)
        code = InstructLLMSampler.trim_preface_of_function(response)
        program = InstructLLMSampler.sample_to_program(code, self._template_program)
        return thought, program

    @classmethod
    def trim_thought_from_response(cls, response: str) -> str | None:
        try:
            pattern = r'\{.*?\}'
            bracketed_texts = re.findall(pattern, response)
            return bracketed_texts[0]
        except:
            return None


prompt_template = [
    {'role': 'system', 'content': 'xxxxxxxxxx'}
]
