# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import logging

from .api_general import InterfaceAPI
from .api_local_llm import InterfaceLocalLLM

logger = logging.getLogger('eoh')


class InterfaceLLM:
    def __init__(self, api_endpoint, api_key, model, use_local, local_url, timeout=60):
        if use_local:
            if not local_url:
                raise ValueError("local_url must be set when use_local=True")
            self.llm = InterfaceLocalLLM(local_url, timeout=timeout)
            logger.info("LLM: local @ %s", local_url)
        else:
            if not api_key or not api_endpoint:
                raise ValueError("api_endpoint and api_key must be set for remote LLM")
            self.llm = InterfaceAPI(api_endpoint, api_key, model, timeout=timeout)
            logger.info("LLM: %s @ %s", model, api_endpoint)

        if self.llm.get_response("1+1=?") is None:
            raise RuntimeError("LLM API check failed. Verify endpoint, key, and model.")
        logger.info("LLM connection verified.")

    def get_response(self, prompt_content: str) -> str:
        return self.llm.get_response(prompt_content)
