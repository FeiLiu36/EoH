# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import json
import logging

import requests

logger = logging.getLogger('eoh')


class InterfaceLocalLLM:
    def __init__(self, url: str, timeout: int = 60):
        self._url = url
        self.timeout = timeout

    def get_response(self, content: str, max_retries: int = 5) -> str | None:
        content = content.strip()
        data = {
            'prompt': content,
            'repeat_prompt': 1,
            'params': {
                'do_sample': True,
                'temperature': None,
                'top_k': None,
                'top_p': None,
                'add_special_tokens': False,
                'skip_special_tokens': True,
            },
        }
        headers = {'Content-Type': 'application/json'}

        for attempt in range(max_retries):
            try:
                response = requests.post(self._url, data=json.dumps(data), headers=headers, timeout=self.timeout)
                if response.status_code == 200:
                    parsed = response.json()
                    content = parsed.get('content')
                    if not content:
                        raise ValueError(f"Local LLM returned no content: {parsed}")
                    return content[0]
                logger.debug("Local LLM error (attempt %d/%d): HTTP %d — %s",
                             attempt + 1, max_retries, response.status_code, response.text[:200])
            except Exception as e:
                logger.debug("Local LLM error (attempt %d/%d): %s", attempt + 1, max_retries, e)

        logger.warning("Local LLM call failed after %d attempts (url=%s).", max_retries, self._url)
        return None
