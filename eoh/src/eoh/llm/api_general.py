# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import http.client
import json
import logging
import time

logger = logging.getLogger('eoh')


class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, timeout=60):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.timeout = timeout

    def get_response(self, prompt_content: str, max_retries: int = 5) -> str | None:
        payload = json.dumps({
            "model": self.model_LLM,
            "messages": [{"role": "user", "content": prompt_content}],
        })
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(max_retries):
            conn = None
            try:
                conn = http.client.HTTPSConnection(self.api_endpoint, timeout=self.timeout)
                conn.request("POST", "/v1/chat/completions", payload, headers)
                data = conn.getresponse().read()
                parsed = json.loads(data)
                choices = parsed.get("choices")
                if not choices:
                    error_msg = parsed.get("error", {}).get("message", str(parsed))
                    raise ValueError(f"API returned no choices: {error_msg}")
                return choices[0]["message"]["content"]
            except Exception as e:
                logger.debug("API error (attempt %d/%d): %s", attempt + 1, max_retries, e)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            finally:
                if conn is not None:
                    conn.close()

        logger.warning("API call failed after %d attempts (endpoint=%s, model=%s).",
                       max_retries, self.api_endpoint, self.model_LLM)
        return None
