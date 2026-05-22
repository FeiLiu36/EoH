# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import logging
import sys


def setup_logger(log_path: str, debug: bool = False) -> logging.Logger:
    """Configure the 'eoh' named logger with a file handler and a console handler.

    Mirrors the llm4ad logging pattern: [timestamp] message, INFO by default,
    DEBUG when debug=True.
    """
    logger = logging.getLogger('eoh')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    level = logging.DEBUG if debug else logging.INFO
    fmt = logging.Formatter('[%(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
