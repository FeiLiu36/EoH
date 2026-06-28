# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import ast
import os
import random
import re
import sys
import signal
import logging
import threading
import warnings
import multiprocessing

import numpy as np

# joblib powers only the legacy generation-batch path (get_algorithm) and
# parallel seed evaluation. The async pipeline in eoh.py does not use it, so the
# dependency is optional — import lazily and fall back to sequential if absent.
try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover
    Parallel = delayed = None

# spawn: safe on all platforms, no background server started at import time.
# forkserver would start a server process at module import, which breaks when
# evolution.py is re-imported inside joblib's loky worker processes.
_MP_CTX = multiprocessing.get_context('spawn')

# Serialises the (rare) global set_start_method call below so concurrent eval
# threads can never race on it. In the async pipeline the start method is set
# once in the main thread, so this branch is normally never taken; the lock only
# matters for the legacy joblib/loky seed path.
_START_METHOD_LOCK = threading.Lock()

logger = logging.getLogger('eoh')


def _eval_worker(queue, problem, code):
    """Subprocess entry point — must be module-level for pickling.

    Calls os.setsid() on Unix to create a new process group, so that any
    child processes spawned here (e.g. a Java/C++ compiler or runner) belong
    to the same group and are killed together on timeout.
    """
    if sys.platform != 'win32':
        os.setsid()
    try:
        queue.put(problem.evaluate(code))
    except Exception:
        queue.put(None)


def _kill_process_tree(p):
    """Hard-stop a timed-out eval process and any children, escalating to SIGKILL.

    On Unix _eval_worker called os.setsid(), so the eval and any compiler/runner
    children share a process group we can signal as a unit. We try SIGTERM first,
    then escalate to SIGKILL, and bound every join so a subprocess that ignores
    SIGTERM can never hang the calling eval thread indefinitely.
    """
    pid = p.pid

    def _signal_group(sig):
        if sys.platform != 'win32' and pid is not None:
            try:
                os.killpg(os.getpgid(pid), sig)
            except (ProcessLookupError, PermissionError, OSError):
                pass

    _signal_group(signal.SIGTERM)
    p.terminate()
    p.join(5)
    if p.is_alive():
        _signal_group(signal.SIGKILL)
        p.kill()
        p.join(5)


def _eval_with_timeout(problem, code, timeout):
    """Run evaluate() in a subprocess and enforce a hard per-eval timeout.

    Module-level so it can be pickled by joblib/loky when called from a
    parallel worker. Returns the fitness value or None on timeout/error.
    """
    # joblib's loky backend registers 'loky' as the global multiprocessing
    # start method inside its worker processes.  spawn.py's get_preparation_data
    # reads this global and embeds it in the child's prep data, so the spawned
    # child calls set_start_method('loky', force=True) — which fails in a fresh
    # interpreter where loky is not registered.  Force 'spawn' before creating
    # the subprocess; each loky worker is an isolated OS process so this only
    # affects the current worker, not the main process or other workers.  The
    # lock makes the (rare) global mutation safe under the threaded eval pool.
    with _START_METHOD_LOCK:
        if multiprocessing.get_start_method(allow_none=False) != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
    q = _MP_CTX.Queue()
    p = _MP_CTX.Process(target=_eval_worker, args=(q, problem, code))
    p.start()
    p.join(timeout)
    if p.is_alive():
        _kill_process_tree(p)
        return None
    try:
        return q.get_nowait()
    except Exception:
        return None


from ..llm.interface_LLM import InterfaceLLM
from ..problem import _get_entry_name, _detect_template_kind, _extract_import_lines


def parent_selection(pop, m):
    if not pop:
        raise ValueError("Cannot select parents from an empty population.")
    ranks = list(range(len(pop)))
    probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
    return random.choices(pop, weights=probs, k=m)


class Evolution:
    """Prompt building, LLM calls, code extraction, and offspring generation."""

    def __init__(self, config, problem):
        self.task = problem.task_description
        self.template = problem.template_program
        self.func_name = _get_entry_name(problem.template_program)
        self._template_kind = _detect_template_kind(problem.template_program)
        self._template_import_prefix = _extract_import_lines(problem.template_program)

        self.interface_eval = problem
        self.debug = config.debug
        self.n_processes = problem.n_processes
        self.timeout = problem.timeout
        self.n_parents = config.n_parents

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.llm = InterfaceLLM(
            config.llm.api_endpoint,
            config.llm.api_key,
            config.llm.model,
            config.llm.use_local,
            config.llm.local_url,
            timeout=config.llm.timeout,
        )

    # ── prompt builders ───────────────────────────────────────────────────────

    def _func_spec(self) -> str:
        if self._template_kind == 'class':
            verb = "implement the following Python class"
        elif self._template_kind == 'multi_function':
            verb = "implement the following Python functions"
        else:
            verb = "implement the following Python function"
        return (
            f"{verb}:\n"
            f"```python\n{self.template.strip()}\n```\n"
            "Do not give additional explanations."
        )

    def _parent_block(self, parents: list) -> str:
        return "\n".join(
            f"No.{i+1} algorithm and the corresponding code are:\n{p['algorithm']}\n{p['code']}"
            for i, p in enumerate(parents)
        )

    def _build_prompt(self, operator: str, parents=None) -> str:
        spec = self._func_spec()
        if operator == "i1":
            return (
                f"{self.task}\n"
                "First, describe your new algorithm and main steps in one sentence. "
                f"The description must be inside a brace. Next, {spec}"
            )
        if operator == "e1":
            block = self._parent_block(parents)
            return (
                f"{self.task}\n"
                f"I have {len(parents)} existing algorithms with their codes as follows:\n{block}\n"
                "Please help me create a new algorithm that has a totally different form from the given ones.\n"
                "First, describe your new algorithm and main steps in one sentence. "
                f"The description must be inside a brace. Next, {spec}"
            )
        if operator == "e2":
            block = self._parent_block(parents)
            return (
                f"{self.task}\n"
                f"I have {len(parents)} existing algorithms with their codes as follows:\n{block}\n"
                "Please help me create a new algorithm that has a totally different form from the given ones "
                "but can be motivated from them.\n"
                "Firstly, identify the common backbone idea in the provided algorithms. "
                "Secondly, based on the backbone idea describe your new algorithm in one sentence. "
                f"The description must be inside a brace. Thirdly, {spec}"
            )
        if operator == "m1":
            return (
                f"{self.task}\n"
                f"I have one algorithm with its code as follows.\n"
                f"Algorithm description: {parents['algorithm']}\nCode:\n{parents['code']}\n"
                "Please assist me in creating a new algorithm that has a different form but can be a "
                "modified version of the algorithm provided.\n"
                "First, describe your new algorithm and main steps in one sentence. "
                f"The description must be inside a brace. Next, {spec}"
            )
        if operator == "m2":
            return (
                f"{self.task}\n"
                f"I have one algorithm with its code as follows.\n"
                f"Algorithm description: {parents['algorithm']}\nCode:\n{parents['code']}\n"
                "Please identify the main algorithm parameters and assist me in creating a new algorithm "
                "that has different parameter settings.\n"
                "First, describe your new algorithm and main steps in one sentence. "
                f"The description must be inside a brace. Next, {spec}"
            )
        if operator == "m3":
            if self._template_kind == 'class':
                keep_str = "keeping the class interface (name, method signatures, inputs, and outputs) unchanged"
            elif self._template_kind == 'multi_function':
                keep_str = "keeping all function names, inputs, and outputs unchanged"
            else:
                keep_str = "keeping the function name, inputs, and outputs unchanged"
            return (
                "First, identify the main components in the code below. "
                "Next, analyze whether any can be overfit to in-distribution instances. "
                "Then, simplify the components to enhance generalization to out-of-distribution instances. "
                f"Finally, provide the revised code {keep_str}.\n"
                f"{parents['code']}\nDo not give additional explanations."
            )
        raise ValueError(f"Unknown operator: {operator}")

    # ── LLM call + extraction ─────────────────────────────────────────────────

    def _extract(self, response: str):
        if not response:
            return [], []

        # ── code ──────────────────────────────────────────────────────────────
        # 1. Fenced code blocks (most reliable)
        code = re.findall(r'```(?:python)?\n(.*?)```', response, re.DOTALL)

        if not code:
            # 2. Locate the first top-level Python statement at the start of a line
            #    (import / from / def / class / decorator), then trim trailing prose
            #    by iteratively removing the last line until the snippet parses.
            start = re.search(r'^(?:import |from |def |class |@)', response, re.MULTILINE)
            if start:
                candidate = response[start.start():].strip()
                lines = candidate.splitlines()
                for trim in range(len(lines)):
                    snippet = '\n'.join(lines[:len(lines) - trim]).strip()
                    if not snippet:
                        break
                    try:
                        ast.parse(snippet)
                        code = [snippet]
                        break
                    except SyntaxError:
                        continue

        # Strip any leading {description} line the LLM sometimes puts inside the code block
        code = [re.sub(r'^\s*\{[^}]*\}\s*\n+', '', c, flags=re.DOTALL).strip() for c in code]
        code = [c for c in code if c]

        # ── algorithm description ──────────────────────────────────────────────
        # Search only in text BEFORE the code to avoid matching Python dict literals.
        if '```' in response:
            pre_code = response[:response.find('```')].strip()
        elif code:
            # Find where the extracted code begins in the original response
            idx = response.find(code[0][:60]) if code[0] else -1
            pre_code = response[:idx].strip() if idx > 0 else response.strip()
        else:
            pre_code = response.strip()

        # Require at least 8 chars to skip empty {}, single-letter vars, dict snippets
        algorithm = re.findall(r'\{([^{}]{8,})\}', pre_code)

        if not algorithm and pre_code:
            # Fall back: everything before the code is treated as the description
            algorithm = [pre_code]

        return algorithm, code

    def _prepend_imports(self, code: str) -> str:
        """Prepend any template import line not already present in code."""
        if not self._template_import_prefix:
            return code
        missing = [
            line for line in self._template_import_prefix.splitlines()
            if line and line not in code
        ]
        if not missing:
            return code
        return "\n".join(missing) + "\n" + code

    def _call_llm(self, prompt: str):
        for attempt in range(4):
            response = self.llm.get_response(prompt)
            if response:
                logger.debug("  [response] attempt %d/4: %.500s", attempt + 1, response)
            algorithm, code = self._extract(response)
            if algorithm and code:
                break
            logger.debug("  [extract] attempt %d/4 failed — no algorithm or code found.", attempt + 1)

        if not algorithm or not code:
            return None, None

        return self._prepend_imports(code[0]), algorithm[0]

    # ── operator dispatch ─────────────────────────────────────────────────────

    def _generate(self, population: list, operator: str):
        if operator == "i1":
            parents = None
            prompt = self._build_prompt("i1")
        elif operator in ("e1", "e2"):
            if not population:
                raise ValueError(f"Operator '{operator}' requires a non-empty population.")
            parents = parent_selection(population, self.n_parents)
            prompt = self._build_prompt(operator, parents)
        elif operator in ("m1", "m2", "m3"):
            if not population:
                raise ValueError(f"Operator '{operator}' requires a non-empty population.")
            parents = parent_selection(population, 1)[0]
            prompt = self._build_prompt(operator, parents)
        else:
            raise ValueError(f"Unknown operator: {operator}")

        logger.debug("  [prompt/%s] %d chars: %.400s", operator, len(prompt), prompt)
        code, algorithm = self._call_llm(prompt)

        if code:
            logger.debug("  [extract] algorithm: %.120r", algorithm)
            logger.debug("  [extract] code (%d chars): %.400s", len(code), code)
        else:
            logger.debug("  [extract] failed — no code extracted.")

        return parents, code, algorithm

    # ── code generation (no evaluation) ───────────────────────────────────────

    def generate_code(self, population: list, operator: str):
        """LLM generation + duplicate-retry, WITHOUT evaluation.

        Returns (parents, code, algorithm), or (None, None, None) on failure.
        This is the producer half used by the async pipeline: evaluation is run
        separately on the evaluation pool so LLM I/O and CPU-bound evaluation
        scale independently of each other and of pop_size.
        """
        parents, code, algorithm = self._generate(population, operator)
        if code is None:
            return None, None, None

        n_retry = 0
        while self._is_duplicate(population, code) and n_retry < 2:
            logger.debug("  [offspring] duplicate — retrying...")
            _, code, algorithm = self._generate(population, operator)
            if code is None:
                return None, None, None
            n_retry += 1

        return parents, code, algorithm

    # ── single offspring ──────────────────────────────────────────────────────

    def get_offspring(self, population: list, operator: str):
        try:
            parents, code, algorithm = self.generate_code(population, operator)
            if code is None:
                return None, None

            # Always isolate evaluation in a subprocess so a per-eval hard timeout
            # applies consistently in both sequential and parallel modes.
            # In parallel mode this function already runs inside a joblib worker
            # process, so the nested subprocess is safe (spawn context, no
            # inherited lock state). os.setsid() in _eval_worker creates a new
            # process group so compiler/interpreter children are killed too.
            fitness = _eval_with_timeout(self.interface_eval, code, self.timeout)
            if fitness is None:
                logger.debug("  [eval] timed out or returned None after %ds", self.timeout)

            if fitness is not None:
                rounded = float(np.round(fitness, 5))
                objective = rounded if np.isfinite(rounded) else None
                if not np.isfinite(rounded):
                    logger.debug("  [eval] non-finite fitness (%s) discarded", fitness)
            else:
                objective = None
            offspring = {
                'algorithm': algorithm,
                'code': code,
                'objective': objective,
                'other_inf': None,
            }
            return parents, offspring

        except Exception as e:
            logger.debug("  [offspring] %s: %s", type(e).__name__, e)
            return None, None

    # ── parallel batch ────────────────────────────────────────────────────────

    def get_algorithm(self, population: list, operators: list):
        """Generate one offspring per entry in operators, optionally in parallel.

        Each eval runs in its own subprocess (see _eval_with_timeout), so per-job
        timeouts are enforced without a batch-level timeout. On any Parallel()
        failure the call falls back to sequential so no offspring are silently lost.
        """
        if self.n_processes == 1 or Parallel is None:
            results = [self.get_offspring(population, op) for op in operators]
        else:
            try:
                results = Parallel(n_jobs=self.n_processes)(
                    delayed(self.get_offspring)(population, op) for op in operators
                )
            except Exception as e:
                logger.warning("  [parallel] %s: %s — falling back to sequential", type(e).__name__, e)
                results = [self.get_offspring(population, op) for op in operators]

        parents = [p for p, _ in results]
        offspring = [o for _, o in results]
        return parents, offspring

    # ── seed evaluation ───────────────────────────────────────────────────────

    def evaluate_seeds(self, seeds: list) -> list:
        _timeout = self.timeout
        _problem = self.interface_eval
        if self.n_processes == 1 or Parallel is None:
            fitness_list = [
                _eval_with_timeout(_problem, s['code'], _timeout) for s in seeds
            ]
        else:
            try:
                fitness_list = Parallel(n_jobs=self.n_processes)(
                    delayed(_eval_with_timeout)(_problem, s['code'], _timeout)
                    for s in seeds
                )
            except Exception as e:
                logger.warning("  [seed parallel] %s — falling back to sequential", e)
                fitness_list = [
                    _eval_with_timeout(_problem, s['code'], _timeout) for s in seeds
                ]
        population = []
        for seed, fitness in zip(seeds, fitness_list):
            if fitness is not None:
                rounded = float(np.round(fitness, 5))
                if not np.isfinite(rounded):
                    logger.debug("  [seed] non-finite fitness (%s) discarded", fitness)
                    continue
                population.append({
                    'algorithm': seed['algorithm'],
                    'code': seed['code'],
                    'objective': rounded,
                    'other_inf': None,
                })
        logger.info("Seeds: %d/%d valid.", len(population), len(seeds))
        return population

    # ── helpers ───────────────────────────────────────────────────────────────

    def _is_duplicate(self, population: list, code: str) -> bool:
        return any(ind['code'] == code for ind in population)
