# Copyright (c) 2026 Fei Liu. MIT License.
# Project: https://github.com/FeiLiu36/EoH
# Citation: Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu,
#           Qingfu Zhang, Evolution of Heuristics: Towards Efficient Automatic Algorithm Design
#           Using Large Language Model, Forty-first International Conference on Machine Learning
#           (ICML), 2024.

import heapq
import json
import os
import time
import random
import logging

import numpy as np

from .evolution import Evolution
from ..utils.logger import setup_logger


def population_management(pop, size):
    pop = [ind for ind in pop if ind['objective'] is not None]
    if not pop:
        return pop
    seen = set()
    unique = []
    for ind in pop:
        if ind['objective'] not in seen:
            seen.add(ind['objective'])
            unique.append(ind)
    return heapq.nsmallest(min(size, len(unique)), unique, key=lambda x: x['objective'])


class EOH:
    """Main EoH evolutionary loop."""

    def __init__(self, config, problem):
        self.operators = config.operators
        self.operator_weights = config.operator_weights
        self.pop_size = config.pop_size
        self.n_pop = config.n_pop
        self.output_path = config.output_dir
        self.config = config
        self.problem = problem

        self._sample_count = 0
        self._best_obj = None
        self._samples_buffer = []
        self._samples_flushed = 0

        log_path = os.path.join(config.output_dir, "results", "run_log.txt")
        self._logger = setup_logger(log_path, config.debug)

        random.seed(2024)
        self.evolution = Evolution(config, problem)

    # ── header ────────────────────────────────────────────────────────────────

    def _log_header(self, cfg):
        ops = " ".join(self.operators)
        llm = cfg.llm
        init_n = 2 * self.pop_size
        for line in [
            "=" * 54,
            "  EoH",
            f"  LLM      : {llm.model} @ {llm.api_endpoint}",
            f"  EC       : gen={self.n_pop}  pop={self.pop_size}  ops=[{ops}]",
            f"  Sampling : init={init_n} (2×pop)  per_gen={self.pop_size}  parallel={self.problem.n_processes}",
            f"  Timeout  : llm={llm.timeout}s  eval={self.problem.timeout}s",
            "=" * 54,
        ]:
            self._logger.info(line)

    # ── per-sample record ─────────────────────────────────────────────────────

    _SAMPLE_BATCH = 200

    def _record(self, op: str, offspring) -> bool:
        """Log one sample line, write to sample files. Returns True if new best."""
        self._sample_count += 1

        if offspring is None:
            score_str = "None (generation failed)"
            obj = None
        elif offspring.get('objective') is None:
            score_str = "None (evaluation failed)"
            obj = None
        else:
            obj = offspring['objective']
            score_str = str(obj)

        is_new_best = obj is not None and (self._best_obj is None or obj < self._best_obj)
        if is_new_best:
            self._best_obj = obj

        best_str = str(self._best_obj) if self._best_obj is not None else "N/A"
        marker = "  *" if is_new_best else ""
        self._logger.info(f"  #{self._sample_count:<4} [{op}]  {score_str:<16}  best={best_str}{marker}")

        self._write_sample(op, offspring, is_new_best)
        return is_new_best

    def _write_sample(self, op: str, offspring, is_new_best: bool):
        record = {
            'sample_order': self._sample_count,
            'operator': op,
            'algorithm': offspring.get('algorithm') if offspring else None,
            'code': offspring.get('code') if offspring else None,
            'objective': offspring.get('objective') if offspring else None,
        }
        self._samples_buffer.append(record)
        if len(self._samples_buffer) >= self._SAMPLE_BATCH:
            self._flush_samples()
        if is_new_best:
            path = os.path.join(self.output_path, "results", "samples", "samples_best.json")
            try:
                with open(path, 'w') as f:
                    json.dump(record, f, indent=4)
            except OSError as e:
                self._logger.warning("Could not write best sample to %s: %s", path, e)

    def _flush_samples(self):
        if not self._samples_buffer:
            return
        lo = self._samples_flushed + 1
        hi = self._samples_flushed + len(self._samples_buffer)
        path = os.path.join(self.output_path, "results", "samples", f"samples_{lo}~{hi}.json")
        try:
            with open(path, 'w') as f:
                json.dump(self._samples_buffer, f, indent=4)
            self._samples_flushed += len(self._samples_buffer)
            self._samples_buffer = []
        except OSError as e:
            self._logger.warning("Could not flush samples to %s: %s", path, e)

    # ── main run ──────────────────────────────────────────────────────────────

    def run(self):
        self._log_header(self.config)
        t0 = time.time()

        population, n_start = self._init_population(t0)

        for gen in range(n_start, self.n_pop):
            self._logger.info(f"\n[Gen {gen+1}/{self.n_pop}]")

            selected_ops = random.choices(
                self.operators, weights=self.operator_weights, k=self.pop_size
            )

            _, offspring = self.evolution.get_algorithm(population, selected_ops)
            for op, off in zip(selected_ops, offspring):
                self._record(op, off)
                if off and off['objective'] is not None:
                    population.append(off)
            population = population_management(population, self.pop_size)

            if population:
                self._save(population, gen + 1)
            elapsed = (time.time() - t0) / 60
            best = population[0]['objective'] if population else 'N/A'
            self._logger.info(f"  --- gen {gen+1} done  pop={len(population)}  best={best}  elapsed={elapsed:.1f}m")

        self._flush_samples()

        elapsed = (time.time() - t0) / 60
        best = population[0]['objective'] if population else 'N/A'
        self._logger.info(f"{'='*54}")
        self._logger.info(f"  Evolution finished.  best={best}  samples={self._sample_count}  time={elapsed:.1f}m")
        self._logger.info(f"{'='*54}\n")

    # ── initialisation ────────────────────────────────────────────────────────

    def _init_population(self, t0):
        cfg = self.config

        if cfg.use_seed:
            try:
                with open(cfg.seed_path, encoding='utf-8') as f:
                    seeds = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Seed file not found: {cfg.seed_path!r}") from None
            except (OSError, json.JSONDecodeError) as e:
                raise RuntimeError(f"Failed to load seed file {cfg.seed_path!r}: {e}") from e
            population = self.evolution.evaluate_seeds(seeds)
            if not population:
                raise RuntimeError("Seed initialization produced no valid individuals.")
            self._save_checkpoint(population, 0)
            return population, 0

        if cfg.use_continue:
            self._logger.info(f"Resuming from {cfg.continue_path}")
            try:
                with open(cfg.continue_path, encoding='utf-8') as f:
                    population = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Continue file not found: {cfg.continue_path!r}") from None
            except (OSError, json.JSONDecodeError) as e:
                raise RuntimeError(f"Failed to load continue file {cfg.continue_path!r}: {e}") from e
            return population, cfg.continue_id

        self._logger.info(f"\n[Init]  ({2 * self.pop_size} samples → pop={self.pop_size})")
        raw_population = []
        init_ops = ['i1'] * self.pop_size
        for _ in range(2):
            _, batch = self.evolution.get_algorithm([], init_ops)
            for ind in batch:
                self._record('i1', ind)
                if ind and ind['objective'] is not None:
                    raw_population.append(ind)

        population = population_management(raw_population, self.pop_size)
        if not population:
            raise RuntimeError(
                "Initial population is empty. Check LLM connectivity, API credentials, "
                "and that evaluate_program() returns a valid float."
            )
        elapsed = (time.time() - t0) / 60
        self._logger.info(
            f"  Init done: {len(raw_population)}/{self._sample_count} evaluated"
            f"  pop={len(population)}  best={population[0]['objective']}  elapsed={elapsed:.1f}m"
        )
        self._save_checkpoint(population, 0)
        return population, 0

    # ── save ──────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, population, gen):
        path = os.path.join(self.output_path, "results", "pops", f"population_generation_{gen}.json")
        try:
            with open(path, 'w') as f:
                json.dump(population, f, indent=4)
        except OSError as e:
            self._logger.warning("Could not save checkpoint to %s: %s", path, e)

    def _save(self, population, gen):
        self._save_checkpoint(population, gen)
        best_path = os.path.join(self.output_path, "results", "pops_best", f"population_generation_{gen}.json")
        try:
            with open(best_path, 'w') as f:
                json.dump(population[0], f, indent=4)
        except OSError as e:
            self._logger.warning("Could not save best individual to %s: %s", best_path, e)
