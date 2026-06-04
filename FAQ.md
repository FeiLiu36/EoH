# EoH Frequently Asked Questions

## Table of Contents

1. [What is EoH and what problems can it solve?](#1-what-is-eoh-and-what-problems-can-it-solve)
2. [How do I install EoH?](#2-how-do-i-install-eoh)
3. [Which LLMs are supported?](#3-which-llms-are-supported)
4. [How do I configure the LLM (API key, endpoint, model)?](#4-how-do-i-configure-the-llm-api-key-endpoint-model)
5. [My API calls keep timing out — how do I fix this?](#5-my-api-calls-keep-timing-out--how-do-i-fix-this)
6. [How do I define my own optimization problem?](#6-how-do-i-define-my-own-optimization-problem)
7. [What are the supported template types?](#7-what-are-the-supported-template-types)
8. [What should I put in the initial template function?](#8-what-should-i-put-in-the-initial-template-function)
9. [Does EoH maximize or minimize the fitness value?](#9-does-eoh-maximize-or-minimize-the-fitness-value)
10. [What are the evolutionary operators (e1, e2, m1, m2)?](#10-what-are-the-evolutionary-operators-e1-e2-m1-m2)
11. [How do I run EoH and what are the key configuration parameters?](#11-how-do-i-run-eoh-and-what-are-the-key-configuration-parameters)
12. [Where are the results saved?](#12-where-are-the-results-saved)
13. [How do I resume a run that was interrupted?](#13-how-do-i-resume-a-run-that-was-interrupted)
14. [How do I seed EoH with hand-crafted algorithms?](#14-how-do-i-seed-eoh-with-hand-crafted-algorithms)
15. [How do I speed up evaluation with parallel workers?](#15-how-do-i-speed-up-evaluation-with-parallel-workers)
16. [Why does EoH produce no valid results or always return None?](#16-why-does-eoh-produce-no-valid-results-or-always-return-none)
17. [How do comments in the template affect EoH performance?](#17-how-do-comments-in-the-template-affect-eoh-performance)
18. [How does EoH compare to FunSearch and AEL?](#18-how-does-eoh-compare-to-funsearch-and-ael)
19. [What are the advantages of LLM-based heuristic design over traditional methods?](#19-what-are-the-advantages-of-llm-based-heuristic-design-over-traditional-methods)
20. [Are there known limitations or failure modes?](#20-are-there-known-limitations-or-failure-modes)

---

## 1. What is EoH and what problems can it solve?

**EoH (Evolution of Heuristics)** is a framework that combines Evolutionary Computation (EC) with Large Language Models (LLMs) to automatically design algorithms and heuristics for search and optimization problems — without requiring manual expert design.

EoH co-evolves both the reasoning behind a heuristic ("thoughts") and its code implementation, using LLMs as intelligent mutation/crossover operators across generations.

It has been applied to 33+ problem types, including:

- **Combinatorial optimization**: TSP, CVRP, bin packing, nurse rostering, circle packing
- **Metaheuristic component design**: PSO velocity updates, DE mutation strategies, SA acceptance criteria, CMA-ES updates
- **Constructive algorithms**: Greedy heuristics for routing and scheduling
- **Machine learning components**: GNN aggregation functions, Bayesian optimization acquisition functions
- **Dynamic/online problems**: Strategies that adapt to changing environments

EoH was accepted at **ICML 2024 (Oral, Top 1.5%)**, set a **world record on the Circle Packing Problem**, and won the **CVRPLib BKS competition** with 51 new Best Known Solutions.

---

## 2. How do I install EoH?

```bash
git clone https://github.com/FeiLiu36/EoH.git
cd eoh
pip install .
```

**Requirements:** Python >= 3.10, `numpy`, `joblib`.

You can verify the installation with:

```python
import eoh
print(eoh.__version__)
```

---

## 3. Which LLMs are supported?

EoH supports any LLM accessible via an **OpenAI-compatible API**, as well as local inference servers.

| Provider | Notes |
|---|---|
| OpenAI (GPT-4o, GPT-4, etc.) | Set `api_endpoint="api.openai.com"` |
| DeepSeek (deepseek-chat, etc.) | Set `api_endpoint="api.deepseek.com"` |
| Any OpenAI-compatible endpoint | Works out of the box |
| Local LLM (HuggingFace models) | Set `use_local=True` and `local_url` to your server URL |

For best results, use a capable code-generation model (GPT-4o, DeepSeek-V3, etc.). Weaker models may produce more syntactically invalid programs.

---

## 4. How do I configure the LLM (API key, endpoint, model)?

Use `LLMConfig` to set up your LLM backend:

```python
from eoh import LLMConfig

llm = LLMConfig(
    api_endpoint="api.deepseek.com",   # host, no https://
    api_key="your-api-key-here",
    model="deepseek-chat",
    timeout=150,                        # seconds per LLM call
)
```

**For local LLMs:**

```python
llm = LLMConfig(
    use_local=True,
    local_url="http://localhost:8080",  # your inference server URL
    model="your-model-name",
    timeout=180,
)
```

Common mistakes:
- Do **not** include `https://` in `api_endpoint` — just the hostname.
- Make sure `api_key` is correct for your provider; DeepSeek keys are different from OpenAI keys.
- If using a proxy or third-party endpoint, verify the URL is exactly correct — even a trailing slash can cause failures.

---

## 5. My API calls keep timing out — how do I fix this?

There are two separate timeouts to be aware of:

| Parameter | What it controls | Where to set it |
|---|---|---|
| `LLMConfig(timeout=...)` | Max seconds to wait for one LLM response | `LLMConfig` |
| `BaseProblem(timeout=...)` | Max seconds allowed to evaluate one generated program | `BaseProblem` subclass |

**For slow models** (e.g., DeepSeek-R1 with chain-of-thought reasoning), the default `timeout=180` is often too short. Try `timeout=300` or higher.

**For slow evaluations** (complex problem instances), increase the problem-level timeout similarly.

Network instability is another common cause of OpenAI API failures. If you see intermittent errors, verify your network connection and consider adding retries at the infrastructure level.

---

## 6. How do I define my own optimization problem?

Subclass `BaseProblem` and implement two required attributes:

```python
from eoh import BaseProblem
import numpy as np

class MyProblem(BaseProblem):
    # The code skeleton the LLM will evolve
    template_program = '''
def heuristic(items: list, capacity: float) -> list:
    """Select items to maximize total value without exceeding capacity."""
    return sorted(items, key=lambda x: x[1] / x[0], reverse=True)
'''
    # Natural-language description of the design goal
    task_description = (
        "Design a heuristic that selects items for a knapsack to maximise "
        "total value without exceeding the weight capacity."
    )

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        """Return a fitness score (lower is better). Return None on failure."""
        try:
            items = [(2, 3), (3, 4), (4, 5), (5, 8), (9, 10)]
            selected = callable_func(items, capacity=10.0)
            return -sum(v for _, v in selected)  # negate: higher value → lower fitness
        except Exception:
            return None
```

Key rules:
- `evaluate_program` must return a **float** (lower is better) or **None** if the program is invalid.
- Keep `task_description` clear and concise — it is injected directly into LLM prompts.
- Instantiate with `MyProblem(timeout=30, n_processes=4)`.

---

## 7. What are the supported template types?

EoH supports three template styles:

**1. Single function** (most common)
The LLM evolves one function. Use this for most heuristic design tasks.

**2. Multi-function**
Multiple cooperating functions; the last defined function is the entry point called by `evaluate_program`. Useful when you want the LLM to design a helper alongside the main heuristic.

**3. Class**
A class template with a designated method as the entry point. Useful for stateful heuristics or when object-oriented structure is natural for the problem.

The template type is inferred automatically from the structure of `template_program`.

---

## 8. What should I put in the initial template function?

**Keep it as simple as possible.** A minimal, correct baseline is better than a complex one. Good choices:

- A trivially correct but naive implementation (e.g., random selection, first-fit, nearest-neighbour)
- A well-known simple algorithm for the domain (e.g., greedy by ratio for knapsack)

Avoid:
- Overly complex implementations that constrain the search space
- Implementations with many comments that encode too much domain knowledge upfront (see Q17)

The initial template sets the function signature and docstring that the LLM must respect — make those clear and informative.

If you have known good algorithms you want EoH to start from, use the seed mechanism (see Q14) instead of encoding them in the template.

---

## 9. Does EoH maximize or minimize the fitness value?

**EoH minimizes** the value returned by `evaluate_program`. Lower return values are considered better.

To **maximize** an objective (e.g., total collected value), simply negate the return value:

```python
def evaluate_program(self, program_str, callable_func):
    value = run_evaluation(callable_func)
    return -value  # negate so that higher value → lower (better) fitness
```

---

## 10. What are the evolutionary operators (e1, e2, m1, m2)?

EoH uses four operators that combine LLM generation with evolutionary search:

| Operator | Type | Description |
|---|---|---|
| `e1` | Crossover | Combines **code** from two parent programs |
| `e2` | Crossover | Combines **thoughts** (reasoning) and code from two parents |
| `m1` | Mutation | Modifies a single program's **code** |
| `m2` | Mutation | Modifies a single program's **thoughts** then regenerates code |

By default all four are used with equal weight: `operators=['e1', 'e2', 'm1', 'm2']`.

You can restrict to a subset or assign custom weights:

```python
eoh = EoH(
    ...,
    operators=['e1', 'm1'],
    operator_weights=[0.7, 0.3],
)
```

---

## 11. How do I run EoH and what are the key configuration parameters?

```python
from eoh import EoH, LLMConfig

llm = LLMConfig(api_endpoint="api.deepseek.com", api_key="...", model="deepseek-chat")
problem = MyProblem(timeout=40, n_processes=4)

eoh = EoH(
    llm=llm,
    problem=problem,
    pop_size=5,          # population size per generation
    n_pop=20,            # number of generations
    operators=['e1', 'e2', 'm1', 'm2'],
    output_dir="./results",
    debug=False,
)
eoh.run()
```

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `pop_size` | 5 | Programs kept per generation |
| `n_pop` | 20 | Number of generations to run |
| `operators` | all four | Evolutionary operators to use |
| `operator_weights` | uniform | Sampling weight per operator |
| `n_parents` | 2 | Parents used for crossover |
| `output_dir` | `"./"` | Directory for logs and results |
| `debug` | `False` | Enable verbose logging |
| `use_seed` | `False` | Load initial population from a seed file |
| `use_continue` | `False` | Resume from a previous run |

---

## 12. Where are the results saved?

Results are written to `output_dir` (default: `./`), structured as:

```
results/
  run_log.txt                          # evolution progress, fitness per generation
  samples/
    samples_0~N.json                   # evaluated programs saved in batches (code + fitness + thoughts)
    ...
    samples_best.json                  # best program found across all generations
  pops/
    population_generation_1.json       # full population snapshot after generation 1
    population_generation_2.json
    ...
  pops_best/
    population_generation_1.json       # best-individual snapshot per generation
    population_generation_2.json
    ...
```

The best solution is always in `samples/samples_best.json`. If this file is missing or empty, check `run_log.txt` for errors — often the LLM timed out before returning any valid program.

---

## 13. How do I resume a run that was interrupted?

Set `use_continue=True` when creating the `EoH` instance and point `output_dir` to the same directory as the previous run:

```python
eoh = EoH(
    llm=llm,
    problem=problem,
    pop_size=5,
    n_pop=20,
    output_dir="./results",   # same as before
    use_continue=True,
)
eoh.run()
```

EoH will load the last saved population from `pops/` and continue from where it left off.

---

## 14. How do I seed EoH with hand-crafted algorithms?

You can provide an initial population of known algorithms so EoH starts from a strong baseline rather than evolving from scratch. Prepare a seed JSON file with the same format as a population snapshot (`pops/population_generation_N.json`), then:

```python
eoh = EoH(
    ...,
    use_seed=True,
    seed_path="./my_seed_algorithms.json",
)
eoh.run()
```

Each entry in the seed file should contain at minimum `"code"` and `"fitness"` fields. See the `examples/` directory for concrete seed file formats.

---

## 15. How do I speed up evaluation with parallel workers?

Set `n_processes` when instantiating your problem:

```python
problem = MyProblem(timeout=40, n_processes=-1)  # -1 uses all available CPUs
```

Each worker evaluates one generated program independently. Parallel evaluation is safe because each program runs in an isolated subprocess with a hard timeout enforced by `joblib`.

Note: if your evaluation function itself uses multiprocessing internally, nest carefully to avoid spawning too many processes.

---

## 16. Why does EoH produce no valid results or always return None?

Common causes:

- **LLM timeout**: The model takes longer than `LLMConfig(timeout=...)` to respond. Increase the timeout (see Q5).
- **Evaluation timeout**: The generated program runs longer than `BaseProblem(timeout=...)`. Increase the problem timeout or simplify your evaluation.
- **Syntax errors in generated code**: Weaker models produce more invalid Python. Switch to a stronger model or add a try/except in `evaluate_program` that returns `None` on exception.
- **Wrong return type**: `evaluate_program` must return a `float` or `None`. Returning a non-numeric type (e.g., a list) will silently fail.
- **Template mismatch**: If `callable_func`'s signature doesn't match how you call it in `evaluate_program`, every program will throw a `TypeError`. Double-check argument names in `template_program`.

Enable `debug=True` in `EoH(...)` to see full LLM responses and tracebacks from failed evaluations.

---

## 17. How do comments in the template affect EoH performance?

Comments in `template_program` are included in the LLM prompt and influence what the model generates. The general principle is: **simpler is better once the essentials are explained.**

- A **one-line docstring** describing the function's purpose and argument types is helpful and should always be present.
- **Inline comments** that explain a subtle constraint or invariant can help the LLM avoid invalid designs.
- **Extensive comments** encoding domain knowledge or a specific algorithmic approach can inadvertently constrain the search space and reduce diversity.

When in doubt, start minimal and add comments only if the LLM consistently misunderstands the task.

---

## 18. How does EoH compare to FunSearch and AEL?

| | EoH | FunSearch | AEL |
|---|---|---|---|
| **Co-evolves thoughts** | Yes | No | No |
| **Operators** | 4 (e1, e2, m1, m2) | mutation only | 3 |
| **LLM queries (bin packing)** | ~500 | ~1000+ | ~500 |
| **Result quality** | Surpasses FunSearch | Strong baseline | Competitive |
| **Venue** | ICML 2024 (Oral) | Nature 2023 | GECCO 2024 |

EoH's key advantage over FunSearch is the co-evolution of "thoughts" (natural-language reasoning) alongside code, which provides richer diversity and better performance with fewer LLM calls. AEL uses a similar LLM-EC combination but without thought evolution.

---

## 19. What are the advantages of LLM-based heuristic design over traditional methods?

- **High automation**: No manual feature engineering or domain-specific tuning required beyond defining the problem interface.
- **No training data needed**: LLMs bring prior knowledge from pretraining; you do not need to collect or label examples.
- **Interpretability**: Generated heuristics are human-readable Python code, unlike neural network policies.
- **Flexibility**: The same framework applies across diverse problem types without algorithmic changes.
- **Speed**: Competitive heuristics are typically found within minutes to a few hours on a laptop.

---

## 20. Are there known limitations or failure modes?

- **Complex problem descriptions**: Extracting a concise, unambiguous task description from a complex real-world problem can be challenging. Poorly written `task_description` values are a common source of low-quality results.
- **Model capability ceiling**: The quality of evolved heuristics is bounded by the LLM's code generation ability. Very weak models may never produce valid programs.
- **Evaluation bottleneck**: If a single evaluation takes minutes, the total wall-clock time grows quickly. Use `n_processes` and keep `timeout` tight.
- **Reproducibility**: LLM outputs are stochastic. Two runs with identical settings will produce different heuristics. Run multiple seeds and take the best for publication.
- **Template design sensitivity**: The choice of initial template and function signature influences what the LLM explores. A poorly designed template (e.g., wrong return type, ambiguous argument names) can significantly degrade performance.
