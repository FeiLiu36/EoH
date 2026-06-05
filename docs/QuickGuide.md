# Step-by-step Guide: Applying EoH to Your Own Problem

This guide walks you through everything needed to use EoH to automatically design a heuristic or algorithm for a problem you define. By the end you will have a working `prob.py`, a `runEoH.py`, and know how to extract and evaluate the result.

---

## Overview

EoH works by repeatedly asking an LLM to improve a Python function (or class). You supply:

1. A **code template** — the function skeleton the LLM will evolve.
2. A **task description** — one or two sentences describing what the function should do.
3. An **evaluation function** — Python code that runs the generated function and returns a fitness score.

EoH handles all LLM calls, evolutionary selection, and logging automatically.

---

## Step 1 — Install EoH

We recommend a conda environment with Python ≥ 3.10.

```bash
git clone https://github.com/FeiLiu36/EoH.git
cd EoH/eoh
pip install .
```

Verify:

```python
import eoh
print(eoh.__version__)
```

---

## Step 2 — Identify the heuristic component to evolve

EoH evolves **one replaceable component** of your algorithm — the part where expert intuition normally lives. Before writing any code, answer these questions:

| Question | Example answer |
|---|---|
| What decision needs to be made at each step? | "Which item to add to the knapsack next?" |
| What inputs are available at that decision point? | Item weights, values, remaining capacity |
| What does the component return? | The index of the chosen item |
| How do you measure quality of a complete solution? | Total value collected |

Keep the component small and focused. If your problem has multiple heuristic choices, start with the most impactful one.

---

## Step 3 — Choose a template type

EoH supports three template styles. Choose the one that matches your component:

| Type | When to use | `callable_func` in `evaluate_program` |
|---|---|---|
| **Single function** | One standalone heuristic decision | The function itself — call it directly |
| **Multi-function** | Main function plus helper sub-functions | The **last** top-level function — helpers are called internally by it |
| **Class** | Stateful heuristic, or OOP structure is natural | The **class** — you must call `callable_func()` to instantiate |

The type is detected automatically from the structure of `template_program`:
- Contains a `class` definition → **class**
- Contains more than one top-level `def` → **multi-function**
- Contains exactly one top-level `def` → **single function**

> **`np` is always available** in generated code — the framework injects `numpy` as `np` automatically. Any additional `import` statements at the top of your `template_program` (e.g., `import math`) are also propagated to generated code.

---

## Step 4 — Write your problem class

Create `prob.py` and subclass `BaseProblem`. You must define `template_program`, `task_description`, and `evaluate_program`.

### 4a — `template_program`: the code skeleton

Write the simplest correct implementation of your heuristic component. This sets:
- The **function/class signature** the LLM must respect.
- The **docstring** explaining inputs, outputs, and units.
- A **baseline implementation** EoH starts from.

```python
template_program = '''
def select_item(items: np.ndarray, remaining_capacity: float) -> int:
    """Select the next item to add to the knapsack.

    Args:
        items: array of shape (n, 2) where items[i] = [weight, value]
        remaining_capacity: remaining weight capacity

    Returns:
        Index of the selected item, or -1 if no item fits.
    """
    feasible = np.where(items[:, 0] <= remaining_capacity)[0]
    if len(feasible) == 0:
        return -1
    return feasible[np.argmax(items[feasible, 1] / items[feasible, 0])]
'''
```

**Tips:**
- Use precise, self-explanatory argument names — the LLM reads them directly.
- Start with a naive but correct baseline (e.g., greedy by ratio, nearest neighbour). A complex template constrains the LLM's search.
- Include type annotations — they help the LLM generate correct code.
- Do not over-comment. A clear docstring is enough. Extensive inline comments encoding domain knowledge can reduce solution diversity.

### 4b — `task_description`: what the LLM should optimise

Write one or two sentences describing the design goal. This is injected directly into the LLM prompt.

```python
task_description = (
    "Design a heuristic function that selects the next item to add to a 0/1 knapsack "
    "to maximise total collected value without exceeding the weight capacity."
)
```

**Tips:**
- Be specific about the objective (maximise value, minimise tour length, etc.).
- Mention any hard constraints (e.g., capacity must not be exceeded).
- Name the classic baseline if one exists — this gives the LLM context without constraining it.
- Avoid vague phrasing like "a good function" — state what "good" means quantitatively.

### 4c — `evaluate_program`: the fitness function

This method receives the generated callable and must return a `float` (lower is better) or `None` if the program is invalid.

**Key rules:**
- Always wrap in `try/except` and return `None` on any exception — LLM-generated code can crash, have type errors, or return wrong shapes.
- EoH **minimises** the return value. Negate objectives you want to maximise.
- Evaluate on **multiple instances** (≥ 8 recommended) so the fitness generalises.
- Keep individual evaluations fast. Use `n_processes` to parallelise.
- **Return `None`** (not `float('inf')`) when evaluation fails — `None` signals an invalid program, not a bad one.

#### For single-function templates

`callable_func` is the function itself. Call it the same way the template signature shows.

```python
def evaluate_program(self, program_str: str, callable_func) -> float | None:
    try:
        total = 0.0
        for items, capacity, optimal in self.instances:
            value = self._run_greedy(callable_func, items, capacity)
            total += value / optimal          # normalise by optimal value
        return -total / len(self.instances)   # negate: higher ratio → lower fitness
    except Exception:
        return None
```

#### For multi-function templates

`callable_func` is the **last top-level function** (the entry point). Any helper functions defined before it in the template live in the same exec namespace and are called directly by the entry function — you do not need to pass them separately.

```python
# template has: compute_scores() then select_next_node()
# callable_func == select_next_node; it calls compute_scores internally.
def evaluate_program(self, program_str: str, callable_func) -> float | None:
    try:
        result = callable_func(current_node, destination_node, unvisited, distance_matrix)
        ...
    except Exception:
        return None
```

#### For class templates

`callable_func` is the **class object** (not an instance). You must instantiate it inside `evaluate_program`.

```python
# template defines: class Metaheuristic with a solve() method
def evaluate_program(self, program_str: str, callable_func) -> float | None:
    try:
        solver = callable_func(self.func, self.dim, self.bounds, self.budget)
        x_best = solver.solve()
        return float(self.func(x_best))
    except Exception:
        return None
```

For stateful heuristics called step-by-step, instantiate once per problem instance:

```python
def evaluate_program(self, program_str: str, callable_func) -> float | None:
    try:
        constructor = callable_func()          # instantiate once per evaluation
        for instance, distance_matrix in self.instance_data:
            next_node = constructor.select_next_node(
                current_node, destination_node, unvisited, distance_matrix
            )
            ...
    except Exception:
        return None
```

### 4d — Designing the fitness value

The choice of fitness representation significantly affects evolution quality.

| Situation | Recommendation |
|---|---|
| Maximise an objective (value, HV) | Return `-objective` |
| Objectives vary greatly in scale across instances | Normalise per instance (e.g., divide by lower bound or optimal) |
| Objective spans many orders of magnitude | Apply `np.log1p(value)` before returning |
| Multi-objective: measure solution quality | Use hypervolume; return `-mean_HV` |
| Hard constraint violations | Return `None` (invalid), not a large penalty |

---

## Step 5 — Complete `prob.py` example (single-function)

```python
import numpy as np
from eoh import BaseProblem


class KnapsackProblem(BaseProblem):
    template_program = '''
def select_item(items: np.ndarray, remaining_capacity: float) -> int:
    """Select the next item to add to the knapsack.

    Args:
        items: array of shape (n, 2) where items[i] = [weight, value]
        remaining_capacity: remaining weight capacity

    Returns:
        Index of the selected item, or -1 if no item fits.
    """
    feasible = np.where(items[:, 0] <= remaining_capacity)[0]
    if len(feasible) == 0:
        return -1
    return feasible[np.argmax(items[feasible, 1] / items[feasible, 0])]
'''
    task_description = (
        "Design a heuristic function that selects the next item to add to a 0/1 knapsack "
        "to maximise total collected value without exceeding the weight capacity."
    )

    def __init__(self, n_instances=16, n_items=50, timeout=30, n_processes=4):
        super().__init__(timeout=timeout, n_processes=n_processes)
        rng = np.random.default_rng(42)
        self.instances = []
        for _ in range(n_instances):
            weights = rng.uniform(1, 10, n_items)
            values  = rng.uniform(1, 10, n_items)
            items   = np.stack([weights, values], axis=1)
            capacity = weights.sum() * 0.5
            optimal = max(self._dp_optimal(items, capacity), 1.0)
            self.instances.append((items, capacity, optimal))

    @staticmethod
    def _dp_optimal(items, capacity):
        cap = int(capacity)
        dp = [0.0] * (cap + 1)
        for w, v in items:
            w = int(w)
            for c in range(cap, w - 1, -1):
                dp[c] = max(dp[c], dp[c - w] + v)
        return dp[cap]

    def _run_greedy(self, func, items, capacity):
        remaining = items.copy()
        mask = np.ones(len(items), dtype=bool)
        cap = capacity
        value = 0.0
        for _ in range(len(items)):
            idx = func(remaining[mask], cap)
            if idx == -1:
                break
            original_idx = np.where(mask)[0][idx]
            w, v = items[original_idx]
            if w > cap:
                break
            cap -= w
            value += v
            mask[original_idx] = False
        return value

    def evaluate_program(self, program_str: str, callable_func) -> float | None:
        try:
            total = 0.0
            for items, capacity, optimal in self.instances:
                v = self._run_greedy(callable_func, items, capacity)
                total += v / optimal
            return -total / len(self.instances)   # negate: higher ratio → lower fitness
        except Exception:
            return None
```

For class and multi-function examples, see `examples/tsp_construct_class` and `examples/tsp_construct_multifunction`.

---

## Step 6 — Configure the LLM

```python
from eoh import LLMConfig

llm = LLMConfig(
    api_endpoint="api.deepseek.com",   # hostname only — no https://
    api_key="your-api-key",
    model="deepseek-chat",             # e.g. "gpt-4o", "deepseek-chat"
    timeout=150,                       # seconds per LLM call; increase for slow/reasoning models
)
```

**Local LLM:**

```python
llm = LLMConfig(
    use_local=True,
    local_url="http://127.0.0.1:8080/completions",
    timeout=180,
)
```

**Common mistakes:**
- Do **not** include `https://` in `api_endpoint` — just the hostname.
- Reasoning models (e.g., DeepSeek-R1) produce long chain-of-thought output; set `timeout=300` or higher.
- The `timeout` here controls how long to wait for a single LLM response, separate from the evaluation timeout on `BaseProblem`.

---

## Step 7 — Create `runEoH.py` and run

```python
from eoh import EoH, LLMConfig
from prob import KnapsackProblem

llm = LLMConfig(
    api_endpoint="api.deepseek.com",
    api_key="your-api-key",
    model="deepseek-chat",
    timeout=150,
)

problem = KnapsackProblem(n_instances=16, n_items=50, timeout=30, n_processes=4)

eoh = EoH(
    llm=llm,
    problem=problem,
    pop_size=5,        # programs kept per generation
    n_pop=20,          # number of generations
    operators=['e1', 'e2', 'm1', 'm2'],
    output_dir="./results",
    debug=False,
)

eoh.run()
```

Run it:

```bash
python runEoH.py
```

**Key parameters:**

| Parameter | Default | Guideline |
|---|---|---|
| `pop_size` | 5 | 5–10 for quick experiments; 10–20 for thorough search |
| `n_pop` | 20 | 10–20 for a quick run; 50+ for thorough search |
| `operators` | all four | Start with `['e1', 'e2', 'm1', 'm2']`; restrict if LLM budget is tight |
| `operator_weights` | uniform | E.g. `[0.3, 0.3, 0.2, 0.2]` to weight crossover more |
| `n_processes` | problem default | Overrides `n_processes` set on the problem instance |
| `debug` | `False` | Set `True` to see LLM responses and evaluation tracebacks |

**Evolutionary operators:**

| Operator | Type | What it does |
|---|---|---|
| `e1` | Crossover | Combines **code** from two parent programs |
| `e2` | Crossover | Combines **thoughts** (reasoning) and code from two parents |
| `m1` | Mutation | Modifies one program's **code** |
| `m2` | Mutation | Modifies one program's **thoughts**, then regenerates code |

---

## Step 8 — Monitor progress

EoH writes results to `output_dir` as it runs:

```
results/
  run_log.txt                     ← fitness per generation (watch this live)
  samples/
    samples_0~N.json              ← all evaluated programs (code + fitness + thoughts)
    samples_best.json             ← best program found so far
  pops/
    population_generation_N.json  ← full population snapshot after generation N
  pops_best/
    population_generation_N.json  ← best individual per generation
```

Watch progress live:

```bash
tail -f results/run_log.txt
```

The best heuristic at any point is in `results/samples/samples_best.json`.

---

## Step 9 — Extract and evaluate the best heuristic

### Extract the code

Open `results/samples/samples_best.json`. Copy the `"code"` field into a file, e.g., `heuristic.py`.

```python
# heuristic.py — paste the evolved function here
import numpy as np

def select_item(items: np.ndarray, remaining_capacity: float) -> int:
    # ... evolved code ...
```

### Evaluate on a held-out test set

Run the heuristic on test instances that were **not** used during evolution:

```python
# runEval.py
import numpy as np
from heuristic import select_item
from prob import KnapsackProblem

test = KnapsackProblem(n_instances=64, n_items=50)   # use a different seed/size
ratios = []
for items, capacity, optimal in test.instances:
    value = test._run_greedy(select_item, items, capacity)
    ratios.append(value / optimal)

print(f"Average optimality ratio: {np.mean(ratios):.4f}")
```

Each example in `examples/*/evaluation/` follows the same pattern — copy your heuristic to `heuristic.py` then run `runEval.py`.

---

## What to do when results are unsatisfying?

> [!TIP]
> Come back to this section whenever a run finishes but the results are not good enough. Work through the checklist in order — most issues are caught by a direct checking of your prob.py and evaluate_program.

Work through this checklist in order:

1. **Check `run_log.txt`** — if fitness never improves or all entries say `None`, most programs are failing. Enable `debug=True` to see why.
2. **Check your `evaluate_program`** — run the template function manually to confirm it returns the right type and your evaluation logic is correct before launching EoH.
3. **Improve the task description** — vague descriptions produce generic code. Add the objective, constraints, and the name of any known baseline.
4. **Simplify the template** — strip to the bare signature + the simplest correct baseline. A complex template constrains the LLM's search space.
5. **Increase compute** — raise `pop_size` and `n_pop`, or switch to a stronger LLM (GPT-4o, DeepSeek-V3).
6. **Seed with known algorithms** — if you already have a good baseline, start EoH from it:

   ```python
   eoh = EoH(..., use_seed=True, seed_path="./my_seed.json")
   ```

   The seed file has the same format as a population snapshot (`pops/population_generation_N.json`). Each entry needs at least `"code"` and `"objective"` fields.

7. **Resume an interrupted run** — point `continue_path` to the last saved population file and set `continue_id` to the last completed generation number:

   ```python
   eoh = EoH(
       ...,
       output_dir="./results",
       use_continue=True,
       continue_path="./results/pops/population_generation_10.json",
       continue_id=10,
   )
   ```

   > The default `continue_path` is `"./results/pops/population_generation_0.json"`. If you used a different `output_dir` or want to resume from a later generation, you must set `continue_path` explicitly.

---

## Common problems and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| Most/all evaluations show `None` in `run_log.txt` | LLM timeout or generated code crashes | Increase `LLMConfig(timeout=...)`; enable `debug=True` to see the traceback |
| Valid programs are generated but every one times out | `BaseProblem(timeout=...)` too short for your evaluation | Increase the problem timeout, or reduce the number/size of instances evaluated per call |
| `evaluate_program` silently fails — returns `None` for every valid-looking program | Returning a non-`float` type (e.g., a numpy array, list, or string) instead of `float` or `None` | Wrap the return value: `return float(score)` |
| `TypeError` on every call to `callable_func` | Argument names or order in the call don't match the `template_program` signature | Check the template signature and align your `evaluate_program` call exactly |
| Class template: every call raises an error | `callable_func` is the class object — it must be instantiated first | Replace `callable_func(args)` with `instance = callable_func(...); instance.method(args)` |
| Multi-function: `NameError` for a helper function | Helper defined **after** the entry function in the template | Move all helper functions above the entry (last top-level) function |
| `continue_path` not found when resuming | Default path `"./results/pops/population_generation_0.json"` doesn't match your `output_dir` or the last saved generation | Set `continue_path` explicitly, e.g. `"./results/pops/population_generation_10.json"`, and set `continue_id=10` |
| LLM connection errors or no response | `https://` included in `api_endpoint`, or wrong API key | Use hostname only: `"api.deepseek.com"`, not `"https://api.deepseek.com"` |
| Poor results after many generations | `task_description` too vague, or template encodes too much domain knowledge | State the objective precisely in `task_description`; strip `template_program` to the simplest correct baseline |

---

## Worked examples in this repository

Study these examples before writing your own:

| Task type | Example | What to study |
|---|---|---|
| Single scoring function | `examples/tsp_construct` | Minimal `prob.py` structure |
| Online packing (gap-ratio fitness) | `examples/bp_online` | EoH paper benchmark; gap-normalised fitness |
| Multi-function template | `examples/tsp_construct_multifunction` | Helper + entry function pattern |
| Class template (stateful) | `examples/tsp_construct_class` | Instantiating `callable_func()` in `evaluate_program` |
| Complete metaheuristic (class) | `examples/bbob_metaheuristic` | Designing a full solver; `log1p` fitness |
| Multi-objective | `examples/moead_decomposition` | Negative hypervolume as fitness |
| Continuous component design | `examples/sa_acceptance`, `examples/de_mutation` | Calibrated fitness, multiple seeds |

Each example folder contains `prob.py`, `runEoH.py`, and an `evaluation/` subfolder for held-out testing.
