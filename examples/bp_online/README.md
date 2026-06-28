# Online Bin Packing (OBP)

EoH evolves a **scoring heuristic** `score(item, bins)` for online bin packing: each
arriving item is placed in the feasible bin with the highest score. The goal is to
minimize the number of bins used.

- `runEoH.py` — evolve a heuristic (trained on a Weibull set, bin capacity **100**).
- `prob.py` / `get_instance.py` — problem definition, training data, L1 lower bound.
- `testingdata/` — Weibull test sets (1k–100k items) and `generate_instances.py`.
- `evaluation/runEval.py` — benchmark heuristics across sizes/capacities; reports
  **excess over the L1 lower bound** (%, lower is better).

> [!IMPORTANT]
> Please read the two issues below before using this example as a benchmark — they
> affect how results are generated and correct a discrepancy with the original EoH paper.

---

## Issue 1 — Item-size scaling across capacities (and an error in the original code)

Item sizes are drawn from `45 × Weibull(shape=3)`, clipped to `[1, capacity]` (mean
≈ 40 at capacity 100). The benchmark then sweeps the **bin capacity** (100, 500) over
the *same* item lists. There are two legitimate but different designs:

1. **Unscaled item sizes** (used here, and intended in EoH) — sizes stay drawn from
   the same distribution regardless of capacity. Items are relatively smaller in a
   large bin, so packing is easier and gaps **shrink as capacity grows**. Varying the
   capacity thus tests the heuristic under **different distributions / packing
   pressures** — expected behavior, not an artifact.
2. **Sizes scaled with capacity** (× `capacity/100`) — every capacity shares the same
   relative distribution, so difficulty stays comparable across capacities.

Both are valid; just be explicit about which you report. This repo uses design (1).

**The bug in the original EoH code:** when generating the five instances for a
non-default capacity, the original `get_instance.py` updated **only the first
instance's capacity; the other four stayed at 100.** Running the original `runEval.py`
reproduces the paper's numbers exactly — but they are **incorrect**, since only 1 of 5
instances used the intended capacity. This repo fixes it (every instance uses the
configured capacity), so the corrected capacity-500 columns differ from the paper.
The fix is being documented in the GitHub repo and the arXiv version of the paper.

*Thanks to Tai (University of St Andrews) for his careful investigation of this issue.*

---

## Issue 2 — Out-of-distribution generalization is unstable

EoH trains on a single distribution (Weibull, capacity 100). **Even on the same
training set, independent runs can produce heuristics with very different OOD
behavior** — training fitness does not predict it, and an evolved score function can
quietly **overfit to the training scale**.

This directory ships two heuristics, both at near-identical *training* fitness
(≈ 0.008 excess). Run `evaluation/runEval.py` to reproduce (excess over L1 bound, %):

| Method                    | 1k_c100 | 5k_c100 | 10k_c100 |  1k_c500 |  5k_c500 | 10k_c500 |
| ------------------------- | ------: | ------: | -------: | -------: | -------: | -------: |
| Best Fit                  |   4.46  |   4.18  |    4.01  |    0.99  |    0.45  |    0.50  |
| First Fit                 |   4.96  |   4.52  |    4.29  |    0.99  |    0.55  |    0.50  |
| EoH (original paper)      |   2.63  |   0.67  |    0.55  |    0.99  |    0.45  |    0.35  |
| EoH (this code, good)     |   2.53  |   0.75  |    0.52  |    0.74  |    0.45  |    0.37  |
| EoH (this code, poor)     |   3.12  |   0.68  |    0.32  | **208.64** | **210.02** | **210.17** |

Both are competitive at capacity 100 (the "poor" one is even best at `10k_c100`), but
at capacity 500 the "poor" heuristic collapses to **>200% excess** (~3× too many bins)
while the "good" one stays ~0.4%. The cause is **scale-invariance**:
`heuristic_EoH_by_this_code.py` normalizes by bin scale (`percentile(bins, 60)`,
`max(bins)`), whereas `heuristic_EoH_by_this_code_poor_generalization.py` uses
absolute-magnitude terms (e.g. `1/(1+remaining)`) tuned to capacity-100 leftovers.

**Guidance:** evaluate on multiple sizes *and* capacities; prefer scale-invariant
scores (normalize by capacity, use ratios); run multiple seeds and select on held-out
OOD performance, not training fitness.

Some solutions to this issue are discussed in EoH-S [1] and MoH [2] etc.

[1] [EoH-S: Evolution of Heuristic Set using LLMs for Automated Heuristic Design](https://ojs.aaai.org/index.php/AAAI/article/view/41038), AAAI 2026
[2] [Generalizable Heuristic Generation Through LLMs with Meta-Optimization](https://openreview.net/forum?id=tIQZ7pVN6S), ICLR 2026

---

## Reproducing

```bash
python runEoH.py            # evolve a heuristic
cd evaluation && python runEval.py   # benchmark; writes results.txt
```
