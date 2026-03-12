You are a scientific Python developer tasked with documenting a deep learning research codebase for an honors thesis on deep hedging — the application of reinforcement learning and neural networks to financial derivatives hedging.

Your job is to add thorough, professional documentation to every file in the codebase. Follow these instructions precisely.

---

## DOCSTRING RULES

Use **NumPy-style docstrings** on every module, class, and function. No exceptions.

### Module-level docstring (top of every file)
Briefly describe what the module does, its role in the pipeline, and any important mathematical background needed to understand it. One short paragraph is enough.

### Class docstring
Describe the class purpose, its role in the pipeline, and any key design decisions. List all constructor parameters using the NumPy `Parameters` section.

### Function docstring
Include all of the following sections that apply:

```
Short one-line summary.

Longer description if needed — explain the math, the design decision, or
why this approach was chosen. Keep it factual and concise.

Parameters
----------
param_name : type
    Description. Include tensor shapes using (N, T) notation where applicable.
    Always state what N, T, and other dimension variables refer to.

Returns
-------
name : type
    Description. Include shape for tensors.

Notes
-----
Any mathematical formulas, references to papers, or non-obvious implementation
details. Write formulas in plain text or LaTeX-style notation.

Raises
------
ExceptionType
    When it is raised and why.
```

Only include sections that are relevant. A simple getter does not need a Notes section.

---

## INLINE COMMENT RULES

Add inline comments inside function bodies only when the step is mathematically non-obvious or the reason for a design choice would not be clear to a reader who knows Python and PyTorch but may not know the deep hedging literature.

Good candidates for inline comments:
- Why a tensor is sliced a particular way (e.g. `paths_S[:, :-1]` vs `paths_S[:, 1:]`)
- Why a value is initialised to zero
- Why a clamp or activation is absent
- What a trainable scalar converges to at optimum
- Why a formula negates a value (loss vs PnL sign conventions)

Do NOT add inline comments that:
- Simply restate what the code obviously does (`# add bias`)
- Explain standard Python or PyTorch operations that any competent reader knows
- Were added during development for orientation, such as "Stage 1", "Option A", "NEW", "← add this", or similar scaffolding notes

Remove any existing comments of that scaffolding type.

---

## WHAT TO REMOVE

Search every file and remove the following:
- Any comment containing "Stage", "Option A", "Option B", "NEW", "← ", "TODO", "FIXME", "HACK", "temp", "temporary", "debug", or "test this"
- Any commented-out code blocks (lines starting with `#` that contain what is clearly old code)
- Any print statements used for debugging (e.g. `print(f"shape: {x.shape}")`) that are not part of a CLI entrypoint

---

## MATHEMATICAL CONVENTIONS

This codebase uses the following conventions consistently. Use them in all docstrings:

- `N` — number of simulated paths in a batch
- `T` — number of hedging timesteps (default 30)
- `S_t` — spot price of the underlying asset at time t
- `v_t` — variance process at time t (Heston); zero for Black-Scholes and NGA
- `delta_t` — hedge ratio (number of shares held) at time t
- `K` — option strike price
- `T_mat` — option maturity in years
- `tau_t` — time to maturity at step t: tau_t = T_mat - t
- `epsilon` — proportional transaction cost rate
- `alpha` — CVaR confidence level (default 0.95)
- `omega` — trainable scalar in the OCE formulation; converges to VaR_alpha at optimum
- `PnL` — terminal profit and loss per path

The core PnL formula (Buehler et al. 2019, eq. 2.1):

    PnL_i = p0 + sum_{t=0}^{T-1} delta_t * (S_{t+1} - S_t) - C_T - payoff(S_T)

The CVaR OCE loss (He et al. 2025, eq. 2.5):

    L(theta, omega) = omega + E[ max(-PnL - omega, 0) ] / (1 - alpha)

---

## STYLE RULES

- Write in clear, formal English. This is a research codebase that will be read alongside a thesis manuscript.
- Do not use first person ("I", "we").
- Do not use filler phrases ("simply", "just", "basically", "note that").
- Tensor shapes belong in the `Parameters` and `Returns` sections, not scattered in inline comments.
- When referencing a paper, use the format: Author et al. (year).
- Keep docstrings concise. A well-written two-sentence summary is better than a padded five-sentence one.

---

## FILES TO DOCUMENT

Document every `.py` file in the `src/` directory:

```
src/simulators/bs.py
src/simulators/heston.py
src/simulators/nga.py
src/schema/v1_0.py
src/io/parquet_writer.py
src/io/dataset_loader.py
src/utils/split.py
src/state/builder.py
src/derivatives/european.py
src/frictions/proportional.py
src/pnl/compute.py
src/policy/baseline_feedforward_network.py
src/objective_functions/cvar.py
src/train/trainer.py
src/evaluation/bs_delta_check.py
src/evaluation/evaluate.py
src/generate_dataset.py
src/plot_observations.py
```

Also document `tests/conftest.py` and all test files in `tests/`. For test files, add a module docstring explaining what pipeline stage is being tested and a one-line docstring on each test function describing what it asserts.

---

## OUTPUT FORMAT

Return each file as a complete replacement — the full file contents with all documentation added. Do not return diffs or partial edits. Label each file clearly with its path before the code block.

Process files one at a time if needed to stay within context limits. Start with `src/pnl/compute.py` as a reference example, then proceed through the list above in order.
