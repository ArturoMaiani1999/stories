# Claude Code Agent Instructions: Upgrade `stories-jax` for Blackwell GPU Support

## Mission

Upgrade the JAX ecosystem dependencies in this repository so that the package
runs on modern NVIDIA GPUs (Ada Lovelace RTX 4xxx, Blackwell RTX 5xxx) without
changing any mathematical logic or algorithms.

---

## Environment context

| Item | Detail |
|---|---|
| Machine | Ubuntu 24.04, NVIDIA GeForce RTX 5070 Ti Laptop GPU (Blackwell, sm_100) |
| Driver | 580.126.09 — supports up to CUDA 13.0, do **not** touch it |
| Conda env | `stories` — Python 3.11, activate before any `pip` commands |
| Repo path | `~/github/stories` |
| Root problem | `jax==0.4.26` bundles a `ptxas` too old to compile for sm_90a/sm_100 |

The NVIDIA driver is fine. Only Python package versions need to change.

---

## What must NOT change

- All files under `stories/steps/` — the mathematical logic is correct
- `stories/loss.py` — the FGW/Sinkhorn solver calls are stable across ott-jax versions
- `stories/tools.py` — no JAX API issues here
- The overall architecture, class interfaces, and public API of `SpaceTime`

---

## Required changes

### 1. `pyproject.toml` — relax version pins

The current pins are exact locks (`jax = "0.4.26"`). Relax them to minimum
bounds so pip can resolve a version set that supports Blackwell.

```toml
# BEFORE
jaxlib = "0.4.26"
jax = "0.4.26"
flax = "0.8.2"
ott-jax = "0.4.6"
numpy = "^1.26.4"

# AFTER (target these minimums, allow newer)
jaxlib = ">=0.4.35"
jax = ">=0.4.35"
flax = ">=0.9.0"
ott-jax = ">=0.4.7"
numpy = ">=1.26.4"
```

Also add `jax[cuda12]` as an optional dependency or document it in the README
install section. Do not pin `jaxlib` separately — it is bundled inside
`jax[cuda12]` in newer releases.

---

### 2. `stories/potentials.py` — fix private import

Line 4 imports `gelu` from a private JAX internal path that was removed:

```python
# BEFORE (broken in jax >= 0.4.28)
from jax._src.nn.functions import gelu

# AFTER
from jax.nn import gelu
```

This is the only change needed in this file.

---

### 3. `stories/spacetime.py` — fix flax + orbax API changes

Three imports/usages broke between flax 0.8 and 0.9+:

#### 3a. `orbax_utils` was removed from flax

```python
# BEFORE
from flax.training import orbax_utils
# (used later as: orbax_utils.save_args_from_target(self.params))

# AFTER — remove the import entirely; it is no longer needed.
# StandardSave/StandardRestore handle this automatically in newer orbax.
```

Remove any call to `orbax_utils.save_args_from_target(...)` in the `fit()`
method. The `StandardSave` and `StandardRestore` args classes handle
serialization automatically now.

#### 3b. `EarlyStopping` moved

```python
# BEFORE
from flax.training.early_stopping import EarlyStopping

# AFTER
from flax.training import EarlyStopping
```

Verify the new path at runtime; if it still fails, implement a minimal inline
replacement:

```python
from dataclasses import dataclass

@dataclass
class EarlyStopping:
    min_delta: float = 0.0
    patience: int = 10
    best_loss: float = float("inf")
    counter: int = 0
    should_stop: bool = False

    def update(self, loss: float) -> "EarlyStopping":
        if loss < self.best_loss - self.min_delta:
            return EarlyStopping(self.min_delta, self.patience, loss, 0, False)
        counter = self.counter + 1
        return EarlyStopping(
            self.min_delta, self.patience, self.best_loss,
            counter, counter >= self.patience
        )
```

Place this class at the top of `spacetime.py` if the flax import fails.

#### 3c. `CheckpointManager` options API

In newer orbax the `best_fn` / `best_mode` kwargs moved into
`CheckpointManagerOptions`. Verify `tools.py::default_checkpoint_manager` still
works. If `CheckpointManagerOptions` raises on unknown kwargs, replace with:

```python
options = CheckpointManagerOptions(
    save_interval_steps=1,
    max_to_keep=1,
)
```

And accept that best-checkpoint selection may fall back to latest. This is a
minor regression acceptable for now — open a follow-up issue if needed.

---

### 4. `stories/steps/icnn_implicit.py` — verify jaxopt API

`jaxopt.OptaxSolver` is used with `implicit_diff=True`. In jaxopt >= 0.8.5 the
`implicit_diff` kwarg was changed to accept a `jaxopt.implicit_diff.ImplicitDiff`
object rather than a bool. Check the installed jaxopt version:

```bash
python -c "import jaxopt; print(jaxopt.__version__)"
```

If version >= 0.8.5, update `self.opt_hyperparams` construction in both
`ICNNImplicitStep` and `MongeImplicitStep`:

```python
# BEFORE
self.opt_hyperparams = {
    "maxiter": maxiter,
    "implicit_diff": implicit_diff,
    "tol": tol,
}

# AFTER (if jaxopt >= 0.8.5)
import jaxopt
self.opt_hyperparams = {
    "maxiter": maxiter,
    "implicit_diff": jaxopt.implicit_diff.CustomLinearSolve() if implicit_diff else False,
    "tol": tol,
}
```

If the old bool API still works, leave it unchanged.

---

## Install sequence

After making the above code changes, reinstall in this exact order:

```bash
conda activate stories

# 1. Uninstall old pinned versions cleanly
pip uninstall -y jax jaxlib jaxlib-cuda12-pjrt jaxlib-cuda12-plugin \
    flax ott-jax orbax-checkpoint stories-jax 2>/dev/null

# 2. Reinstall the package in editable mode (pulls new deps from updated pyproject.toml)
cd ~/github/stories
pip install -e .

# 3. Install the CUDA 12 JAX build last, so it wins over any CPU jax pulled in step 2
pip install "jax[cuda12]>=0.4.35"

# 4. Verify GPU is detected
python -c "import jax; print(jax.devices())"
# Expected: [CudaDevice(id=0)]

# 5. Verify stories imports cleanly
python -c "import stories; print('ok')"
```

---

## Verification checklist

Run these checks in order. Each must pass before moving to the next.

```bash
# GPU detected
python -c "import jax; assert str(jax.devices()[0]).startswith('cuda'), 'No GPU!'; print('GPU ok')"

# stories imports
python -c "import stories; print('import ok')"

# Core classes instantiate
python -c "
from stories import SpaceTime
from stories.steps import ExplicitStep, MongeImplicitStep, ICNNImplicitStep
m = SpaceTime()
print('SpaceTime ok')
"

# JAX jit + grad work on GPU (smoke test)
python -c "
import jax, jax.numpy as jnp
f = jax.jit(jax.grad(lambda x: jnp.sum(x**2)))
print(f(jnp.ones(4)))
print('jit+grad ok')
"
```

If any check fails, read the traceback carefully. Paste it back to Claude with
the message: _"Check N failed with this error — what do I fix?"_

---

## What to do if ott-jax breaks

`ott-jax` is the highest-risk dependency because Fused Gromov-Wasserstein is
its core feature and the API surface is non-trivial. If `loss.py` fails after
the upgrade:

1. Check the ott-jax changelog: `pip show ott-jax` to get the version, then
   look at https://github.com/ott-jax/ott/releases
2. The most likely breakage is in how `GromovWasserstein` returns its result
   object. In newer ott-jax the field may be `.reg_gw_cost` → `.primal_cost`.
   Search `loss.py` for `.reg_gw_cost` and `.reg_ot_cost` and check the ott-jax
   docs for the installed version.
3. Do not change the mathematical structure — only the attribute name used to
   extract the scalar loss.

---

## Out of scope

Do not touch:

- The NVIDIA driver or system CUDA installation
- The conda base environment
- Any other conda environments (`diarydb-llm`, `bench_cifar`, etc.)
- The test suite in `tests/` — run it after changes but do not modify it
- Documentation files
---

## Reporting back

When done, output a brief summary in this format:
```
FILES CHANGED:
- pyproject.toml: relaxed 4 version pins
- stories/potentials.py: 1 import fix
- stories/spacetime.py: removed orbax_utils, fixed EarlyStopping import
- stories/steps/icnn_implicit.py: [changed / no change needed]

VERSIONS INSTALLED:
- jax: X.X.X
- jaxlib: X.X.X
- flax: X.X.X
- ott-jax: X.X.X

VERIFICATION:
- GPU detected: yes/no
- stories import: ok/failed
- SpaceTime instantiation: ok/failed
- jit+grad smoke test: ok/failed
```
