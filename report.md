# Issue Report: Modern JAX / Blackwell GPU Compatibility

## Status

Resolved locally and verified in the `stories` conda environment.

## Problem

The repository was pinned to an older JAX stack that no longer works well on
modern NVIDIA GPUs such as Ada Lovelace RTX 4xxx and Blackwell RTX 5xxx.

In practice, this caused several issues:

- older JAX/CUDA builds were not suitable for modern GPU targets
- a private JAX import had been removed in newer JAX versions
- Flax / Orbax checkpoint APIs had changed
- the OTT quadratic solver API had changed
- some transitive dependencies selected by `pip` were incompatible with the
  resolved JAX version

## What was changed

### Dependency updates

Updated `pyproject.toml` to move away from the old hard-pinned stack:

- removed `jaxlib = "0.4.26"`
- changed `jax` to `>=0.4.35,<0.7`
- changed `flax` to `>=0.9.0`
- changed `ott-jax` to `>=0.4.7`
- changed `numpy` to `>=1.26.4`
- changed `orbax-checkpoint` to `>=0.11.0`
- added explicit runtime dependencies:
  - `equinox >= 0.11.11`
  - `lineax >= 0.0.8`
  - `diffrax >= 0.7.0`

The temporary `<0.7` cap on JAX is intentional. The current OTT / Equinox stack
used by the project is not yet compatible with `jax 0.7.x`.

### Code fixes

Updated `stories/potentials.py`:

- replaced the removed private import
  - `from jax._src.nn.functions import gelu`
- with the public import
  - `from jax.nn import gelu`

Updated `stories/spacetime.py`:

- removed the obsolete `flax.training.orbax_utils` import
- replaced the old `EarlyStopping` import path
- added a small fallback `EarlyStopping` implementation for compatibility

Updated `stories/loss.py`:

- fixed the `ott-jax 0.6.x` FGW API change
- passed an explicit `linear_solver=Sinkhorn(threshold=1e-3)` into
  `GromovWasserstein(...)`
- changed `relative_epsilon=False` to `relative_epsilon=None`

Updated `stories/tools.py`:

- fixed a `ZeroDivisionError` in `DataLoader.train_or_val()` when
  `train_val_split=1.0`

Updated `README.md`:

- replaced the outdated CUDA install example with:

```bash
pip install stories-jax "jax[cuda12]>=0.4.35,<0.7"
```

## Environment work performed

The local `stories` environment was updated to validate the fix end to end:

1. uninstalled the older JAX / Flax / Orbax / OTT packages
2. reinstalled the project in editable mode
3. installed a CUDA 12 JAX build compatible with the updated constraints
4. upgraded missing or incompatible transitive dependencies
5. reinstalled the project in editable mode again so the environment matched the
   final `pyproject.toml`

## Verified working versions

- `jax 0.6.2`
- `jaxlib 0.6.2`
- `flax 0.11.2`
- `ott-jax 0.6.0`
- `orbax-checkpoint 0.11.33`
- `equinox 0.13.6`
- `lineax 0.1.0`
- `diffrax 0.7.2`

## Verification

The following checks passed in the updated environment:

- `import stories`
- `jax.devices()` returned `CudaDevice(id=0)`
- checkpoint manager creation succeeded
- a small end-to-end `SpaceTime.fit(...); transform(...)` smoke test succeeded
- the quadratic FGW training path also succeeded with:

```python
scheduler = optax.cosine_decay_schedule(1e-2, 10_000)
model.fit(
    adata=adata,
    time_key="time",
    omics_key="X_pca_harmony",
    space_key="spatial",
    weight_key="growth",
    optimizer=optax.adamw(scheduler),
    checkpoint_manager="...",
)
```

This specifically confirmed that the previous OTT error:

```text
TypeError: GromovWasserstein.__init__() missing 1 required positional argument: 'linear_solver'
```

is resolved by the current code.

## Notes

- `pytest` was not installed in the `stories` conda environment, so I used
  import checks and smoke tests for verification instead.
- JAX emitted `hwloc` and XLA autotuning warnings during GPU runs, but these did
  not block training, checkpointing, or inference.
- unrelated existing workspace changes were left untouched

## Suggested closeout

The Blackwell / modern CUDA 12 compatibility issue is resolved locally.

The project now:

- installs against a modern JAX stack
- imports successfully
- detects the CUDA device
- saves checkpoints correctly
- runs both the standard training/inference path and the quadratic FGW training
  path successfully
