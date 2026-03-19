import stories
import stories.steps
import anndata as ad
import jax
import numpy as np
import tempfile

key = jax.random.PRNGKey(0)
key_0, key_1, key_2 = jax.random.split(key, 3)

x_0 = jax.random.normal(key_0, shape=(100, 20))
x_1 = jax.random.normal(key_1, shape=(100, 20))
x_2 = jax.random.normal(key_2, shape=(100, 20))

adata_0 = ad.AnnData(np.array(x_0))
adata_0.obs["time"] = 0
adata_1 = ad.AnnData(np.array(x_1))
adata_1.obs["time"] = 1
adata_2 = ad.AnnData(np.array(x_2))
adata_2.obs["time"] = 2

adata = ad.concat((adata_0, adata_1, adata_2))
adata.obsm["X_pca"] = adata.X.copy()
adata.obsm["spatial"] = adata.X[:, :2].copy()


def test_model_explicit():
    step = stories.steps.ExplicitStep()
    model = stories.SpaceTime(proximal_step=step)
    model.fit(
        adata,
        time_key="time",
        omics_key="X_pca",
        space_key="spatial",
        batch_size=50,
        max_iter=5,
    )
    model.transform(adata, omics_key="X_pca", tau=1.0, batch_size=50)


def test_model_linear():
    step = stories.steps.ExplicitStep()
    model = stories.SpaceTime(proximal_step=step, quadratic=False)
    model.fit(
        adata,
        time_key="time",
        omics_key="X_pca",
        space_key="spatial",
        batch_size=50,
        max_iter=5,
    )
    model.transform(adata, omics_key="X_pca", tau=1.0, batch_size=50)


def test_model_10steps():
    step = stories.steps.ExplicitStep()
    model = stories.SpaceTime(proximal_step=step, quadratic=False, n_steps=10)
    model.fit(
        adata,
        time_key="time",
        omics_key="X_pca",
        space_key="spatial",
        batch_size=50,
        max_iter=5,
    )
    model.transform(adata, omics_key="X_pca", tau=1.0, batch_size=50)


def test_model_ICNN_implicit():
    step = stories.steps.ICNNImplicitStep()
    model = stories.SpaceTime(proximal_step=step)
    model.fit(
        adata,
        time_key="time",
        omics_key="X_pca",
        space_key="spatial",
        batch_size=50,
        max_iter=5,
    )
    model.transform(adata, omics_key="X_pca", tau=1.0, batch_size=50)


def test_model_monge_implicit():
    step = stories.steps.MongeImplicitStep()
    model = stories.SpaceTime(proximal_step=step)
    model.fit(
        adata,
        time_key="time",
        omics_key="X_pca",
        space_key="spatial",
        batch_size=50,
        max_iter=5,
    )
    model.transform(adata, omics_key="X_pca", tau=1.0, batch_size=50)


def test_model_load_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = f"{tmpdir}/ckpt"

        model = stories.SpaceTime(proximal_step=stories.steps.ExplicitStep())
        model.fit(
            adata,
            time_key="time",
            omics_key="X_pca",
            space_key="spatial",
            batch_size=50,
            max_iter=2,
            checkpoint_manager=checkpoint_path,
        )
        original_pred = model.transform(adata, omics_key="X_pca", tau=1.0, batch_size=50)

        loaded_model = stories.SpaceTime(proximal_step=stories.steps.ExplicitStep())
        loaded_model.load(
            adata,
            omics_key="X_pca",
            checkpoint_manager=checkpoint_path,
        )
        loaded_pred = loaded_model.transform(
            adata, omics_key="X_pca", tau=1.0, batch_size=50
        )

        np.testing.assert_allclose(original_pred, loaded_pred)
