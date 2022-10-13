from copy import deepcopy
import logging
import warnings
from typing import Optional, Sequence, Union, List

import jax
import jax.numpy as jnp
import numpy as np
from anndata import AnnData
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
)
from scvi.dataloaders import DataSplitter
from scvi.module.base import JaxModuleWrapper
from scvi.train import JaxModuleInit, JaxTrainingPlan, TrainRunner
from scvi.utils import setup_anndata_dsp

from scvi.model.base import BaseModelClass
from scvi.model._utils import _init_library_size

from ._jaxmodule import JaxMrVAE


logger = logging.getLogger(__name__)

DEFAULT_TRAIN_KWARGS = dict(
    early_stopping=True,
    early_stopping_patience=15,
    check_val_every_n_epoch=1,
    batch_size=256,
    train_size=0.9,
    plan_kwargs=dict(
        lr=1e-2,
        n_epochs_kl_warmup=20,
        do_comp=False,
        lambd=0.1,
    ),
)

class JaxTrainingMixin:
    """General purpose train method for Jax-backed modules."""

    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[bool] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        early_stopping: bool = False,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """
        Train the model.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        use_gpu
            Whether or not to use GPU resources. If None, will use GPU if available.
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        n_cells = self.adata.n_obs
        if max_epochs is None:
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        if use_gpu is None or use_gpu is True:
            try:
                self.module.to(jax.devices("gpu")[0])
                logger.info(
                    "Jax module moved to GPU. "
                    "Note: Pytorch lightning will show GPU is not being used for the Trainer."
                )
            except RuntimeError:
                logger.debug("No GPU available to Jax.")
        else:
            cpu_device = jax.devices("cpu")[0]
            self.module.to(cpu_device)
            logger.info("Jax module moved to CPU.")

        data_splitter = DataSplitter(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            # for pinning memory only
            use_gpu=False,
            iter_ndarray=True,
        )

        self.training_plan = JaxTrainingPlan(
            self.module, **plan_kwargs
        )
        if "callbacks" not in trainer_kwargs.keys():
            trainer_kwargs["callbacks"] = []
        trainer_kwargs["callbacks"].append(JaxModuleInit())

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )

        # Ignore Pytorch Lightning warnings for Jax workarounds.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, module=r"pytorch_lightning.*"
            )
            runner = TrainRunner(
                self,
                training_plan=self.training_plan,
                data_splitter=data_splitter,
                max_epochs=max_epochs,
                use_gpu=False,
                **trainer_kwargs,
            )
            runner()

        self.is_trained_ = True
        self.module.eval()


class JaxMrVI(JaxTrainingMixin, BaseModelClass):
    def __init__(
        self,
        adata: AnnData,
        **model_kwargs,
    ):
        super().__init__(adata)

        n_cats_per_nuisance_keys = (
            self.adata_manager.get_state_registry(
                "categorical_nuisance_keys"
            ).n_cats_per_key
            if "categorical_nuisance_keys" in self.adata_manager.data_registry
            else []
        )

        n_cats_per_bio_keys = (
            self.adata_manager.get_state_registry(
                "categorical_biological_keys"
            ).n_cats_per_key
            if "categorical_biological_keys" in self.adata_manager.data_registry
            else []
        )
        n_batch = self.summary_stats.n_batch
        library_log_means, library_log_vars = _init_library_size(
            self.adata_manager, n_batch
        )
        n_obs_per_batch = (
            adata.obs.groupby(
                self.adata_manager.get_state_registry("batch")["original_key"]
            )
            .size()
            .loc[self.adata_manager.get_state_registry("batch")["categorical_mapping"]]
            .values
        )
        n_obs_per_batch = jnp.array(n_obs_per_batch)
        self.data_splitter = None
        self.module = JaxModuleWrapper(
            JaxMrVAE,
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_cats_per_nuisance_keys=n_cats_per_nuisance_keys,
            n_cats_per_bio_keys=n_cats_per_bio_keys,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            n_obs_per_batch=n_obs_per_batch,
            **model_kwargs,
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        categorical_nuisance_keys: Optional[List[str]] = None,
        categorical_biological_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        %(summary)s.
        Parameters
        ----------
        %(param_layer)s
        %(param_batch_key)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
            CategoricalJointObsField(
                "categorical_nuisance_keys", categorical_nuisance_keys
            ),
            CategoricalJointObsField(
                "categorical_biological_keys", categorical_biological_keys
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        early_stopping: bool = False,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        train_kwargs = dict(
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            early_stopping=early_stopping,
            **trainer_kwargs,
        )
        train_kwargs = dict(deepcopy(DEFAULT_TRAIN_KWARGS), **train_kwargs)
        plan_kwargs = plan_kwargs or {}
        train_kwargs["plan_kwargs"] = dict(
            deepcopy(DEFAULT_TRAIN_KWARGS["plan_kwargs"]), **plan_kwargs
        )
        super().train(**train_kwargs)

    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        mc_samples: int = 5000,
        batch_size: Optional[int] = None,
        give_z: bool = False,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.
        This is denoted as :math:`z_n` in our manuscripts.
        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, iter_ndarray=True
        )
        run_inference = self.module.get_inference_fn(mc_samples=mc_samples)
        u = []
        z = []
        for array_dict in tqdm(scdl):
            outputs = run_inference(array_dict)
            u.append(outputs["u"].mean)
            z.append(outputs["z"].mean)
        u = np.array(jax.device_get(jnp.concatenate(u, axis=0)))
        z = np.array(jax.device_get(jnp.concatenate(z, axis=0)))        
        return z if give_z else u


    def get_cf_degs(
        self,
        adata: Optional[AnnData] = None,
        indices=None,
        batch_size: Optional[int] = None,
    ):
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, iter_ndarray=True
        )
        run_inference = self.module.get_inference_fn()
        cf_degs = []
        for array_dict in tqdm(scdl):
            outputs = run_inference(array_dict)
            cf_degs.append(outputs["cf_degs"])
        cf_degs = np.array(jax.device_get(jnp.concatenate(cf_degs, axis=0)))
        return cf_degs


    @staticmethod
    def compute_distance_matrix_from_representations(
        representations: np.ndarray, metric: str = "cosine"
    ) -> np.ndarray:
        """Compute distance matrices from representations of shape (n_cells, n_donors, n_features)"""
        n_cells, n_donors, _ = representations.shape
        pairwise_dists = np.zeros((n_cells, n_donors, n_donors))
        for i, cell_rep in enumerate(representations):
            d_ = pairwise_distances(cell_rep, metric=metric)
            pairwise_dists[i, :, :] = d_
        return pairwise_dists

    # @torch.no_grad()
    def get_local_sample_representation(
        self,
        adata=None,
        batch_size=256,
        mc_samples: int = 10,
        x_space=False,
        x_log=True,
        x_dim=50,
        eps=1e-6,
        return_distances=False,
    ):
        # TODO: write
        pass

    def to_device(self, device):  # noqa: D102
        pass

    @property
    def device(self):  # noqa: D102
        return self.module.device