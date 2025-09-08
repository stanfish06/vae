from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.module import VAE, VAEC


class SCVI(UnsupervisedTrainingMixin, BaseModelClass, VAEMixin):
    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 256,
        n_layers: int = 1,
        n_latent: int = 30,
        use_vamp_prior: bool = True,
        **model_kwargs,
    ):
        super().__init__(adata)
        if use_vamp_prior:
            self.module = VAEC(
                n_input=self.summary_stats.n_vars,
                n_batch=getattr(self.summary_stats, "n_batch", 0),
                n_labels=self.summary_stats.n_labels,
                n_hidden=n_hidden,
                n_latent=n_latent,
                n_layers=n_layers,
                **model_kwargs,
            )
        else:
            self.module = VAE(
                n_input=self.summary_stats["n_vars"],
                n_batch=self.summary_stats["n_batch"],
                n_latent=n_latent,
                **model_kwargs,
            )
        self._model_summary_string = (
            f"SCVI model with the following parameters: \nn_latent {n_latent}"
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: str | None = None,
        layer: str | None = None,
        **kwargs,
    ) -> AnnData | None:
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            # Dummy fields required for VAE class.
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, None, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, None),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, None),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
