from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from flax import linen as nn
from flax.linen.initializers import variance_scaling

from scvi import REGISTRY_KEYS
from scvi.distributions import JaxNegativeBinomialMeanDisp as NegativeBinomial
from scvi.module.base import JaxBaseModuleClass, LossRecorder

import pdb

DEFAULT_PX_HIDDEN = 32
DEFAULT_PZ_LAYERS = 1
DEFAULT_PZ_HIDDEN = 32


@jax.jit
def exp_activation(x):
    return jnp.exp(x)


class JaxResnetFC(nn.Module):
    n_in: int
    n_out: int
    n_hidden: int = 128
    activation: str = "softmax"

    def setup(self):
        if self.activation == "softmax":
            self.activation_ = nn.softmax
        elif self.activation == "softplus":
            self.activation_ = nn.softplus
        elif self.activation == "exp":
            self.activation_ = jnp.exp
        elif self.activation == "sigmoid":
            self.activation_ = nn.sigmoid
        elif self.activation == "relu":
            self.activation_ = nn.relu
        else:
            raise ValueError("activation not implemented")
        self.dense1 = nn.Dense(self.n_hidden)
        self.dense2 = nn.Dense(self.n_out)
        self.batchnorm1 = nn.BatchNorm()
        self.batchnorm2 = nn.BatchNorm()
        if self.n_in != self.n_hidden:
            self.id_map1 = nn.Dense(self.n_hidden)
        else:
            self.id_map1 = None
        
    def __call__(self, inputs, training=False):
        need_reshaping = False
        if inputs.ndim == 3:
            n_d1, nd2 = inputs.shape[:2]
            inputs = inputs.reshape(n_d1 * nd2, -1)
            need_reshaping = True
        h = self.dense1(inputs)
        h = self.batchnorm1(h, use_running_average=not training)
        h = nn.relu(h)
        if self.id_map1 is not None:
            h = h + self.id_map1(inputs)
        h = self.dense2(inputs)
        h = self.batchnorm2(h, use_running_average=not training)
        if need_reshaping:
            h = h.reshape(n_d1, nd2, -1)
        if self.activation is not None:
            return self.activation_(h)
        return h


class _JaxNormalNN(nn.Module):
    n_in: int
    n_out: int
    n_hidden: int = 128
    n_layers: int = 1
    use_batch_norm: bool = True
    use_layer_norm: bool = False
    do_orthogonal: bool = False

    def setup(self):
        self.hidden = JaxResnetFC(self.n_in, n_out=self.n_hidden, activation="relu")
        self._mean = nn.Dense(self.n_out)
        self._var = nn.Sequential([
            nn.Dense(self.n_out),
            nn.softplus,
        ])

    def __call__(self, inputs):
        if self.n_layers >= 1:
            h = self.hidden(inputs)
            mean = self._mean(h)
            var = self._var(h)
        else:
            mean = self._mean(inputs)
            k = mean.shape[0]
            var = self._var[None].expand(k, -1)
        return mean, var


class JaxNormalNN(nn.Module):
    n_in: int
    n_out: int
    n_categories: int
    n_hidden: int = 128
    n_layers: int = 1
    use_batch_norm: bool = True
    use_layer_norm: bool = False

    def setup(self):
        nn_kwargs = dict(
            n_in=self.n_in,
            n_out=self.n_out,
            n_hidden=self.n_hidden,
            n_layers=self.n_layers,
            use_batch_norm=self.use_batch_norm,
            use_layer_norm=self.use_layer_norm,
        )
        self._modules = [_JaxNormalNN(**nn_kwargs) for _ in range(self.n_categories)]

    def __call__(self, inputs, categories=None):
        means = []
        vars = []
        for module in self._modules:
            _means, _vars = module(inputs)
            means.append(_means[..., None])
            vars.append(_vars[..., None])
        means = jnp.concatenate(means, axis=-1)
        vars = jnp.concatenate(vars, axis=-1)
        if categories is not None:
            # categories (minibatch, 1)
            n_batch = categories.shape[0]
            cat_ = categories.unsqueeze(-1).expand(n_batch, -1, self.n_out, 1)
            if means.ndim == 4:
                d1, n_batch, _, _ = means.shape
                cat_ = categories[None, :, None].expand(d1, n_batch, self.n_out, 1)
            means = jnp.take_along_axis(means, cat_, axis=-1)
            vars = jnp.take_along_axis(vars, cat_, axis=-1)
        means = means.squeeze(-1)
        vars = vars.squeeze(-1)
        return dist.Normal(means, vars+1e-5)


class JaxConditionalBatchNorm1d(nn.Module):
    num_features: int
    num_classes: int

    def setup(self):
        # self.bn = nn.BatchNorm(self.num_features)
        self.bn = nn.BatchNorm()
        self.embed = nn.Embed(self.num_classes, self.num_features * 2)
        # TODO init scale of weights
        # self.embed.weight.data[:, :self.num_features].normal_(
        #     1, 0.02
        # ) # Initialise scale at N(1, 0.02)
        # self.embed.weight.data[:, self.num_features:].zero_() # Initialise bias at 0

    def __call__(self, x, y, training=False):
        is_eval = not training

        need_reshaping = False
        if x.ndim == 3:
            n_d1, nd2 = x.shape[:2]
            x = x.reshape(n_d1 * nd2, -1)
            need_reshaping = True

            y = y[None].expand(n_d1, nd2, -1)
            y = y.contiguous().view(n_d1 * nd2, -1)
        
        out = self.bn(x, use_running_average=is_eval)
        gamma, beta = jnp.array_split(self.embed(y.squeeze(-1)), 2, axis=1)
        out = gamma * out + beta

        if need_reshaping:
            out = out.reshape(n_d1, nd2, -1)
        
        return out


class Dense(nn.Dense):
    """Jax dense layer."""

    def __init__(self, *args, **kwargs):
        # scale set to reimplement pytorch init
        scale = 1 / 3
        kernel_init = variance_scaling(scale, "fan_in", "uniform")
        # bias init can't see input shape so don't include here
        kwargs.update({"kernel_init": kernel_init})
        super().__init__(*args, **kwargs)


class FlaxEncoder(nn.Module):
    """Encoder for Jax VAE."""

    n_input: int
    n_latent: int
    n_hidden: int
    dropout_rate: int

    def setup(self):
        """Setup encoder."""
        self.dense1 = Dense(self.n_hidden)
        self.dense2 = Dense(self.n_hidden)
        self.dense3 = Dense(self.n_latent)
        self.dense4 = Dense(self.n_latent)

        # self.batchnorm1 = nn.BatchNorm(momentum=0.99, epsilon=0.001)
        # self.batchnorm2 = nn.BatchNorm(momentum=0.99, epsilon=0.001)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropout2 = nn.Dropout(self.dropout_rate)

    def __call__(self, x: jnp.ndarray, training: bool = False):
        """Forward pass."""
        x_ = jnp.log1p(x)

        h = self.dense1(x_)
        # h = self.batchnorm1(h, use_running_average=is_eval)
        h = nn.relu(h)
        h = self.dropout1(h, deterministic=not training)
        h = self.dense2(h)
        # h = self.batchnorm2(h, use_running_average=is_eval)
        h = nn.relu(h)
        h = self.dropout2(h, deterministic=not training)

        mean = self.dense3(h)
        log_var = self.dense4(h)

        return mean, jnp.exp(log_var)


class FlaxDecoderZX(nn.Module):
    """Parameterizes the counts likelihood for the data given the latent variables."""

    n_in: int
    n_out: int
    n_nuisance: int
    linear_decoder: bool
    n_hidden: int
    activation: str = "softmax"
    
    def setup(self):
        if self.activation == "softmax":
            self.activation_ = nn.softmax
        elif self.activation == "softplus":
            self.activation_ = nn.softplus
        elif self.activation == "exp":
            self.activation_ = jnp.exp
        elif self.activation == "sigmoid":
            self.activation_ = nn.sigmoid
        elif self.activation == "relu":
            self.activation_ = nn.relu
        else:
            raise ValueError("activation not implemented")
        self.n_latent = self.n_in - self.n_nuisance
        if self.linear_decoder:
            self.amat = Dense(self.n_out, use_bias=False)
            self.amat_site = self.param(
                "amat_site", lambda rng, shape: jax.random.normal(rng, shape), (self.n_nuisance, self.n_latent, self.n_out)
            )
            self.offsets = self.param(
                "offsets", lambda rng, shape: jax.random.normal(rng, shape), (self.n_nuisance, self.n_out)
            )
            self.dropout_ = nn.Dropout(0.1)
        else:
            self.px_mean = JaxResnetFC(
                n_in=self.n_in,
                n_out=self.n_out,
                n_hidden=self.n_hidden,
                activation=self.activation
            )
        self.px_r = self.param(
            "px_r", lambda rng, shape: jax.random.normal(rng, shape), (self.n_out,)
        )

    def __call__(self, z, size_factor, training = False):
        if self.linear_decoder:
            nuisance_oh = z[..., -self.n_nuisance :]
            z0 = z[..., : -self.n_nuisance]
            x1 = self.amat(z0)

            nuisance_ids = jnp.argmax(nuisance_oh, axis=-1)
            As = self.amat_site[nuisance_ids]
            z0_detach = self.dropout_(jax.lax.stop_gradient(z0), deterministic=not training)[..., None]
            x2 = (As * z0_detach).sum(axis=-2)
            offsets = self.offsets[nuisance_ids]
            mu = x1 + x2 + offsets
            mu = self.activation_(mu)
        else:
            mu = self.px_mean(z)
        mu = mu * size_factor
        return NegativeBinomial(mean=mu, inverse_dispersion=jnp.exp(self.px_r))


class FlaxLinearDecoderUZ(nn.Module):
    
    n_latent: int
    n_donors: int
    n_out: int
    scaler: bool = False
    scaler_n_hidden: int = 32

    def setup(self):
        self.amat_sample = self.param(
            "amat_sample", lambda rng, shape: jax.random.normal(rng, shape), (self.n_donors, self.n_latent, self.n_out)
        )
        self.offsets = self.param(
            "offsets", lambda rng, shape: jax.random.normal(rng, shape), (self.n_donors, self.n_out)
        )
        self.scaler_ = None
        if self.scaler:
            self.scaler_ = nn.Sequential([
                nn.Dense(self.scaler_n_hidden),
                nn.LayerNorm(),
                nn.relu,
                nn.Dense(1),
                nn.sigmoid
            ])

    def __call__(self, u, donor_id, training=False):
        donor_id_ = donor_id.squeeze()
        As = self.amat_sample[donor_id_]

        u_detach = jax.lax.stop_gradient(u)[..., None]
        z2 = (As * u_detach).sum(axis=-2)
        offsets = self.offsets[donor_id_]
        delta = z2 + offsets
        if self.scaler_ is not None:
            donor_oh = jax.nn.one_hot(donor_id, self.n_donors)
            if u.ndim != donor_oh.ndim:
                # donor_oh = donor_oh[None].expand(u.shape[0], *donor_oh.shape)
                donor_oh = jax.lax.broadcast(donor_oh, (u.shape[0],))
            inputs = jnp.concatenate([jax.lax.stop_gradient(u), donor_oh], axis=-1)
            delta = delta * self.scaler_(inputs)
        return u + delta


class FlaxDecoderUZ(nn.Module):

    n_latent: int
    n_latent_donor: int
    n_out: int
    unnormalized_scaler: bool = False
    dropout_rate: float = 0.0
    n_layers: int = 1
    n_hidden: int = 128

    def setup(self):
        self.n_in = self.n_latent + self.n_latent_donor
        # mod
        self.mod_dense1 = nn.Dense(self.n_hidden)
        self.mod_dense2 = nn.Dense(self.n_hidden)
        self.mod_dense3 = nn.Dense(self.n_out, use_bias=False)
        self.mod_batchnorm1 = nn.BatchNorm()
        self.mod_batchnorm2 = nn.BatchNorm()
        self.mod_dropout1 = nn.Dropout(self.dropout_rate)
        # scale
        self.scale_dense1 = nn.Dense(self.n_hidden)
        self.scale_dense2 = nn.Dense(self.n_hidden)
        self.scale_dense3 = nn.Dense(1)
        self.scale_batchnorm1 = nn.BatchNorm()
        self.scale_batchnorm2 = nn.BatchNorm()
        self.scale_dropout1 = nn.Dropout(self.dropout_rate)
        if not self.unnormalized_scaler:
            self.activation = nn.sigmoid
        else:
            self.activation = nn.softplus

    def __call__(self, u, training=False):
        # TODO: generalize to use more than 1 layer
        u_ = jnp.copy(u)
        if u_.ndim == 3:
            n_samples, n_cells, n_features = u_.shape
            u0_ = u_[:, :, : self.n_latent].reshape(-1, self.n_latent)
            u_ = u_.reshape(-1, n_features)
            # mod
            h_mod = self.mod_dense1(u_)
            h_mod = self.mod_batchnorm1(h_mod, use_running_average=not training)
            h_mod = self.mod_dropout1(h_mod, deterministic=not training)
            h_mod = nn.relu(h_mod)
            h_mod = self.mod_dense2(h_mod)
            h_mod = self.mod_batchnorm2(h_mod, use_running_average=not training)
            h_mod = nn.relu(h_mod)
            h_mod = self.mod_dense3(h_mod)
            # scale
            h_scale = self.scale_dense1(u0_)
            h_scale = self.scale_batchnorm1(h_scale, use_running_average=not training)
            h_scale = self.scale_dropout1(h_scale, deterministic=not training)
            h_scale = nn.relu(h_scale)
            h_scale = self.scale_dense2(h_scale)
            h_scale = self.scale_batchnorm2(h_scale, use_running_average=not training)
            h_scale = nn.relu(h_scale)
            h_scale = self.scale_dense3(h_scale)
            pred_ = h_mod.reshape(n_samples, n_cells, -1)
            scaler_ = h_scale.reshape(n_samples, n_cells, -1)
        else:
            # mod
            h_mod = self.mod_dense1(u)
            h_mod = self.mod_batchnorm1(h_mod, use_running_average=not training)
            h_mod = self.mod_dropout1(h_mod, deterministic=not training)
            h_mod = nn.relu(h_mod)
            h_mod = self.mod_dense2(h_mod)
            h_mod = self.mod_batchnorm2(h_mod, use_running_average=not training)
            h_mod = nn.relu(h_mod)
            h_mod = self.mod_dense3(h_mod)
            # scale
            h_scale = self.scale_dense1(u[:, : self.n_latent])
            h_scale = self.scale_batchnorm1(h_scale, use_running_average=not training)
            h_scale = self.scale_dropout1(h_scale, deterministic=not training)
            h_scale = nn.relu(h_scale)
            h_scale = self.scale_dense2(h_scale)
            h_scale = self.scale_batchnorm2(h_scale, use_running_average=not training)
            h_scale = nn.relu(h_scale)
            h_scale = self.scale_dense3(h_scale)
            pred_ = h_mod
            scaler_ = h_scale
        # TODO: add this
        # if self.unnormalized_scaler:
        #     pred_ = F.normalize(pred_, p=2, dim=-1)
        mean = u[..., : self.n_latent] + scaler_ * pred_
        return mean

class JaxMrVAE(JaxBaseModuleClass):
    """Variational autoencoder model."""

    n_input: int
    n_batch: int
    n_obs_per_batch: int
    n_cats_per_nuisance_keys: int
    n_cats_per_bio_keys: int
    library_log_means: jnp.ndarray
    library_log_vars: jnp.ndarray
    n_hidden: int = 128
    n_latent: int = 10
    n_latent_donor: int = 2
    observe_library_sizes: bool = True
    linear_decoder_zx: bool = True
    linear_decoder_uz: bool = True
    linear_decoder_uz_scaler: bool = False
    linear_decoder_uz_scaler_n_hidden: int = 32
    unnormalized_scaler: bool = False
    px_kwargs: dict = None
    pz_kwargs: dict =None
    max_batches_comp: int = 30
    dropout_rate: float = 0.0
    n_layers: int = 1
    gene_likelihood: str = "nb"
    eps: float = 1e-8

    def setup(self):
        """Setup model."""
        px_kwargs = dict(n_hidden=DEFAULT_PX_HIDDEN)
        if px_kwargs is not None:
            px_kwargs.update(px_kwargs)
        pz_kwargs = dict(n_layers=DEFAULT_PZ_LAYERS, n_hidden=DEFAULT_PZ_HIDDEN)
        if pz_kwargs is not None:
            pz_kwargs.update(pz_kwargs)

        self.max_batches_comp_ = jnp.minimum(self.max_batches_comp, self.n_batch)
        assert self.n_latent_donor != 0

        self.donor_embeddings = nn.Embed(self.n_batch, self.n_latent_donor)

        # self.register_buffer(
        #     "library_log_means", torch.from_numpy(library_log_means).float()
        # )
        # self.register_buffer(
        #     "library_log_vars", torch.from_numpy(library_log_vars).float()
        # )
        self.n_nuisance = sum(self.n_cats_per_nuisance_keys)
        # Generative model
        self.px = FlaxDecoderZX(
            n_in=self.n_latent + self.n_nuisance,
            n_out=self.n_input,
            n_nuisance=self.n_nuisance,
            linear_decoder=self.linear_decoder_zx,
            **px_kwargs
        )
        self.qu = JaxNormalNN(128 + self.n_latent_donor, self.n_latent, n_categories=1)
        self.ql = JaxNormalNN(self.n_input, 1, n_categories=1)

        if self.linear_decoder_uz:
            self.pz = FlaxLinearDecoderUZ(
                n_latent=self.n_latent,
                n_donors=self.n_batch,
                n_out=self.n_latent,
                scaler=self.linear_decoder_uz_scaler,
                scaler_n_hidden=self.linear_decoder_uz_scaler_n_hidden
            )
        else:
            self.pz = FlaxDecoderUZ(
                n_latent=self.n_latent,
                n_latent_donor=self.n_latent_donor,
                n_out=self.n_latent,
                unnormalized_scaler=self.unnormalized_scaler,
                **pz_kwargs,
            )
        self.x_featurizer = nn.Sequential([
            nn.Dense(128),
            nn.relu
        ])
        self.bnn = JaxConditionalBatchNorm1d(128, self.n_batch)
        self.x_featurizer2 = nn.Sequential([
            nn.Dense(128),
            nn.relu
        ])
        self.bnn2 = JaxConditionalBatchNorm1d(128, self.n_batch)

    @property
    def required_rngs(self):  # noqa: D102
        return ("params", "dropout", "z", "l", "u")

    def _get_inference_input(self, tensors: Dict[str, jnp.ndarray]):
        """Get input for inference."""
        x = tensors[REGISTRY_KEYS.X_KEY]
        # needs to be int for embedding
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY].astype(jnp.int32)
        categorical_nuisance_keys = tensors["categorical_nuisance_keys"]
        return dict(
            x=x,
            batch_index=batch_index,
            categorical_nuisance_keys=categorical_nuisance_keys,
        )

    def inference(
        self, 
        x, 
        batch_index,
        categorical_nuisance_keys,
        n_samples=1,
        cf_batch=None,
        use_mean=False,
    ) -> dict:
        """Run inference model."""
        x_ = jnp.log1p(x)

        batch_index_cf = batch_index if cf_batch is None else cf_batch
        zdonor = self.donor_embeddings(batch_index_cf.squeeze(-1))
        zdonor_ = zdonor
        if n_samples >=2:
            # zdonor_ = zdonor[None].expand(n_samples, *zdonor.shape)
            zdonor_ = jax.lax.broadcast(zdonor, (n_samples,))
        qzdonor = None

        sample_shape = () if n_samples == 1 else (n_samples,)

        nuisance_oh = []
        for dim in range(categorical_nuisance_keys.shape[-1]):
            nuisance_oh.append(
                jax.nn.one_hot(
                    categorical_nuisance_keys[:, dim], 
                    self.n_cats_per_nuisance_keys[dim]
                )
            )
        nuisance_oh = jnp.concatenate(nuisance_oh, axis=-1)

        x_feat = self.x_featurizer(x_)
        x_feat = self.bnn(x_feat, batch_index)
        x_feat = self.x_featurizer2(x_feat)
        x_feat = self.bnn2(x_feat, batch_index)
        if x_.ndim != zdonor_.ndim:
            # x_feat_ = x_feat[None].expand(n_samples, *x_feat.shape)
            x_feat_ = jax.lax.broadcast(x_feat, (n_samples,))
            # nuisance_oh = nuisance_oh[None].expand(n_samples, *nuisance_oh.shape)
            nuisance_oh = jax.lax.broadcast(nuisance_oh, (n_samples,))
        else:
            x_feat_ = x_feat
        
        inputs = jnp.concatenate([x_feat_, zdonor_], axis=-1)
        qu = self.qu(inputs)
        if use_mean:
            u = qu.mean
        else:
            u_rng = self.make_rng("u")
            u = qu.rsample(u_rng, sample_shape=sample_shape)
        
        # if jnp.isinf(u).any():
        #     pdb.set_trace()
        if self.linear_decoder_uz:
            z = self.pz(u, batch_index_cf)
        else:
            inputs = jnp.concatenate([u, zdonor_], axis=-1)
            z = self.pz(inputs)
        if self.observe_library_sizes:
            library = jnp.log(x.sum(1))[..., None]#.unsqueeze(1)
            ql = None
        else:
            ql = self.ql(x_)
            l_rng = self.make_rng("l")
            library = ql.rsample(l_rng, sample_shape=sample_shape)

        return dict(
            qu=qu,
            qzdonor=qzdonor,
            ql=ql,
            u=u,
            z=z,
            zdonor=zdonor,
            library=library,
            nuisance_oh=nuisance_oh,
        )

    def get_z(self, u, zdonor=None, batch_index=None):
        if batch_index is not None:
            zdonor = self.donor_embeddings(batch_index.squeeze(-1))
            zdonor_ = zdonor
        else:
            zdonor_ = zdonor
        inputs = jnp.concatenate([u, zdonor_], axis=-1)
        z = self.pz(inputs)
        return z
        

    def _get_generative_input(
        self,
        tensors: Dict[str, jnp.ndarray],
        inference_outputs: Dict[str, jnp.ndarray],
        **kwargs,
    ):
        """Get input for generative model."""
        categorical_nuisance_keys = tensors["categorical_nuisance_keys"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        return dict(
            z=inference_outputs["z"],
            zdonor=inference_outputs["zdonor"],
            library=inference_outputs["library"],
            batch_index=batch_index,
            categorical_nuisance_keys=categorical_nuisance_keys,
            nuisance_oh=inference_outputs["nuisance_oh"],
        )

    def generative(
        self, 
        z, 
        library,
        batch_index,
        zdonor,
        categorical_nuisance_keys,
        nuisance_oh,
    ) -> dict:
        """Run generative model."""
        pzdonor = None
        inputs = jnp.concatenate([z, nuisance_oh], axis=-1)
        px = self.px(inputs, size_factor=jnp.exp(library))
        h = px.mean / jnp.exp(library)

        (
            local_library_log_means,
            local_library_log_vars,
        ) = self._compute_local_library_params(batch_index)
        pl = (
            None if self.observe_library_sizes
            else dist.Normal(local_library_log_means, local_library_log_vars.sqrt())
        )
        
        pu = dist.Normal(0, 1)
        return dict(
            pzdonor=pzdonor,
            px=px,
            pl=pl,
            pu=pu,
            h=h,
        )

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        lambd=None,
        do_comp=False,
        **kwargs, # TODO: should remove but gives error if I do
    ):
        """Compute loss"""
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch = tensors[REGISTRY_KEYS.BATCH_KEY]

        reconstruction_loss = -generative_outputs["px"].log_prob(x).sum(-1)
        kl_u = dist.kl_divergence(inference_outputs["qu"], generative_outputs["pu"]).sum(-1)
        kl_local_for_warmup = kl_u

        if self.observe_library_sizes:
            kl_local_no_warmup = 0.0
        else:
            kl_local_no_warmup = dist.kl_divergence(
                inference_outputs["ql"], generative_outputs["pl"]
            ).sum(-1)
        
        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup
        loss = jnp.mean(reconstruction_loss + weighted_kl_local)

        pen = 0.0
        # TODO: add back
        # if (lambd is not None) and do_comp:
        #     log_qz = (
        #         inference_outputs["qu"]
        #         .log_prob(inference_outputs["u"][:, None])
        #         .sum(-1)
        #     )
        #     samples = batch.squeeze()
        #     unique_samples = jnp.unique(samples)
        #     n_batches_to_take = jnp.minimum(
        #         unique_samples.shape[0], self.max_batches_comp_
        #     )
        #     unique_samples = np.random.choice(
        #         unique_samples, size=n_batches_to_take, replace=False
        #     )
        #     for batch_index in unique_samples:
        #         set_s = samples == batch_index
        #         set_ms = samples != batch_index
        #         log_qz_foreg = (
        #             jax.nn.logsumexp(log_qz[set_s][:, set_s], 1) - set_s.sum().log()
        #         )
        #         log_qz_backg = (
        #             jax.nn.logsumexp(log_qz[set_s][:, set_ms], 1) - set_ms.sum().log()
        #         )
        #         pen += (log_qz_foreg - log_qz_backg).sum()
        #     pen = lambd * pen / unique_samples.shape[0]
        loss += pen

        kl_local = 0.0
        kl_global = 0.0
        return LossRecorder(
            loss, 
            reconstruction_loss, 
            kl_local,
            kl_global,
            pen=pen
        )

    def _compute_local_library_params(self, batch_index):
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = jax.nn.one_hot(batch_index.squeeze(), n_batch) @ self.library_log_means.T
        local_library_log_vars = jax.nn.one_hot(batch_index.squeeze(), n_batch) @ self.library_log_vars.T
        return local_library_log_means, local_library_log_vars