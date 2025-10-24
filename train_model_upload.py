import os
import netket as nk
import numpy as np
import netket.nn as nknn
from flax import nnx
import flax.linen as nn
import jax.numpy as jnp

os.environ["JAX_PLATFORM_NAME"] = "cpu"                                                    
os.environ["JAX_PLATFORMS"] = "cpu"


# most basic netket training boilerplate
class CNN(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 4, kernel_size=(2, 2), rngs=rngs, padding='CIRCULAR', param_dtype=jnp.complex64)
        self.linear1 = nnx.Linear(6 * 6 * 4, 10, rngs=rngs, param_dtype=jnp.complex64)
    def __call__(self, x):
        batch_size = x.shape[0]
        x = 2 * (x.reshape(batch_size, Lx, Ly, 1) - 1)
        x = nk.nn.activation.log_cosh(self.conv1(x))
        x = nk.nn.activation.log_cosh(self.linear1(x.reshape(batch_size, -1)))
        return jnp.sum(x, axis=-1)


Lx, Ly = 6, 6
lattice = nk.graph.Grid(extent=(Lx, Ly), pbc=True, max_neighbor_order=2)                                          
hi = nk.hilbert.Spin(s=1/2, N=lattice.n_nodes, total_sz=0)
hamiltonian = nk.operator.Heisenberg(hilbert=hi, graph=lattice, J=[1.0, 0.4], sign_rule=[True, False]).to_jax_operator()
model = CNN(rngs=nnx.Rngs(0))
sampler = nk.sampler.MetropolisExchange(hilbert=hi,
                                        graph=lattice,
                                        d_max=2,
                                        n_chains_per_rank=512)
vstate = nk.vqs.MCState(                                                                                      
            sampler=sampler,
            model=model,
            n_samples=512,
            n_discard_per_chain=0,                                                                                    
        )

opt = nk.optimizer.Sgd(learning_rate=0.001)                                                                   
sr = nk.optimizer.SR(diag_shift=0.1, holomorphic=True)
gs = nk.VMC(hamiltonian=hamiltonian,
                        optimizer=opt,
                        variational_state=vstate,
                         preconditioner=sr)

gs.run(n_iter=20)

# now vstate.parameters contains final iteration model params

# flax library contains serialization routines:
import flax
with open('model_params.msgpack', 'wb') as f:
    f.write(flax.serialization.msgpack_serialize(vstate.parameters))

# we can upload to huggingface repositories from python code:
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
api.upload_file(
    path_or_fileobj="model_params.msgpack",
    path_in_repo="model_params_uploaded.msgpack",
    repo_id="lprimeau/test_model",
    repo_type="model",
)

# and download...
model_dir = hf_hub_download(repo_id="lprimeau/test_model", filename="model_params_uploaded.msgpack")

with open(model_dir, 'rb') as f:
    new_pytree = flax.serialization.msgpack_restore(f.read())

# you can use the new pytree for a fresh vmc run. 
