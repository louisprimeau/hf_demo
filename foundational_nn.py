

from functools import partial
import numpy as np

import jax
import jax.numpy as jnp
import netket as nk
from huggingface_hub import hf_hub_download

import flax
from flax.training import checkpoints


flax.config.update('flax_use_orbax_checkpointing', False)

lattice = nk.graph.Hypercube(length=10, n_dim=2, pbc=True, max_neighbor_order=2)

J2 = 0.5

assert J2 >= 0.4 and J2 <= 0.6 #* the model has been trained on this interval

from transformers import FlaxAutoModel
wf = FlaxAutoModel.from_pretrained("nqs-models/j1j2_square_fnqs", trust_remote_code=True)
N_params = nk.jax.tree_size(wf.params)
print('Number of parameters = ', N_params, flush=True)

hilbert = nk.hilbert.Spin(s=1/2, N=lattice.n_nodes, total_sz=0)
hamiltonian = nk.operator.Heisenberg(hilbert=hilbert, 
                                    graph=lattice, 
                                    J=[1.0, J2], 
                                    sign_rule=[False, False]).to_jax_operator() # No Marshall sign rule

sampler = nk.sampler.MetropolisExchange(hilbert=hilbert,
                                        graph=lattice,
                                        d_max=2,
                                        n_chains=16000,
                                        sweep_size=lattice.n_nodes)

key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key, 2)
vstate = nk.vqs.MCState(sampler=sampler, 
                        apply_fun=partial(wf.__call__, coups=J2), 
                        sampler_seed=subkey,
                        n_samples=16000, 
                        n_discard_per_chain=0,
                        variables=wf.params,
                        chunk_size=16000)

# Overwrite samples with already thermalized ones
path = hf_hub_download(repo_id="nqs-models/j1j2_square_fnqs", filename="spins")
samples = checkpoints.restore_checkpoint(ckpt_dir=path, prefix="spins", target=None)
samples = jnp.array(samples, dtype='int8')
vstate.sampler_state = vstate.sampler_state.replace(σ = samples)


import time
# Sample the model
for _ in range(10):
    start = time.time()
    E = vstate.expect(hamiltonian)
    vstate.sample()
    
    print("Mean: ", E.mean.real / lattice.n_nodes / 4, "\t time=", time.time()-start)
