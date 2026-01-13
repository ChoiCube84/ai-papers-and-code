import jax
import jax.numpy as jnp
from jax import random

from RPE import attend_one_to_every, relative_index_matrix

key = random.PRNGKey(0)
subkeys = random.split(key, 10)

num_token : int = 10

input_dim : int = 8
output_dim : int = 16

maximum_distance : int = 6

x : jnp.ndarray = random.normal(subkeys[0], shape=(num_token, input_dim))

W_Q : jnp.ndarray = random.normal(subkeys[1], shape=(input_dim, output_dim))
W_K : jnp.ndarray = random.normal(subkeys[2], shape=(input_dim, output_dim))
W_V : jnp.ndarray = random.normal(subkeys[3], shape=(input_dim, output_dim))

a_K : jnp.ndarray = random.normal(subkeys[4], shape=(2 * maximum_distance + 1, output_dim)) # d_a = d_z in the paper
a_V : jnp.ndarray = random.normal(subkeys[5], shape=(2 * maximum_distance + 1, output_dim))

clipped = relative_index_matrix(num_token, maximum_distance)

attend_all = jax.vmap(
    attend_one_to_every,
    in_axes=(None, None, None, None, None, None, None, None, 0)
)
attend_all_jit = jax.jit(attend_all)

z = attend_all_jit(
    x, W_Q, W_K, W_V, a_K, a_V, clipped, output_dim,
    jnp.arange(num_token)
)

print(z.shape)