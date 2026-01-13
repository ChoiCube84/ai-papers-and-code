import jax.numpy as jnp
from jax.lax import rsqrt
from jax.nn import softmax

def relative_index_matrix(num_token: int, maximum_distance: int) -> jnp.ndarray:
    i = jnp.arange(num_token)[:, None]
    j = jnp.arange(num_token)[None, :]

    diff = j - i

    return jnp.clip(diff, -maximum_distance, maximum_distance) + maximum_distance

def get_query(x : jnp.ndarray, W_Q : jnp.ndarray) -> jnp.ndarray:
    return x @ W_Q

def get_key(x : jnp.ndarray, W_K : jnp.ndarray, a_K : jnp.ndarray, query_index : int, clipped : jnp.ndarray) -> jnp.ndarray:
    product : jnp.ndarray = x @ W_K
    result : jnp.ndarray = product + a_K[clipped[query_index]]

    return result

def get_value(x : jnp.ndarray, W_V : jnp.ndarray, a_V : jnp.ndarray, query_index : int, clipped : jnp.ndarray) -> jnp.ndarray:
    product : jnp.ndarray = x @ W_V
    result : jnp.ndarray = product + a_V[clipped[query_index]]
        
    return result

def query_key_product(q : jnp.ndarray, k : jnp.ndarray, output_dim : int) -> jnp.ndarray:
    return q @ k.transpose() * rsqrt(jnp.float32(output_dim))

def attend_one_to_every(x : jnp.ndarray, 
                        W_Q : jnp.ndarray, W_K : jnp.ndarray, W_V : jnp.ndarray, 
                        a_K : jnp.ndarray, a_V : jnp.ndarray, 
                        clipped : jnp.ndarray, output_dim : int,
                        query_index : int) -> jnp.ndarray:
    query = get_query(x[query_index], W_Q) # (output_dim, )
    key = get_key(x, W_K, a_K, query_index, clipped) # (num_token, output_dim)
    value = get_value(x, W_V, a_V, query_index, clipped) # (num_token, output_dim)

    e_i = query_key_product(query, key, output_dim) # (num_token, )
    z_i = softmax(e_i) @ value

    return z_i