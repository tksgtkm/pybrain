import numpy as np
jax_enable = True
try:
    import jax as jnp
    jax = jnp
except ImportError:
    jax_enable = False

from pybrain import Variable

def get_array_module(x):
    if isinstance(x, Variable):
        x = x.data

    if not jax_enable:
        return np
    xp = jnp.numpy.array(xp)
    return xp

def as_numpy(x):
    if isinstance(x, Variable):
        x = x.data

    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x

def as_jax(x):
    if isinstance(x, Variable):
        x = x.data

    if not jax_enable:
        raise Exception('Jax cannnot be loaded')
    return jnp.numpy.array(x)

