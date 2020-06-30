from numpy import *
_empty = empty
def empty(*args, **kwargs):
    kwargs.update(dtype=float128)
    _empty(*args, **kwargs)