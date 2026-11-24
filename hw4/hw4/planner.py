"""Select the configured HW4 mode without falling back to another behavior."""

from hw4 import fast, safe


def get_mode(name):
    """Return the mode module, rejecting the unimplemented safe skeleton."""
    if name == 'fast':
        return fast
    if name == 'safe':
        raise NotImplementedError('Safe mode is a skeleton; select navigation.mode: fast for run 1.')
    raise ValueError('navigation.mode must be fast or safe.')
