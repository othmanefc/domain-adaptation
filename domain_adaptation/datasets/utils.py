import functools

import tensorflow as tf


def random_apply(p=0.5):

    def rand_apply_with_param(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            do_it = tf.random.uniform((1,)) > p
            assert "img" in kwargs.keys()
            if not do_it:
                return kwargs["img"]
            else:
                return func(*args, **kwargs)

        return wrapper

    return rand_apply_with_param
