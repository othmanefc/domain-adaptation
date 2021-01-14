import functools

import tensorflow as tf  # type: ignore


def random_apply(p=0.5):

    def rand_apply_with_param(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            do_it = tf.random.uniform((1,)) > (1 - p)
            assert "img" in kwargs.keys()
            if not do_it:
                return kwargs["img"]
            else:
                return func(*args, **kwargs)

        return wrapper

    return rand_apply_with_param


def decode_img(img, width, height, channels):
    img = tf.image.decode_jpeg(img, channels=channels)
    return tf.image.resize(img, [width, height])


def process_path(file_path, width, height, channels, with_label=False):
    img = tf.io.read_file(file_path)
    # if tf.io.is_jpeg(img):
    img = decode_img(img, width, height, channels)
    if with_label:
        label = tf.strings.split(file_path)[-2]
        return img, label
    else:
        return img
