import tensorflow as tf
import tensorflow.data as tfds

def load():
    return tfds.load(
        name='eurosat/rgb',
        data_dir='.data')