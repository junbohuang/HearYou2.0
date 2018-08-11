import functools
import tensorflow as tf


def top_k_accuracy():
    top2_acc = functools.partial(tf.keras.metrics.top_k_categorical_accuracy, k=2)
    top3_acc = functools.partial(tf.keras.metrics.top_k_categorical_accuracy, k=3)
    # top4_acc = functools.partial(tf.keras.metrics.top_k_categorical_accuracy, k=4)
    # top5_acc = functools.partial(tf.keras.metrics.top_k_categorical_accuracy, k=5)

    top2_acc.__name__ = 'Top2Acc'
    top3_acc.__name__ = 'Top3Acc'
    # top4_acc.__name__ = 'Top4Acc'
    # top5_acc.__name__ = 'Top5Acc'

    return ['accuracy', top2_acc, top3_acc] #, top4_acc, top5_acc]

