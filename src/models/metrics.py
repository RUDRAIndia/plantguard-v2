"""Keras 3 training-time metrics for a monitored quantity that isn't
misleading under CLAUDE.md's 36:1 class imbalance: macro-F1 (mean of each
class's F1, unweighted by class size) and per-class recall. Both accumulate
a running confusion matrix across batches within an epoch (stateful Keras
Metric objects, reset at each epoch boundary by model.fit()) rather than
averaging per-batch scores, which would weight small final-batch remainders
identically to full batches and drift from the true epoch-level value.

y_true is expected as sparse integer class indices (config.PLANTVILLAGE_
CLASS_NAMES' index order — CLAUDE.md rule 7), matching
src/data/pipeline.py's integer labels; y_pred is the model's softmax output.
"""

import tensorflow as tf
from tensorflow import keras


class MacroF1Score(keras.metrics.Metric):
    """Macro-averaged F1: unweighted mean of every class's F1, so a
    handful of misclassified examples in a rare class count exactly as much
    as the same mistake in a common one — unlike accuracy, which a 36:1
    imbalance lets a model dominate by doing well only on the largest
    classes. Suitable as EarlyStopping/ReduceLROnPlateau's `monitor`.
    """

    def __init__(self, num_classes: int, name: str = "macro_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion = self.add_weight(
            name="confusion", shape=(num_classes, num_classes), initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_pred_labels = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        batch_confusion = tf.math.confusion_matrix(
            y_true, y_pred_labels, num_classes=self.num_classes, dtype=self.confusion.dtype
        )
        self.confusion.assign_add(batch_confusion)

    def result(self) -> tf.Tensor:
        true_positives = tf.linalg.diag_part(self.confusion)
        predicted_positives = tf.reduce_sum(self.confusion, axis=0)
        actual_positives = tf.reduce_sum(self.confusion, axis=1)

        precision = tf.math.divide_no_nan(true_positives, predicted_positives)
        recall = tf.math.divide_no_nan(true_positives, actual_positives)
        f1 = tf.math.divide_no_nan(2 * precision * recall, precision + recall)
        return tf.reduce_mean(f1)

    def reset_state(self) -> None:
        self.confusion.assign(tf.zeros((self.num_classes, self.num_classes)))

    def get_config(self) -> dict:
        config = super().get_config()
        config["num_classes"] = self.num_classes
        return config


class PerClassRecall(keras.metrics.Metric):
    """Recall for every class individually (a length-num_classes vector),
    for logging alongside macro_f1 — CLAUDE.md rule 4 names per-class
    recall and worst-class recall as primary metrics, not just an aggregate.
    Not meant as a callback `monitor` target (those need a scalar); computed
    every epoch and written into src/train.py's history JSON instead.
    """

    def __init__(self, num_classes: int, name: str = "per_class_recall", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(
            name="true_positives", shape=(num_classes,), initializer="zeros"
        )
        self.actual_positives = self.add_weight(
            name="actual_positives", shape=(num_classes,), initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_pred_labels = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        correct = tf.cast(tf.equal(y_true, y_pred_labels), self.true_positives.dtype)
        y_true_one_hot = tf.one_hot(y_true, self.num_classes, dtype=self.true_positives.dtype)

        self.true_positives.assign_add(tf.reduce_sum(y_true_one_hot * correct[:, tf.newaxis], axis=0))
        self.actual_positives.assign_add(tf.reduce_sum(y_true_one_hot, axis=0))

    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self.true_positives, self.actual_positives)

    def reset_state(self) -> None:
        self.true_positives.assign(tf.zeros((self.num_classes,)))
        self.actual_positives.assign(tf.zeros((self.num_classes,)))

    def get_config(self) -> dict:
        config = super().get_config()
        config["num_classes"] = self.num_classes
        return config
