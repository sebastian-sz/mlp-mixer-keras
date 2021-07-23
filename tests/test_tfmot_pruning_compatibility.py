from typing import Callable

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from absl.testing import absltest, parameterized

from tests.test_models import TEST_PARAMS


class TestEfficientNetV2PruningWrapper(parameterized.TestCase):
    @parameterized.named_parameters(TEST_PARAMS)
    def test_tfmot_pruning_entire_model(self, model_fn: Callable):
        self.skipTest(
            "The entire model cannot be wrapped in pruning wrapper"
            "as gelu activation is not supported."
            "This test might succeed once the TF-MOT package is updated."
        )
        model = model_fn(weights=None)
        tfmot.sparsity.keras.prune_low_magnitude(model)

    @parameterized.named_parameters(TEST_PARAMS)
    def test_tfmot_pruning_some_layers(self, model_fn: Callable):
        model = model_fn(weights=None)

        def apply_pruning(layer):
            if isinstance(layer, tf.keras.layers.Dense):
                return tfmot.sparsity.keras.prune_low_magnitude(layer)
            elif isinstance(layer, tf.keras.layers.Conv2D):
                return tfmot.sparsity.keras.prune_low_magnitude(layer)
            return layer

        tf.keras.models.clone_model(model, clone_function=apply_pruning)


if __name__ == "__main__":
    absltest.main()
