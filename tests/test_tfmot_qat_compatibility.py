from collections import Callable

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from absl.testing import absltest, parameterized

from tests.test_models import TEST_PARAMS


class TestEfficientNetV2QATWrap(parameterized.TestCase):
    @parameterized.named_parameters(TEST_PARAMS)
    def test_qat_wrapping_entire_model(self, model_fn: Callable):
        self.skipTest(
            "The entire model cannot be wrapped in Quantization Aware Training."
            "LayerNormalization and gelu activation function are not supported."
            "This test might succeed, once TF-MOT package will be updated."
        )
        model = model_fn(weights=None)
        tfmot.quantization.keras.quantize_model(model)

    @parameterized.named_parameters(TEST_PARAMS)
    def test_applying_qat_to_some_layers(self, model_fn: Callable):
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
