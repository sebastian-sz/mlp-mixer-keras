import os
import tempfile
from typing import Callable, Tuple

import tensorflow as tf
from absl.testing import absltest, parameterized

from mlp_mixer.mlp_mixer import MLPMixer_B16, MLPMixer_B32, MLPMixer_L16

INPUT_SHAPE = (224, 224, 3)
TEST_PARAMS = [
    {"testcase_name": "b16", "model_fn": MLPMixer_B16},
    {"testcase_name": "b32", "model_fn": MLPMixer_B32},
    {"testcase_name": "l16", "model_fn": MLPMixer_L16},
]


class TestModelsUnit(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    model_path = os.path.join(tempfile.mkdtemp(), "model.h5")

    @parameterized.named_parameters(TEST_PARAMS)
    def test_model_inference(self, model_fn: Callable[[], tf.keras.Model]):
        model = model_fn()  # TODO: arguments
        mock_inputs = self.rng.uniform(shape=(1, *INPUT_SHAPE), dtype=tf.float32)
        outputs = model(mock_inputs, training=False)

        self.assertTrue(isinstance(outputs, tf.Tensor))
        self.assertEqual(outputs.shape, (1, 1000))

    @parameterized.named_parameters(TEST_PARAMS)
    def test_model_keras_serialization(self, model_fn: Callable[[], tf.keras.Model]):
        model = model_fn()  # TODO: arguments.
        tf.keras.models.save_model(
            model=model, filepath=self.model_path, save_format="h5"
        )

        self.assertTrue(os.path.exists(self.model_path))

        loaded = tf.keras.models.load_model(self.model_path)
        self.assertTrue(isinstance(loaded, tf.keras.Model))

    def tearDown(self) -> None:
        if os.path.exists(self.model_path):
            os.remove(self.model_path)


if __name__ == "__main__":
    absltest.main()
