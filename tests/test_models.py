import os
import tempfile
from typing import Callable

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
    def test_model_inference(self, model_fn: Callable):
        model = model_fn(weights=None)
        mock_inputs = self.rng.uniform(shape=(1, *INPUT_SHAPE), dtype=tf.float32)
        outputs = model(mock_inputs, training=False)

        self.assertTrue(isinstance(outputs, tf.Tensor))
        self.assertEqual(outputs.shape, (1, 1000))

    @parameterized.named_parameters(TEST_PARAMS)
    def test_model_keras_serialization(self, model_fn: Callable):
        model = model_fn(weights=None)
        tf.keras.models.save_model(
            model=model, filepath=self.model_path, save_format="h5"
        )

        self.assertTrue(os.path.exists(self.model_path))

        loaded = tf.keras.models.load_model(self.model_path)
        self.assertTrue(isinstance(loaded, tf.keras.Model))

    def tearDown(self) -> None:
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    @parameterized.named_parameters(TEST_PARAMS)
    def test_models_not_being_created_with_invalid_input_shapes(
        self, model_fn: Callable
    ):
        # Provided input_shape doesn't match pretrained input shape
        with self.assertRaises(ValueError):
            model_fn(input_shape=(160, 160, 3))

        # Provided input_shape has invalid number of params:
        with self.assertRaises(ValueError):
            model_fn(input_shape=(1, 224, 224, 3), weights=None)

        # Invalid number of channels
        with self.assertRaises(ValueError):
            model_fn(input_shape=(224, 224, 2), weights=None)  # Should be either 1 or 3

        # Input shape is below minimum
        with self.assertRaises(ValueError):
            model_fn(input_shape=(8, 8, 3), weights=None)

    @parameterized.named_parameters(TEST_PARAMS)
    def test_models_work_with_smaller_input_shapes(self, model_fn: Callable):
        input_shape = (160, 160, 3)
        noise = self.rng.uniform(shape=(1, *input_shape))
        model = model_fn(weights=None, input_shape=input_shape)

        model(noise, training=False)

    @parameterized.named_parameters(TEST_PARAMS)
    def test_models_work_with_single_channel_input(self, model_fn: Callable):
        input_shape = (160, 160, 1)
        noise = self.rng.uniform(shape=(1, *input_shape))
        model = model_fn(weights=None, input_shape=input_shape)

        model(noise, training=False)

    @parameterized.named_parameters(TEST_PARAMS)
    def test_models_with_custom_head_sequential(self, model_fn: Callable):
        num_classes = 10
        backbone = model_fn(weights=None, include_top=False, pooling="avg")

        new_model = tf.keras.Sequential([backbone, tf.keras.layers.Dense(num_classes)])

        noise = self.rng.uniform(shape=(1, 224, 224, 3))
        output = new_model(noise, training=False)

        self.assertTrue(isinstance(output, tf.Tensor))
        self.assertTrue(output.shape == (1, num_classes))

    @parameterized.named_parameters(TEST_PARAMS)
    def test_models_with_custom_head_functional(self, model_fn: Callable):
        input_shape = (160, 160, 3)
        num_classes = 10

        # Make model:
        backbone = model_fn(weights=None, include_top=False, input_shape=input_shape)
        backbone.trainable = False
        x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
        outputs = tf.keras.layers.Dense(num_classes)(x)
        model = tf.keras.Model(inputs=[backbone.inputs], outputs=[outputs])

        # Run inference
        noise = self.rng.uniform(shape=(1, *input_shape))
        output = model(noise, training=False)

        self.assertTrue(isinstance(output, tf.Tensor))
        self.assertEqual(output.shape, (1, 10))


if __name__ == "__main__":
    absltest.main()
