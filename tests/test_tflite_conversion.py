import os
import tempfile
from typing import Callable

import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized
from psutil import virtual_memory

from tests.test_models import INPUT_SHAPE, TEST_PARAMS

# Assure the machine has enough memory for the conversion. Not enough RAM will kill CI.
VARIANT_TO_MIN_MEMORY = {"mlp-mixer-b16": 8, "mlp-mixer-b32": 8, "mlp-mixer-l16": 17}


class TestTFLiteConversion(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    tflite_path = os.path.join(tempfile.mkdtemp(), "model.tflite")

    def tearDown(self) -> None:
        if os.path.exists(self.tflite_path):
            os.remove(self.tflite_path)

    @parameterized.named_parameters(TEST_PARAMS)
    def test_tflite_conversion(self, model_fn: Callable):
        model = model_fn(weights=None, input_shape=INPUT_SHAPE)

        if not self._enough_memory_to_convert(model.name):
            self.skipTest(
                "Not enough memory to convert to tflite. Need at least "
                f"{VARIANT_TO_MIN_MEMORY[model.name]} GB. Skipping... ."
            )

        self._convert_and_save_tflite(model)
        self.assertTrue(os.path.isfile(self.tflite_path))

        # Check outputs:
        mock_input = self.rng.uniform(shape=(1, *INPUT_SHAPE), dtype=tf.float32)
        original_output = model.predict(mock_input)
        tflite_output = self._run_tflite_inference(mock_input)

        tf.debugging.assert_near(original_output, tflite_output)

    def _convert_and_save_tflite(self, model: tf.keras.Model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(self.tflite_path, "wb") as file:
            file.write(tflite_model)

    def _run_tflite_inference(self, inputs: tf.Tensor) -> np.ndarray:
        interpreter = tf.lite.Interpreter(model_path=self.tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]["index"], inputs.numpy())
        interpreter.invoke()

        return interpreter.get_tensor(output_details[0]["index"])

    @staticmethod
    def _enough_memory_to_convert(model_name: str) -> bool:
        total_ram = virtual_memory().total / (1024.0 ** 3)
        required_ram = VARIANT_TO_MIN_MEMORY[model_name]
        return total_ram >= required_ram


if __name__ == "__main__":
    absltest.main()
