import os
import shutil
import subprocess
import tempfile
from typing import Callable

import onnxruntime
import tensorflow as tf
from absl.testing import absltest, parameterized
from psutil import virtual_memory

from tests.test_models import INPUT_SHAPE, TEST_PARAMS

# Assure the machine has enough memory for the conversion. Not enough RAM will kill CI.
VARIANT_TO_MIN_MEMORY = {"mlp-mixer-b16": 14, "mlp-mixer-b32": 15, "mlp-mixer-l16": 25}


class TestONNXConversion(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    saved_model_path = os.path.join(tempfile.mkdtemp(), "saved_model")
    onnx_model_path = os.path.join(tempfile.mkdtemp(), "model.onnx")

    def tearDown(self) -> None:
        if os.path.exists(self.onnx_model_path):
            os.remove(self.onnx_model_path)
        if os.path.exists(self.saved_model_path):
            shutil.rmtree(self.saved_model_path)

    @parameterized.named_parameters(TEST_PARAMS)
    def test_model_onnx_conversion(self, model_fn: Callable):
        model = model_fn(weights=None)

        if not self._enough_memory_to_convert(model.name):
            self.skipTest(
                "Not enough memory to convert to onnx. Need at least "
                f"{VARIANT_TO_MIN_MEMORY[model.name]} GB. Skipping... ."
            )

        model.save(self.saved_model_path)

        self._convert_onnx()
        self.assertTrue(os.path.isfile(self.onnx_model_path))

        # Compare outputs:
        mock_input = self.rng.uniform(shape=(1, *INPUT_SHAPE), dtype=tf.float32)
        original_output = model.predict(mock_input)

        onnx_session = onnxruntime.InferenceSession(self.onnx_model_path)
        onnx_inputs = {onnx_session.get_inputs()[0].name: mock_input.numpy()}
        onnx_output = onnx_session.run(None, onnx_inputs)

        tf.debugging.assert_near(original_output, onnx_output)

    @staticmethod
    def _enough_memory_to_convert(model_name: str) -> bool:
        total_ram = virtual_memory().total / (1024.0 ** 3)
        required_ram = VARIANT_TO_MIN_MEMORY[model_name]
        return total_ram >= required_ram

    def _convert_onnx(self):
        command = (
            f"python -m tf2onnx.convert "
            f"--saved-model {self.saved_model_path} "
            f"--output {self.onnx_model_path} "
        )
        subprocess.run(command, shell=True, check=True)


if __name__ == "__main__":
    absltest.main()
