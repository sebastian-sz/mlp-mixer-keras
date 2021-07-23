import os
from typing import Callable

import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized

from mlp_mixer.mlp_mixer import MLPMixer_B16, MLPMixer_B32, MLPMixer_L16
from tests._root_dir import ROOT_DIR
from tests.test_models import INPUT_SHAPE, TEST_PARAMS

CONSISTENCY_TEST_PARAMS = [
    {
        "testcase_name": "b16-imagenet1k",
        "model_fn": MLPMixer_B16,
        "original_outputs": os.path.join(
            ROOT_DIR, "assets/original_outputs/b16_imagenet1k.npy"
        ),
    },
    {
        "testcase_name": "b16-sam",
        "model_fn": MLPMixer_B16,
        "original_outputs": os.path.join(
            ROOT_DIR, "assets/original_outputs/b16_sam.npy"
        ),
    },
    {
        "testcase_name": "b32-sam",
        "model_fn": MLPMixer_B32,
        "original_outputs": os.path.join(
            ROOT_DIR, "assets/original_outputs/b32_sam.npy"
        ),
    },
    {
        "testcase_name": "l16-imagenet1k",
        "model_fn": MLPMixer_L16,
        "original_outputs": os.path.join(
            ROOT_DIR, "assets/original_outputs/l16_imagenet1k.npy"
        ),
    },
]


class TestKerasVSOriginalOutputConsistency(parameterized.TestCase):
    image_path = os.path.join(ROOT_DIR, "assets/panda.jpg")
    image = tf.image.decode_png(tf.io.read_file(image_path))
    image = tf.expand_dims(image, axis=0)

    @parameterized.named_parameters(CONSISTENCY_TEST_PARAMS)
    def test_output_consistency(self, model_fn: Callable, original_outputs: str):
        model = model_fn()

        input_tensor = tf.image.resize(self.image, INPUT_SHAPE[:2])
        input_tensor = self._pre_process_image(input_tensor)

        output = model(input_tensor, training=False)
        original_output = np.load(original_outputs)

        tf.debugging.assert_near(output, original_output)

    @staticmethod
    def _pre_process_image(img: tf.Tensor) -> tf.Tensor:
        return (img / 128) - 1

    @parameterized.named_parameters(TEST_PARAMS)
    def test_feature_extraction_with_pretrained_weights(self, model_fn: Callable):
        model = model_fn(include_top=False)

        input_tensor = tf.image.resize(self.image, INPUT_SHAPE[:2])
        input_tensor = self._pre_process_image(input_tensor)

        out = model(input_tensor, training=False)

        self.assertTrue(isinstance(out, tf.Tensor))
        self.assertEqual(len(out.shape), 4)


if __name__ == "__main__":
    absltest.main()
