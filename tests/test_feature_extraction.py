import os
from typing import Callable

import tensorflow as tf
from absl.testing import absltest, parameterized

from mlp_mixer.mlp_mixer import MLPMixer_B16, MLPMixer_B32, MLPMixer_L16
from tests._root_dir import ROOT_DIR

INPUT_SHAPE = (224, 224, 3)
WEIGHTS_DIR = "/".join(ROOT_DIR.split("/")[:-1]) + "/weights"

FEATURE_EXTRACTION_TEST_PARAMS = [
    {
        "testcase_name": "b16-imagenet1k",
        "model_fn": MLPMixer_B16,
        "weights_path": os.path.join(WEIGHTS_DIR, "mlp-mixer-b16_notop.h5"),
    },
    {
        "testcase_name": "b16-sam",
        "model_fn": MLPMixer_B16,
        "weights_path": os.path.join(WEIGHTS_DIR, "mlp-mixer-b16-sam_notop.h5"),
    },
    {
        "testcase_name": "b32-sam",
        "model_fn": MLPMixer_B32,
        "weights_path": os.path.join(WEIGHTS_DIR, "mlp-mixer-b32-sam_notop.h5"),
    },
    {
        "testcase_name": "l16-imagenet1k",
        "model_fn": MLPMixer_L16,
        "weights_path": os.path.join(WEIGHTS_DIR, "mlp-mixer-l16_notop.h5"),
    },
]


class TestFeatureExtraction(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()

    @parameterized.named_parameters(FEATURE_EXTRACTION_TEST_PARAMS)
    def test_models_can_be_used_as_pretrained_feature_extractors(
        self, model_fn: Callable, weights_path: str
    ):
        if not os.path.exists(weights_path):
            self.skipTest("No weights present in repo. Skipping... .")

        model = model_fn(include_top=False)
        model.load_weights(weights_path)

        noise = self.rng.uniform((1, *INPUT_SHAPE))
        out = model(noise, training=False)

        self.assertTrue(isinstance(out, tf.Tensor))
        self.assertEqual(len(out.shape), 4)


if __name__ == "__main__":
    absltest.main()
