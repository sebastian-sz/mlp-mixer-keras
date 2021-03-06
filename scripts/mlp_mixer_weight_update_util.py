"""Utility to convert weights from JAX (.npz files) to Keras .h5."""
from argparse import ArgumentParser

import numpy as np

from mlp_mixer.mlp_mixer import MLPMixer_B16, MLPMixer_B32, MLPMixer_L16


def parse_args():
    """Parse command line input."""
    parser = ArgumentParser()
    parser.add_argument("--model", "-m", help="Model type to use")
    parser.add_argument("--input", "-i", help="Path to jax (.npz) checkpoint.")
    parser.add_argument("--output", "-o", help="Where to save .h5 converted weights.")
    parser.add_argument(
        "--notop", action="store_true", help="Whether to build model with head."
    )

    return parser.parse_args()


def jax_weight_names(jax_ckpt):
    """List all variable names in jax checkpoint (.npz)."""
    return [x for x in jax_ckpt.files]


def keras_variable_names(model):
    """List all variable names in Keras Model."""
    return [x.name for x in model.variables]


def make_weights_map(jax_names):
    """Create a dictionary mapping jax variable name to keras variable name."""
    results = {}
    for name in jax_names:

        # Handle head:
        if name.startswith("head"):
            results.update({name: f"{name}:0"})

        # Handle pre-head layer norm:
        if name.startswith("pre_head_layer_norm"):

            if name.endswith("bias"):
                results.update({name: "pre_head_layer_norm/beta:0"})
            elif name.endswith("scale"):
                results.update({name: "pre_head_layer_norm/gamma:0"})

        # Handle stem:
        elif name.startswith("stem"):
            results.update({name: f"{name}:0"})

        # Handle Blocks:
        elif name.startswith("MixerBlock"):

            # Handle channel and token mixing:
            if name.split("/")[1].endswith("mixing"):
                results.update({name: f"{name}:0"})

            # Handle layer normalization:

            if name.split("/")[1].startswith("LayerNorm"):

                name_parted = name.split("/")

                # Handle layer norm bias
                if name_parted[-1] == "bias":
                    results.update(
                        {name: "/".join(name_parted[:-1]) + "/beta:0"}
                        # Replace bias with beta:0
                    )

                # Handle layer norm scale
                elif name_parted[-1] == "scale":
                    results.update(
                        {name: "/".join(name_parted[:-1]) + "/gamma:0"}
                        # Replace scale with gamma
                    )

    return results


def main():
    """Run weight conversion script for given model variant and JAX checkpoint."""
    args = parse_args()
    arg_to_model = {"b16": MLPMixer_B16, "b32": MLPMixer_B32, "l16": MLPMixer_L16}

    model = arg_to_model[args.model](weights=None, include_top=not args.notop)

    jax_variables = dict(np.load(args.input))
    jax_names = list(jax_variables.keys())

    jax_to_keras_weights_mapping = make_weights_map(jax_names)
    keras_to_jax_reversed_weights_map = {
        y: x for (x, y) in jax_to_keras_weights_mapping.items()
    }

    # Assign jax weights to keras model:
    for v in model.variables:
        jax_name = keras_to_jax_reversed_weights_map[v.name]
        jax_variable = jax_variables[jax_name]

        assert (
            v.shape == jax_variable.shape
        ), f"Shape mismatch for {v.name}. Keras {v.shape}, jax: {jax_variable.shape}"
        v.assign(jax_variable)

    model.save(filepath=args.output, save_format=".h5", include_optimizer=False)

    print("OK. All variables matched.")
    print(f"Converted weights saved at {args.output}")


if __name__ == "__main__":
    main()
