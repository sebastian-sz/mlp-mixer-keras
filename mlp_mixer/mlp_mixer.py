"""MLP-Mixer models, reimplemented in Keras."""

import tensorflow as tf
from keras.applications import imagenet_utils
from keras.utils import layer_utils
from tensorflow.keras import backend
from tensorflow.python.lib.io import file_io

POSSIBLE_WEIGHT_VARIANTS = {
    "b16": ["imagenet", "sam"],
    "b32": ["sam"],
    "l16": ["imagenet"],
}


def MLPMixer(
    num_blocks: int,
    patch_size: int,
    hidden_size: int,
    tokens_mlp_dim: int,
    channels_mlp_dim: int,
    default_size: int,
    dropout_rate=0.2,
    epsilon=1e-06,
    name="",
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    """Create MLP-Mixer model, based on provided parameters.

    # TODO: extend this docstring, with parameter description.
    """
    # Handle weights
    model_variant = name.split("-")[-1]  # b16 or b32 or l16
    possible_weights = POSSIBLE_WEIGHT_VARIANTS[model_variant]
    if not (weights in {None, *possible_weights} or file_io.file_exists_v2(weights)):
        raise ValueError(
            f"The `weights` argument should be either "
            "`None` (random initialization),"
            f"{possible_weights}, (pretrained on Imagenet)"
            "or the path to the weights file to be loaded."
        )

    if weights in {"imagenet", "sam"} and include_top and classes != 1000:
        raise ValueError(
            'If using `weights` as `"imagenet"` or `"sam"` with `include_top`'
            " as true, `classes` should be 1000"
        )

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )

    # Define input
    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Build Stem (extract patches)
    x = tf.keras.layers.Conv2D(
        filters=hidden_size,
        kernel_size=(patch_size, patch_size),
        strides=(patch_size, patch_size),
        padding="same",
        name="stem",
    )(img_input)

    x = tf.keras.layers.Reshape(target_shape=(-1, x.shape[-1]))(x)

    # Build Blocks:
    for i in range(num_blocks):
        x = mixer_block(
            x,
            tokens_mlp_dim=tokens_mlp_dim,
            channels_mlp_dim=channels_mlp_dim,
            dropout_rate=dropout_rate,
            name=f"MixerBlock_{i}/",
        )

    # Build Head
    x = tf.keras.layers.LayerNormalization(epsilon=epsilon, name="pre_head_layer_norm")(
        x
    )

    if include_top:
        x = tf.keras.layers.GlobalAveragePooling1D(name="avg_pool")(x)
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate, name="top_dropout")(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = tf.keras.layers.Dense(
            classes,
            activation=classifier_activation,
            kernel_initializer="zeros",
            name="head",
        )(x)
    else:
        if pooling == "avg":
            x = tf.keras.layers.GlobalAveragePooling1D(name="avg_pool")(x)
        elif pooling == "max":
            x = tf.keras.layers.GlobalMaxPooling1D(name="max_pool")(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model
    model = tf.keras.Model(inputs=[inputs], outputs=[x], name=name)

    # Load weights.
    # TODO: add logic for weights download.

    return model


def mixer_block(
    inputs,
    tokens_mlp_dim: int,
    channels_mlp_dim: int,
    dropout_rate: float = 0.2,
    epsilon: float = 1e-6,
    name: str = "",
):
    """Build a single mixer block."""
    x = tf.keras.layers.LayerNormalization(epsilon=epsilon, name=name + "LayerNorm_0/")(
        inputs
    )
    x = tf.keras.layers.Permute((2, 1))(x)

    x = mlp_block(
        x,
        mlp_dim=tokens_mlp_dim,
        dropout_rate=dropout_rate,
        name=name + "token_mixing/",
    )

    x = tf.keras.layers.Permute((2, 1))(x)
    out = tf.keras.layers.Add()([x, inputs])

    x = tf.keras.layers.LayerNormalization(epsilon=epsilon, name=name + "LayerNorm_1/")(
        out
    )
    x = mlp_block(
        x,
        mlp_dim=channels_mlp_dim,
        dropout_rate=dropout_rate,
        name=name + "channel_mixing/",
    )

    return tf.keras.layers.Add()([x, out])


def mlp_block(inputs, mlp_dim: int, dropout_rate: float = 0.2, name: str = ""):
    """Single MLP block."""
    x = tf.keras.layers.Dense(mlp_dim, name=name + "Dense_0")(inputs)
    x = tf.keras.activations.gelu(x, approximate=True, name=name + "gelu_activation")

    units = inputs.shape[-1]
    x = tf.keras.layers.Dense(units=units, name=name + "Dense_1")(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    return x


def MLPMixer_B16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    """Build B16 MLP-Mixer model variant."""
    return MLPMixer(
        num_blocks=12,
        patch_size=16,
        hidden_size=768,
        tokens_mlp_dim=384,
        channels_mlp_dim=3072,
        default_size=224,
        name="mlp-mixer-b16",
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )


def MLPMixer_B32(
    include_top=True,
    weights="sam",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    """Build B32 MLP-Mixer model variant."""
    return MLPMixer(
        num_blocks=12,
        patch_size=32,
        hidden_size=768,
        tokens_mlp_dim=384,
        channels_mlp_dim=3072,
        name="mlp-mixer-b32",
        default_size=224,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )


def MLPMixer_L16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    """Build L16 MLP-Mixer model variant."""
    return MLPMixer(
        num_blocks=24,
        patch_size=16,
        hidden_size=1024,
        tokens_mlp_dim=512,
        channels_mlp_dim=4096,
        name="mlp-mixer-l16",
        default_size=224,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )
