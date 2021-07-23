"""MLP-Mixer models, reimplemented in Keras."""
import math
from typing import Tuple, Union

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


def _validate_input_shape(
    input_shape: Union[None, Tuple[int, int, int]],
    default_shape: Tuple[int, int, int],
    minimum_shape: int,
    pretrained: bool,
):
    if input_shape is None:
        return default_shape

    if pretrained:
        if input_shape != default_shape:
            raise ValueError(
                "When passing weights `imagenet` or `sam` input shape must be"
                f"equal to default input shape: {default_shape} or `None`."
            )
    else:
        if len(input_shape) != 3:
            raise ValueError("Input shape must be tuple of 3 integer.")
        elif input_shape[-1] not in {1, 3}:
            raise ValueError("Input shape must have 1 or 3 channels in the last dim.")
        elif input_shape[0] < minimum_shape or input_shape[1] < minimum_shape:
            raise ValueError(f"Minimum value for input height/width is {minimum_shape}")
        else:
            return input_shape


def MLPMixer(
    num_blocks: int,
    patch_size: int,
    hidden_size: int,
    tokens_mlp_dim: int,
    channels_mlp_dim: int,
    dropout_rate=0.2,
    epsilon=1e-06,
    default_shape: Tuple[int, int, int] = (224, 224, 3),
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

    :arg num_blocks: how many times to repeat mixer block in the model architecture.
    :arg patch_size: how big (in pixels) the extracted patches should be.
    :arg hidden_size: number of filters in STEM conv2d layer.
    :arg tokens_mlp_dim: number of dense layer units in token-mixing part of the block.
    :arg channels_mlp_dim: number of dense layer units in channel-mixing part of the
        block.
    :arg dropout_rate: at what rate to apply dropout to layers. This was not present
        in the original repo, but I added it as a regularization technique.
    :arg epsilon: epsilon argument for LayerNormalization.
    :arg default_shape: shape that the network has been trained with, in the original
        repository. Also, for this shape the official weights have been released.

    :arg name: name of the model. Useful to determine which model variant is currently
        used.
    :arg include_top: whether to include avg_pooling + prediction head.
        Defaults to False.
    :arg weights: path to .h5 file or "imagenet" or "sam" (depending on the model
        variant). Not all combinations variant-weights have been released. Defaults to
        `imagenet` or `sam`, if imagenet is not available.
    :arg input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use
        as image input for the model. Defaults to None.
    :arg input_shape: input shape to build the model with. Defaults to None.
    :arg pooling: only works when `include_top=False`. Which pooling to apply after last
        block and normalization. Can be one of {"avg", "max" or None}. Defaults to None.
    :arg classes: only works when include_top=False. Number of classes for prediction
        head. Defaults to 1000.
    :arg classifier_activation: function to apply to classifier head. Defaults to
        "softmax".
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
    input_shape = _validate_input_shape(
        input_shape=input_shape,
        default_shape=default_shape,
        pretrained=weights in ["imagenet", "sam"],
        minimum_shape=patch_size,
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

    # Reshape output from [batch_size, N, M] to [batch_size, sqrt(N), sqrt(N), M]:
    feature_size = int(math.sqrt(x.shape[-2]))
    x = tf.keras.layers.Reshape(target_shape=(feature_size, feature_size, -1))(x)

    if include_top:
        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
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
            x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = tf.keras.layers.GlobalMaxPooling2D(name="max_pool")(x)

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
        default_shape=(224, 224, 3),
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
        default_shape=(224, 224, 3),
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
        default_shape=(224, 224, 3),
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )
