"""MLP-Mixer models, reimplemented in Keras."""


import tensorflow as tf


def MLPMixer(
    num_classes: int,
    num_blocks: int,
    patch_size: int,
    hidden_size: int,
    tokens_mlp_dim: int,
    channels_mlp_dim: int,
    image_size: int,  # TODO: input_shape Tuple[int, int, int]?
    dropout_rate=0.2,
    epsilon=1e-06,
    name="",
):
    """Create MLP-Mixer model, based on provided parameters.

    # TODO: extend this docstring, with parameter description.
    """
    # Define Input
    inputs = tf.keras.Input(
        shape=(image_size, image_size, 3)
    )  # TODO: input_shape Tuple[int, int, int]?

    # Build Stem (extract patches)
    x = tf.keras.layers.Conv2D(
        filters=hidden_size,
        kernel_size=(patch_size, patch_size),
        strides=(patch_size, patch_size),
        padding="same",
        name="stem",
    )(inputs)

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
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(
        units=num_classes, name="head", kernel_initializer="zeros"
    )(x)

    return tf.keras.Model(inputs=[inputs], outputs=[x], name=name)


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


def MLPMixer_B16():
    """Build B16 MLP-Mixer model variant."""
    # TODO: keras-like params.
    return MLPMixer(
        num_classes=1000,
        num_blocks=12,
        patch_size=16,
        hidden_size=768,
        tokens_mlp_dim=384,
        channels_mlp_dim=3072,
        image_size=224,
    )


def MLPMixer_B32():
    """Build B32 MLP-Mixer model variant."""
    # TODO: keras-like params.
    return MLPMixer(
        num_classes=1000,
        num_blocks=12,
        patch_size=32,
        hidden_size=768,
        tokens_mlp_dim=384,
        channels_mlp_dim=3072,
        image_size=224,
    )


def MLPMixer_L16():
    """Build L16 MLP-Mixer model variant."""
    # TODO: keras-like params.
    return MLPMixer(
        num_classes=1000,
        num_blocks=24,
        patch_size=16,
        hidden_size=1024,
        tokens_mlp_dim=512,
        channels_mlp_dim=4096,
        image_size=224,
    )
