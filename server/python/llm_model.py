from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import tensorflow as tf
from tensorflow import keras


@dataclass
class GPTConfig:
    vocabulary_size: int
    context_length: int = 128
    embedding_size: int = 128
    attention_heads: int = 4
    transformer_layers: int = 4
    feed_forward_size: int = 384
    dropout: float = 0.1
    positional_encoding: str = "learned"
    pad_token_id: int = 0


@keras.utils.register_keras_serializable(package="ai_generator")
class LearnedPositionEmbedding(keras.layers.Layer):
    def __init__(self, context_length: int, embedding_size: int, **kwargs):
        super().__init__(**kwargs)
        self.context_length = int(context_length)
        self.embedding_size = int(embedding_size)
        self.position_embedding = keras.layers.Embedding(
            input_dim=self.context_length,
            output_dim=self.embedding_size,
            name="position_embedding",
        )

    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        embedded = self.position_embedding(positions)
        return inputs + embedded

    def get_config(self):
        return {
            **super().get_config(),
            "context_length": self.context_length,
            "embedding_size": self.embedding_size,
        }


@keras.utils.register_keras_serializable(package="ai_generator")
class SinusoidalPositionEncoding(keras.layers.Layer):
    def __init__(self, context_length: int, embedding_size: int, **kwargs):
        super().__init__(**kwargs)
        self.context_length = int(context_length)
        self.embedding_size = int(embedding_size)

    def call(self, inputs):
        sequence_length = tf.shape(inputs)[1]
        positions = tf.cast(tf.range(sequence_length)[:, tf.newaxis], tf.float32)
        channels = tf.cast(tf.range(self.embedding_size)[tf.newaxis, :], tf.float32)
        angle_rates = 1.0 / tf.pow(10000.0, (2.0 * tf.floor(channels / 2.0)) / tf.cast(self.embedding_size, tf.float32))
        angles = positions * angle_rates
        sines = tf.sin(angles[:, 0::2])
        cosines = tf.cos(angles[:, 1::2])
        encoding = tf.concat([sines, cosines], axis=-1)
        encoding = encoding[tf.newaxis, :, :]
        return inputs + tf.cast(encoding, inputs.dtype)

    def get_config(self):
        return {
            **super().get_config(),
            "context_length": self.context_length,
            "embedding_size": self.embedding_size,
        }


@keras.utils.register_keras_serializable(package="ai_generator")
class TransformerBlock(keras.layers.Layer):
    def __init__(self, embedding_size: int, attention_heads: int, feed_forward_size: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.embedding_size = int(embedding_size)
        self.attention_heads = int(attention_heads)
        self.feed_forward_size = int(feed_forward_size)
        self.dropout = float(dropout)

        self.attention = keras.layers.MultiHeadAttention(
            num_heads=self.attention_heads,
            key_dim=max(1, self.embedding_size // self.attention_heads),
            dropout=self.dropout,
            name="self_attention",
        )
        self.dropout_attention = keras.layers.Dropout(self.dropout)
        self.norm_attention = keras.layers.LayerNormalization(epsilon=1e-5)

        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(self.feed_forward_size, activation="gelu"),
                keras.layers.Dropout(self.dropout),
                keras.layers.Dense(self.embedding_size),
            ],
            name="feed_forward",
        )
        self.dropout_ffn = keras.layers.Dropout(self.dropout)
        self.norm_ffn = keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, inputs, training: Optional[bool] = None):
        attention_out = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            training=training,
            use_causal_mask=True,
        )
        attention_out = self.dropout_attention(attention_out, training=training)
        hidden = self.norm_attention(inputs + attention_out)

        ffn_out = self.ffn(hidden, training=training)
        ffn_out = self.dropout_ffn(ffn_out, training=training)
        return self.norm_ffn(hidden + ffn_out)

    def get_config(self):
        return {
            **super().get_config(),
            "embedding_size": self.embedding_size,
            "attention_heads": self.attention_heads,
            "feed_forward_size": self.feed_forward_size,
            "dropout": self.dropout,
        }


def build_gpt_model(config: GPTConfig) -> keras.Model:
    token_ids = keras.Input(shape=(config.context_length,), dtype="int32", name="token_ids")
    x = keras.layers.Embedding(
        input_dim=config.vocabulary_size,
        output_dim=config.embedding_size,
        name="token_embedding",
    )(token_ids)

    if config.positional_encoding == "sinusoidal":
        x = SinusoidalPositionEncoding(
            context_length=config.context_length,
            embedding_size=config.embedding_size,
            name="sinusoidal_position_encoding",
        )(x)
    else:
        x = LearnedPositionEmbedding(
            context_length=config.context_length,
            embedding_size=config.embedding_size,
            name="learned_position_encoding",
        )(x)

    x = keras.layers.Dropout(config.dropout, name="embedding_dropout")(x)
    for index in range(config.transformer_layers):
        x = TransformerBlock(
            embedding_size=config.embedding_size,
            attention_heads=config.attention_heads,
            feed_forward_size=config.feed_forward_size,
            dropout=config.dropout,
            name=f"transformer_block_{index + 1}",
        )(x)

    x = keras.layers.LayerNormalization(epsilon=1e-5, name="final_layer_norm")(x)
    logits = keras.layers.Dense(config.vocabulary_size, name="lm_head")(x)
    return keras.Model(inputs=token_ids, outputs=logits, name="custom_gpt")


def build_model_config(payload: dict, vocabulary_size: int, pad_token_id: int = 0) -> GPTConfig:
    embedding_size = max(32, int(payload.get("embeddingSize", 128)))
    attention_heads = max(1, int(payload.get("attentionHeads", 4)))
    while attention_heads > 1 and embedding_size % attention_heads != 0:
        attention_heads -= 1

    return GPTConfig(
        vocabulary_size=max(32, int(vocabulary_size)),
        context_length=max(16, int(payload.get("sequenceLength", 128))),
        embedding_size=embedding_size,
        attention_heads=attention_heads,
        transformer_layers=max(1, int(payload.get("transformerLayers", 4))),
        feed_forward_size=max(embedding_size, int(payload.get("feedForwardSize", embedding_size * 4))),
        dropout=min(0.6, max(0.0, float(payload.get("dropout", 0.1)))),
        positional_encoding=str(payload.get("positionalEncoding", "learned") or "learned"),
        pad_token_id=max(0, int(pad_token_id)),
    )


def count_model_parameters(model: keras.Model) -> int:
    return int(sum(int(tf.size(weight)) for weight in model.trainable_weights))
