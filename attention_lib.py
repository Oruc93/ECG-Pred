# this library is for constructing a attention neural network
# we code the encoder part of the transformer tutorial of TensorFlow https://www.tensorflow.org/text/tutorials/transformer#the_transformer

import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow_text

def positional_encoding(length, depth, dtype=tf.float32):
    """ this function creates array of shape (length, depth)
    along length sinus and cosines oscillate 
    along depth sinus and cosines has different frequencies
    
    Is needed for positional encoding for attention input

    Args:
        length (int): length of time series after downsampling
        depth (int): number of filters of convolution

    Returns:
        tensor: positional embedding
    """
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1) 

    return tf.cast(pos_encoding, dtype=dtype)

class PositionalEmbedding(tf.keras.layers.Layer):
    """ This Layer adds positional embedding on input"""
    def __init__(self, d_model, length):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = positional_encoding(length=length, depth=d_model, dtype=self.compute_dtype)

    def call(self, x):
        length = tf.shape(x)[1]
        # This factor sets the relative scale of the embedding and positional_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, self.compute_dtype))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x
    
class BaseAttention(tf.keras.layers.Layer):
    """ base class for attention use-cases"""
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

class GlobalSelfAttention(BaseAttention):
    """ this attention layer considers all timesteps
    regardless of causality or distance"""
    def call(self, x):
        # multi-head attention
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        # residual connection to input before mha
        # this way the gradient can flow around mha, if attention does not work
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
    
class CausalSelfAttention(BaseAttention):
    """ this attention layer considers all timesteps before current timestep
    this way causality is valid in analysis
    regardless of distance"""
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask = True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class SlidingWindowSelfAttention(BaseAttention):
    """ this attention layer considers all timesteps in window around current step
    regardless of causality
    
    w_2 is half window size"""
    def call(self, x, w_2):
        # sliding window mask
        mask = get_sliding_window_attention_mask(x, w_2)
        # multi-head attention
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            attention_mask= mask)
        # residual connection to input before mha
        # this way the gradient can flow around mha, if attention does not work
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
    
class FeedForward(tf.keras.layers.Layer):
    """ two point-wise Dense layers with dropout layer
    followed by residual connections
    
    dff = length of timeseries after downsampling"""
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model),
        tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x
    
class EncoderLayer(tf.keras.layers.Layer):
    """ Encoder with multi-head attention and point-wise feedforward layer"""
    def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        self.sliding_window_self_attention = SlidingWindowSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        self.ffn = FeedForward(d_model, dff)

    def call(self, x, w_2=0):
        if w_2 == 0:
            x = self.self_attention(x)
        else:
            x = self.sliding_window_self_attention(x, w_2)
        x = self.ffn(x)
        return x
    
class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, length, num_heads,
                dff, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(d_model=d_model, length=length)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                        num_heads=num_heads,
                        dff=dff,
                        dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, w_2=0):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, w_2=w_2)
            
        return x # Shape `(batch_size, seq_len, d_model)`.

def get_sliding_window_attention_mask(inputs, w_2):
    """Generates sliding window mask for multi-head attention layers

    Args:
        inputs (tensor): input for multi-head attention layer (self-attention)
                            Just need shape of tensor. Expected axis (Batch, sequence length, features)
        w_2 (int): half window size. how many steps left and right of current time step are looked at

    Returns:
        tensor: sliding window mask
    """
    input_shape = tf.shape(inputs) # get shape
    batch_size, sequence_length = input_shape[0], input_shape[1] # deconstruct meaning

    i = tf.range(sequence_length)[:, tf.newaxis]
    j = tf.range(sequence_length)

    mask = tf.cast(tf.abs(i-j)<=w_2, dtype="int32")
    mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))

    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
        axis=0,
    )

    # print(tf.tile(mask, mult)[0,:,:])
    # exit()
    return tf.tile(mask, mult)