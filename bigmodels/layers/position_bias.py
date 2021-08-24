from .base import Layer
from .embedding import Embedding
from ..allocator import Allocator
from ..functions.scale_copy import elementwise_copy
import math
import numpy as np
import cupy


class PositionBias(Layer):
    def __init__(self, num_buckets, num_heads, is_decoder):
        self.num_buckets = num_buckets
        self.is_decoder = is_decoder
        self.num_heads = num_heads
        self.embedding = Embedding(num_buckets, num_heads)

    
    def _relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).astype(np.int32) * num_buckets
            relative_position = np.abs(relative_position)
        else:
            relative_position = -np.clip(relative_position, None, 0)
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            np.log(relative_position.astype(np.float32) / max_exact + 1e-6)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(np.int32)
        relative_postion_if_large = np.clip(relative_postion_if_large, 0, num_buckets - 1)
        relative_buckets += np.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def forward(self, allocator : Allocator, query_len, key_len):
        context_position = np.arange(query_len, dtype=np.int32)[:, np.newaxis]
        memory_position = np.arange(key_len, dtype=np.int32)[np.newaxis, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets= self.num_buckets,
        )
        out = self.embedding.forward(allocator, relative_position_bucket)
        assert out.shape == (query_len, key_len, self.num_heads)
        out = out.transpose((2, 1, 0))[cupy.newaxis]
        return out  # (1, num_heads, key_len, query_len)
