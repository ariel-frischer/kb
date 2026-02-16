"""Tests for kb.embed — serialization and embedding."""

import struct
from unittest.mock import MagicMock

from kb.config import Config
from kb.embed import embed_batch, serialize_f32


class TestSerializeF32:
    def test_roundtrip(self):
        vec = [1.0, 2.5, -3.14, 0.0]
        serialized = serialize_f32(vec)
        assert isinstance(serialized, bytes)
        assert len(serialized) == 4 * len(vec)
        unpacked = list(struct.unpack(f"{len(vec)}f", serialized))
        for a, b in zip(vec, unpacked):
            assert abs(a - b) < 1e-5

    def test_empty_vector(self):
        assert serialize_f32([]) == b""

    def test_single_element(self):
        serialized = serialize_f32([42.0])
        assert len(serialized) == 4


class TestEmbedBatch:
    def test_calls_openai_correctly(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.data = [
            MagicMock(embedding=[0.1, 0.2]),
            MagicMock(embedding=[0.3, 0.4]),
        ]
        mock_client.embeddings.create.return_value = mock_resp

        cfg = Config(embed_model="test-model", embed_dims=2)
        result = embed_batch(mock_client, ["text1", "text2"], cfg)

        mock_client.embeddings.create.assert_called_once_with(
            model="test-model", input=["text1", "text2"], dimensions=2
        )
        assert result == [[0.1, 0.2], [0.3, 0.4]]
