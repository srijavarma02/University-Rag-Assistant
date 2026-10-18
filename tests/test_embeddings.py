"""Unit tests for the embeddings module (Ollama mocked)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.embeddings import embed_text, embed_texts


def _fake_embed_response(texts):
    """Return a fake response shaped like ollama.embed output."""
    return {"embeddings": [[float(len(t))] * 4 for t in texts]}


class TestEmbedTexts:
    @patch("app.embeddings.ollama.embed")
    def test_single_text(self, mock_embed):
        mock_embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        result = embed_texts(["hello"])
        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]

    @patch("app.embeddings.ollama.embed")
    def test_batch(self, mock_embed):
        texts = [f"text_{i}" for i in range(5)]

        def fake_embed(model=None, input=None, **kwargs):
            return {"embeddings": [[float(len(t))] * 4 for t in input]}

        mock_embed.side_effect = fake_embed
        result = embed_texts(texts, batch_size=3)
        assert len(result) == 5
        # Should have called embed twice (3 + 2)
        assert mock_embed.call_count == 2

    @patch("app.embeddings.ollama.embed")
    def test_large_batch_split(self, mock_embed):
        texts = [f"t{i}" for i in range(10)]
        mock_embed.return_value = _fake_embed_response(texts[:4])
        # batch_size=4 → 3 calls (4+4+2)
        embed_texts(texts, batch_size=4)
        assert mock_embed.call_count == 3

    @patch("app.embeddings.ollama.embed")
    def test_retry_on_failure(self, mock_embed):
        texts = ["hello"]
        # Fail twice, succeed on third try
        mock_embed.side_effect = [
            ConnectionError("conn refused"),
            ConnectionError("conn refused"),
            {"embeddings": [[0.5, 0.5]]},
        ]
        result = embed_texts(texts, max_retries=3)
        assert result == [[0.5, 0.5]]
        assert mock_embed.call_count == 3

    @patch("app.embeddings.ollama.embed")
    def test_retry_exhausted(self, mock_embed):
        mock_embed.side_effect = ConnectionError("conn refused")
        with pytest.raises(ConnectionError):
            embed_texts(["fail"], max_retries=2)
        assert mock_embed.call_count == 2

    @patch("app.embeddings.ollama.embed")
    def test_empty_list(self, mock_embed):
        result = embed_texts([])
        assert result == []
        mock_embed.assert_not_called()


class TestEmbedText:
    @patch("app.embeddings.ollama.embed")
    def test_single(self, mock_embed):
        mock_embed.return_value = {"embeddings": [[0.1, 0.2, 0.3, 0.4]]}
        result = embed_text("hello")
        assert len(result) == 4
        assert result == [0.1, 0.2, 0.3, 0.4]
