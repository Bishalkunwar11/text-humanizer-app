"""Tests for text_humanizer.py

These tests mock external API calls so they can run without real API keys.
"""

import json
from unittest import mock

import pytest

import text_humanizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_detection_response(score: float):
    """Return a mock ``requests.Response`` whose JSON matches GPTZero format."""
    resp = mock.MagicMock()
    resp.status_code = 200
    resp.raise_for_status = mock.MagicMock()
    resp.json.return_value = {
        "documents": [{"completely_generated_prob": score}]
    }
    return resp


# ---------------------------------------------------------------------------
# Module-level configuration tests
# ---------------------------------------------------------------------------

class TestConfiguration:
    def test_api_key_placeholders_exist(self):
        assert hasattr(text_humanizer, "OPENAI_API_KEY")
        assert hasattr(text_humanizer, "GPTZERO_API_KEY")

    def test_max_iterations_is_ten(self):
        assert text_humanizer.MAX_ITERATIONS == 10

    def test_threshold_is_five_percent(self):
        assert text_humanizer.AI_THRESHOLD == 0.05

    def test_original_essay_placeholder_exists(self):
        assert hasattr(text_humanizer, "original_essay")
        assert isinstance(text_humanizer.original_essay, str)


# ---------------------------------------------------------------------------
# detect_ai_probability tests
# ---------------------------------------------------------------------------

class TestDetectAIProbability:
    @mock.patch("text_humanizer.requests.post")
    def test_returns_score_on_success(self, mock_post):
        mock_post.return_value = _mock_detection_response(0.72)
        score = text_humanizer.detect_ai_probability("some text")
        assert score == pytest.approx(0.72)

    @mock.patch("text_humanizer.requests.post")
    def test_returns_1_on_timeout(self, mock_post):
        import requests as _req
        mock_post.side_effect = _req.exceptions.Timeout("timed out")
        score = text_humanizer.detect_ai_probability("some text")
        assert score == 1.0

    @mock.patch("text_humanizer.requests.post")
    def test_returns_1_on_connection_error(self, mock_post):
        import requests as _req
        mock_post.side_effect = _req.exceptions.ConnectionError("failed")
        score = text_humanizer.detect_ai_probability("some text")
        assert score == 1.0

    @mock.patch("text_humanizer.requests.post")
    def test_returns_1_on_bad_json(self, mock_post):
        resp = mock.MagicMock()
        resp.raise_for_status = mock.MagicMock()
        resp.json.return_value = {"unexpected": "format"}
        mock_post.return_value = resp
        score = text_humanizer.detect_ai_probability("some text")
        assert score == 1.0


# ---------------------------------------------------------------------------
# paraphrase_text tests
# ---------------------------------------------------------------------------

class TestParaphraseText:
    @mock.patch("text_humanizer.openai.OpenAI")
    def test_returns_rewritten_text(self, mock_openai_cls):
        mock_client = mock.MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_choice = mock.MagicMock()
        mock_choice.message.content = "Rewritten essay."
        mock_client.chat.completions.create.return_value = mock.MagicMock(
            choices=[mock_choice]
        )

        result = text_humanizer.paraphrase_text("Original essay.")
        assert result == "Rewritten essay."
        # Verify the model is 'clude'
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "clude"

    @mock.patch("text_humanizer.openai.OpenAI")
    def test_returns_original_on_auth_error(self, mock_openai_cls):
        mock_client = mock.MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = (
            text_humanizer.openai.AuthenticationError(
                message="bad key",
                response=mock.MagicMock(status_code=401),
                body=None,
            )
        )

        result = text_humanizer.paraphrase_text("Original essay.")
        assert result == "Original essay."


# ---------------------------------------------------------------------------
# humanize loop tests
# ---------------------------------------------------------------------------

class TestHumanizeLoop:
    @mock.patch("text_humanizer.paraphrase_text")
    @mock.patch("text_humanizer.detect_ai_probability")
    def test_exits_immediately_when_score_below_threshold(
        self, mock_detect, mock_paraphrase
    ):
        """If the very first detection is <= 0.05, the loop should exit
        without calling paraphrase at all."""
        mock_detect.return_value = 0.03
        result = text_humanizer.humanize("Good essay.")
        assert result == "Good essay."
        mock_paraphrase.assert_not_called()

    @mock.patch("text_humanizer.paraphrase_text")
    @mock.patch("text_humanizer.detect_ai_probability")
    def test_rewrites_until_score_drops(self, mock_detect, mock_paraphrase):
        """Score starts high, then drops below threshold on the second check."""
        mock_detect.side_effect = [0.80, 0.04]
        mock_paraphrase.return_value = "Improved essay."

        result = text_humanizer.humanize("AI essay.")
        assert result == "Improved essay."
        assert mock_paraphrase.call_count == 1
        assert mock_detect.call_count == 2

    @mock.patch("text_humanizer.paraphrase_text")
    @mock.patch("text_humanizer.detect_ai_probability")
    def test_respects_max_iterations(self, mock_detect, mock_paraphrase):
        """If the score never drops, the loop should stop after MAX_ITERATIONS."""
        mock_detect.return_value = 0.90
        mock_paraphrase.side_effect = lambda t: t  # identity

        text_humanizer.humanize("Stubborn essay.")
        assert mock_detect.call_count == text_humanizer.MAX_ITERATIONS
        assert mock_paraphrase.call_count == text_humanizer.MAX_ITERATIONS

    @mock.patch("text_humanizer.paraphrase_text")
    @mock.patch("text_humanizer.detect_ai_probability")
    def test_prints_5_when_passing(self, mock_detect, mock_paraphrase, capsys):
        """When the threshold is met the script must print the number 5."""
        mock_detect.return_value = 0.02
        text_humanizer.humanize("Essay.")
        captured = capsys.readouterr()
        assert "5" in captured.out
