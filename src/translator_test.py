import os
import sys
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

# Mock 'openai' module so we can test without it being installed
mock_openai = MagicMock()
mock_async_openai = MagicMock()
mock_openai.AsyncOpenAI = mock_async_openai
sys.modules['openai'] = mock_openai

from translator import _get_client, translate_text, translate_srt

class TestTranslator(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        # Reset environment
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        if "OPENAI_BASE_URL" in os.environ:
            del os.environ["OPENAI_BASE_URL"]
        if "TRANSLATE_MODEL" in os.environ:
            del os.environ["TRANSLATE_MODEL"]
            
        # Reset mocks
        mock_async_openai.reset_mock()
            
    def test_get_client_defaults(self):
        """Test client initialization with default empty keys."""
        client = _get_client()
        mock_async_openai.assert_called_with(api_key="EMPTY")
        self.assertIsNotNone(client)

    def test_get_client_custom_env(self):
        """Test client initialization with explicit keys."""
        os.environ["OPENAI_API_KEY"] = "sk-test1234"
        os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
        _get_client()
        mock_async_openai.assert_called_with(api_key="sk-test1234", base_url="http://localhost:11434/v1")

    @patch('translator._get_client')
    async def test_translate_text_empty(self, mock_get_client):
        """Empty texts should return immediately."""
        result = await translate_text("   ", "en")
        self.assertEqual(result, "   ")
        mock_get_client.assert_not_called()

    @patch('translator._get_client')
    async def test_translate_text_success(self, mock_get_client):
        """Test normal text translation behavior."""
        mock_client = MagicMock()
        mock_completions = AsyncMock()
        mock_client.chat.completions.create = mock_completions
        mock_get_client.return_value = mock_client
        
        # Mocking the response
        mock_choice = MagicMock()
        mock_choice.message.content = "  Hello translated world!  "
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_completions.return_value = mock_response

        # Execute
        result = await translate_text("Bonjour le monde", "en")
        
        # Verify
        self.assertEqual(result, "Hello translated world!")
        # We expect temperature=0.3
        mock_completions.assert_called_once()
        args, kwargs = mock_completions.call_args
        self.assertEqual(kwargs['temperature'], 0.3)
        self.assertEqual(kwargs['model'], 'gpt-3.5-turbo')
        # Check that proper roles exist
        self.assertTrue(len(kwargs['messages']) >= 2)
        self.assertTrue(kwargs['messages'][1]['role'] == 'user')
        self.assertIn("English", kwargs['messages'][1]['content'])
        self.assertIn("Bonjour le monde", kwargs['messages'][1]['content'])

    @patch('translator._get_client')
    async def test_translate_srt_empty(self, mock_get_client):
        """Empty SRTs should return immediately."""
        result = await translate_srt("", "zh")
        self.assertEqual(result, "")
        mock_get_client.assert_not_called()

    @patch('translator._get_client')
    async def test_translate_srt_strips_markdown(self, mock_get_client):
        """Test if markdown output tags from LLMs are correctly stripped."""
        mock_client = MagicMock()
        mock_completions = AsyncMock()
        mock_client.chat.completions.create = mock_completions
        mock_get_client.return_value = mock_client
        
        mock_choice = MagicMock()
        mock_choice.message.content = "```srt\n1\n00:00:01,000 --> 00:00:02,000\n你好\n```"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_completions.return_value = mock_response

        # Execute
        srt_input = "1\n00:00:01,000 --> 00:00:02,000\nHello"
        result = await translate_srt(srt_input, "zh")
        
        # Verify markdown stripping logic works correctly without breaking timestamps
        self.assertEqual(result, "1\n00:00:01,000 --> 00:00:02,000\n你好")
        # Ensure it passes "Chinese" appropriately down the chain
        args, kwargs = mock_completions.call_args
        self.assertIn("Chinese", kwargs['messages'][1]['content'])
        self.assertEqual(kwargs['temperature'], 0.1)

    @patch('translator._get_client')
    async def test_translate_fails_if_no_choices(self, mock_get_client):
        """Test that translation accurately propagates a ValueError if API returns empty"""
        mock_client = MagicMock()
        mock_completions = AsyncMock()
        mock_client.chat.completions.create = mock_completions
        mock_get_client.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = []
        mock_completions.return_value = mock_response

        with self.assertRaises(ValueError):
            await translate_text("Some text", "en")

if __name__ == '__main__':
    unittest.main()
