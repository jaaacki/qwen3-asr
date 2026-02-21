import logging
import unittest
from unittest.mock import patch, MagicMock
from loguru import logger
import sys

# Import our custom setup
from logger import setup_logger, InterceptHandler

class TestLoguruIntegration(unittest.TestCase):
    
    def setUp(self):
        # Reset the logging handlers to avoid test interference
        logger.remove()
        logging.root.handlers = []

    def test_intercept_handler_emits_to_loguru(self):
        """Test that the InterceptHandler properly bridges python standard logging into Loguru."""
        
        # We need to capture loguru's output to verify
        captured = []
        logger.add(lambda msg: captured.append(msg), format="{message}")
        
        handler = InterceptHandler()
        logging.root.addHandler(handler)
        
        # Emit a standard log
        standard_logger = logging.getLogger("test_standard_logger")
        standard_logger.setLevel(logging.INFO)
        standard_logger.info("This is a standard log message")
        
        # Check if Loguru sink caught it
        self.assertTrue(len(captured) > 0)
        self.assertIn("This is a standard log message", captured[0])

    def test_setup_logger_intercepts_uvicorn(self):
        """Test that setup_logger properly overwrites uvicorn's default handlers."""
        # Add a dummy handler to uvicorn to simulate it being configured
        uvicorn_logger = logging.getLogger("uvicorn.access")
        uvicorn_logger.addHandler(logging.StreamHandler(sys.stdout))
        uvicorn_logger.propagate = True
        
        # Run our setup
        custom_logger = setup_logger()
        
        # Verify uvicorn.access logger was hijacked
        hijacked_logger = logging.getLogger("uvicorn.access")
        self.assertEqual(len(hijacked_logger.handlers), 1)
        self.assertIsInstance(hijacked_logger.handlers[0], InterceptHandler)
        self.assertFalse(hijacked_logger.propagate)
        
    def test_loguru_receives_direct_logs(self):
        """Test that loguru direct logs work correctly."""
        captured = []
        logger.add(lambda msg: captured.append(msg), format="{level} | {message}")
        
        logger.error("Direct Error Log")
        
        self.assertTrue(len(captured) > 0)
        self.assertIn("Direct Error Log", captured[0])
        self.assertIn("ERROR", captured[0])

if __name__ == '__main__':
    unittest.main()
