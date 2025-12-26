import logging
import json
import unittest
from io import StringIO
from config.logging_config import setup_logging

class TestLoggingSecurity(unittest.TestCase):
    def test_authorize_token_redaction(self):
        """Verify that 'authorize' tokens are redacted from logs."""
        
        # Setup Logger capturing output to string
        logger = logging.getLogger("test_security")
        logger.setLevel(logging.INFO)
        
        # Clear handlers
        logger.handlers = []
        
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        # IMPORTANT: We must manually attach the filter if we build handler manually
        # But setup_logging does this. Let's reuse setup_logging logic pattern
        # or just import TokenSanitizer if we can export it?
        # TokenSanitizer is in config.logging_config
        
        from config.logging_config import TokenSanitizer, JsonFormatter
        
        sanitizer = TokenSanitizer()
        handler.addFilter(sanitizer)
        handler.setFormatter(JsonFormatter())
        
        logger.addHandler(handler)
        
        # Test Case 1: Direct JSON message
        sensitive_json = '{"authorize": "sensitive_api_token_123"}'
        logger.info(sensitive_json)
        
        output = stream.getvalue()
        log_json = json.loads(output)
        
        self.assertIn('"authorize": "***"', log_json["message"])
        self.assertNotIn("sensitive_api_token_123", log_json["message"])
        
        # Test Case 2: Dictionary conversion (if app logs dict as string)
        # Often apps log: logger.info(f"Sending: {msg}")
        # Note: The filter works on the message STRING. 
        # So if we format before logging, it works.
        
        stream.truncate(0)
        stream.seek(0)
        
        msg_dict = {"authorize": "another_token_456", "req_id": 1}
        logger.info(f"Sending: {json.dumps(msg_dict)}")
        
        output = stream.getvalue()
        log_json = json.loads(output)
        
        self.assertIn('"authorize": "***"', log_json["message"])
        self.assertNotIn("another_token_456", log_json["message"])

if __name__ == "__main__":
    unittest.main()
