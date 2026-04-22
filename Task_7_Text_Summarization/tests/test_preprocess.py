import sys
import os
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from preprocess import truncate_for_t5

def test_truncate_for_t5():
    # Mocking a tokenizer to return ids
    class MockTokenizer:
        def encode(self, text, truncation, max_length):
            return [1, 2, 3] # mock input ids
        def decode(self, ids, skip_special_tokens):
            return "This is a short sequence."
            
    tokenizer = MockTokenizer()
    long_text = "This should practically be a long text but the mock returns short."
    
    truncated = truncate_for_t5(long_text, tokenizer)
    assert truncated == "This is a short sequence."
