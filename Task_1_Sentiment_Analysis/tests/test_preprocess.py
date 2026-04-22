import sys
import os
import pytest
from pandas import DataFrame

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from preprocess import preprocess_text

def test_preprocess_text():
    raw_text = "ThIs Is A <br /> wOrSt movie EVER! 123"
    cleaned = preprocess_text(raw_text)
    
    # Check lowercasing
    assert "THIS" not in cleaned
    # Check punctuation removal
    assert "<br />" not in cleaned
    # Check stopword removal
    assert "is" not in cleaned
    # It should correctly process to 'worst movie ever'
    assert "worst" in cleaned
