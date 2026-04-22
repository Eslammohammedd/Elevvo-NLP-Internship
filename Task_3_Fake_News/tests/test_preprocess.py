import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from preprocess import preprocess_text, to_binary

def test_preprocess_text():
    raw_text = "ThIs Is a FAKE stAtemEnt! 100%"
    cleaned = preprocess_text(raw_text)
    
    assert "THIS" not in cleaned
    assert "100%" not in cleaned
    assert "fake statement" in cleaned

def test_to_binary():
    # 0, 4, 5 are fake (0)
    # 1, 2, 3 are real (1)
    assert to_binary(0) == 0
    assert to_binary(5) == 0
    assert to_binary(3) == 1
    assert to_binary(1) == 1
