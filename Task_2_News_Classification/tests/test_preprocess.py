import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from preprocess import preprocess_text
from train import LABEL_MAP

def test_preprocess_text():
    raw_text = "ThIs Is a NEWs aRtIcle! 2024"
    cleaned = preprocess_text(raw_text)
    
    assert "THIS" not in cleaned
    assert "2024" not in cleaned
    assert "is" not in cleaned
    assert "news article" in cleaned

def test_label_map():
    assert LABEL_MAP[0] == 'World'
    assert LABEL_MAP[1] == 'Sports'
    assert LABEL_MAP[2] == 'Business'
    assert LABEL_MAP[3] == 'Sci/Tech'
