import pytest
import numpy as np
from starter_code_HNSW import evaluate_hnsw

def test_evaluate_hnsw():
    
    expected = [932085, 934876, 561813, 708177, 706771, 695756, 435345, 701258, 455537, 872728]
    
    result = evaluate_hnsw()
    
    assert np.array_equal(expected, result), "The retrieved indices do not match the expected values."
