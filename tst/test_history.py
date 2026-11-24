import os
import tempfile

from syne_tune.experiments import load_experiment

from syne_tune.config_space import randint, uniform, choice, finrange
from syne_tune.constants import SYNE_TUNE_ENV_FOLDER

from open_optformer.history import History, Trial, encode, quantize, dequantize

def test_quantize():
    assert quantize(0.5, 0, 1, q=1000) == 500
    assert quantize(0, 0, 1, q=1000) == 0
    assert quantize(1, 0, 1, q=1000) == 999


def test_quantize_max_below_q():
    """quantize(x_max, ...) must always be < q, matching tokenizer's range(q)."""
    assert quantize(1, 0, 1, q=1000) < 1000
    assert quantize(10, 0, 10, q=1000) < 1000
    assert quantize(100, 0, 100, q=1000) < 1000
    # log scale
    assert quantize(1, 0.01, 1, q=1000, log_scale=True) < 1000


def test_quantize_dequantize_roundtrip():
    """dequantize(quantize(x)) should approximate x."""
    import numpy as np
    test_cases = [
        (0.5, 0.0, 1.0, False),
        (0.0, 0.0, 1.0, False),
        (1.0, 0.0, 1.0, False),
        (5, 0, 10, False),
        (100, 0, 200, False),
        (0.1, 0.01, 1, True),
        (0.5, 0.01, 1, True),
    ]
    q = 1000
    for x, x_min, x_max, log_scale in test_cases:
        x_q = quantize(x, x_min, x_max, q, log_scale)
        x_rt = dequantize(x_q, x_min, x_max, q, log_scale)
        if log_scale:
            tol = (np.log(x_max + 1e-10) - np.log(x_min + 1e-10)) / (2 * (q - 1))
            assert abs(np.log(x + 1e-10) - np.log(x_rt + 1e-10)) <= tol + 1e-9, \
                f"Round-trip failed for x={x}, got {x_rt}"
        else:
            tol = (x_max - x_min) / (2 * (q - 1))
            assert abs(x - x_rt) <= tol + 1e-9, \
                f"Round-trip failed for x={x}, got {x_rt}"


def test_encode():
    assert encode(0.5, uniform(0, 1), q=1000) == 500
    assert encode(5, randint(0, 10), q=1000) == 500
    assert encode('a', choice(['a', 'b', 'c']), q=1000) == '<0>'
    assert encode('c', choice(['a', 'b', 'c']), q=1000) == '<2>'

def test_history():
    config_space = {
        'x': uniform(0, 1),
        'y': randint(0, 10),
        'z': choice(['a', 'b', 'c'])
    }
    history = History(name='test', algorithm='test', config_space=config_space, num_numeric_tokens=1000)
    history.add_trial({'x': 0.5, 'y': 5, 'z': 'a'}, 0.5)
    history.add_trial({'x': 0.6, 'y': 6, 'z': 'b'}, 0.6)
    prompt = history.get_prompt()
    print(prompt)
    assert isinstance(prompt, str)
    assert 'benchmark:test' in prompt
    assert 'algorithm:test' in prompt
    assert '{name:x,type:UNI,min_value:0,max_value:1,linear_scale}' in prompt
    assert '{name:y,type:INT,min_value:0,max_value:10,linear_scale}' in prompt
    assert "{name:z,type:CAT,categories:[0,1,2]}" in prompt
    assert '500,500,<0>*0|599,599,<1>*999|' in prompt
    
def test_trial():
    trial = Trial(config={'x': 0.5}, metric=0.5)
    assert trial.config == {'x': 0.5}
    assert trial.metric == 0.5
