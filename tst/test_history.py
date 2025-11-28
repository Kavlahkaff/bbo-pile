import os
import tempfile

from syne_tune.experiments import load_experiment

from syne_tune.config_space import randint, uniform, choice, finrange
from syne_tune.constants import SYNE_TUNE_ENV_FOLDER

from open_optformer.history import History, Trial, encode, quantize, preprocess

def test_quantize():
    assert quantize(0.5, 0, 1, q=1000) == 500
    assert quantize(0, 0, 1, q=1000) == 0
    assert quantize(1, 0, 1, q=1000) == 1000


def test_encode():
    assert encode(0.5, uniform(0, 1), q=1000) == 500
    assert encode(5, randint(0, 10), q=1000) == 500
    assert encode('a', choice(['a', 'b', 'c']), q=1000) == 0
    assert encode('c', choice(['a', 'b', 'c']), q=1000) == 2


def test_encode_hp_config_space():
    config_space = {
        'x': uniform(-1, 3),
        'y': randint(-2, 5),
        'z': choice(['a', 'b', 'c']),
        't': finrange(0, 3, 4),
    }
    history = History(name='test', algorithm='test', config_space=config_space, num_numeric_tokens=1000)

    expecteds = {
        "x": "{name:x,type:UNI,min_value:-1,max_value:3,}",
        "y": "{name:y,type:INT,min_value:-2,max_value:5,}",
        "z": "{name:z,type:CAT,categories:['a','b','c'],}",
        "t": "{name:t,type:UNI,min_value:0,max_value:3,}",
    }
    for hp, expected in expecteds.items():
        got = history._encode_hp_config_space(hp_name=hp)
        print(hp)
        print(expected)
        print(got)
        assert got == expected


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
    assert isinstance(prompt, str)
    assert 'benchmark:test' in prompt
    assert 'algorithm:test' in prompt
    assert 'parameter:{name:x,type:UNI,min_value:0,max_value:1,}' in prompt
    assert 'parameter:{name:y,type:INT,min_value:0,max_value:10,}' in prompt
    assert "parameter:{name:z,type:CAT,categories:['a','b','c'],}" in prompt
    ("benchmark:test,algorithm:test,parameter:{name:x,type:UNI,min_value:0,max_value:1,}parameter:{name:y,type:INT,min_value:0,max_value:10,}parameter:{name:z,type:CAT,categories:['a','b','c'],}&"
     "*0|,500*0|,0*0|*1000|,600*1000|,1*1000|")
    assert '&500,500,0*0|600,600,1*1000|' in prompt
    
def test_trial():
    trial = Trial(config={'x': 0.5}, metric=0.5)
    assert trial.config == {'x': 0.5}
    assert trial.metric == 0.5

def test_preprocess():
    prompt = 'parameter "trial" '
    processed_prompt = preprocess(prompt)
    assert processed_prompt == ''

def test_from_syne_tune_experiment():
    from syne_tune import Tuner, StoppingCriterion
    from syne_tune.backend import PythonBackend
    from syne_tune.config_space import randint
    from syne_tune.optimizer.baselines import RandomSearch

    def train_height(steps: int, width: float, height: float):
        """
        The function to be tuned, note that import must be in PythonBackend and no global variable are allowed,
        more details on requirements of tuned functions can be found in
        :class:`~syne_tune.backend.PythonBackend`.
        """
        from syne_tune import Reporter

        reporter = Reporter()
        for step in range(steps):
            dummy_score = (0.1 + width * step / 100) ** (-1) + height * 0.1
            # Feed the score back to Syne Tune.
            reporter(step=step, mean_loss=dummy_score)
            #time.sleep(0.1)

    config_space = {
        "steps": 100,
        "width": randint(0, 20),
        "height": randint(-100, 100),
    }

    metric = "mean_loss"
    scheduler = RandomSearch(
        config_space,
        metrics=[metric],
    )


    stop_criterion = StoppingCriterion(
        max_num_trials_completed=1,
    )

    with tempfile.TemporaryDirectory() as local_path:
        os.environ[SYNE_TUNE_ENV_FOLDER] = local_path
        backend = PythonBackend(tune_function=train_height, config_space=config_space)
        backend.set_path(results_root=local_path)
        tuner = Tuner(
            trial_backend=backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=1,
            save_tuner=False,
            
        )
        tuner.run()
        experiment = load_experiment(tuner_name=tuner.name, local_path=local_path)
        history = History.from_syne_tune_experiment(experiment)

        assert len(history.trials) == 1