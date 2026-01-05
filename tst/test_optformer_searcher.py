import numpy as np
import pathlib
import tempfile

from litgpt.scripts.download import download_from_hub
from syne_tune.config_space import randint, uniform, choice

from open_optformer.optformer_searcher import OptFormerSearcher


def test_optformer_searcher():

    config_space = {
    #    "a": choice([0, 1, 2, 3, 4]),  # we use a special encoding for categorical variables which does not work with the Pythia tokenizer
        'b': randint(1, 100),
        'c': uniform(0, 1)
    }

    with tempfile.TemporaryDirectory() as tmp_dir:

        checkpoint_dir = pathlib.Path(tmp_dir)

        download_from_hub('EleutherAI/pythia-14m', checkpoint_dir=checkpoint_dir)
        searcher = OptFormerSearcher(config_space=config_space,
                                     checkpoint_dir=checkpoint_dir / 'EleutherAI' / 'pythia-14m')

        for i in range(5):
            config = searcher.suggest()
 #           assert config['a'] in config_space['a'].categories
            assert config_space['b'].lower <= config['b'] <= config_space['b'].upper
            assert config_space['c'].lower <= config['c'] <= config_space['c'].upper
            searcher.on_trial_complete(i, config, metric=np.random.rand())