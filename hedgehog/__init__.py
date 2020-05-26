import os

from .bayes_net import BayesNet
from .examples import load_alarm
from .examples import load_asia
from .examples import load_grades
from .examples import load_sprinkler


__all__ = [
    'BayesNet',
    'load_alarm',
    'load_asia',
    'load_grades',
    'load_sprinkler'
]


def cli_hook():
    here = os.path.dirname(os.path.realpath(__file__))
    os.system(f'streamlit run {here}/gui.py')
