import os

from .bayes_net import BayesNet
from . import examples
from . import structure


__all__ = [
    'examples',
    'BayesNet',
    'structure'
]


def cli_hook():
    here = os.path.dirname(os.path.realpath(__file__))
    os.system(f'streamlit run {here}/gui.py')
