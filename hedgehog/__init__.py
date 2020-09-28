import os

from . import examples
from . import structure
from .bayes_net import BayesNet


__all__ = [
    'examples',
    'BayesNet',
    'structure'
]


def cli_hook():
    here = os.path.dirname(os.path.realpath(__file__))
    os.system(f'streamlit run {here}/gui.py')
