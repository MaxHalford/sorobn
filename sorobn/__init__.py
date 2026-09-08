import os

from . import examples
from . import structure
from .bayes_net import BayesNet
from .discretization import Discretizer
from .predicates import (
    Between, Eq, Ge, Glob, Gt, In, IsNotNull, IsNull, Le, Like, Lt, Ne,
    Predicate, Regex,
)


__all__ = [
    'BayesNet',
    'Discretizer',
    'Predicate', 'Eq', 'Ne', 'Lt', 'Le', 'Gt', 'Ge', 'In', 'Between',
    'Like', 'Regex', 'Glob', 'IsNull', 'IsNotNull',
    'examples',
    'structure'
]


def cli_hook():
    here = os.path.dirname(os.path.realpath(__file__))
    os.system(f'streamlit run {here}/gui.py')
