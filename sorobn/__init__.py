import os

from . import examples
from . import structure
from .bayes_net import BayesNet
from .compaction import Compactor, OTHER
from .discretization import Discretizer
from .predicates import (
    Between, Eq, Ge, Glob, Gt, In, IsNotNull, IsNull, Le, Like, Lt, Ne,
    MISSING, Predicate, Regex,
)


__all__ = [
    'BayesNet',
    'Compactor',
    'OTHER',
    'Discretizer',
    'Predicate', 'Eq', 'Ne', 'Lt', 'Le', 'Gt', 'Ge', 'In', 'Between',
    'Like', 'Regex', 'Glob', 'IsNull', 'IsNotNull', 'MISSING',
    'examples',
    'structure'
]


def cli_hook():
    here = os.path.dirname(os.path.realpath(__file__))
    os.system(f'streamlit run {here}/gui.py')
