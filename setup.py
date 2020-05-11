import os
from setuptools import setup
import numpy as np


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='rev',
    version='0.0.1',
    author='Max Halford',
    license='MIT',
    author_email='maxhalford25@gmail.com',
    description='Bayesian networks in Python',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/MaxHalford/rev',
    packages=['rev'],
    install_requires=[
        'pandas',
        'git+https://github.com/MaxHalford/vose'
    ],
    extras_require={'dev': ['pytest']},
    python_requires='>=3.6'
)

