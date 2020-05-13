<div align="center">
    <h1>Bayesian networks in Python</h1>
</div>
</br>

This is an unambitious Python library for working with [Bayesian networks](https://www.wikiwand.com/en/Bayesian_network). For serious usage, you should probably be using a more established project, such as [pomegranate](https://pomegranate.readthedocs.io/en/latest/), [PyMC](https://docs.pymc.io/), [Stan](https://mc-stan.org/), [Edward](http://edwardlib.org/), and [Pyro](https://pyro.ai/).

The main goal of this project is to be used for educational purposes. As such, more emphasis is put on tidyness and conciseness than on performance. Nonetheless, it is reasonably efficient and should be able to satisfy most usecases.

## Table of contents

- [Table of contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
  - [Manual definition](#manual-definition)
  - [Probabilistic inference](#probabilistic-inference)
  - [Missing value imputation](#missing-value-imputation)
  - [Random sampling](#random-sampling)
  - [Parameter estimation](#parameter-estimation)
  - [Structure learning](#structure-learning)
  - [Visualization](#visualization)
- [Examples](#examples)
- [Development](#development)
- [License](#license)

## Installation

You should be able to install and use this library with any Python version above 3.5.

```sh
$ pip install git+https://github.com/MaxHalford/hedgehog --upgrade
```

## Usage

### Manual definition

As an example, let's use Judea Pearl's famous alarm network. The central construct in `hedgehog` is the `BayesNet` class. The edges of the network can be provided manually when initialising a `BayesNet`:

```python
>>> import hedgehog as hh

>>> bn = hh.BayesNet(
...     ('Burglary', 'Alarm'),
...     ('Earthquake', 'Alarm'),
...     ('Alarm', 'John calls'),
...     ('Alarm', 'Mary calls')
... )

```

In Judea Pearl's example, the [conditional probability tables](https://www.wikiwand.com/en/Conditional_probability_table) are given. Therefore, we can set them manually by accessing the `cpts` attribute:

```python
>>> import pandas as pd

# P(Burglary)
>>> bn.cpts['Burglary'] = pd.Series({False: .999, True: .001})

# P(Earthquake)
>>> bn.cpts['Earthquake'] = pd.Series({False: .998, True: .002})

# P(Alarm | Burglary, Earthquake)
>>> bn.cpts['Alarm'] = pd.Series({
...     (True, True, True): .95,
...     (True, True, False): .05,
...
...     (True, False, True): .94,
...     (True, False, False): .06,
...
...     (False, True, True): .29,
...     (False, True, False): .71,
...
...     (False, False, True): .001,
...     (False, False, False): .999
... })

# P(John calls | Alarm)
>>> bn.cpts['John calls'] = pd.Series({
...     (True, True): .9,
...     (True, False): .1,
...     (False, True): .05,
...     (False, False): .95
... })

# P(Mary calls | Alarm)
>>> bn.cpts['Mary calls'] = pd.Series({
...     (True, True): .7,
...     (True, False): .3,
...     (False, True): .01,
...     (False, False): .99
... })

```

The `prepare` method has to be called whenever the structure and/or the CPTs are manually specified. This will do some house-keeping and make sure everything is sound. Just like washing your hands, it is highly recommended but not compulsory.

```python
>>> bn.prepare()

```

### Probabilistic inference

A Bayesian network is a [generative model](https://www.wikiwand.com/en/Generative_model). Therefore, it can be used for many purposes. First of all, it can answer probabilistic queries, such as *what is the likelihood of there being a burglary if both John and Mary call?*. This can done via the `query` method, which returns the probability distribution of the possible outcomes.

```python
>>> bn.query('Burglary', event={'Mary calls': True, 'John calls': True})
Burglary
False    0.715828
True     0.284172
Name: P(Burglary), dtype: float64

```

By default, the answer is found via an exact inference method. For small networks this isn't very expensive to perform. However, for larger networks, you might want to prefer using [approximate inference](https://www.wikiwand.com/en/Approximate_inference). The latter is a class of methods that randomly sample the network and return an estimate of the answer. The quality of the estimate increases with the number of iterations that are performed. For instance, you can use [Gibbs sampling](https://www.wikiwand.com/en/Gibbs_sampling):

```python
>>> import numpy as np
>>> np.random.seed(42)

>>> bn.query(
...     'Burglary',
...     event={'Mary calls': True, 'John calls': True},
...     algorithm='gibbs',
...     n_iterations=1000
... )
Burglary
False    0.73
True     0.27
Name: P(Burglary), dtype: float64

```

The supported inference methods are:

- `exact` for [variable elimination](https://www.wikiwand.com/en/Variable_elimination).
- `gibbs` for [Gibbs sampling](https://www.wikiwand.com/en/Gibbs_sampling).
- `likekihood` for [likelihood weighting](https://artint.info/2e/html/ArtInt2e.Ch8.S6.SS4.html).
- `rejection` for [rejection sampling](https://www.wikiwand.com/en/Rejection_sampling).

### Missing value imputation

A usecase for probabilistic inference is to impute missing values. The `impute` method replaces the missing values with the most likely replacements given the present information. This is usually more accurate than simply replacing by the mean or the most common value. Additionally, such an approach can be much more efficient than [model-based iterative imputation](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer).

```python
>>> from pprint import pprint

>>> sample = {
...     'Alarm': True,
...     'Burglary': True,
...     'Earthquake': False,
...     'John calls': None,  # missing
...     'Mary calls': None   # missing
... }

>>> sample = bn.impute(sample)
>>> pprint(sample)
{'Alarm': True,
 'Burglary': True,
 'Earthquake': False,
 'John calls': True,
 'Mary calls': True}

```

Note that the `impute` method can be seen as the equivalent of [`pomegranate`'s `predict` method](https://pomegranate.readthedocs.io/en/latest/BayesianNetwork.html#prediction).

### Random sampling

You can use a Bayesian network to generate random samples. The samples will follow the distribution induced by the network's structure and it's conditional probability tables.

```python
>>> pprint(bn.sample())
{'Alarm': False,
 'Burglary': False,
 'Earthquake': False,
 'John calls': False,
 'Mary calls': False}

>>> bn.sample(5)
    Alarm  Burglary  Earthquake  John calls  Mary calls
0   False     False       False       False       False
1   False     False       False       False       False
2   False     False       False       False       False
3   False     False       False        True       False
4   False     False       False       False       False

```

### Parameter estimation

You can determine the values of the CPTs from a dataset. This is a straightforward procedure, as it only requires perfoming a [`groupby`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html) followed by a [`value_counts`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html) for each CPT.

```python
>>> samples = bn.sample(1000)
>>> bn = bn.fit(samples)

```

Note that in this case you do not have to call the `prepare` method, as this is done for you implicitely.

### Structure learning

On the way.

### Visualization

On the way.

## Examples

Several premade networks are available to fool around with:

- `load_alarm`
- `load_sprinkler`

Here is some example usage:

```python
>>> bn = hh.load_sprinkler()
>>> bn.nodes
['Cloudy', 'Rain', 'Sprinkler', 'Wet grass']

>>> pprint(bn.parents)
{'Rain': ['Cloudy'],
 'Sprinkler': ['Cloudy'],
 'Wet grass': ['Sprinkler', 'Rain']}

>>> pprint(bn.children)
{'Cloudy': ['Sprinkler', 'Rain'],
 'Rain': ['Wet grass'],
 'Sprinkler': ['Wet grass']}

```

## Development

```sh
> git clone https://github.com/MaxHalford/hedgehog
> cd hedgehog
> pip install -e ".[dev]"
> python setup.py develop
> pytest
```

## License

This project is free and open-source software licensed under the [MIT license](https://github.com/MaxHalford/hedgehog/blob/master/LICENSE).
