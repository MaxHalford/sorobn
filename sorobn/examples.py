import pandas as pd

from .bayes_net import BayesNet


__all__ = [
    'alarm',
    'asia',
    'grades',
    'sprinkler'
]


def alarm() -> BayesNet:
    """Load Judea Pearl's famous example.

    At the time of writing his seminal paper on Bayesian networks, Judea Pearl lived in California,
    where earthquakes are quite common.

    Examples
    --------

    >>> import sorobn as hh

    >>> bn = hh.examples.alarm()

    >>> bn.query('John calls', 'Mary calls', event={'Burglary': True, 'Earthquake': False})
    John calls  Mary calls
    False       False         0.08463
                True          0.06637
    True        False         0.25677
                True          0.59223
    Name: P(John calls, Mary calls), dtype: float64

    """

    bn = BayesNet(
        ('Burglary', 'Alarm'),
        ('Earthquake', 'Alarm'),
        ('Alarm', 'John calls'),
        ('Alarm', 'Mary calls')
    )

    T = True
    F = False

    # P(Burglary)
    bn.P['Burglary'] = pd.Series({F: .999, T: .001})

    # P(Earthquake)
    bn.P['Earthquake'] = pd.Series({F: .998, T: .002})

    # P(Alarm | Burglary, Earthquake)
    bn.P['Alarm'] = pd.Series({
        (T, T, T): .95,
        (T, T, F): .05,

        (T, F, T): .94,
        (T, F, F): .06,

        (F, T, T): .29,
        (F, T, F): .71,

        (F, F, T): .001,
        (F, F, F): .999
    })

    # P(John calls | Alarm)
    bn.P['John calls'] = pd.Series({
        (T, T): .9,
        (T, F): .1,
        (F, T): .05,
        (F, F): .95
    })

    # P(Mary calls | Alarm)
    bn.P['Mary calls'] = pd.Series({
        (T, T): .7,
        (T, F): .3,
        (F, T): .01,
        (F, F): .99
    })

    bn.prepare()

    return bn


def asia() -> BayesNet:
    """Load the Asia network.

    Examples
    --------

    >>> import sorobn as hh

    >>> bn = hh.examples.asia()

    >>> bn.query('Lung cancer', event={'Visit to Asia': True, 'Smoker': False})
    Lung cancer
    False    0.99
    True     0.01
    Name: P(Lung cancer), dtype: float64

    """

    bn = BayesNet(
        ('Visit to Asia', 'Tuberculosis'),
        ('Smoker', ['Lung cancer', 'Bronchitis']),
        (['Tuberculosis', 'Lung cancer'], 'TB or cancer'),
        ('TB or cancer', ['Positive X-ray', 'Dispnea']),
        ('Bronchitis', 'Dispnea')
    )

    T = True
    F = False

    # P(Visit to Asia)
    bn.P['Visit to Asia'] = pd.Series({T: .01, False: .99})

    # P(Tuberculosis | Visit to Asia)
    bn.P['Tuberculosis'] = pd.Series({
        (T, T): .05,
        (T, False): .95,
        (False, T): .01,
        (False, False): .99
    })

    # P(Smoker)
    bn.P['Smoker'] = pd.Series({T: .5, F: .5})

    # P(Lung cancer | Smoker)
    bn.P['Lung cancer'] = pd.Series({
        (T, T): .1,
        (T, F): .9,
        (F, T): .01,
        (F, F): .99
    })

    # P(Bronchitis | Smoker)
    bn.P['Bronchitis'] = pd.Series({
        (T, T): .6,
        (T, F): .4,
        (F, T): .3,
        (F, F): .7
    })

    # P(TB or cancer | Tuberculosis, Lung cancer)
    bn.P['TB or cancer'] = pd.Series({
        (T, T, T): 1,
        (T, T, F): 0,

        (T, F, T): 1,
        (T, F, F): 0,

        (F, T, T): 1,
        (F, T, F): 0,

        (F, F, T): 0,
        (F, F, F): 1
    })

    # P(Positive X-ray | TB or cancer)
    bn.P['Positive X-ray'] = pd.Series({
        (T, T): .98,
        (T, F): .02,
        (F, T): .05,
        (F, F): .95
    })

    # P(Dispnea | TB or cancer, Bronchitis)
    bn.P['Dispnea'] = pd.Series({
        (T, T, T): .9,
        (T, T, F): .1,

        (F, T, T): .7,
        (F, T, F): .3,

        (T, F, T): .8,
        (T, F, F): .2,

        (F, F, T): .1,
        (F, F, F): .9
    })

    bn.prepare()

    return bn


def sprinkler() -> BayesNet:
    """Load the water sprinkler network.

    This example is taken from figure 14.12(a) of Artificial Intelligence: A Modern Approach.

    Examples
    --------

    >>> import sorobn as hh

    >>> bn = hh.examples.sprinkler()

    >>> bn.query('Rain', event={'Sprinkler': True})
    Rain
    False    0.7
    True     0.3
    Name: P(Rain), dtype: float64

    """

    bn = BayesNet(
        ('Cloudy', 'Sprinkler'),
        ('Cloudy', 'Rain'),
        ('Sprinkler', 'Wet grass'),
        ('Rain', 'Wet grass')
    )

    T = True
    F = False

    # P(Cloudy)
    bn.P['Cloudy'] = pd.Series({F: .5, T: .5})

    # P(Sprinkler | Cloudy)
    bn.P['Sprinkler'] = pd.Series({
        (T, T): .1,
        (T, F): .9,
        (F, T): .5,
        (F, F): .5
    })

    # P(Rain | Cloudy)
    bn.P['Rain'] = pd.Series({
        (T, T): .8,
        (T, F): .2,
        (F, T): .2,
        (F, F): .8
    })

    # P(Wet grass | Sprinkler, Rain)
    bn.P['Wet grass'] = pd.Series({
        (T, T, T): .99,
        (T, T, F): .01,

        (T, F, T): .9,
        (T, F, F): .1,

        (F, T, T): .9,
        (F, T, F): .1,

        (F, F, T): 0,
        (F, F, F): 1
    })

    bn.prepare()

    return bn


def grades():
    """Load the student grades network.

    Examples
    --------

    >>> import sorobn as hh

    >>> bn = hh.examples.grades()

    >>> bn.nodes
    ['Difficulty', 'Intelligence', 'Grade', 'SAT', 'Letter']

    >>> bn.children
    {'Difficulty': ['Grade'], 'Intelligence': ['Grade', 'SAT'], 'Grade': ['Letter']}

    >>> bn.parents
    {'Grade': ['Difficulty', 'Intelligence'], 'SAT': ['Intelligence'], 'Letter': ['Grade']}

    >>> bn.query('Letter', 'SAT', event={'Intelligence': 'Smart'})
    Letter  SAT
    Strong  Failure    0.153544
            Success    0.614176
    Weak    Failure    0.046456
            Success    0.185824
    Name: P(Letter, SAT), dtype: float64

    """

    bn = BayesNet(
        ('Difficulty', 'Grade'),
        ('Intelligence', 'Grade'),
        ('Intelligence', 'SAT'),
        ('Grade', 'Letter')
    )

    # P(Difficulty)
    bn.P['Difficulty'] = pd.Series({'Easy': .6, 'Hard': .4})

    # P(Intelligence)
    bn.P['Intelligence'] = pd.Series({'Average': .7, 'Smart': .3})

    # P(Grade | Difficult, Intelligence)
    bn.P['Grade'] = pd.Series({
        ('Easy', 'Average', 'A'): .3,
        ('Easy', 'Average', 'B'): .4,
        ('Easy', 'Average', 'C'): .3,

        ('Easy', 'Smart', 'A'): .9,
        ('Easy', 'Smart', 'B'): .08,
        ('Easy', 'Smart', 'C'): .02,

        ('Hard', 'Average', 'A'): .05,
        ('Hard', 'Average', 'B'): .25,
        ('Hard', 'Average', 'C'): .7,

        ('Hard', 'Smart', 'A'): .5,
        ('Hard', 'Smart', 'B'): .3,
        ('Hard', 'Smart', 'C'): .2
    })

    # P(SAT | Intelligence)
    bn.P['SAT'] = pd.Series({
        ('Average', 'Failure'): .95,
        ('Average', 'Success'): .05,
        ('Smart', 'Failure'): .2,
        ('Smart', 'Success'): .8
    })

    # P(Letter | Grade)
    bn.P['Letter'] = pd.Series({
        ('A', 'Weak'): .1,
        ('A', 'Strong'): .9,
        ('B', 'Weak'): .4,
        ('B', 'Strong'): .6,
        ('C', 'Weak'): .99,
        ('C', 'Strong'): .01
    })

    bn.prepare()

    return bn
