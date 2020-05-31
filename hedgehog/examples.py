import pandas as pd

import hedgehog as hh


__all__ = [
    'load_alarm',
    'load_asia',
    'load_grades',
    'load_sprinkler'
]


def load_alarm() -> hh.BayesNet:
    """Load Judea Pearl's famous example.

    At the time of writing his seminal paper on Bayesian networks, Judea Pearl lived in California,
    where earthquakes are quite common.

    Example:

        >>> import hedgehog as hh

        >>> bn = hh.load_alarm()

        >>> bn.query('John calls', 'Mary calls', event={'Burglary': True, 'Earthquake': False})
        John calls  Mary calls
        False       False         0.08463
                    True          0.06637
        True        False         0.25677
                    True          0.59223
        Name: P(John calls, Mary calls), dtype: float64

    """

    bn = hh.BayesNet(
        ('Burglary', 'Alarm'),
        ('Earthquake', 'Alarm'),
        ('Alarm', 'John calls'),
        ('Alarm', 'Mary calls')
    )

    # P(Burglary)
    bn.P['Burglary'] = pd.Series({False: .999, True: .001})

    # P(Earthquake)
    bn.P['Earthquake'] = pd.Series({False: .998, True: .002})

    # P(Alarm | Burglary, Earthquake)
    bn.P['Alarm'] = pd.Series({
        (True, True, True): .95,
        (True, True, False): .05,

        (True, False, True): .94,
        (True, False, False): .06,

        (False, True, True): .29,
        (False, True, False): .71,

        (False, False, True): .001,
        (False, False, False): .999
    })

    # P(John calls | Alarm)
    bn.P['John calls'] = pd.Series({
        (True, True): .9,
        (True, False): .1,
        (False, True): .05,
        (False, False): .95
    })

    # P(Mary calls | Alarm)
    bn.P['Mary calls'] = pd.Series({
        (True, True): .7,
        (True, False): .3,
        (False, True): .01,
        (False, False): .99
    })

    bn.prepare()

    return bn


def load_asia() -> hh.BayesNet:
    """Load the Asia network.

    Example:

        >>> import hedgehog as hh

        >>> bn = hh.load_asia()

        >>> bn.query('Lung cancer', event={'Visit to Asia': True, 'Smoker': False})
        Lung cancer
        False    0.99
        True     0.01
        Name: P(Lung cancer), dtype: float64

    """

    bn = hh.BayesNet(
        ('Visit to Asia', 'Tuberculosis'),
        ('Smoker', ('Lung cancer', 'Bronchitis')),
        (('Tuberculosis', 'Lung cancer'), 'TB or cancer'),
        ('TB or cancer', ('Positive X-ray', 'Dispnea')),
        ('Bronchitis', 'Dispnea')
    )

    # P(Visit to Asia)
    bn.P['Visit to Asia'] = pd.Series({True: .01, False: .99})

    # P(Tuberculosis | Visit to Asia)
    bn.P['Tuberculosis'] = pd.Series({
        (True, True): .05,
        (True, False): .95,
        (False, True): .01,
        (False, False): .99
    })

    # P(Smoker)
    bn.P['Smoker'] = pd.Series({True: .5, False: .5})

    # P(Lung cancer | Smoker)
    bn.P['Lung cancer'] = pd.Series({
        (True, True): .1,
        (True, False): .9,
        (False, True): .01,
        (False, False): .99
    })

    # P(Bronchitis | Smoker)
    bn.P['Bronchitis'] = pd.Series({
        (True, True): .6,
        (True, False): .4,
        (False, True): .3,
        (False, False): .7
    })

    # P(TB or cancer | Tuberculosis, Lung cancer)
    bn.P['TB or cancer'] = pd.Series({
        (True, True, True): 1,
        (True, True, False): 0,

        (True, False, True): 1,
        (True, False, False): 0,

        (False, True, True): 1,
        (False, True, False): 0,

        (False, False, True): 0,
        (False, False, False): 1
    })

    # P(Positive X-ray | TB or cancer)
    bn.P['Positive X-ray'] = pd.Series({
        (True, True): .98,
        (True, False): .02,
        (False, True): .05,
        (False, False): .95
    })

    # P(Dispnea | TB or cancer, Bronchitis)
    bn.P['Dispnea'] = pd.Series({
        (True, True, True): .9,
        (True, True, False): .1,

        (True, False, True): .7,
        (True, False, False): .3,

        (False, True, True): .8,
        (False, True, False): .2,

        (False, False, True): .1,
        (False, False, False): .9
    })

    bn.prepare()

    return bn


def load_sprinkler() -> hh.BayesNet:
    """Load the water sprinkler network.

    This example is taken from figure 14.12(a) of Artificial Intelligence: A Modern Approach.

    Example:

        >>> import hedgehog as hh

        >>> bn = hh.load_sprinkler()

        >>> bn.query('Rain', event={'Sprinkler': True})
        Rain
        False    0.7
        True     0.3
        Name: P(Rain), dtype: float64

    """

    bn = hh.BayesNet(
        ('Cloudy', 'Sprinkler'),
        ('Cloudy', 'Rain'),
        ('Sprinkler', 'Wet grass'),
        ('Rain', 'Wet grass')
    )

    # P(Cloudy)
    bn.P['Cloudy'] = pd.Series({False: .5, True: .5})

    # P(Sprinkler | Cloudy)
    bn.P['Sprinkler'] = pd.Series({
        (True, True): .1,
        (True, False): .9,
        (False, True): .5,
        (False, False): .5
    })

    # P(Rain | Cloudy)
    bn.P['Rain'] = pd.Series({
        (True, True): .8,
        (True, False): .2,
        (False, True): .2,
        (False, False): .8
    })

    # P(Wet grass | Sprinkler, Rain)
    bn.P['Wet grass'] = pd.Series({
        (True, True, True): .99,
        (True, True, False): .01,

        (True, False, True): .9,
        (True, False, False): .1,

        (False, True, True): .9,
        (False, True, False): .1,

        (False, False, True): 0,
        (False, False, False): 1
    })

    bn.prepare()

    return bn


def load_grades():
    """Load the student grades network.

    Example:

        >>> import hedgehog as hh

        >>> bn = hh.load_grades()

        >>> bn.nodes
        ['Difficulty', 'Grade', 'Intelligence', 'Letter', 'SAT']

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

    bn = hh.BayesNet(
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
