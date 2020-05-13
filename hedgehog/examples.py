import pandas as pd

import hedgehog as hh


__all__ = ['load_alarm']


def load_alarm() -> hh.BayesNet:
    """Load Judea Pearl's famous example.

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
    bn.cpts['Burglary'] = pd.Series({False: .999, True: .001})

    # P(Earthquake)
    bn.cpts['Earthquake'] = pd.Series({False: .998, True: .002})

    # P(Alarm | Burglary, Earthquake)
    bn.cpts['Alarm'] = pd.Series({
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
    bn.cpts['John calls'] = pd.Series({
        (True, True): .9,
        (True, False): .1,
        (False, True): .05,
        (False, False): .95
    })

    # P(Mary calls | Alarm)
    bn.cpts['Mary calls'] = pd.Series({
        (True, True): .7,
        (True, False): .3,
        (False, True): .01,
        (False, False): .99
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
    bn.cpts['Cloudy'] = pd.Series({False: .5, True: .5})

    # P(Sprinkler | Cloudy)
    bn.cpts['Sprinkler'] = pd.Series({
        (True, True): .1,
        (True, False): .9,
        (False, True): .5,
        (False, False): .5
    })

    # P(Rain | Cloudy)
    bn.cpts['Rain'] = pd.Series({
        (True, True): .8,
        (True, False): .2,
        (False, True): .2,
        (False, False): .8
    })

    # P(Wet grass | Sprinkler, Rain)
    bn.cpts['Wet grass'] = pd.Series({
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
