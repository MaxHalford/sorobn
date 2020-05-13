import pandas as pd

from . import bayes_net


__all__ = ['load_alarm']


def load_alarm() -> bayes_net.BayesNet:
    """Load Judea Pearl's famous example.

    """

    bn = bayes_net.BayesNet(
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


def load_sprinkler() -> bayes_net.BayesNet:
    """Load the water sprinkler network.

    This example is taken from figure 14.12(a) of Artificial Intelligence: A Modern Approach.

    """

    bn = bayes_net.BayesNet(
        ('Cloudy', 'Sprinkler'),
        ('Cloudy', 'Rain'),
        ('Sprinkler', 'Wet grass'),
        ('Rain', 'Wet grass')
    )

    bn.cpts['Cloudy'] = pd.Series({False: .5, True: .5})

    bn.cpts['Sprinkler'] = pd.Series({
        (True, True): .1,
        (True, False): .9,
        (False, True): .5,
        (False, False): .5
    })

    bn.cpts['Rain'] = pd.Series({
        (True, True): .8,
        (True, False): .2,
        (False, True): .2,
        (False, False): .8
    })

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
