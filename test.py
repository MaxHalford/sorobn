import rev

bn = rev.BayesNet(
    ('Burglary', 'Alarm'),
    ('Earthquake', 'Alarm'),
    ('Alarm', 'John calls'),
    ('Alarm', 'Mary calls')
)

import pandas as pd

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

sample = {
    'Alarm': True,
    'Burglary': True,
    'Earthquake': False,
    'John calls': None,  # missing
    'Mary calls': None   # missing
}

sample = bn.impute(sample)

from pprint import pprint
pprint(sample)
