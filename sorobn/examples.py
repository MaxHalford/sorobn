import pandas as pd

from .bayes_net import BayesNet

__all__ = ["alarm", "asia", "grades", "sprinkler"]


def alarm(**kwargs) -> BayesNet:
    """Load Judea Pearl's famous example.

    At the time of writing his seminal paper on Bayesian networks, Judea Pearl lived in California,
    where earthquakes are quite common.

    Examples
    --------

    >>> import sorobn

    >>> bn = sorobn.examples.alarm()

    >>> bn.query('John calls', 'Mary calls', event={'Burglary': True, 'Earthquake': False})
    John calls  Mary calls
    False       False         0.08463
                True          0.06637
    True        False         0.25677
                True          0.59223
    Name: P(John calls, Mary calls), dtype: float64

    """

    bn = BayesNet(
        ("Burglary", "Alarm"),
        ("Earthquake", "Alarm"),
        ("Alarm", "John calls"),
        ("Alarm", "Mary calls"),
        **kwargs,
    )

    T = True
    F = False

    # P(Burglary)
    bn.P["Burglary"] = pd.Series({F: 0.999, T: 0.001})

    # P(Earthquake)
    bn.P["Earthquake"] = pd.Series({F: 0.998, T: 0.002})

    # P(Alarm | Burglary, Earthquake)
    bn.P["Alarm"] = pd.DataFrame(
        {
            "Burglary": [T, T, T, T, F, F, F, F],
            "Earthquake": [T, T, F, F, T, T, F, F],
            "Alarm": [T, F, T, F, T, F, T, F],
            "p": [0.95, 0.05, 0.94, 0.06, 0.29, 0.71, 0.001, 0.999],
        }
    )

    # P(John calls | Alarm)
    bn.P["John calls"] = pd.DataFrame(
        {
            "Alarm": [T, T, F, F],
            "John calls": [T, F, T, F],
            "p": [0.9, 0.1, 0.05, 0.95],
        }
    )

    # P(Mary calls | Alarm)
    bn.P["Mary calls"] = pd.DataFrame(
        {
            "Alarm": [T, T, F, F],
            "Mary calls": [T, F, T, F],
            "p": [0.7, 0.3, 0.01, 0.99],
        }
    )

    bn.prepare()

    return bn


def asia(**kwargs) -> BayesNet:
    """Load the Asia network.

    Examples
    --------

    >>> import sorobn

    >>> bn = sorobn.examples.asia()

    >>> bn.query('Lung cancer', event={'Visit to Asia': True, 'Smoker': False})
    Lung cancer
    False    0.99
    True     0.01
    Name: P(Lung cancer), dtype: float64

    """

    bn = BayesNet(
        ("Visit to Asia", "Tuberculosis"),
        ("Smoker", ["Lung cancer", "Bronchitis"]),
        (["Tuberculosis", "Lung cancer"], "TB or cancer"),
        ("TB or cancer", ["Positive X-ray", "Dispnea"]),
        ("Bronchitis", "Dispnea"),
        **kwargs,
    )

    T = True
    F = False

    # P(Visit to Asia)
    bn.P["Visit to Asia"] = pd.Series({T: 0.01, F: 0.99})

    # P(Tuberculosis | Visit to Asia)
    bn.P["Tuberculosis"] = pd.DataFrame(
        {
            "Visit to Asia": [T, T, F, F],
            "Tuberculosis": [T, F, T, F],
            "p": [0.05, 0.95, 0.01, 0.99],
        }
    )

    # P(Smoker)
    bn.P["Smoker"] = pd.Series({T: 0.5, F: 0.5})

    # P(Lung cancer | Smoker)
    bn.P["Lung cancer"] = pd.DataFrame(
        {
            "Smoker": [T, T, F, F],
            "Lung cancer": [T, F, T, F],
            "p": [0.1, 0.9, 0.01, 0.99],
        }
    )

    # P(Bronchitis | Smoker)
    bn.P["Bronchitis"] = pd.DataFrame(
        {
            "Smoker": [T, T, F, F],
            "Bronchitis": [T, F, T, F],
            "p": [0.6, 0.4, 0.3, 0.7],
        }
    )

    # P(TB or cancer | Tuberculosis, Lung cancer)
    bn.P["TB or cancer"] = pd.DataFrame(
        {
            "Lung cancer": [T, T, T, T, F, F, F, F],
            "Tuberculosis": [T, T, F, F, T, T, F, F],
            "TB or cancer": [T, F, T, F, T, F, T, F],
            "p": [1, 0, 1, 0, 1, 0, 0, 1],
        }
    )

    # P(Positive X-ray | TB or cancer)
    bn.P["Positive X-ray"] = pd.DataFrame(
        {
            "TB or cancer": [T, T, F, F],
            "Positive X-ray": [T, F, T, F],
            "p": [0.98, 0.02, 0.05, 0.95],
        }
    )

    # P(Dispnea | TB or cancer, Bronchitis)
    bn.P["Dispnea"] = pd.DataFrame(
        {
            "Bronchitis": [T, T, T, T, F, F, F, F],
            "TB or cancer": [T, T, F, F, T, T, F, F],
            "Dispnea": [T, F, T, F, T, F, T, F],
            "p": [0.9, 0.1, 0.7, 0.3, 0.8, 0.2, 0.1, 0.9],
        }
    )

    bn.prepare()

    return bn


def sprinkler(**kwargs) -> BayesNet:
    """Load the water sprinkler network.

    This example is taken from figure 14.12(a) of Artificial Intelligence: A Modern Approach.

    Examples
    --------

    >>> import sorobn

    >>> bn = sorobn.examples.sprinkler()

    >>> bn.query('Rain', event={'Sprinkler': True})
    Rain
    False    0.7
    True     0.3
    Name: P(Rain), dtype: float64

    """

    bn = BayesNet(
        ("Cloudy", "Sprinkler"),
        ("Cloudy", "Rain"),
        ("Sprinkler", "Wet grass"),
        ("Rain", "Wet grass"),
        **kwargs,
    )

    T = True
    F = False

    # P(Cloudy)
    bn.P["Cloudy"] = pd.Series({F: 0.5, T: 0.5})

    # P(Sprinkler | Cloudy)
    bn.P["Sprinkler"] = pd.DataFrame(
        {
            "Cloudy": [T, T, F, F],
            "Sprinkler": [T, F, T, F],
            "p": [0.1, 0.9, 0.5, 0.5],
        }
    )

    # P(Rain | Cloudy)
    bn.P["Rain"] = pd.DataFrame(
        {
            "Cloudy": [T, T, F, F],
            "Rain": [T, F, T, F],
            "p": [0.8, 0.2, 0.2, 0.8],
        }
    )

    # P(Wet grass | Sprinkler, Rain)
    bn.P["Wet grass"] = pd.DataFrame(
        {
            "Rain": [T, T, T, T, F, F, F, F],
            "Sprinkler": [T, T, F, F, T, T, F, F],
            "Wet grass": [T, F, T, F, T, F, T, F],
            "p": [0.99, 0.01, 0.9, 0.1, 0.9, 0.1, 0, 1],
        }
    )

    bn.prepare()

    return bn


def grades(**kwargs):
    """Load the student grades network.

    Examples
    --------

    >>> import sorobn

    >>> bn = sorobn.examples.grades()

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
        ("Difficulty", "Grade"),
        ("Intelligence", "Grade"),
        ("Intelligence", "SAT"),
        ("Grade", "Letter"),
        **kwargs,
    )

    # P(Difficulty)
    bn.P["Difficulty"] = pd.Series({"Easy": 0.6, "Hard": 0.4})

    # P(Intelligence)
    bn.P["Intelligence"] = pd.Series({"Average": 0.7, "Smart": 0.3})

    # P(Grade | Difficulty, Intelligence)
    bn.P["Grade"] = pd.DataFrame(
        {
            "Difficulty": [
                "Easy", "Easy", "Easy", "Easy", "Easy", "Easy",
                "Hard", "Hard", "Hard", "Hard", "Hard", "Hard",
            ],
            "Intelligence": [
                "Average", "Average", "Average", "Smart", "Smart", "Smart",
                "Average", "Average", "Average", "Smart", "Smart", "Smart",
            ],
            "Grade": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C"],
            "p": [0.3, 0.4, 0.3, 0.9, 0.08, 0.02, 0.05, 0.25, 0.7, 0.5, 0.3, 0.2],
        }
    )

    # P(SAT | Intelligence)
    bn.P["SAT"] = pd.DataFrame(
        {
            "Intelligence": ["Average", "Average", "Smart", "Smart"],
            "SAT": ["Failure", "Success", "Failure", "Success"],
            "p": [0.95, 0.05, 0.2, 0.8],
        }
    )

    # P(Letter | Grade)
    bn.P["Letter"] = pd.DataFrame(
        {
            "Grade": ["A", "A", "B", "B", "C", "C"],
            "Letter": ["Weak", "Strong", "Weak", "Strong", "Weak", "Strong"],
            "p": [0.1, 0.9, 0.4, 0.6, 0.99, 0.01],
        }
    )

    bn.prepare()

    return bn
