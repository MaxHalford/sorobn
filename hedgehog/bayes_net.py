import collections
import functools
import itertools
import random
import typing

import numpy as np
import pandas as pd
import vose


__all__ = ['BayesNet']


@pd.api.extensions.register_series_accessor('cpt')
class CPTAccessor:
    """

    Adds utilities to a pandas.Series to help manipulate it as a conditional probability
    table (CPT).

    """

    def __init__(self, series: pd.Series):
        self.series = series
        self.sampler = None

    def sample(self):
        """Sample a row at random.

        The `sample` method of a series is very slow. Additionally, it is not designed to be used
        repetitively and requires O(n) steps every time it is called. Instead, we use a Cython
        implemention of Vose's alias method that takes O(n) time to build and O(1) time to query.

        """
        if self.sampler is None:
            self.sampler = vose.Sampler(
                weights=self.series.to_numpy(),
                seed=np.random.randint(2 ** 16)
            )
        idx = self.sampler.sample()
        return self.series.index[idx]

    @functools.lru_cache(maxsize=256)
    def __getitem__(self, idx):
        """Cached row accessor.

        Accessing a row of pandas.Series is very inefficient. This method caches the row accesses
        and therefore circumvents the issue.

        """
        return self.series[idx]


def pointwise_mul(left: pd.Series, right: pd.Series):
    """Pointwise multiplication of two series.

    Examples:

        Example taken from figure 14.10 of Artificial Intelligence: A Modern Approach.

        >>> a = pd.Series({
        ...     ('T', 'T'): .3,
        ...     ('T', 'F'): .7,
        ...     ('F', 'T'): .9,
        ...     ('F', 'F'): .1
        ... })
        >>> a.index.names = ['A', 'B']

        >>> b = pd.Series({
        ...     ('T', 'T'): .2,
        ...     ('T', 'F'): .8,
        ...     ('F', 'T'): .6,
        ...     ('F', 'F'): .4
        ... })
        >>> b.index.names = ['B', 'C']

        >>> pointwise_mul(a, b)
        B  A  C
        F  T  T    0.42
              F    0.28
           F  T    0.06
              F    0.04
        T  T  T    0.06
              F    0.24
           F  T    0.18
              F    0.72
        dtype: float64

        This method returns the Cartesion product in case two don't share any part of their index
        in common.

        >>> a = pd.Series({
        ...     ('T', 'T'): .3,
        ...     ('T', 'F'): .7,
        ...     ('F', 'T'): .9,
        ...     ('F', 'F'): .1
        ... })
        >>> a.index.names = ['A', 'B']

        >>> b = pd.Series({
        ...     ('T', 'T'): .2,
        ...     ('T', 'F'): .8,
        ...     ('F', 'T'): .6,
        ...     ('F', 'F'): .4
        ... })
        >>> b.index.names = ['C', 'D']

        >>> pointwise_mul(a, b)
        A  B  C  D
        T  T  F  F    0.12
                 T    0.18
              T  F    0.24
                 T    0.06
           F  F  F    0.28
                 T    0.42
              T  F    0.56
                 T    0.14
        F  T  F  F    0.36
                 T    0.54
              T  F    0.72
                 T    0.18
           F  F  F    0.04
                 T    0.06
              T  F    0.08
                 T    0.02
        dtype: float64

        Here is an example where both series have a one-dimensional index:

        >>> a = pd.Series({
        ...     'T': .3,
        ...     'F': .7
        ... })
        >>> a.index.names = ['A']

        >>> b = pd.Series({
        ...     'T': .2,
        ...     'F': .8
        ... })
        >>> b.index.names = ['B']

        >>> pointwise_mul(a, b)
        A  B
        T  T    0.06
           F    0.24
        F  T    0.14
           F    0.56
        dtype: float64

        Finally, here is an example when only one of the series has a MultiIndex.

        >>> a = pd.Series({
        ...     'T': .3,
        ...     'F': .7
        ... })
        >>> a.index.names = ['A']

        >>> b = pd.Series({
        ...     ('T', 'T'): .2,
        ...     ('T', 'F'): .8,
        ...     ('F', 'T'): .6,
        ...     ('F', 'F'): .4
        ... })
        >>> b.index.names = ['B', 'C']

        >>> pointwise_mul(a, b)
        A  B  C
        T  F  F    0.12
              T    0.18
           T  F    0.24
              T    0.06
        F  F  F    0.28
              T    0.42
           T  F    0.56
              T    0.14
        dtype: float64

    """

    # Return the Cartesion product if the index names have nothing in common with each other
    if not set(left.index.names) & set(right.index.names):
        cart = pd.DataFrame(np.outer(left, right), index=left.index, columns=right.index)
        return cart.stack(list(range(cart.columns.nlevels)))

    index, l_idx, r_idx, = left.index.join(right.index, how='inner', return_indexers=True)
    if l_idx is None:
        l_idx = np.arange(len(left))
    if r_idx is None:
        r_idx = np.arange(len(right))
    return pd.Series(left.iloc[l_idx].values * right.iloc[r_idx].values, index=index)


def sum_out(cdt: pd.Series, var: str) -> pd.Series:
    nodes = list(cdt.index.names)
    nodes.remove(var)
    return cdt.groupby(nodes).sum()


class BayesNet:
    """Bayesian network.

    """

    def __init__(self, *edges):

        # Convert edges into children and parent connections
        parents = collections.defaultdict(list)
        children = collections.defaultdict(list)
        for parent, child in collections.OrderedDict.fromkeys(edges):  # remove duplicates
            parents[child].append(parent)
            children[parent].append(child)
        self.parents = dict(parents)
        self.children = dict(children)

        self.nodes = sorted({*parents.keys(), *children.keys()})
        self.cpts = {}

    def prepare(self):
        """Perform house-keeping.

        It is highly recommended to call this method whenever the structure and/or the parameters
        of the Bayesian network are set manually.

        """

        for node, cpt in self.cpts.items():
            cpt.sort_index(inplace=True)
            cpt.index.rename(
                [*self.parents[node], node] if node in self.parents else node,
                inplace=True
            )
            cpt.name = (
                f'P({node} | {", ".join(self.parents[node])})'
                if node in self.parents else
                f'P({node})'
            )

    def _sample(self, init: dict = None):
        """

        This method is not public because setting fixed values doesn't produce samples that follow
        the network's distribution, which is an easy pit to fall into for users.

        """

        def sample_node_value(node, sample, visited):

            # If this node has been visisted, then that implies
            # that it's parents and ancestors have been too, which means
            # there is no point going further up the network.
            if node in visited:
                return
            visited.add(node)

            # At this point, we need to ensure that the sample contains a value for each
            # parent node
            for parent in self.parents.get(node, ()):
                sample_node_value(node=parent, sample=sample, visited=visited)

            if node not in sample:
                cpt = self.cpts[node]
                if node in self.parents:
                    condition = tuple(sample[parent] for parent in self.parents[node])
                    cpt = cpt.cpt[condition]
                sample[node] = cpt.cpt.sample()

        sample = init.copy() if init else {}
        visited = set()

        for node in self.nodes:
            if node not in self.children:  # is a leaf
                sample_node_value(node=node, sample=sample, visited=visited)

        return sample

    def sample(self, n=1):
        """Generate a new sample at random by using forward sampling.

        Although the idea is to implement forward sampling, the implementation
        actually works backwards, starting from the leaf nodes. For every node, we recursively
        check that values have been sampled for each parent node. Once a value has been chosen for
        each parent, we can pick the according distribution and sample from it.

        Parameters:
            n: Number of samples to produce. A dataframe is returned if `n > 1`. A dictionary is
                returned if `n <= 1`.

        """
        if n > 1:
            return pd.DataFrame(self._sample() for _ in range(n)).sort_index(axis='columns')
        return self._sample()

    def fit(self, X: pd.DataFrame):
        """Find the values of each conditional distribution."""

        # Compute conditional distribution for each child
        for child, parents in self.parents.items():
            self.cpts[child] = X.groupby(parents)[child].value_counts(normalize=True)

        # Compute distribution for each orphan (i.e. the roots)
        for orphan in set(self.nodes) - set(self.parents):
            self.cpts[orphan] = X[orphan].value_counts(normalize=True)

        self.prepare()

        return self

    def _rejection_sampling(self, *query, event, n_iterations):
        """Answer a query using rejection sampling.

        This is probably the easiest approximate inference method to understand. The idea is simply
        to produce a random sample and keep if it satisfies the specified event. The sample is
        rejected if any part of the event is not consistent with the sample. The downside of this
        method is that it can potentially reject many samples, and therefore requires a large `n`
        in order to produce reliable estimates.

        Example:

            >>> import hedgehog as hh
            >>> import numpy as np

            >>> np.random.seed(42)

            >>> bn = hh.load_sprinkler()

            >>> event = {'Sprinkler': True}
            >>> bn.query('Rain', event=event, algorithm='rejection', n_iterations=500)
            Rain
            False    0.691781
            True     0.308219
            Name: P(Rain), dtype: float64

        """

        # We don't know many samples we won't reject, therefore we cannot preallocate arrays
        samples = {var: [] for var in query}

        for _ in range(n_iterations):
            sample = self.sample()

            # Reject if the sample is not consistent with the specified events
            if any(sample[var] != val for var, val in event.items()):
                continue

            for var in query:
                samples[var].append(sample[var])

        # Aggregate and normalize the obtained samples
        samples = pd.DataFrame(samples)
        return samples.groupby(list(query)).size() / len(samples)

    def _llh_weighting(self, *query, event, n_iterations):
        """Likelihood weighting.

        Likelihood weighting is a particular instance of importance sampling. The idea is to
        produce random samples, and weight each sample according to it's likelihood.

        Example:

            >>> import hedgehog as hh
            >>> import numpy as np

            >>> np.random.seed(42)

            >>> bn = hh.load_sprinkler()

            >>> event = {'Sprinkler': True}
            >>> bn.query('Rain', event=event, algorithm='likelihood', n_iterations=500)
            Rain
            False    0.714862
            True     0.285138
            Name: P(Rain), dtype: float64

        """

        samples = {var: [None] * n_iterations for var in query}
        weights = [None] * n_iterations

        for i in range(n_iterations):

            # Sample by using the events as fixed values
            sample = self._sample(init=event)

            # Compute the likelihood of this sample
            weight = 1.
            for var, val in event.items():
                cpt = self.cpts[var]
                if var in self.parents:
                    condition = tuple(sample[p] for p in self.parents[var])
                    cpt = cpt.cpt[condition]
                weight *= cpt.get(val, 0)

                if weight == 0:
                    break

            for var in query:
                samples[var][i] = sample[var]
                weights[i] = weight

        # Now we aggregate the resulting samples according to their associated likelihoods
        results = pd.DataFrame({'weight': weights, **samples})
        results = results.groupby(list(query))['weight'].mean()
        results /= results.sum()

        return results

    def _gibbs_sampling(self, *query, event, n_iterations):
        """Gibbs sampling.

        The mathematical details of why this works is quite involved, but the idea is quite simple.
        We start with a random sample where the event variables are specified. Every iteration,
        we pick a random variable that is not part of the event variables, and sample it randomly.
        The sampling is conditionned on the current state of the sample, which requires computing
        the conditional distribution of each variable with respect to it's Markov blanket. Every
        time a random value is sampled, we update the current state and record it.

        Example:

            >>> import hedgehog as hh
            >>> import numpy as np

            >>> np.random.seed(42)

            >>> bn = hh.load_sprinkler()

            >>> event = {'Sprinkler': True}
            >>> bn.query('Rain', event=event, algorithm='gibbs', n_iterations=500)
            Rain
            False    0.726
            True     0.274
            Name: P(Rain), dtype: float64

        """

        # We start by computing the conditional distributions for each node that is not part of
        # the event. Each relevant node is therefore conditioned on its Markov blanket. Refer to
        # equation 14.12 of Artificial Intelligence: A Modern Approach for more detail.
        posteriors = {}
        blankets = {}
        nonevents = sorted(set(self.nodes) - set(event))
        for node in nonevents:

            post = self.cpts[node]
            for child in self.children.get(node, ()):
                post = pointwise_mul(post, self.cpts[child])

            blanket = list(post.index.names)  # Markov blanket
            blanket.remove(node)
            post = post.groupby(blanket).apply(lambda g: g / g.sum())
            post = post.reorder_levels([*blanket, node])
            post = post.sort_index()
            posteriors[node] = post
            blankets[node] = blanket

        # Start with a random sample
        state = self._sample(init=event)

        samples = {var: [None] * n_iterations for var in query}
        cycle = itertools.cycle(nonevents)  # arbitrary order, it doesn't matter

        for i in range(n_iterations):
            # Go to the next variable
            var = next(cycle)

            # Sample from P(var | blanket(var))
            cpt = posteriors[var]
            condition = tuple(state[node] for node in blankets[var])
            if condition:
                cpt = cpt.cpt[condition]
            val = cpt.cpt.sample()
            state[var] = val

            # Record the current state
            for var in query:
                samples[var][i] = state[var]

        # Aggregate and normalize the obtained samples
        samples = pd.DataFrame(samples)
        return samples.groupby(list(query)).size() / len(samples)

    def _variable_elimination(self, *query, event):
        """Variable elimination.

        See figure 14.11 of Artificial Intelligence: A Modern Approach for more detail.

        Example:

            >>> import hedgehog as hh
            >>> import numpy as np

            >>> np.random.seed(42)

            >>> bn = hh.load_sprinkler()

            >>> event = {'Sprinkler': True}
            >>> bn.query('Rain', event=event, algorithm='exact')
            Rain
            False    0.7
            True     0.3
            Name: P(Rain), dtype: float64

        """

        # We start by determining which nodes can be discarded. We can remove any leaf node that is
        # part of query variable(s) or the event variable(s). After a leaf node has been removed,
        # there might be some more leaf nodes to be remove, etc. Differently put, we can ignore
        # every node that isn't an ancestor of the query variable(s) or the event variable(s).
        relevant = {*query, *event}
        for node in list(relevant):
            relevant |= self.ancestors(node)
        hidden = relevant - {*query, *event}

        factors = []
        for node in relevant:
            factor = self.cpts[node].copy()
            # Filter each factor according to the event
            for var, val in event.items():
                if var in factor.index.names:
                    factor = factor[factor.index.get_level_values(var) == val]

            factors.append(factor)

        # Sum-out the hidden variables from the factors in which they appear
        for node in hidden:
            prod = functools.reduce(
                pointwise_mul,
                (
                    factors.pop(i)
                    for i in reversed(range(len(factors)))
                    if node in factors[i].index.names
                )
            )
            prod = sum_out(prod, node)
            factors.append(prod)

        # Pointwise multiply the rest of the factors and normalize the result
        posterior = functools.reduce(pointwise_mul, factors)
        posterior = posterior / posterior.sum()
        posterior.index = posterior.index.droplevel(list(set(posterior.index.names) - set(query)))
        return posterior

    def ancestors(self, node):
        parents = self.parents.get(node, ())
        if parents:
            return set(parents) | set.union(*[self.ancestors(p) for p in parents])
        return set()

    def query(self, *query: typing.Tuple[str], event: dict, algorithm='exact',
              n_iterations=100) -> pd.Series:
        """Answer a probabilistic query.

        Exact inference is performed by default. However, this might be too slow depending on the
        graph structure. In that case, it is more suitable to use one of the approximate inference
        methods. Provided `n` is "large enough", approximate inference methods are usually very
        reliable.

        Parameters:
            query: The variables for which the posterior distribution is inferred.
            event: The information on which to condition the answer. This is also referred to as
                the "evidence".
            algorithm: Inference method to use.
            n_iterations: Number of iterations to perform when using an approximate inference
                method.

        """

        if algorithm == 'exact':
            answer = self._variable_elimination(*query, event=event)

        elif algorithm == 'gibbs':
            answer = self._gibbs_sampling(*query, event=event, n_iterations=n_iterations)

        elif algorithm == 'likelihood':
            answer = self._llh_weighting(*query, event=event, n_iterations=n_iterations)

        elif algorithm == 'rejection':
            answer = self._rejection_sampling(*query, event=event, n_iterations=n_iterations)

        else:
            raise ValueError('Unknown algorithm, must be one of: exact, gibbs, likelihood, ' +
                             'rejection')

        return answer.rename(f'P({", ".join(query)})')

    def impute(self, sample: dict, **query_params) -> dict:
        """Replace missing values with the most probable possibility.

        This method returns a fresh copy and does not modify the input.

        Parameters:
            sample: The sample for which the missing values need replacing. The missing values are
                expected to be represented with `None`.
            query_params: The rest of the keyword arguments for specifying what parameters to call
                the `query` method with.

        """

        # Determine which variables are missing and which ones are not
        missing = []
        event = sample.copy()
        for k, v in sample.items():
            if v is None:
                missing.append(k)
                del event[k]

        # Compute the likelihood of each possibility
        posterior = self.query(*missing, event=event, **query_params)

        # Replace the missing values with the most likely values
        for k, v in zip(posterior.index.names, posterior.idxmax()):
            event[k] = v

        return event
