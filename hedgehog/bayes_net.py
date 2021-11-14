import collections
import functools
import itertools
import graphlib
import queue
import typing

import numpy as np
import pandas as pd
import vose


__all__ = ["BayesNet"]


@pd.api.extensions.register_series_accessor("cdt")
class CDTAccessor:
    """

    Adds utilities to a pandas.Series to help manipulate it as a conditional probability
    table (CDT).

    """

    def __init__(self, series: pd.Series):
        self.series = series
        self.sampler = None

    def sample(self):
        """Sample a row at random.

        The `sample` method of a Series is very slow. Additionally, it is not designed to be used
        repetitively and requires O(n) steps every time it is called. Instead, we use a Cython
        implemention of Vose's alias method that takes O(n) time to build and O(1) time to query.

        """
        if self.sampler is None:
            self.sampler = vose.Sampler(
                weights=self.series.to_numpy(dtype=float),
                seed=np.random.randint(2 ** 16),
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

    def sum_out(self, *variables):
        """Sums out a variable from a multi-indexed series.

        Examples
        --------

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

        >>> ab = pointwise_mul_two(a, b)
        >>> ab
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

        >>> ab.cdt.sum_out('B')
        A  C
        F  F    0.76
           T    0.24
        T  F    0.52
           T    0.48
        dtype: float64

        """
        nodes = list(self.series.index.names)
        for var in variables:
            nodes.remove(var)
        return self.series.groupby(nodes).sum()


def pointwise_mul_two(left: pd.Series, right: pd.Series):
    """Pointwise multiplication of two series.

    Examples
    --------

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

    >>> pointwise_mul_two(a, b)
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

    >>> pointwise_mul_two(a, b)
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

    >>> pointwise_mul_two(a, b)
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

    >>> pointwise_mul_two(a, b)
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
        cart = pd.DataFrame(
            np.outer(left, right), index=left.index, columns=right.index
        )
        return cart.stack(list(range(cart.columns.nlevels)))

    (
        index,
        l_idx,
        r_idx,
    ) = left.index.join(right.index, how="inner", return_indexers=True)
    if l_idx is None:
        l_idx = np.arange(len(left))
    if r_idx is None:
        r_idx = np.arange(len(right))

    return pd.Series(left.iloc[l_idx].values * right.iloc[r_idx].values, index=index)


def pointwise_mul(cdts, keep_zeros=False):
    if not keep_zeros:
        cdts = (cdt[cdt > 0] for cdt in cdts)
    return functools.reduce(pointwise_mul_two, cdts)


class BayesNet:
    """Bayesian network.

    Parameters
    ----------
    structure
        Each tuple denotes a (parent, child) connection. A CycleError is raised if the
        structure is not acyclic.

    prior_count
        If provided, artificial samples will be used to compute each conditional
        probability distribution, in addition to provided samples. As a consequence, each
        combination of parent(s)/child(ren) values will appear prior_count times. The
        justification for doing so is related to Laplace's rule of succession and to Bayesian
        statistics in general.

    Attributes
    ----------
    nodes (list)
        The node names sorted in topological order. Iterating over this is equivalent to performing
        a breadth-first search.

    """

    def __init__(self, *structure, prior_count: int = None):

        self.prior_count = prior_count

        def coerce_list(obj):
            if isinstance(obj, list):
                return obj
            return [obj]

        # The structure is made up of nodes (scalars) and edges (tuples)
        edges = (e for e in structure if isinstance(e, tuple))
        nodes = set(e for e in structure if not isinstance(e, tuple))

        # Convert edges into children and parent connections
        self.parents = collections.defaultdict(set)
        self.children = collections.defaultdict(set)

        for parents, children in edges:
            for parent, child in itertools.product(
                coerce_list(parents), coerce_list(children)
            ):
                self.parents[child].add(parent)
                self.children[parent].add(child)

        # collections.defaultdict(set) -> dict(list)
        self.parents = {node: sorted(parents) for node, parents in self.parents.items()}
        self.children = {
            node: sorted(children) for node, children in self.children.items()
        }

        # The nodes are sorted in topological order. Nodes of the same level are sorted in
        # lexicographic order.
        ts = graphlib.TopologicalSorter()
        for node in sorted({*self.parents.keys(), *self.children.keys(), *nodes}):
            ts.add(node, *self.parents.get(node, []))
        self.nodes = list(ts.static_order())

        self.P = {}
        self._P_sizes = {}

    def prepare(self):
        """Perform house-keeping.

        It is highly recommended to call this method whenever the structure and/or the parameters
        of the Bayesian network are set manually.

        """

        for node, P in self.P.items():
            if node not in self.parents:
                P.index.name = node
            elif set(P.index.names) == set([*self.parents[node], node]):
                P = P.reorder_levels([*self.parents[node], node])
            else:
                P.index.names = [*self.parents[node], node]
            P.sort_index(inplace=True)
            P.name = (
                f'P({node} | {", ".join(map(str, self.parents[node]))})'
                if node in self.parents
                else f"P({node})"
            )

    def ancestors(self, node):
        """Return a node's ancestors."""
        parents = self.parents.get(node, ())
        if parents:
            return set(parents) | set.union(*[self.ancestors(p) for p in parents])
        return set()

    @property
    def roots(self):
        """Return the network's roots.

        A root is a node that has no parent.

        """
        return [node for node in self.nodes if node not in self.parents]

    def full_joint_dist(self, event: dict = None, keep_zeros=False) -> pd.DataFrame:
        """Return the full joint distribution.

        The full joint distribution is obtained by pointwise multiplying all the conditional
        probability tables with each other and normalizing the result.

        Parameters
        ----------
        event
            An optional set of values certain variables must take on.
        keep_zeros
            Determines whether or not to include value combinations that don't occur together.

        Examples
        --------

        >>> import hedgehog as hh

        >>> bn = hh.examples.sprinkler()

        >>> bn.full_joint_dist()
        Cloudy  Rain   Sprinkler  Wet grass
        False   False  False      False        0.2000
                       True       False        0.0200
                                  True         0.1800
                True   False      False        0.0050
                                  True         0.0450
                       True       False        0.0005
                                  True         0.0495
        True    False  False      False        0.0900
                       True       False        0.0010
                                  True         0.0090
                True   False      False        0.0360
                                  True         0.3240
                       True       False        0.0004
                                  True         0.0396
        Name: P(Cloudy, Rain, Sprinkler, Wet grass), dtype: float64

        The cases that don't occur are excluded by default. They can be included by setting
        the `keep_zeros` parameter to `True`.

        >>> bn.full_joint_dist(keep_zeros=True)
        Cloudy  Rain   Sprinkler  Wet grass
        False   False  False      False        0.2000
                                  True         0.0000
                       True       False        0.0200
                                  True         0.1800
                True   False      False        0.0050
                                  True         0.0450
                       True       False        0.0005
                                  True         0.0495
        True    False  False      False        0.0900
                                  True         0.0000
                       True       False        0.0010
                                  True         0.0090
                True   False      False        0.0360
                                  True         0.3240
                       True       False        0.0004
                                  True         0.0396
        Name: P(Cloudy, Rain, Sprinkler, Wet grass), dtype: float64

        """

        fjd = pointwise_mul(self.P.values(), keep_zeros=keep_zeros)
        fjd = fjd.reorder_levels(sorted(fjd.index.names))
        fjd = fjd.sort_index()
        fjd.name = f'P({", ".join(fjd.index.names)})'
        return fjd / fjd.sum()

    def partial_fit(self, X: pd.DataFrame):
        """Update the parameters of each conditional distribution."""

        # Compute the conditional distribution for each node that has parents
        for child, parents in self.parents.items():

            # If a P already exists, then we update it incrementally...
            if child in self.P:
                old_counts = self.P[child] * self._P_sizes[child]
                new_counts = X.groupby(parents + [child]).size()
                counts = old_counts.add(new_counts, fill_value=0)

            # ... else we compute it from scratch
            else:
                counts = X.groupby(parents + [child]).size()
                if self.prior_count:
                    combos = itertools.product(
                        *[X[var].unique() for var in parents + [child]]
                    )
                    prior = pd.Series(
                        1, pd.MultiIndex.from_tuples(combos, names=parents + [child])
                    )
                    counts = counts.add(prior, fill_value=0)

            # Normalize
            self._P_sizes[child] = counts.groupby(parents).sum()
            self.P[child] = counts / self._P_sizes[child]

        # Compute the distribution for each root
        for root in self.roots:

            # Incremental update
            if root in self.P:
                old_counts = self.P[root] * self._P_sizes[root]
                new_counts = X[root].value_counts()
                counts = old_counts.add(new_counts, fill_value=0)
                self._P_sizes[root] += len(X)
                self.P[root] = counts / self._P_sizes[root]

            # From scratch
            else:
                self._P_sizes[root] = len(X)
                self.P[root] = X[root].value_counts(normalize=True)

        self.prepare()

        return self

    def fit(self, X: pd.DataFrame):
        """Find the values of each conditional distribution."""
        self.P = {}
        self._P_sizes = {}
        return self.partial_fit(X)

    def _forward_sample(
        self, init: dict = None
    ) -> typing.Iterator[typing.Tuple[dict, float]]:
        """Perform forward sampling.

        This is also known as "ancestral sampling" or "prior sampling".

        """

        init = init or {}

        while True:

            sample = {}
            likelihood = 1.0

            for node in self.nodes:

                # Access P(node | parents(node))
                P = self.P[node]
                if node in self.parents:
                    condition = tuple(sample[parent] for parent in self.parents[node])
                    P = P.cdt[condition]

                if node in init:
                    node_value = init[node]
                else:
                    node_value = P.cdt.sample()

                sample[node] = node_value
                likelihood *= P.get(node_value, 0)

            yield sample, likelihood

    def _backward_sample(
        self, init: dict = None
    ) -> typing.Iterator[typing.Tuple[dict, float]]:
        """Perform backward sampling."""
        return self._forward_sample(init)

    def sample(self, n=1, init: dict = None, method="forward"):
        """Generate a new sample at random by using forward sampling.

        Parameters
        ----------
        n
            Number of samples to produce. A DataFrame is returned if `n > 1`. A dictionary is
            returned if not.
        init
            Allows forcing certain variables to take on given values.
        method
            The sampling method to use. Possible choices are: forward, backward.

        """

        if method == "forward":
            sampler = (sample for sample, _ in self._forward_sample(init))

        elif method == "backward":
            sampler = (sample for sample, _ in self._backward_sample(init))

        else:
            raise ValueError("Unknown method, must be one of: forward, backward")

        if n > 1:
            return pd.DataFrame(next(sampler) for _ in range(n)).sort_index(
                axis="columns"
            )
        return next(sampler)

    def _rejection_sampling(self, *query, event, n_iterations):
        """Answer a query using rejection sampling.

        This is probably the easiest approximate inference method to understand. The idea is simply
        to produce a random sample and keep it if it satisfies the specified event. The sample is
        rejected if any part of the event is not consistent with the sample. The downside of this
        method is that it can potentially reject many samples, and therefore requires a large `n`
        in order to produce reliable estimates.

        Examples
        --------

        >>> import hedgehog as hh
        >>> import numpy as np

        >>> np.random.seed(42)

        >>> bn = hh.examples.sprinkler()

        >>> event = {'Sprinkler': True}
        >>> bn.query('Rain', event=event, algorithm='rejection', n_iterations=100)
        Rain
        False    0.678571
        True     0.321429
        Name: P(Rain), dtype: float64

        """

        # We don't know many samples we won't reject, therefore we cannot preallocate arrays
        samples = {var: [] for var in query}
        sampler = (sample for sample, _ in self._forward_sample())

        for _ in range(n_iterations):
            sample = next(sampler)

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
        produce random samples, and weight each sample according to its likelihood.

        Examples
        --------

        >>> import hedgehog as hh
        >>> import numpy as np

        >>> np.random.seed(42)

        >>> bn = hh.examples.sprinkler()

        >>> event = {'Sprinkler': True}
        >>> bn.query('Rain', event=event, algorithm='likelihood', n_iterations=500)
        Rain
        False    0.765995
        True     0.234005
        Name: P(Rain), dtype: float64

        """

        samples = {var: [None] * n_iterations for var in query}
        likelihoods = [None] * n_iterations

        sampler = self._forward_sample(init=event)

        for i in range(n_iterations):

            # Sample by using the events as fixed values
            sample, likelihood = next(sampler)

            # Compute the likelihood of this sample
            for var in query:
                samples[var][i] = sample[var]
            likelihoods[i] = likelihood

        # Now we aggregate the resulting samples according to their associated likelihoods
        results = pd.DataFrame({"likelihood": likelihoods, **samples})
        results = results.groupby(list(query))["likelihood"].mean()
        results /= results.sum()

        return results

    def _gibbs_sampling(self, *query, event, n_iterations):
        """Gibbs sampling.

        The mathematical details of why this works are quite involved, but the idea is quite
        simple. We start with a random sample where the event variables are specified. Every
        iteration, we pick a random variable that is not part of the event variables, and sample it
        randomly. The sampling is conditionned on the current state of the sample, which requires
        computing the conditional distribution of each variable with respect to it's Markov
        blanket. Every time a random value is sampled, we update the current state and record it.

        Examples
        --------

        >>> import hedgehog as hh
        >>> import numpy as np

        >>> np.random.seed(42)

        >>> bn = hh.examples.sprinkler()

        >>> event = {'Sprinkler': True}
        >>> bn.query('Rain', event=event, algorithm='gibbs', n_iterations=500)
        Rain
        False    0.726
        True     0.274
        Name: P(Rain), dtype: float64

        """

        # We start by computing the conditional distributions for each node that is not part of
        # the event. Each relevant node is therefore conditioned on its Markov boundary. Refer to
        # equation 14.12 of Artificial Intelligence: A Modern Approach for more detail.
        posteriors = {}
        boundaries = {}
        nonevents = sorted(set(self.nodes) - set(event))

        for node in nonevents:

            post = pointwise_mul(
                self.P[node] for node in [node, *self.children.get(node, [])]
            )

            if boundary := self.markov_boundary(node):
                post = post.groupby(boundary).apply(lambda g: g / g.sum())
                post = post.reorder_levels([*boundary, node])

            post = post.sort_index()
            posteriors[node] = post
            boundaries[node] = boundary

        # Start with a random sample
        state = self.sample(init=event)

        samples = {var: [None] * n_iterations for var in query}
        cycle = itertools.cycle(nonevents)  # arbitrary order, it doesn't matter

        for i in range(n_iterations):

            # Go to the next variable
            var = next(cycle)

            # Sample from P(var | boundary(var))
            P = posteriors[var]
            condition = tuple(state[node] for node in boundaries[var])
            if condition:
                P = P.cdt[condition]
            state[var] = P.cdt.sample()

            # Record the current state
            for var in query:
                samples[var][i] = state[var]

        # Aggregate and normalize the obtained samples
        samples = pd.DataFrame(samples)
        return samples.groupby(list(query)).size() / len(samples)

    def _variable_elimination(self, *query, event):
        """Variable elimination.

        See figure 14.11 of Artificial Intelligence: A Modern Approach for more detail.

        Examples
        --------

        >>> import hedgehog as hh

        >>> bn = hh.examples.sprinkler()

        >>> bn.query('Rain', event={'Sprinkler': True}, algorithm='exact')
        Rain
        False    0.7
        True     0.3
        Name: P(Rain), dtype: float64

        """

        # We start by determining which nodes can be discarded. We can remove any leaf node that is
        # part of query variable(s) or the event variable(s). After a leaf node has been removed,
        # there might be some more leaf nodes to be remove, etc. Said otherwise, we can ignore each
        # node that isn't an ancestor of the query variable(s) or the event variable(s).
        relevant = {*query, *event}
        for node in list(relevant):
            relevant |= self.ancestors(node)
        hidden = relevant - {*query, *event}

        factors = []
        for node in relevant:
            factor = self.P[node].copy()
            # Filter each factor according to the event
            for var, val in event.items():
                if var in factor.index.names:
                    factor = factor[factor.index.get_level_values(var) == val]

            factors.append(factor)

        # Sum-out the hidden variables from the factors in which they appear
        for node in hidden:
            prod = pointwise_mul(
                factors.pop(i)
                for i in reversed(range(len(factors)))
                if node in factors[i].index.names
            )
            prod = prod.cdt.sum_out(node)
            factors.append(prod)

        # Pointwise multiply the rest of the factors and normalize the result
        posterior = pointwise_mul(factors)
        posterior = posterior / posterior.sum()
        posterior.index = posterior.index.droplevel(
            list(set(posterior.index.names) - set(query))
        )
        return posterior

    def query(
        self,
        *query: typing.Tuple[str],
        event: dict,
        algorithm="exact",
        n_iterations=100,
    ) -> pd.Series:
        """Answer a probabilistic query.

        Exact inference is performed by default. However, this might be too slow depending on the
        graph structure. In that case, it is more suitable to use one of the approximate inference
        methods. Provided `n` is "large enough", approximate inference methods are usually very
        reliable.

        Parameters
        ----------

        query
            The variables for which the posterior distribution is inferred.
        event
            The information on which to condition the answer. This can also called the "evidence".
        algorithm
            Inference method to use. Possible choices are: exact, gibbs, likelihood, rejection.
        n_iterations
            Number of iterations to perform when using an approximate inference method.

        Examples
        --------

        >>> import hedgehog as hh

        >>> bn = hh.examples.asia()

        >>> event = {'Visit to Asia': True, 'Smoker': True}
        >>> bn.query('Lung cancer', 'Tuberculosis', event=event)
        Lung cancer  Tuberculosis
        False        False           0.855
                     True            0.045
        True         False           0.095
                     True            0.005
        Name: P(Lung cancer, Tuberculosis), dtype: float64

        """

        if not query:
            raise ValueError("At least one query variable has to be specified")

        for q in query:
            if q in event:
                raise ValueError("A query variable cannot be part of the event")

        if algorithm == "exact":
            answer = self._variable_elimination(*query, event=event)

        elif algorithm == "gibbs":
            answer = self._gibbs_sampling(
                *query, event=event, n_iterations=n_iterations
            )

        elif algorithm == "likelihood":
            answer = self._llh_weighting(*query, event=event, n_iterations=n_iterations)

        elif algorithm == "rejection":
            answer = self._rejection_sampling(
                *query, event=event, n_iterations=n_iterations
            )

        else:
            raise ValueError(
                "Unknown algorithm, must be one of: exact, gibbs, likelihood, "
                + "rejection"
            )

        answer = answer.rename(f'P({", ".join(query)})')

        # We sort the index levels if there are multiple query variables
        if isinstance(answer.index, pd.MultiIndex):
            answer = answer.reorder_levels(sorted(answer.index.names))

        return answer.sort_index()

    def impute(self, sample: dict, **query_params) -> dict:
        """Replace missing values with the most probable possibility.

        This method returns a fresh copy and does not modify the input.

        Parameters
        ----------
        sample
            The sample for which the missing values need replacing. The missing values are expected
            to be represented with `None`.
        query_params
            The rest of the keyword arguments for specifying what parameters to call the `query`
            method with.

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

    def graphviz(self):
        """Export to Graphviz.

        The graphviz module is imported during this function call. Therefore it isn't a hard
        requirement. Instead the user has to install it by herself.

        """

        import graphviz

        G = graphviz.Digraph()

        for node in self.nodes:
            G.node(str(node))

        for node, children in self.children.items():
            for child in children:
                G.edge(str(node), str(child))

        return G

    def _repr_svg_(self):
        return self.graphviz()._repr_svg_()

    def predict_proba(self, X: typing.Union[dict, pd.DataFrame]):
        """Return likelihood estimates.

        The probabilities are obtained by first computing the full joint distribution. Then, the
        likelihood of a sample is retrieved by accessing the relevant row in the full joint
        distribution.

        This method is a stepping stone for other functionalities, such as computing the
        log-likelihood. The latter can in turn be used for structure learning.

        Parameters
        ----------
        X
            One or more samples.

        """

        if isinstance(X, dict):
            return self.predict_proba(pd.DataFrame([X])).iloc[0]

        fjd = self.full_joint_dist()

        if unobserved := set(fjd.index.names) - set(X.columns):
            fjd = fjd.droplevel(list(unobserved))
            fjd = fjd.groupby(fjd.index.names).sum()

        if len(fjd.index.names) > 1:
            return fjd[pd.MultiIndex.from_frame(X)]
        return fjd

    def predict_log_proba(self, X: typing.Union[dict, pd.DataFrame]):
        """Return log-likelihood estimates.

        Parameters
        ----------
        X
            One or more samples.

        """
        return np.log(self.predict_proba(X))

    @property
    def is_tree(self):
        """Indicate whether or not the network is a tree.

        Each node in a tree has at most one parent. Therefore, the network is not a tree if any of
        its nodes has two or more parents.

        Examples
        --------

        >>> import hedgehog as hh

        >>> hh.BayesNet(
        ...     ('a', 'b'),
        ...     ('a', 'c')
        ... ).is_tree
        True

        >>> hh.BayesNet(
        ...     ('a', 'c'),
        ...     ('b', 'c')
        ... ).is_tree
        False

        """
        return not any(len(parents) > 1 for parents in self.parents.values())

    def markov_boundary(self, node):
        """Return the Markov boundary of a node.

        In a Bayesian network, the Markov boundary is a minimal Markov blanket. The Markov boundary
        of a node includes its parents, children and the other parents of all of its children.

        Examples
        --------

        The following article is taken from the Markov blanket Wikipedia article.

        >>> import hedgehog as hh

        >>> bn = hh.BayesNet(
        ...     (0, 3),
        ...     (1, 4),
        ...     (2, 5),
        ...     (3, 6),
        ...     (4, 6),
        ...     (5, 8),
        ...     (6, 8),
        ...     (6, 9),
        ...     (7, 9),
        ...     (7, 10),
        ...     (8, 11),
        ...     (8, 12)
        ... )

        >>> bn.markov_boundary(6)  # corresponds to node A on Wikipedia
        [3, 4, 5, 7, 8, 9]

        """
        children = self.children.get(node, [])
        return sorted(
            set(self.parents.get(node, []))
            | set(children)
            | set().union(*[self.parents[child] for child in children]) - {node}
        )

    def iter_dfs(self):
        """Iterate over the nodes in depth-first search fashion.

        Examples
        --------

        >>> import hedgehog as hh

        >>> bn = hh.examples.asia()

        >>> for node in bn.iter_dfs():
        ...     print(node)
        Smoker
        Bronchitis
        Dispnea
        Lung cancer
        TB or cancer
        Positive X-ray
        Visit to Asia
        Tuberculosis

        """

        def bfs(node, visited):

            yield node
            visited.add(node)

            for child in self.children.get(node, []):
                if child not in visited:
                    yield from bfs(child, visited)

        visited = set()

        for root in self.roots:
            yield from bfs(root, visited)
