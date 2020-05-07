import collections

import pandas as pd

from rev import dist


__all__ = ['BayesNet']


class BayesNet:
    """Bayesian network.

    """

    def __init__(self, *edges):
        parents = collections.defaultdict(list)
        children = collections.defaultdict(list)

        for parent, child in set(edges):
            parents[child].append(parent)
            children[parent].append(parent)

        self.parents = dict(parents)
        self.children = dict(children)
        self.nodes = {*parents.keys(), *children.keys()}
        self.dists = {}

    def _sample(self, init=None):
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
                condition = tuple(sample[parent] for parent in self.parents.get(node, ()))
                dist = self.dists[node][condition]
                sample[node] = dist.sample()

        sample = init.copy() if init else {}
        visited = set()

        for leaf in self.nodes - set(self.children):
            sample_node_value(node=leaf, sample=sample, visited=visited)

        return sample

    def sample(self):
        """Generate a new sample at random by using forward sampling.

        Although the idea is to implement forward sampling, the implementation
        actually works backwards, starting from the leaf nodes. For every node, we recursively
        check that values have been sampled for each parent node. Once a value has been chosen for
        each parent, we can pick the according distribution and sample from it.

        """
        return self._sample()

    def fit(self, X: pd.DataFrame):
        """Find the values of each conditional distribution."""

        # Compute conditional distribution for each child
        for child, parents in self.parents.items():

            self.dists[child] = {}

            # Determine what kind of distribution to use
            dist = dist.Multinomial
            if X[child].dtype == bool:
                dist = dist.Bernoulli

            for condition, group in X.groupby(parents):
                condition = condition if isinstance(condition, tuple) else (condition,)
                self.dists[child][condition] = dist(None).fit(samples=X[child])

        # Compute distribution for each orphan
        for orphan in self.nodes - set(self.parents):

            # Determine what kind of distribution to use
            dist = dist.Multinomial
            if X[child].dtype == bool:
                dist = dist.Bernoulli

            self.dists[orphan] = {(): dist(None).fit(samples=X[orphan])}

        return self

    def _get_dist(self, var):
        return next(iter(self.dists[var].values())).__class__

    def _rejection_sampling(self, *query, event, n=100):
        """Answer a query using rejection sampling.

        """

        # We will store the outcomes for each query variable in lists
        samples = {var: [] for var in query}

        for _ in range(n):
            sample = self.sample()

            # Reject if the sample is not consistent with the specified events
            if any(sample[var] != val for var, val in event.items()):
                continue

            for var in query:
                samples[var].append(sample[var])

        return {
            var: self._get_dist(var)(None).fit(samples=samples[var])
            for var in query
        }

    def _llh_weighting(self, *query, n=100, event):
        """Answers a query using likelihood weighting.

        Likelihood weighting is a particular instance of importance sampling.

        """

        samples = {var: [None] * n for var in query}
        weights = {var: [None] * n for var in query}

        for i in range(n):

            # Sample by using the events as fixed values
            sample = self._sample(init=event)

            # Compute the likelihood of this sample
            weight = 1.
            for var, val in event.items():
                condition = tuple(sample[parent] for parent in self.parents.get(var, ()))
                weight *= self.dists[var][condition].P(val)

            for var in query:
                samples[var][i] = sample[var]
                weights[var][i] = weight

        return {
            var: self._get_dist(var)(None).fit(samples=samples[var], weights=weights[var])
            for var in query
        }

    def query(self, *query, event, algorithm='rejection', n=100):
        if algorithm == 'likelihood':
            return self._llh_weighting(*query, event=event, n=n)

        elif algorithm == 'rejection':
            return self._rejection_sampling(*query, event=event, n=n)

        raise ValueError('Unknown algorithm, must be one of: likelihood, rejection')


    def impute(self, sample):
        raise NotImplementedError
