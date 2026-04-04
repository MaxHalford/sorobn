import collections
import itertools

import pandas as pd


# ---------------------------------------------------------------------------
# Graph algorithms for junction trees
# ---------------------------------------------------------------------------

def moralize(parents, children, nodes):
    """Convert a DAG to its moral graph (undirected, co-parents connected)."""
    adj = collections.defaultdict(set)
    for node in nodes:
        for parent in parents.get(node, []):
            adj[node].add(parent)
            adj[parent].add(node)
        node_parents = parents.get(node, [])
        for p1, p2 in itertools.combinations(node_parents, 2):
            adj[p1].add(p2)
            adj[p2].add(p1)
    for node in nodes:
        adj.setdefault(node, set())
    return dict(adj)


def triangulate(adj):
    """Triangulate an undirected graph using greedy min-degree elimination."""
    work = {n: set(nbs) for n, nbs in adj.items()}
    tri_adj = {n: set(nbs) for n, nbs in adj.items()}
    order = []
    while work:
        node = min(work, key=lambda n: (len(work[n]), n))
        neighbors = work[node]
        for n1, n2 in itertools.combinations(neighbors, 2):
            work[n1].add(n2)
            work[n2].add(n1)
            tri_adj[n1].add(n2)
            tri_adj[n2].add(n1)
        for nb in neighbors:
            work[nb].discard(node)
        del work[node]
        order.append(node)
    return tri_adj, order


def find_cliques(tri_adj, order):
    """Extract maximal cliques from a triangulated graph."""
    remaining = set(tri_adj)
    cliques = []
    for node in order:
        clique = frozenset({node} | (tri_adj[node] & remaining))
        cliques.append(clique)
        remaining.discard(node)
    maximal = []
    for c in cliques:
        if not any(c < other for other in cliques):
            if c not in maximal:
                maximal.append(c)
    return maximal


def build_junction_tree(cliques):
    """Build a junction tree (max spanning tree over cliques weighted by sepset size)."""
    if len(cliques) == 1:
        return {cliques[0]: []}

    edges = sorted(
        itertools.combinations(cliques, 2),
        key=lambda e: len(e[0] & e[1]),
        reverse=True,
    )

    parent = {id(c): c for c in cliques}
    rank = {id(c): 0 for c in cliques}

    def find(c):
        cid = id(c)
        while parent[cid] is not c:
            c = parent[cid]
            cid = id(c)
        return c

    def union(a, b):
        aid, bid = id(a), id(b)
        if rank[aid] < rank[bid]:
            parent[aid] = b
        elif rank[aid] > rank[bid]:
            parent[bid] = a
        else:
            parent[bid] = a
            rank[aid] += 1

    tree_adj = collections.defaultdict(list)
    n_edges = 0
    for c1, c2 in edges:
        r1, r2 = find(c1), find(c2)
        if r1 is not r2:
            tree_adj[c1].append(c2)
            tree_adj[c2].append(c1)
            union(r1, r2)
            n_edges += 1
            if n_edges == len(cliques) - 1:
                break

    for c in cliques:
        tree_adj.setdefault(c, [])
    return dict(tree_adj)


def root_tree(tree_adj, root):
    """Root a tree and return (children dict, BFS order)."""
    children = collections.defaultdict(list)
    order = [root]
    visited = {root}
    queue = collections.deque([root])
    while queue:
        node = queue.popleft()
        for nb in tree_adj.get(node, []):
            if nb not in visited:
                visited.add(nb)
                children[node].append(nb)
                order.append(nb)
                queue.append(nb)
    return dict(children), order


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _factor_names(factor):
    """Return the variable names of a factor as a list."""
    if isinstance(factor.index, pd.MultiIndex):
        return list(factor.index.names)
    return [factor.index.name]


def _normalize_conditional(joint, conditioning_vars, query_vars):
    """Normalize a joint into a conditional P(query | conditioning).

    The result has index levels [*conditioning_vars, *query_vars] and sums to 1
    for each conditioning combination, making it compatible with CDTAccessor's
    cached __getitem__ and Vose sampler.

    """
    if conditioning_vars:
        level_order = list(conditioning_vars) + list(query_vars)
        if isinstance(joint.index, pd.MultiIndex):
            joint = joint.reorder_levels(level_order)
        joint = joint.sort_index()
        # Normalize: divide by sum within each conditioning group
        normalizer = joint.groupby(level=list(conditioning_vars)).transform("sum")
        cond = joint / normalizer
        # Drop entries where normalizer was 0 (impossible conditioning combos)
        cond = cond[cond.notna() & (cond > 0)]
    else:
        cond = joint / joint.sum()
        cond = cond.sort_index()
    return cond


# ---------------------------------------------------------------------------
# Path sampler
# ---------------------------------------------------------------------------

class PathSampler:
    """Exact sampler that walks a DFS path through the undirected graph skeleton.

    Precomputes a conditional table for each node in DFS order. At sample time,
    these are accessed via CDTAccessor's cached __getitem__ and Vose sampler for
    O(1) lookups and sampling.

    """

    def __init__(self, bn):
        self.bn = bn
        self._tables = None

    def invalidate(self):
        self._tables = None

    def _build(self):
        from .bayes_net import pointwise_mul

        bn = self.bn
        order = bn._dfs_order()

        # Undirected adjacency for the skeleton
        adj = collections.defaultdict(set)
        for child, parents in bn.parents.items():
            for parent in parents:
                adj[child].add(parent)
                adj[parent].add(child)

        tables = []
        sampled_so_far = set()

        for node in order:
            # Start with the CPT for this node
            factors = [bn.P[node][bn.P[node] > 0].copy()]

            # Add CPTs of visited children (they contain info about `node`)
            for child in bn.children.get(node, []):
                if child in sampled_so_far:
                    factors.append(bn.P[child][bn.P[child] > 0].copy())

            # Eliminate hidden variables.  Adding a hidden variable's CPT
            # may introduce further hidden parents, so we iterate until
            # no new hidden variables remain.
            eliminated = set()
            while True:
                factor_vars = set()
                for f in factors:
                    factor_vars |= set(_factor_names(f))
                to_eliminate = factor_vars - {node} - sampled_so_far - eliminated
                if not to_eliminate:
                    break
                for h in list(to_eliminate):
                    h_cpt = bn.P[h][bn.P[h] > 0].copy()
                    factors.append(h_cpt)
                    relevant = [
                        factors.pop(i)
                        for i in reversed(range(len(factors)))
                        if h in _factor_names(factors[i])
                    ]
                    if relevant:
                        prod = pointwise_mul(relevant)
                        prod = prod.cdt.sum_out(h)
                        factors.append(prod)
                    eliminated.add(h)

            joint = pointwise_mul(factors)

            joint_vars = set(
                joint.index.names if isinstance(joint.index, pd.MultiIndex)
                else [joint.index.name]
            )
            effective_cond = sorted(joint_vars - {node})

            cond = _normalize_conditional(joint, effective_cond, [node])
            tables.append((node, effective_cond, cond))
            sampled_so_far.add(node)

        self._tables = tables

    def _ensure_built(self):
        if self._tables is None:
            self._build()

    def iter_samples(self, rng, init=None):
        """Yield samples as dicts, indefinitely."""
        self._ensure_built()
        init = init or {}

        while True:
            sample = dict(init)

            for node, conditioning_vars, cond in self._tables:
                if node in sample:
                    continue

                if conditioning_vars:
                    condition = tuple(sample[v] for v in conditioning_vars)
                    P = cond.cdt[condition]
                else:
                    P = cond

                sample[node] = P.cdt.sample(rng=rng)

            yield sample


# ---------------------------------------------------------------------------
# Junction tree sampler
# ---------------------------------------------------------------------------

class JunctionTreeSampler:
    """Exact sampler using a calibrated junction tree.

    Builds and calibrates the junction tree once, then converts each clique's
    calibrated potential into a conditional table. At sample time, these are
    accessed via CDTAccessor for O(1) lookups and sampling.

    """

    def __init__(self, bn):
        self.bn = bn
        self._data = None  # list of (new_vars, sepset_vars, conditional)

    def invalidate(self):
        self._data = None

    def _build(self):
        from .bayes_net import pointwise_mul_two

        bn = self.bn
        adj = moralize(bn.parents, bn.children, bn.nodes)
        tri_adj, elim_order = triangulate(adj)
        cliques = find_cliques(tri_adj, elim_order)
        tree_adj = build_junction_tree(cliques)

        # Assign each CPT to the smallest containing clique
        potentials = {c: None for c in cliques}
        for node in bn.nodes:
            cpt = bn.P[node]
            cpt_vars = frozenset(_factor_names(cpt))
            target = min((c for c in cliques if cpt_vars <= c), key=len)
            if potentials[target] is None:
                potentials[target] = cpt.copy()
            else:
                potentials[target] = pointwise_mul_two(potentials[target], cpt)

        # Cliques with no CPT get a uniform potential
        for c in cliques:
            if potentials[c] is None:
                domains = {}
                for var in c:
                    for pot in potentials.values():
                        if pot is not None and var in _factor_names(pot):
                            domains[var] = pot.index.get_level_values(var).unique()
                            break
                idx = pd.MultiIndex.from_product(
                    [domains[v] for v in sorted(c)], names=sorted(c)
                )
                potentials[c] = pd.Series(1.0, index=idx)

        # Root the tree and calibrate
        root = cliques[0]
        tree_children, order = root_tree(tree_adj, root)
        potentials = self._calibrate(tree_children, order, potentials)

        # Convert calibrated potentials into conditionals for fast sampling.
        # For each clique in BFS order, the sepset with its parent contains the
        # already-sampled variables (by the running intersection property).
        parent_of = {}
        for node in order:
            for child in tree_children.get(node, []):
                parent_of[child] = node

        tables = []
        for clique in order:
            pot = potentials[clique]
            if clique in parent_of:
                sepset = sorted(clique & parent_of[clique])
                new_vars = sorted(clique - parent_of[clique])
            else:
                sepset = []
                new_vars = sorted(clique)

            cond = _normalize_conditional(pot, sepset, new_vars)
            tables.append((new_vars, sepset, cond))

        self._data = tables

    @staticmethod
    def _calibrate(tree_children, order, potentials):
        """Hugin-style collect/distribute message passing."""
        from .bayes_net import pointwise_mul_two

        messages_up = {}

        for node in reversed(order):
            for child in tree_children.get(node, []):
                sep = node & child
                eliminate = list(child - sep)
                msg = potentials[child]
                for var in eliminate:
                    msg = msg.cdt.sum_out(var)
                messages_up[(child, node)] = msg
                potentials[node] = pointwise_mul_two(potentials[node], msg)

        for node in order:
            for child in tree_children.get(node, []):
                sep = node & child
                eliminate = list(node - sep)
                msg_down = potentials[node]
                for var in eliminate:
                    msg_down = msg_down.cdt.sum_out(var)
                msg_up = messages_up[(child, node)]
                msg_net = msg_down / msg_up.reindex(msg_down.index, fill_value=0)
                msg_net = msg_net.fillna(0)
                potentials[child] = pointwise_mul_two(potentials[child], msg_net)

        return potentials

    def _ensure_built(self):
        if self._data is None:
            self._build()

    def iter_samples(self, rng, init=None):
        """Yield samples as dicts, indefinitely."""
        self._ensure_built()
        init = init or {}

        while True:
            sample = dict(init)

            for new_vars, sepset, cond in self._data:
                if all(v in sample for v in new_vars):
                    continue

                if sepset:
                    condition = tuple(sample[v] for v in sepset)
                    P = cond.cdt[condition]
                else:
                    P = cond

                chosen = P.cdt.sample(rng=rng)
                if len(new_vars) == 1:
                    sample[new_vars[0]] = chosen
                else:
                    if not isinstance(chosen, tuple):
                        chosen = (chosen,)
                    for name, val in zip(P.index.names, chosen):
                        if name not in sample:
                            sample[name] = val

            yield sample
