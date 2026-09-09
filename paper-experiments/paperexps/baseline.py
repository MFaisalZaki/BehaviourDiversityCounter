"""The plan-level (metric-based) diversity of the existing literature, for
Experiment 1: the stability, state and uniqueness distances of Srivastava et
al. and Katz and Sohrabi, the average-pairwise plan-set score, and the greedy
selection that maximises it.  Implemented here, over the same pool the
behaviour-space selections see, so both selections meet identical candidates.
"""

import numpy as np

METRICS = ('stability', 'state', 'uniqueness')


def action_set(plan):
    """The ground actions of a plan, as a set."""
    return frozenset(str(action) for action in plan.actions)


def visited_states(states):
    """The states a trace passes through, initial state included, each reduced
    to a hash of its true atoms.

    A ``UPState`` stores only what its own action changed and reaches the rest
    through its ancestors, so the full assignment is rebuilt by folding every
    state's own values over the previous ones.  Boolean atoms that are false
    are dropped, which is what makes two STRIPS states with the same true
    atoms equal; numeric fluents keep their value.
    """
    assignment, visited = {}, set()
    for state in states:
        for fluent, value in state._values.items():
            if value.is_bool_constant():
                if value.is_true():
                    assignment[str(fluent)] = str(fluent)
                else:
                    assignment.pop(str(fluent), None)
            else:
                assignment[str(fluent)] = f'{fluent}={value}'
        visited.add(hash(frozenset(assignment.values())))
    return frozenset(visited)


def _jaccard_distance(a, b):
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


class PlanMetric:
    """One plan-level distance over a pool, with its n x n matrix."""

    def __init__(self, name, plans, trace=None):
        if name not in METRICS:
            raise ValueError(f'unknown plan metric {name!r}; one of {METRICS}')
        self.name = name
        if name == 'state':
            if trace is None:
                raise ValueError('the state metric needs the plans\' state traces')
            self.features = [visited_states(trace[id(plan)][0]) for plan in plans]
        else:
            self.features = [action_set(plan) for plan in plans]

    def distance(self, i, j):
        a, b = self.features[i], self.features[j]
        if self.name == 'uniqueness':
            return 0.0 if a == b else 1.0
        return _jaccard_distance(a, b)

    def matrix(self):
        n = len(self.features)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i, j] = matrix[j, i] = self.distance(i, j)
        return matrix


def plan_set_score(matrix, indices):
    """The metric-based score: the average pairwise distance over the set."""
    indices = list(indices)
    if len(indices) < 2:
        return 0.0
    rows, cols = np.triu_indices(len(indices), k=1)
    sub = matrix[np.ix_(indices, indices)]
    return float(sub[rows, cols].mean())


def greedy_select(matrix, k):
    """Farthest pair first, then the plan maximising the average pairwise
    distance of the combined set -- which, the pairs already held being fixed,
    is the plan of greatest summed distance to the selection.  Ties fall to
    the lowest index, so with a cost-sorted pool to the cheapest plan."""
    n = len(matrix)
    if k <= 0 or n == 0:
        return []
    if k == 1 or n == 1:
        return [0]
    rows, cols = np.triu_indices(n, k=1)
    best = int(np.argmax(np.round(matrix[rows, cols], 9)))
    selected = [int(rows[best]), int(cols[best])]
    candidates = np.setdiff1d(np.arange(n), selected)
    while len(selected) < k and len(candidates):
        gains = matrix[np.ix_(candidates, selected)].sum(axis=1)
        position = int(np.argmax(np.round(gains, 9)))
        selected.append(int(candidates[position]))
        candidates = np.delete(candidates, position)
    return selected
