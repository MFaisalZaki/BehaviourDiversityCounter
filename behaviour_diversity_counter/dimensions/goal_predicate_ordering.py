from collections import defaultdict

from behaviour_diversity_counter.dimensions.base import BehaviourDimension, declared_weight, options


class GoalPredicatesOrderingDimension(BehaviourDimension):
    """``go``: the order in which the goal atoms are first achieved.

    The goal atoms are taken in a canonical order -- the order in which the
    problem's goal expressions introduce them, sorted by expression identity so
    that it is the same in every process. Atoms achieved by the same action
    keep that order between them. ``addinfo['max-goals']`` caps the atoms at
    the first ``m`` of that order; the cap is recorded as ``self.max_goals``
    and the atoms actually used as ``self.vars``.
    """

    def __init__(self, task, addinfo=None):
        super().__init__(task, 'go', addinfo, declared_weight(addinfo))
        from unified_planning.model.walkers.free_vars import FreeVarsExtractor
        atoms = []
        for goal in self.task.goals:
            atoms.extend(sorted(FreeVarsExtractor().get(goal), key=lambda atom: atom.node_id))
        self.goal_atoms = len(atoms)
        self.max_goals = options(addinfo).get('max-goals')
        self.vars = atoms if self.max_goals is None else atoms[:self.max_goals]

    def extract(self, plan):
        _time_step_history = defaultdict(list)
        for t, state in enumerate(plan.states):
            for g in self.vars:
                _time_step_history[g].append(state.get_value(g).is_true())
        val = '->'.join(map(lambda e: str(e[0]), sorted([(g, next((i for i, x in enumerate(_time_step_history[g]) if x), -1)) for g in self.vars], key=lambda e:e[1])))
        self.domain.add(val)
        return f'{self.name}:' + val

    def _ordering(self, behaviour):
        return self.payload(behaviour).replace(' ', '').split('->')

    def dissimilarity(self, b1, b2):
        # The Hamming distance between the two orderings, divided by the number
        # of goals so that it lies in [0, 1] (Def. feature).
        ordering1, ordering2 = self._ordering(b1), self._ordering(b2)
        hamming = sum(x != y for x, y in zip(ordering1, ordering2))
        return self.weight * (hamming / len(ordering1) if ordering1 else 0.0)
