from behaviour_diversity_counter.dimensions.base import BehaviourDimension, declared_weight

#: Joins the action strings inside the token. ' $$ ' separates dimensions, so
#: it cannot also separate actions within this one.
SEPARATOR = ' ; '


class StabilityDimension(BehaviourDimension):
    """``stability``: the literature's model, stated as one feature. Its
    extracting function reads a plan's set of actions, its dimension is the
    set of such sets, and its dissimilarity is the stability distance of
    Srivastava et al. (2007), one minus the Jaccard measure of two action
    sets. It is definite on action sets, as Def. feature requires, so two
    plans are twins under it exactly when they have the same action set; it
    is not a metric-based model in the paper's sense, since two orderings of
    one action set are at distance zero. B-MaxSum selection under it is the
    post-hoc selection of Katz and Sohrabi (2020).
    """

    def __init__(self, task, addinfo=None):
        super().__init__(task, 'stability', addinfo, declared_weight(addinfo))

    def plan_behaviour(self, plan):
        value = SEPARATOR.join(sorted({str(action) for action in plan.actions}))
        self.domain.add(value)
        return f'{self.name}:{value}'

    def _actions(self, behaviour):
        return set(filter(None, self.payload(behaviour).split(SEPARATOR)))

    def distance(self, b1, b2):
        a1, a2 = self._actions(b1), self._actions(b2)
        if not a1 and not a2:
            return 0.0
        return self.weight * (1.0 - len(a1 & a2) / len(a1 | a2))
