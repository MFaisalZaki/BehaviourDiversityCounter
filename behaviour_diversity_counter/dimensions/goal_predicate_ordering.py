from behaviour_diversity_counter.dimensions.landmark_predicate_ordering import LandmarkPredicatesOrderingDimension

class GoalPredicatesOrderingDimension(LandmarkPredicatesOrderingDimension):
    def __init__(self, task, addinfo=None):
        super().__init__(task, 'go', addinfo)