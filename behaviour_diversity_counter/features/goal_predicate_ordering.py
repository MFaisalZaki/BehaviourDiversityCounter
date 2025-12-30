from behaviour_diversity_counter.features.landmark_predicate_ordering import LandmarkPredicatesOrderingSimulator

class GoalPredicatesOrderingSimulator(LandmarkPredicatesOrderingSimulator):
    def __init__(self, task, addinfo=None):
        super().__init__(task, 'go', addinfo)