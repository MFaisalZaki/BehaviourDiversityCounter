
class DimensionConstructorSimulator:
    def __init__(self, task, name, addinfo):
        self.task = task
        self.name = name
        self.addinfo = addinfo
        self.domain = set()
        self.estimated_domain_size = set()
    
    def _estimate_domain(self):
        assert False, 'This method should be implemented by the child class.'