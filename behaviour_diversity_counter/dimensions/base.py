from collections.abc import Mapping


def declaration_source(addinfo):
    """The declaration file a dimension was pointed at.

    Either the path on its own, or the ``'file'`` entry of a mapping that also
    carries the dimension's weight.
    """
    return addinfo.get('file') if isinstance(addinfo, Mapping) else addinfo


class BehaviourDimension:
    def __init__(self, task, name, addinfo, weight=1.0):
        self.task    = task
        self.name    = name
        self.addinfo = addinfo
        self.weight  = weight
        self.domain  = set()

    def distance(self, b1, b2):
        """This dimension's term of ``d(b, b') = sum_i w_i * d_i(b_i, b'_i)``.

        An implementation scores the pair in [0, 1] and scales by ``self.weight``.
        """
        assert False, 'This method should be implemented by the child class.'
