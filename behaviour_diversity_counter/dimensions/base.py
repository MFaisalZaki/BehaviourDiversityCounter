from collections.abc import Mapping


def options(addinfo):
    """A dimension's ``addinfo`` as a mapping; ``None`` means nothing was supplied."""
    if addinfo is None:
        return {}
    if not isinstance(addinfo, Mapping):
        raise TypeError(f'addinfo must be a mapping or None; got {addinfo!r}')
    return addinfo


def token_payload(behaviour, name):
    """The value one dimension wrote into a behaviour string, or None.

    Matched on the token *prefix*, not as a substring: predicate and object
    names in other tokens may contain a dimension's name ('truck1' holds 'ru').
    """
    for part in behaviour.split('$$'):
        part = part.strip()
        if part.startswith(name + ':'):
            return part[len(name) + 1:].strip()
    return None

class BehaviourDimension:
    """One feature ``<Delta, extract, psi>`` of the paper's Def. feature:
    the values a plan can take on the dimension (``domain``), the extracting
    function (``extract``), and the dissimilarity ``psi`` on those values,
    in ``[0, 1]`` (``dissimilarity``). The counter averages the dimensions'
    dissimilarities into ``psi_M``.
    """

    def __init__(self, task, name, addinfo):
        self.task    = task
        self.name    = name
        self.addinfo = addinfo
        self.domain  = set()

    def payload(self, behaviour):
        """This dimension's own token value out of a joined behaviour string."""
        value = token_payload(behaviour, self.name)
        assert value is not None, 'The dimension value should be present in the plan behaviour.'
        return value

    def dissimilarity(self, pi1, pi2):
        """psi_i on two behaviours: the dimension's dissimilarity function."""
        # This is a placeholder implementation; each child class should override this.
        assert False, 'dissimilarity() must be implemented in subclasses of BehaviourDimension.'