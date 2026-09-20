from collections.abc import Mapping


def options(addinfo):
    """A dimension's ``addinfo`` as a mapping.

    ``None`` means nothing was supplied; a bare string is the path of the
    dimension's declaration file; a mapping is taken as it is.
    """
    if addinfo is None:
        return {}
    if isinstance(addinfo, Mapping):
        return addinfo
    return {'file': addinfo}


def declaration_source(addinfo):
    """The declaration file a dimension was pointed at, if any."""
    return options(addinfo).get('file')


def declared_weight(addinfo):
    """The weight a dimension's ``addinfo`` declares, or None when it does not.

    The counter turns None into the uniform ``1/n`` when no dimension declares
    a weight, so this must stay distinguishable from an explicit ``1.0``.
    """
    return options(addinfo).get('weight')


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
    """One feature ``<Delta, extract, psi, w>`` of the paper's Def. feature:
    the values a plan can take on the dimension (``domain``), the extracting
    function (``extract``), the dissimilarity ``psi`` on those values, in
    ``[0, 1]`` before the weight is applied (``dissimilarity``), and the
    weight ``w``.
    """

    def __init__(self, task, name, addinfo, weight=None):
        self.task    = task
        self.name    = name
        self.addinfo = addinfo
        self.declared_weight = weight is not None
        self.weight  = 1.0 if weight is None else float(weight)
        self.domain  = set()

    def payload(self, behaviour):
        """This dimension's own token value out of a joined behaviour string."""
        value = token_payload(behaviour, self.name)
        assert value is not None, 'The dimension value should be present in the plan behaviour.'
        return value

    def dissimilarity(self, b1, b2):
        """This dimension's term of ``psi_M(b, b') = sum_i w_i * psi_i(b_i, b'_i)``.

        An implementation scores the pair in [0, 1] and scales by ``self.weight``.
        """
        assert False, 'This method should be implemented by the child class.'
