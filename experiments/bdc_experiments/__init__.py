"""The empirical evaluation of "Behaviour Spaces for Diversity Planning".

Six questions (E1-E6), each one sentence of the paper, over pools of plans a
top-quality planner produces. Two task kinds run on a cluster -- the selection
sweep and E6's timing -- and the six reports are pure functions of their
results. ``cli.main`` is the ``bdcexp`` entry point.
"""

__all__ = ['SCHEMA_VERSION']

#: Bumped when the on-disk shape of a raw dump changes. Every JSON this package
#: writes carries it next to a ``schema`` name.
SCHEMA_VERSION = 2
