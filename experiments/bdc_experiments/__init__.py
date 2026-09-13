"""The empirical evaluation of "Behaviour Spaces for Diversity Planning".

Six experiments (E1-E6), each one sentence of the paper, over pools of plans a
top-quality planner produces. ``cli.main`` is the ``bdcexp`` entry point.
"""

__all__ = ['SCHEMA_VERSION']

#: Bumped when the on-disk shape of a raw dump changes. Every JSON this package
#: writes carries it next to a ``schema`` name.
SCHEMA_VERSION = 1
