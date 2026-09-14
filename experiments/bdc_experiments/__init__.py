"""The empirical evaluation of "Behaviour Spaces for Diversity Planning".

Three questions (E1-E3) over the pools a top-quality planner produced once.
Two task kinds run on a cluster -- the selection sweep and E3's timing -- and
the three reports are pure functions of their results. ``cli.main`` is the
``bdcexp`` entry point.
"""

#: Bumped when the on-disk shape of a raw dump changes. Every JSON this package
#: writes carries it next to a ``schema`` name.
SCHEMA_VERSION = 2
