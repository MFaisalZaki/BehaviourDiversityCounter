"""Shared fixtures: a small deterministic transport task and plans over it.

The task is deliberately tiny so that every expected behaviour string in the
suite can be worked out by hand:

    locations : l0, l1, l2      trucks : tr1, tr2
    fluents   : at(truck, location), delivered(location), fuel [0..100]
    actions   : move(t, from, to)  -- decreases fuel by 10
                drop(t, l)        -- sets delivered(l)
    goals     : delivered(l1), delivered(l2)
"""

import pytest

from unified_planning.shortcuts import (
    BoolType,
    Fluent,
    InstantaneousAction,
    IntType,
    Object,
    ObjectExp,
    Problem,
    UserType,
)
from unified_planning.plans import ActionInstance, SequentialPlan


@pytest.fixture
def domain():
    """The transport task plus the handles tests need to build expressions."""
    location = UserType('Location')
    truck = UserType('Truck')

    at = Fluent('at', BoolType(), t=truck, l=location)
    delivered = Fluent('delivered', BoolType(), l=location)
    fuel = Fluent('fuel', IntType(0, 100))

    move = InstantaneousAction('move', t=truck, f=location, to=location)
    t, f, to = move.parameter('t'), move.parameter('f'), move.parameter('to')
    move.add_precondition(at(t, f))
    move.add_effect(at(t, f), False)
    move.add_effect(at(t, to), True)
    move.add_decrease_effect(fuel, 10)

    drop = InstantaneousAction('drop', t=truck, l=location)
    dt, dl = drop.parameter('t'), drop.parameter('l')
    drop.add_precondition(at(dt, dl))
    drop.add_effect(delivered(dl), True)

    problem = Problem('transport')
    problem.add_fluent(at, default_initial_value=False)
    problem.add_fluent(delivered, default_initial_value=False)
    problem.add_fluent(fuel, default_initial_value=100)
    problem.add_action(move)
    problem.add_action(drop)

    objects = {name: Object(name, location) for name in ('l0', 'l1', 'l2')}
    objects.update({name: Object(name, truck) for name in ('tr1', 'tr2')})
    problem.add_objects(objects.values())

    problem.set_initial_value(at(objects['tr1'], objects['l0']), True)
    problem.set_initial_value(at(objects['tr2'], objects['l0']), True)

    # Goal order matters: dimensions derive their variable order from problem.goals.
    problem.add_goal(delivered(objects['l1']))
    problem.add_goal(delivered(objects['l2']))

    return {
        'problem': problem,
        'move': move,
        'drop': drop,
        'at': at,
        'delivered': delivered,
        'fuel': fuel,
        **objects,
    }


@pytest.fixture
def task(domain):
    return domain['problem']


def _plan(*steps):
    return SequentialPlan(
        [
            ActionInstance(action, tuple(ObjectExp(p) for p in params))
            for action, params in steps
        ]
    )


@pytest.fixture
def make_plan():
    """Build a SequentialPlan from (action, (objects...)) pairs."""
    return _plan


@pytest.fixture
def plan_l1_then_l2(domain):
    """tr1 delivers l1 first, then l2. Four actions."""
    move, drop = domain['move'], domain['drop']
    tr1, l0, l1, l2 = domain['tr1'], domain['l0'], domain['l1'], domain['l2']
    return _plan(
        (move, (tr1, l0, l1)),
        (drop, (tr1, l1)),
        (move, (tr1, l1, l2)),
        (drop, (tr1, l2)),
    )


@pytest.fixture
def plan_l2_then_l1(domain):
    """Same cost and same resources as plan_l1_then_l2, reversed goal order."""
    move, drop = domain['move'], domain['drop']
    tr1, l0, l1, l2 = domain['tr1'], domain['l0'], domain['l1'], domain['l2']
    return _plan(
        (move, (tr1, l0, l2)),
        (drop, (tr1, l2)),
        (move, (tr1, l2, l1)),
        (drop, (tr1, l1)),
    )


@pytest.fixture
def plan_two_trucks(domain):
    """tr1 delivers l1, tr2 delivers l2. Same goal order as plan_l1_then_l2,
    but a different resource set -- separates the 'ru'/'rc' dimensions."""
    move, drop = domain['move'], domain['drop']
    tr1, tr2 = domain['tr1'], domain['tr2']
    l0, l1, l2 = domain['l0'], domain['l1'], domain['l2']
    return _plan(
        (move, (tr1, l0, l1)),
        (drop, (tr1, l1)),
        (move, (tr2, l0, l2)),
        (drop, (tr2, l2)),
    )


@pytest.fixture
def inapplicable_plan(domain):
    """drop(tr1, l1) without moving tr1 to l1 first: the precondition fails."""
    return _plan((domain['drop'], (domain['tr1'], domain['l1'])))


@pytest.fixture
def resource_file(tmp_path):
    """A (:resource ...) file naming both trucks as resources."""
    path = tmp_path / 'resources.txt'
    path.write_text('(:resource tr1 0 10 1)\n(:resource tr2 0 10 1)\n')
    return str(path)


@pytest.fixture
def function_file(tmp_path):
    """A (:function ...) file binning fuel from 0 to 100 in steps of 10."""
    path = tmp_path / 'functions.txt'
    path.write_text('(:function fuel 0 100 10)\n')
    return str(path)


@pytest.fixture
def utility_goals(domain):
    """Utility weights keyed by goal expression, as UtilityValueDimension expects."""
    delivered, l1, l2 = domain['delivered'], domain['l1'], domain['l2']
    return {'utility-goals': {delivered(l1): 5, delivered(l2): 3}}
