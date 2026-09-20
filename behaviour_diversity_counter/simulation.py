"""Replaying a plan against its task: the state trace and the plan's cost.

Every dimension reads the state trace (an extracting function is written over
the plan, but the transition function is deterministic, so a plan fixes the
states it passes through), and two places read the cost: the ``cb`` dimension,
and B-Coverage selection, which keeps the cheapest plan per behaviour.
"""

from unified_planning.engines.sequential_simulator import evaluate_quality_metric
from unified_planning.shortcuts import SequentialSimulator


class InapplicablePlanError(ValueError):
    """A plan could not be simulated against the task.

    The state trace is what every dimension reads, so a plan that cannot be
    replayed has no behaviour to report. Counting it anyway would silently
    inflate the diversity count with a behaviour no valid plan produced.
    """


def cost_metric(task):
    """The task's ``MinimizeActionCosts`` metric, or None under unit cost."""
    return next((metric for metric in task.quality_metrics
                 if metric.is_minimize_action_costs()), None)


def simulate(task, plan, simulator=None):
    """``(states, cost)`` for a plan: its state trace, initial state first, and
    its cost in the sense of the paper's Def. plan -- the sum of its action
    costs under the task's ``MinimizeActionCosts`` metric, which is the plan
    length when the task declares none (unit cost).
    """
    simulator = simulator if simulator is not None else SequentialSimulator(problem=task)
    metric = cost_metric(task)
    state = simulator.get_initial_state()
    states, cost = [state], 0
    for step, action_instance in enumerate(plan.actions):
        next_state = simulator.apply(state, action_instance)
        if next_state is None:
            raise InapplicablePlanError(
                f'{action_instance} at step {step} is not applicable to the state '
                f'it reaches; the plan cannot be simulated against this task.'
            )
        if metric is not None:
            cost = evaluate_quality_metric(
                simulator, metric, cost, state,
                action_instance.action, action_instance.actual_parameters, next_state)
        state = next_state
        states.append(state)
    if metric is None:
        cost = len(plan.actions)
    return states, cost


def plan_cost(task, plan, simulator=None):
    """The plan's cost alone; see :func:`simulate`."""
    return simulate(task, plan, simulator)[1]
