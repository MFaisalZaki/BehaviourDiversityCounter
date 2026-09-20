from behaviour_diversity_counter.dimensions.base import (
    BehaviourDimension, declaration_source, declared_weight)
from behaviour_diversity_counter.dimensions.declaration_file import parse_declaration_file


def _normalised(name):
    return str(name).replace('(', '_').replace(')', '').replace(' ', '_').replace(',', '')


class NumericFunctionDimension(BehaviourDimension):
    """``fn``: the final value of each declared numeric fluent, quantised into
    bins of the user's width.

    A dimension is a finite set, so a numeric criterion enters the space only
    after quantisation, and the paper leaves the bin width to the user: a
    declaration ``(:function f min max delta)`` bins ``[min, max)`` into
    ``ceil((max - min) / delta)`` bins of width ``delta``. Values below ``min``
    fall into the first bin and values at or above ``max`` into the last.
    """

    def __init__(self, task, addinfo=None):
        super().__init__(task, 'fn',
                         parse_declaration_file(declaration_source(addinfo), 'function'),
                         declared_weight(addinfo))

    @staticmethod
    def _bin_count(fn):
        return len(range(fn['min'], fn['max'], fn['delta']))

    def _bin_of(self, fn, value):
        count = self._bin_count(fn)
        if count == 0:
            return 0
        index = int((value - fn['min']) // fn['delta'])
        return min(max(index, 0), count - 1)

    def extract(self, plan):
        # A lazy state holds only the fluents its own action changed, so the
        # final value of a fluent is the last value seen anywhere in the trace.
        final_values = {}
        for state in plan.states:
            for expr in state._values:
                final_values[str(expr)] = final_values[_normalised(expr)] = state.get_value(expr)

        bins = {}
        for fn in self.addinfo.values():
            if fn['name'] not in final_values:
                continue
            bins[fn['name']] = self._bin_of(fn, final_values[fn['name']].constant_value())

        val = ','.join(f'{name}={index}' for name, index in bins.items())
        self.domain.add(val)
        return f'{self.name}:' + val

    def _bins(self, behaviour):
        return {name: int(index) for name, index in
                (item.split('=') for item in self.payload(behaviour).split(',') if item)}

    def dissimilarity(self, b1, b2):
        # Per function, the bin distance |i - i'| / (bins - 1), which respects
        # the bin order as the paper requires of a quantised dimension; averaged
        # over the declared functions so the term stays in [0, 1].
        if not self.addinfo:
            return 0.0
        bins1, bins2 = self._bins(b1), self._bins(b2)
        terms = []
        for name, fn in self.addinfo.items():
            count = self._bin_count(fn)
            if count < 2 or name not in bins1 or name not in bins2:
                terms.append(0.0)
                continue
            terms.append(abs(bins1[name] - bins2[name]) / (count - 1))
        return self.weight * (sum(terms) / len(terms))
