"""Tests for the (:resource ...) and (:function ...) file parsers."""

import pytest

from behaviour_diversity_counter.dimensions.functions import parse_functions_file
from behaviour_diversity_counter.dimensions.resources import parse_resource_file


class TestParseResourceFile:
    def test_parses_fields_in_declaration_order(self, resource_file):
        parsed = parse_resource_file(resource_file)

        assert set(parsed) == {'tr1', 'tr2'}
        assert parsed['tr1'] == {'name': 'tr1', 'min': 0, 'max': 10, 'delta': 1}

    def test_no_input_file_yields_empty_mapping(self):
        assert parse_resource_file(None) == {}

    def test_missing_file_is_rejected(self, tmp_path):
        with pytest.raises(AssertionError, match='does not exist'):
            parse_resource_file(str(tmp_path / 'nope.txt'))

    def test_parses_parenthesised_names(self, tmp_path):
        path = tmp_path / 'r.txt'
        path.write_text('(:resource fuel(tr1) 0 100 10)\n')

        assert parse_resource_file(str(path))['fuel(tr1)']['max'] == 100

    def test_parses_multiple_resources(self, tmp_path):
        path = tmp_path / 'r.txt'
        path.write_text('(:resource a 0 1 1)\n(:resource b 2 3 1)\n(:resource c 4 5 1)\n')

        assert set(parse_resource_file(str(path))) == {'a', 'b', 'c'}


class TestParseFunctionsFile:
    def test_no_input_file_yields_empty_mapping(self):
        assert parse_functions_file(None) == {}

    def test_missing_file_is_rejected(self, tmp_path):
        with pytest.raises(AssertionError, match='does not exist'):
            parse_functions_file(str(tmp_path / 'nope.txt'))

    def test_parses_name_and_delta(self, function_file):
        parsed = parse_functions_file(function_file)

        assert set(parsed) == {'fuel'}
        assert parsed['fuel']['name'] == 'fuel'
        assert parsed['fuel']['delta'] == 10

    def test_min_and_max_are_not_swapped(self, function_file):
        """Regression: min was read from token[2] and max from token[1],
        inverting the grammar order NAME MIN MAX DELTA."""
        parsed = parse_functions_file(function_file)

        # Declared as: (:function fuel 0 100 10)
        assert parsed['fuel']['min'] == 0
        assert parsed['fuel']['max'] == 100

    def test_reads_the_same_grammar_as_the_resource_parser(self, tmp_path):
        """The two parsers share a grammar shape and must agree on it."""
        fn_path = tmp_path / 'f.txt'
        fn_path.write_text('(:function x 0 100 10)\n')
        rc_path = tmp_path / 'r.txt'
        rc_path.write_text('(:resource x 0 100 10)\n')

        fn = parse_functions_file(str(fn_path))['x']
        rc = parse_resource_file(str(rc_path))['x']

        assert (fn['min'], fn['max'], fn['delta']) == (0, 100, 10)
        assert (fn['min'], fn['max'], fn['delta']) == (rc['min'], rc['max'], rc['delta'])

    def test_parses_parenthesised_names(self, tmp_path):
        path = tmp_path / 'f.txt'
        path.write_text('(:function fuel(tr1) 0 100 10)\n')

        assert parse_functions_file(str(path))['fuel(tr1)']['max'] == 100
