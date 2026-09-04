"""Tests for the (:resource ...) / (:function ...) declaration file parser.

Both files are the same format under two keywords, and are read by one parser.
"""

import pytest

from behaviour_diversity_counter.dimensions.declaration_file import parse_declaration_file


class TestParseDeclarationFile:
    def test_parses_fields_in_declaration_order(self, resource_file):
        parsed = parse_declaration_file(resource_file, 'resource')

        assert set(parsed) == {'tr1', 'tr2'}
        assert parsed['tr1'] == {'name': 'tr1', 'min': 0, 'max': 10, 'delta': 1}

    def test_no_input_file_yields_empty_mapping(self):
        assert parse_declaration_file(None, 'resource') == {}
        assert parse_declaration_file(None, 'function') == {}

    def test_missing_file_is_rejected(self, tmp_path):
        with pytest.raises(AssertionError, match='does not exist'):
            parse_declaration_file(str(tmp_path / 'nope.txt'), 'resource')

    def test_parses_parenthesised_names(self, tmp_path):
        path = tmp_path / 'r.txt'
        path.write_text('(:resource fuel(tr1) 0 100 10)\n')

        assert parse_declaration_file(str(path), 'resource')['fuel(tr1)']['max'] == 100

    def test_parses_multiple_declarations(self, tmp_path):
        path = tmp_path / 'r.txt'
        path.write_text('(:resource a 0 1 1)\n(:resource b 2 3 1)\n(:resource c 4 5 1)\n')

        assert set(parse_declaration_file(str(path), 'resource')) == {'a', 'b', 'c'}

    def test_min_and_max_are_not_swapped(self, function_file):
        """Regression: the function parser read min from token[2] and max from
        token[1], inverting the grammar order NAME MIN MAX DELTA."""
        parsed = parse_declaration_file(function_file, 'function')

        # Declared as: (:function fuel 0 100 10)
        assert parsed['fuel'] == {'name': 'fuel', 'min': 0, 'max': 100, 'delta': 10}

    def test_the_two_keywords_read_the_same_grammar(self, tmp_path):
        """The keyword is the only thing that varies, so the two files must
        parse identically -- this is what the two separate copies had to agree
        on by hand, and what templating one grammar makes structural."""
        fn_path = tmp_path / 'f.txt'
        fn_path.write_text('(:function x 0 100 10)\n')
        rc_path = tmp_path / 'r.txt'
        rc_path.write_text('(:resource x 0 100 10)\n')

        fn = parse_declaration_file(str(fn_path), 'function')['x']
        rc = parse_declaration_file(str(rc_path), 'resource')['x']

        assert (fn['min'], fn['max'], fn['delta']) == (0, 100, 10)
        assert fn == rc

    def test_the_keyword_is_required_to_match(self, tmp_path):
        """A resource file is not a function file: the keyword is part of the
        grammar, not decoration."""
        path = tmp_path / 'r.txt'
        path.write_text('(:resource x 0 100 10)\n')

        with pytest.raises(Exception):
            parse_declaration_file(str(path), 'function')
