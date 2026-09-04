import os

from collections import defaultdict
from lark import Lark, Transformer, v_args

#: ``(:<keyword> <name> <min> <max> <delta>)``, one declaration per line.
#:
#: The ``(:resource ...)`` and ``(:function ...)`` files are the same format
#: under two keywords -- same fields, same order, same ``name -> fields``
#: mapping -- so one grammar reads both, templated on the keyword.
_GRAMMAR = r'''
    start: declaration+
    declaration: "(:KEYWORD" (NAME | NAME_WITH_PARENTHESIS) MIN MAX DELTA ")"
    NAME: /[a-zA-Z_][\w-]*/
    NAME_WITH_PARENTHESIS: /[a-zA-Z_]\w*\([^)]*\)/
    MIN: /[0-9]+/
    MAX: /[0-9]+/
    DELTA: /[0-9]+/
    %ignore /\s+/
'''


class _DeclarationTransformer(Transformer):
    def declaration(self, token):
        # Grammar order is NAME MIN MAX DELTA.
        return {
            'name':  token[0].value,
            'min':   int(token[1].value),
            'max':   int(token[2].value),
            'delta': int(token[3].value)
        }


def parse_declaration_file(inputfile, keyword):
    """Read a ``(:<keyword> ...)`` file into ``name -> {name, min, max, delta}``.

    No file means the dimension declares nothing, and gets an empty mapping.
    """
    assert inputfile is not None, f'The {inputfile} file should not be None.'
    declarations = defaultdict(dict)
    if not inputfile:
        return declarations
    assert os.path.exists(inputfile), f'The {keyword} file {inputfile} does not exist.'
    with open(inputfile, 'r') as f:
        text = f.read()
    parser = Lark(_GRAMMAR.replace('KEYWORD', keyword), parser='lalr',
                  transformer=v_args(inline=True))
    for declaration in _DeclarationTransformer().transform(parser.parse(text)).children:
        declarations[declaration['name']] = declaration
    return declarations
