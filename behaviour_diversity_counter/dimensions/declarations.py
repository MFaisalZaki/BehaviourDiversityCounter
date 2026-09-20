from collections import defaultdict
from lark import Lark, Transformer, v_args

#: ``(:<keyword> <name> <min> <max> <delta>)``, one declaration per string.
#:
#: The ``(:resource ...)`` and ``(:function ...)`` declarations are the same
#: format under two keywords -- same fields, same order, same ``name ->
#: fields`` mapping -- so one grammar reads both, templated on the keyword.
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


def parse_declarations(declarations, keyword):
    """Parse ``(:<keyword> ...)`` strings into ``name -> {name, min, max, delta}``.

    ``declarations`` is a list of strings, one declaration each. None or no
    strings means the dimension declares nothing, and gets an empty mapping.
    """
    parsed = defaultdict(dict)
    declarations = declarations or []
    if isinstance(declarations, str) or not all(isinstance(d, str) for d in declarations):
        raise TypeError(f'{keyword} declarations must be a list of strings, '
                        f'one (:{keyword} ...) each; got {declarations!r}')
    text = '\n'.join(declarations)
    if not text.strip():
        return parsed
    parser = Lark(_GRAMMAR.replace('KEYWORD', keyword), parser='lalr',
                  transformer=v_args(inline=True))
    for declaration in _DeclarationTransformer().transform(parser.parse(text)).children:
        parsed[declaration['name']] = declaration
    return parsed
