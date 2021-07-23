import logging

from nltk.sem.logic import LogicParser
from nltk.sem.logic import LogicalExpressionException

logic_parser = LogicParser(type_check=False)
def lexpr(formula_str):
    try:
        expr = logic_parser.parse(formula_str)
    except LogicalExpressionException as e:
        logging.error('Failed to parse {0}. Error: {1}'.format(formula_str, e))
        raise
    return expr