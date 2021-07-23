
from nltk.sem.logic import ApplicationExpression
from nltk.sem.logic import AndExpression
from nltk.sem.logic import OrExpression
from nltk.sem.logic import ImpExpression
from nltk.sem.logic import NegatedExpression
from nltk.sem.logic import ExistsExpression
from nltk.sem.logic import AllExpression


def rev_pol(value):
    if value == 0:
        out = 1
    else:
        out = 0
    return out


def calculate_polarity(expression):
    pols = polarity_expr(expression, 1)
    return pols


def polarity_expr(expression, value):
    if isinstance(expression, ApplicationExpression):
        pols = polarity_application_expr(expression, value)
    elif isinstance(expression, AndExpression):
        pols = polarity_and_expr(expression, value)
    elif isinstance(expression, OrExpression):
        pols = polarity_or_expr(expression, value)
    elif isinstance(expression, ImpExpression):
        pols = polarity_imp_expr(expression, value)
    elif isinstance(expression, NegatedExpression):
        pols = polarity_not_expr(expression, value)
    elif isinstance(expression, ExistsExpression):
        pols = polarity_exists_expr(expression, value)
    elif isinstance(expression, AllExpression):
        pols = polarity_all_expr(expression, value)
    else:
        pols = []
    return pols


def polarity_application_expr(expression, value):
    fun, args = expression.uncurry()
    res = [(str(fun), value)]
    return res


def polarity_and_expr(expression, value):
    left = polarity_expr(expression.first, value)
    right = polarity_expr(expression.second, value)
    pols = left + right
    return pols


def polarity_or_expr(expression, value):
    left = polarity_expr(expression.first, value)
    right = polarity_expr(expression.second, value)
    pols = left + right
    return pols


def polarity_imp_expr(expression, value):
    left = polarity_expr(expression.first, rev_pol(value))
    right = polarity_expr(expression.second, value)
    pols = left + right
    return pols


def polarity_not_expr(expression, value):
    pols = polarity_expr(expression.term, rev_pol(value))
    return pols


def polarity_exists_expr(expression, value):
    pols = polarity_expr(expression.term, value)
    return pols


def polarity_all_expr(expression, value):
    pols = polarity_expr(expression.term, value)
    return pols
