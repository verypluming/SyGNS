from nltk.sem.drt import DRS
from nltk.sem.drt import DrtApplicationExpression
from nltk.sem.drt import DrtNegatedExpression
from nltk.sem.drt import DrtOrExpression

from logic_parser import lexpr
from nltk2drs import convert_to_drs


# Example of clausal form (clf)
# %%% I have n't touched anything .
# % I [0...1]
# % have [2...6]
# b1 NEGATION b2         % n't [6...9]
# b1 REF t1              % touched [10...17]
# b1 TPR t1 "now"        % touched [10...17]
# b1 time "n.08" t1      % touched [10...17]
# b2 REF e1              % touched [10...17]
# b2 Agent e1 "speaker"  % touched [10...17]
# b2 Experiencer e1 x1   % touched [10...17]
# b2 Time e1 t1          % touched [10...17]
# b2 touch "v.01" e1     % touched [10...17]
# b2 REF x1              % anything [18...26]
# b2 entity "n.01" x1    % anything [18...26]
# % . [26...27]


def convert_to_clausal_forms(drs):
    _, cls = convert_to_clf(1, [], drs)
    return cls


def is_variable(drs_str):
    prefix = ['x', 'e']
    if len(drs_str) <= 1:
        return False
    elif drs_str[0] in prefix and drs_str[1].isdigit():
        return True
    else:
        return False


def check_constant_and_add_quotes(drs_str):
    if is_variable(drs_str):
        return drs_str
    else:
        return '"' + drs_str + '"'


def convert_to_clf(idx, clfs, drs):
    if isinstance(drs, str):
        clfs.append(drs)
    elif isinstance(drs, DRS):
        refs = drs.refs
        conds = drs.conds
        if drs.consequent:
            head = len(clfs)
            consequent = drs.consequent
            boxvar = 'b' + str(idx)
            idx = idx + 1
            boxarg1 = 'b' + str(idx)
            antecedent = DRS(refs, conds)
            idx, clfs = convert_to_clf(idx, clfs, antecedent)
            idx = idx + 1
            boxarg2 = 'b' + str(idx)
            idx, clfs = convert_to_clf(idx, clfs, consequent)
            res = boxvar + ' IMP ' + boxarg1 + ' ' + boxarg2
            clfs.insert(head, res)
            return idx, clfs
        else:
            boxvar = 'b' + str(idx)
            for ref in refs:
                clf = boxvar + ' REF ' + str(ref)
                clfs.append(clf)
            for cond in conds:
                idx, clfs = convert_to_clf(idx, clfs, cond)
            return idx, clfs
    elif isinstance(drs, DrtApplicationExpression):
        predicate = drs.uncurry()[0]
        args = drs.uncurry()[1]
        op = str(predicate)
        boxvar = 'b' + str(idx)
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, DRS):
                next_idx = idx + 1
                boxarg = 'b' + str(next_idx)
                res = boxvar + ' ' + op + ' ' + boxarg
                idx, clfs = convert_to_clf(next_idx, clfs, arg)
            else:
                out = check_constant_and_add_quotes(str(arg))
                res = boxvar + ' ' + op + ' ' + out
                # adding dummy synset
                # res = boxvar + ' ' + op + ' "n.01" ' + out
        if len(args) == 2:
            arg1 = args[0]
            arg2 = args[1]
            if isinstance(arg2, DRS):
                out = check_constant_and_add_quotes(str(arg1))
                next_idx = idx + 1
                box_arg = 'b' + str(next_idx)
                res = boxvar + ' ' + op + ' ' + out + ' ' + box_arg
                idx, clfs = convert_to_clf(next_idx, clfs, arg2)
            else:
                out1 = check_constant_and_add_quotes(str(arg1))
                out2 = check_constant_and_add_quotes(str(arg2))
                # if op in roles:
                res = 'b' + str(idx) + ' ' + op + ' ' + out1 + ' ' + out2
        clfs.append(res)
        return idx, clfs
    elif isinstance(drs, DrtNegatedExpression):
        boxvar = 'b' + str(idx)
        idx = idx + 1
        drs_var = 'b' + str(idx)
        neg = boxvar + ' NOT ' + drs_var
        clfs.append(neg)
        term = drs.term
        idx, clfs = convert_to_clf(idx, clfs, term)
        return idx, clfs
    elif isinstance(drs, DrtOrExpression):
        head = len(clfs)
        boxvar = 'b' + str(idx)
        idx = idx + 1
        boxarg1 = 'b' + str(idx)
        idx, clfs = convert_to_clf(idx, clfs, drs.first)
        idx = idx + 1
        boxarg2 = 'b' + str(idx)
        idx, clfs = convert_to_clf(idx, clfs, drs.second)
        res = boxvar + ' DIS ' + boxarg1 + ' ' + boxarg2
        clfs.insert(head, res)
        return idx, clfs
#     elif isinstance(drs, DrtLambdaExpression):
#     elif isinstance(drs, DrtEqualityExpression):
    else:
        return idx, str(drs)
    return idx, clfs
