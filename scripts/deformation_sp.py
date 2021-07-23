
import argparse
import pandas as pd
from nltk2normal import rename
from nltk2drs import convert_to_drs
from nltk2pol import calculate_polarity
from drs2clf import convert_to_clausal_forms
from logic_parser import lexpr


def conv_to_nltk(expr):
    try:
        formula = rename(lexpr(expr.lower()))
    except AttributeError as error:
        print('{0}: {1}'.format(error, expr))
        formula = ""
    return formula


def conv_to_clf(formula):
    try:
        drs = convert_to_drs(formula)
        clauses = convert_to_clausal_forms(drs)
        clf = ' [SEP] '.join(clauses)
    except AttributeError as error:
        print('{0}: {1}'.format(error, formula))
        formula = ""
    return clf


def assign_fol_pol(formula):
    try:
        res = calculate_polarity(formula)
    except AttributeError as error:
        print('{0}: {1}'.format(error, formula))
        res = ""
    return res


def main():
    parser = argparse.ArgumentParser('')
    parser.add_argument('FILE')
    args = parser.parse_args()

    tsv_file = args.FILE
    df = pd.read_csv(tsv_file, sep="\t")

    formulas = [conv_to_nltk(f) for f in df['fol']]
    df['fol'] = formulas

    clfs = [conv_to_clf(f) for f in df['fol']]
    df['drs'] = clfs

    pols = [assign_fol_pol(f) for f in df['fol']]
    df.insert(4, 'folpol', pols)

    fout = tsv_file.replace('.tmp', '')
    df.to_csv(fout, sep="\t", index=False)


if __name__ == '__main__':
    main()
