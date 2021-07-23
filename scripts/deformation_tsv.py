
import argparse
import sys

from nltk2drs import convert_to_drs
from nltk2tptp import convert_to_tptp
from drs2clf import convert_to_clausal_forms
from nltk2pol import calculate_polarity
from nltk2normal import rename
from logic_parser import lexpr
import glob

import pandas as pd
import re


def extract_formula(expr):
    expr = expr.lower()
    formula = lexpr(expr)
    return formula

def compute_f(match_num, test_num, gold_num):
    if test_num == 0 or gold_num == 0:
            return 0.00, 0.00, 0.00
    precision = round(float(match_num) / float(test_num), 2)
    recall = round(float(match_num) / float(gold_num), 2)
    if precision < 0.0 or precision > 1.0 or recall < 0.0 or recall > 1.0:
        raise ValueError("Precision and recall should never be outside (0.0-1.0), now {0} and {1}".format(precision, recall))

    if (precision + recall) != 0:
        f_score = round(2 * precision * recall / (precision + recall), 2)
        return precision, recall, f_score
    else:
        return precision, recall, 0.00

def format_clf(idx, genre, df, folder):
    with open(folder+"/pred_"+str(idx)+"_"+str(genre)+".clf", "w") as pp, open(folder+"/gold_"+str(idx)+"_"+str(genre)+".clf", "w") as pg:
        for idx, row in df.iterrows():
            clauses = [], "", ""
            try:
                pp.write("%%% "+row["phenomena_tags"]+" "+row["sentence"]+" "+row["sentence_fol_pred"]+"\n")
                clauses = row["sentence_fol_pred"].split(" [SEP] ")
                for clause in clauses:
                    pp.write(clause+"\n")
            except Exception as error:
                pp.write("%%% "+row["phenomena_tags"]+" "+row["sentence"]+" "+row["sentence_fol_pred"]+" Error: "+str(error)+"\n")
            pp.write("\n")
            clauses, formula, drs = [], "", ""
            try:
                pg.write("%%% "+row["phenomena_tags"]+" "+row["sentence"]+" "+row["sentence_fol_gold"]+"\n")
                clauses = row["sentence_fol_gold"].split(" [SEP] ")
                for clause in clauses:
                    pg.write(clause+"\n")
            except Exception as error:
                pg.write("%%% "+row["phenomena_tags"]+" "+row["sentence"]+" "+row["sentence_fol_gold"]+" Error: "+str(error)+"\n")
            pg.write("\n")

def phenomena(test, pheno):
    if pheno == "adj":
        return test.query('phenomena_tags.str.contains("adjective:yes") and \
                               phenomena_tags.str.contains("adverb:no") and \
                               phenomena_tags.str.contains("conjunction:no") and \
                               phenomena_tags.str.contains("disjunction:no") and \
                               phenomena_tags.str.contains("negation:no")')
    elif pheno == "adj_neg":
        return test.query('phenomena_tags.str.contains("adjective:yes") and \
                                   phenomena_tags.str.contains("adverb:no") and \
                                   phenomena_tags.str.contains("conjunction:no") and \
                                   phenomena_tags.str.contains("disjunction:no") and \
                                   phenomena_tags.str.contains("negation:yes")')
    elif pheno == "adv":
        return test.query('phenomena_tags.str.contains("adjective:no") and \
                               phenomena_tags.str.contains("adverb:yes") and \
                               phenomena_tags.str.contains("conjunction:no") and \
                               phenomena_tags.str.contains("disjunction:no") and \
                               phenomena_tags.str.contains("negation:no")')
    elif pheno == "adv_neg":
        return test.query('phenomena_tags.str.contains("adjective:no") and \
                                   phenomena_tags.str.contains("adverb:yes") and \
                                   phenomena_tags.str.contains("conjunction:no") and \
                                   phenomena_tags.str.contains("disjunction:no") and \
                                   phenomena_tags.str.contains("negation:yes")')
    elif pheno == "conj_disj":
        return test.query('phenomena_tags.str.contains("adjective:no") and \
                                phenomena_tags.str.contains("adverb:no") and \
                                phenomena_tags.str.contains("negation:no")')
    elif pheno == "conj_disj_neg":
        return test.query('phenomena_tags.str.contains("adjective:no") and \
                                    phenomena_tags.str.contains("adverb:no") and \
                                    phenomena_tags.str.contains("negation:yes")')

def main(args = None):
    parser = argparse.ArgumentParser('')
    parser.add_argument("-i", "--input", type=str, nargs="?", help="Input tsv file")
    parser.add_argument("-f", "--format", type=str, default="drs",
                        choices=["normal", "plnormal", "drs", "drsbox", "clf",
                                 "coq", "tptp", "pol", "pol_eval", "clf_eval"],
                        help="Output format (default: drs).")
    args = parser.parse_args()

    if args.format == "pol_eval":
        df = pd.read_csv(args.input, sep="\t")
        output = re.sub(".tsv", "_polacc.tsv", args.input)
        df_ps = df.sentence
        df_pg = df.sentence_fol_gold
        df_pp = df.sentence_fol_pred
        df_ph = df.phenomena_tags
        tprec, trec, tf1, pgolds, ppreds = [], [], [], [], []
        for idx in range(df.shape[0]):
            pol_pgold, pol_ppred, pgold, ppred = "", "", "", ""
            try:
                pgold = extract_formula(str(df_pg.iloc[idx]))
            except:
                pgold = ""
            pol_pgold = calculate_polarity(pgold)
            pgolds.append(pol_pgold)
            try:
                ppred = extract_formula(str(df_pp.iloc[idx]))
            except:
                ppred = ""
            pol_ppred = calculate_polarity(ppred)
            ppreds.append(pol_ppred)
            pmatch = set(pol_ppred) & set(pol_pgold)
            prec, rec, f1 = compute_f(len(pmatch), len(pol_ppred), len(pol_pgold))
            tprec.append(prec)
            trec.append(rec)
            tf1.append(f1)
        aprec = sum(tprec) / len(tprec)
        arec = sum(trec) / len(trec)
        af1 = sum(tf1) / len(tf1)
        with open(output, "w") as f:
            f.write("id\tprec\trec\tf1\tgold\tpred\tphenomena_tags\n")
            idx = 1
            for pr, tr, fs, pg, pp, ph in zip(tprec, trec, tf1, pgolds, ppreds, df_ph.values.tolist()):
                f.write("{0}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4}\t{5}\t{6}".format(idx, pr, tr, fs, str(pg), str(pp), ph))
                f.write("\n")
                idx+=1
            f.write("average: {0:.2f}\t{1:.2f}\t{2:.2f}".format(aprec, arec, af1))
    if args.format == "pol_free_eval":
        df = pd.read_csv(args.input, sep="\t")
        output = re.sub(".tsv", "_polacc.tsv", args.input)
        df_ps = df.sentence
        df_pg = df.sentence_fol_gold
        df_pp = df.sentence_fol_pred
        df_ph = df.phenomena_tags
        tprec, trec, tf1, pgolds, ppreds = [], [], [], [], []
        for idx in range(df.shape[0]):
            pol_pgold, pol_ppred, pgold, ppred = "", "", "", ""
            try:
                pgold = extract_formula(str(df_pg.iloc[idx]))
            except:
                pgold = ""
            pol_pgold = calculate_polarity(pgold)
            pgolds.append(pol_pgold)
            try:
                ppred = extract_formula(str(df_pp.iloc[idx]))
            except:
                ppred = ""
            pol_ppred = calculate_polarity(ppred)
            ppreds.append(pol_ppred)
            pmatch = set(pol_ppred) & set(pol_pgold)
            prec, rec, f1 = compute_f(len(pmatch), len(pol_ppred), len(pol_pgold))
            tprec.append(prec)
            trec.append(rec)
            tf1.append(f1)
        aprec = sum(tprec) / len(tprec)
        arec = sum(trec) / len(trec)
        af1 = sum(tf1) / len(tf1)
        with open(output, "w") as f:
            f.write("id\tprec\trec\tf1\tgold\tpred\tphenomena_tags\n")
            idx = 1
            for pr, tr, fs, pg, pp, ph in zip(tprec, trec, tf1, pgolds, ppreds, df_ph.values.tolist()):
                f.write("{0}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4}\t{5}\t{6}".format(idx, pr, tr, fs, str(pg), str(pp), ph))
                f.write("\n")
                idx+=1
            f.write("average: {0:.2f}\t{1:.2f}\t{2:.2f}".format(aprec, arec, af1))

    if args.format == "tptp":
        df = pd.read_csv(args.input, sep="\t")
        output = re.sub(".tsv", "_proof.tsv", args.input)
        df_ps = df.sentence_fol_gold
        df_hs = df.sentence_fol_pred
        new_df = df.copy()
        for idx in range(df.shape[0]):
            try:
                formula = extract_formula(str(df_ps.iloc[idx]))
                new_df["sentence_fol_gold"][idx] = convert_to_tptp(formula)
                formula = extract_formula(str(df_hs.iloc[idx]))
                new_df["sentence_fol_pred"][idx] = convert_to_tptp(formula)
            except:
                new_df["sentence_fol_gold"][idx] = ""
                new_df["sentence_fol_pred"][idx] = ""
        new_df.to_csv(output, sep="\t", index=False)

    if args.format == "clf_eval":
        results = glob.glob(args.input+"/prediction*eval[1-5].tsv")
        folder = args.input
        phenos = ["adj", "adj_neg", "adv", "adv_neg", "conj_disj", "conj_disj_neg"]
        #phenos = ["adj_neg", "adv_neg", "conj_disj_neg"]
        for i, res in enumerate(results):
            d = pd.read_csv(res, sep="\t")
            for pheno in phenos:
                tsv = phenomena(d, pheno)
                format_clf(i+1, pheno, tsv, folder)
            # exi = d.query('sentence.str.contains("one ") or sentence.str.contains("a ")')
            # format_clf(i+1, "exi", exi, folder)
            # num = d.query('sentence.str.contains("two ") or sentence.str.contains("three ")')
            # format_clf(i+1, "num", num, folder)
            # uni = d.query('sentence.str.contains("every ") or sentence.str.startswith("all ") or sentence.str.contains(" all ")')
            # format_clf(i+1, "uni", uni, folder)
            # for j in range(1,5):
            #     tsv = d.query("depth==@j")
            #     format_clf(i+1, j, tsv, folder)




if __name__ == '__main__':
    main()
