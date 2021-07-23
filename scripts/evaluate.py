# -*- coding: utf-8 -*-
import pandas as pd
from statistics import stdev
import argparse
import glob

parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--outdir", nargs='?', type=str, help="output dir")
parser.add_argument("--setting", nargs='?', type=str, default=None, help="setting")
parser.add_argument("--format", nargs='?', type=str, default="fol", help="format")
args = parser.parse_args()

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

def form_tags(a):
    fir = []
    sec = []
    for i, tmp in enumerate(a):
        if i%2 == 0:
            fir.append(tmp.replace(' ', ''))
        else:
            fir.append(tmp.replace(' ', ''))
            sec.append(tuple(fir))
            fir = []
    return sec

def select_elem(taga, tagb, pol="1"):
    newtaga, newtagb, words = [], [], []
    for tmp in taga:
        if tmp[1] == pol:
            newtaga.append(tmp)
            words.append(tmp[0])
    for tmp in tagb:
        if tmp[0] in words:
            newtagb.append(tmp)
    return newtaga, newtagb

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

def main():
    tsvs = glob.glob(args.outdir+"/prediction_eval[1-5].tsv")
    phenos = ["adj", "adj_neg", "adv", "adv_neg", "conj_disj", "conj_disj_neg"]
    uniave, exiave, numave = [], [], []
    for pheno in phenos:
        print(f'phenomena: {pheno}')
        totave = []
        for tsv in tsvs:
            totacc = 0 
            tmptsv = pd.read_csv(tsv, sep="\t")
            etsv = phenomena(tmptsv, pheno)
            if args.format == "fol" or args.format == "free" or args.format == "clf":
                totacc = len(etsv[etsv['sentence_fol_gold'] == etsv['sentence_fol_pred']])/len(etsv)
            else:
                totacc = len(etsv[etsv['sentence'] == etsv['pred']])/len(etsv)
            totave.append(totacc)
        print(f'total ave: {sum(totave)/len(totave)*100:.1f}, stdev: {stdev(totave):.1f}')
        with open(args.outdir+"/pheno.txt", "a") as f:
            f.write(f'{sum(totave)/len(totave)*100:.1f}')
            f.write('\n')
    for tsv in tsvs:
        uniacc, exiacc, numacc = 0, 0, 0
        etsv = pd.read_csv(tsv, sep="\t")
        uni = etsv.query('sentence.str.contains("every ") or sentence.str.startswith("all ") or sentence.str.contains(" all ")')
        exi = etsv.query('sentence.str.contains("one ") or sentence.str.contains("a ")')
        num = etsv.query('sentence.str.contains("two ") or sentence.str.contains("three ")')
        if args.format == "fol" or args.format == "free" or args.format == "clf":
            uniacc = len(uni[uni['sentence_fol_gold'] == uni['sentence_fol_pred']])/len(uni)
            exiacc = len(exi[exi['sentence_fol_gold'] == exi['sentence_fol_pred']])/len(exi)
            numacc = len(num[num['sentence_fol_gold'] == num['sentence_fol_pred']])/len(num)
        else:
            uniacc = len(uni[uni['sentence'] == uni['pred']])/len(uni)
            exiacc = len(exi[exi['sentence'] == exi['pred']])/len(exi)
            numacc = len(num[num['sentence'] == num['pred']])/len(num)
        uniave.append(uniacc)
        exiave.append(exiacc)
        numave.append(numacc)
    print(f'existential ave: {sum(exiave)/len(exiave)*100:.1f}, stdev: {stdev(exiave):.1f}')
    print(f'numeral ave: {sum(numave)/len(numave)*100:.1f}, stdev: {stdev(numave):.1f}')
    print(f'universal ave: {sum(uniave)/len(uniave)*100:.1f}, stdev: {stdev(uniave):.1f}')
    with open(args.outdir+"/quant.txt", "w") as f:
        f.write(f'{sum(exiave)/len(exiave)*100:.1f}')
        f.write('\n')
        f.write(f'{sum(numave)/len(numave)*100:.1f}')
        f.write('\n')
        f.write(f'{sum(uniave)/len(uniave)*100:.1f}')
        f.write('\n')


    if args.setting == "depth":
        tsvs = glob.glob(outdir+"/"+setting+"/prediction_eval[1-5].tsv")
        for i in range(2,5):
            totave = []
            print(i)
            for tsv in tsvs:
                print(tsv)
                tmp = pd.read_csv(tsv, sep="\t")
                etsv = tmp.query("depth==@i")
                if args.format == "fol" or args.format == "free" or args.format == "clf":
                    totacc = len(etsv[etsv['sentence_fol_gold'] == etsv['sentence_fol_pred']])/len(etsv)
                else:
                    totacc = len(etsv[etsv['sentence'] == etsv['pred']])/len(etsv)
                totave.append(totacc)
            print(f'total ave: {sum(totave)/len(totave)*100:.2f}, stdev: {stdev(totave):.2f}')

    if setting == "polarity":
        tsvs = glob.glob(outdir+"/"+setting+"/prediction_eval[1-5]_polacc.tsv")
        aveprec, averec, avef1 = [], [], []
        for tsv in tsvs:
            tprec, trec, tf1 = [], [], []
            for idx, row in tsv.iterrows():
                a = re.sub("\[|\]|\(|\)|\'", "", row['gold']).split(",")
                taga = form_tags(a)
                b = re.sub("\[|\]|\(|\)|\'", "", row['pred']).split(",")
                tagb = form_tags(b)
                taga, tagb = select_elem(taga, tagb, "0")
                if len(taga) == 0:
                    continue
                pmatch = set(tagb) & set(taga)
                prec, rec, f1 = compute_f(len(pmatch), len(tagb), len(taga))
                tprec.append(prec)
                trec.append(rec)
                tf1.append(f1)
            aprec = sum(tprec)*100 / len(tprec)
            arec = sum(trec)*100 / len(trec)
            af1 = sum(tf1)*100 / len(tf1)
            aveprec.append(aprec)
            averec.append(arec)
            avef1.append(af1)
        print(f'{sum(aveprec)/len(aveprec)*100:.2f}, {sum(averec)/len(averec)*100:.2f}, {sum(avef1)/len(avef1)*100:.2f}')

if __name__ == '__main__':
    main()
