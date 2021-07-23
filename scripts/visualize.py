# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from statistics import stdev
import argparse
import glob
import matplotlib.pyplot as plt
import re
parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--outdir", nargs='?', type=str, help="output dir")
parser.add_argument("--format", nargs='?', type=str, default="fol", help="format")
args = parser.parse_args()

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def main():
    train_sizes = []
    exi_test_scores_mean, uni_test_scores_mean, num_test_scores_mean = [], [], []
    exi_test_scores_std, uni_test_scores_std, num_test_scores_std = [], [], []
    txts = glob.glob(args.outdir+"/*/quant.txt")
    for txt in sorted(txts, key=natural_keys):
        #print(txt)
        size = re.search("/([0-9]*)/quant.txt", txt).group(1)
        with open(txt, "r") as f:
            accs = f.readlines()
        uniscore, unistd = accs[2].split("\t")
        exiscore, existd = accs[0].split("\t")
        numscore, numstd = accs[1].split("\t")
        uni_test_scores_mean.append(float(uniscore))
        uni_test_scores_std.append(float(unistd))
        exi_test_scores_mean.append(float(exiscore))
        exi_test_scores_std.append(float(existd))
        num_test_scores_mean.append(float(numscore))
        num_test_scores_std.append(float(numstd))
        train_sizes.append(int(size))
        
    plt.rcParams["font.size"] = 18
    plt.figure(figsize=(5.0, 4.0))
    plt.title("Quantifiers (Transformer)")
    plt.xlabel("Training examples")
    #plt.ylabel("Accuracy")
    cmap2 = plt.get_cmap("plasma")
    #plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(np.array(train_sizes), np.array(exi_test_scores_mean), 'o-', color=cmap2(float(1)/3), label="Existential")
    plt.plot(np.array(train_sizes), np.array(uni_test_scores_mean), 'o-', color=cmap2(float(2)/3), label="Universal")
    plt.plot(np.array(train_sizes), np.array(num_test_scores_mean), 'o-', color=cmap2(float(3)/3), label="Numeral")
    #plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color="r", alpha=0.2)
    plt.fill_between(np.array(train_sizes), np.array(exi_test_scores_mean) - np.array(exi_test_scores_std), np.array(exi_test_scores_mean) + np.array(exi_test_scores_std), color=cmap2(float(1)/3), alpha=0.2)
    plt.fill_between(np.array(train_sizes), np.array(uni_test_scores_mean) - np.array(uni_test_scores_std), np.array(uni_test_scores_mean) + np.array(uni_test_scores_std), color=cmap2(float(2)/3), alpha=0.2)
    plt.fill_between(np.array(train_sizes), np.array(num_test_scores_mean) - np.array(num_test_scores_std), np.array(num_test_scores_mean) + np.array(num_test_scores_std), color=cmap2(float(3)/3), alpha=0.2)

    plt.xlim(train_sizes[0], train_sizes[-1])
    plt.ylim(0, 100)
    #plt.legend(loc="best", markerscale=1, handlelength=0.7).get_frame().set_alpha(0.3)
    plt.savefig(args.outdir+'/lc_quant.png', bbox_inches='tight', dpi=150)

    train_sizes = []
    adj_mean, adjneg_mean, adv_mean, advneg_mean, conj_mean, conjneg_mean = [], [], [], [], [], []
    adj_std, adjneg_std, adv_std, advneg_std, conj_std, conjneg_std = [], [], [], [], [], []
    txts = glob.glob(args.outdir+"/*/pheno.txt")
    for txt in sorted(txts, key=natural_keys):
        #print(txt)
        size = re.search("/([0-9]*)/pheno.txt", txt).group(1)
        with open(txt, "r") as f:
            accs = f.readlines()

        adj_mean.append(float(accs[0].split("\t")[0]))
        adj_std.append(float(accs[0].split("\t")[1]))
        adjneg_mean.append(float(accs[1].split("\t")[0]))
        adjneg_std.append(float(accs[1].split("\t")[1]))
        adv_mean.append(float(accs[2].split("\t")[0]))
        adv_std.append(float(accs[2].split("\t")[1]))
        advneg_mean.append(float(accs[3].split("\t")[0]))
        advneg_std.append(float(accs[3].split("\t")[1]))
        conj_mean.append(float(accs[4].split("\t")[0]))
        conj_std.append(float(accs[4].split("\t")[1]))
        conjneg_mean.append(float(accs[5].split("\t")[0]))
        conjneg_std.append(float(accs[5].split("\t")[1]))
        train_sizes.append(int(size))
        
        
    plt.figure(figsize=(5.0, 4.0))
    plt.title("Modifiers (Transformer)")
    plt.xlabel("Training examples")
    #plt.ylabel("Accuracy")
    cmap = plt.get_cmap("viridis")
    plt.plot(np.array(train_sizes), np.array(adj_mean), 'o-', c=cmap(float(1)/6), label="Adj")
    plt.plot(np.array(train_sizes), np.array(adjneg_mean), 'o-', c=cmap(float(2)/6), label="Adj+Neg")
    plt.plot(np.array(train_sizes), np.array(adv_mean), 'o-', c=cmap(float(3)/6), label="Adv")
    plt.plot(np.array(train_sizes), np.array(advneg_mean), 'o-', c=cmap(float(4)/6), label="Adv+Neg")
    plt.plot(np.array(train_sizes), np.array(conj_mean), 'o-', c=cmap(float(5)/6), label="Con")
    plt.plot(np.array(train_sizes), np.array(conjneg_mean), 'o-', c=cmap(float(6)/6), label="Con+Neg")
    plt.fill_between(np.array(train_sizes), np.array(adj_mean) - np.array(adj_std), np.array(adj_mean) + np.array(adj_std), color=cmap(float(1)/6), alpha=0.2)
    plt.fill_between(np.array(train_sizes), np.array(adjneg_mean) - np.array(adjneg_std), np.array(adjneg_mean) + np.array(adjneg_std), color=cmap(float(2)/6), alpha=0.2)
    plt.fill_between(np.array(train_sizes), np.array(adv_mean) - np.array(adv_std), np.array(adv_mean) + np.array(adv_std), color=cmap(float(3)/6), alpha=0.2)
    plt.fill_between(np.array(train_sizes), np.array(advneg_mean) - np.array(advneg_std), np.array(advneg_mean) + np.array(advneg_std), color=cmap(float(4)/6), alpha=0.2)
    plt.fill_between(np.array(train_sizes), np.array(conj_mean) - np.array(conj_std), np.array(conj_mean) + np.array(conj_std), color=cmap(float(5)/6), alpha=0.2)
    plt.fill_between(np.array(train_sizes), np.array(conjneg_mean) - np.array(conjneg_std), np.array(conjneg_mean) + np.array(conjneg_std), color=cmap(float(6)/6), alpha=0.2)

    plt.xlim(train_sizes[0], train_sizes[-1])
    plt.ylim(0, 100)
    #plt.legend(loc="best", ncol=2, markerscale=1, handlelength=0.7).get_frame().set_alpha(0.3)
    plt.savefig(args.outdir+'/lc_pheno.png', bbox_inches='tight', dpi=150)

if __name__ == '__main__':
    main()
