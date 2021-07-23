# -*- coding: utf-8 -*-

import pandas as pd
import itertools
import argparse

parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--outdir", nargs='?', type=str, default="tmp_results", help="input output directory")
parser.add_argument("--input", nargs='?', type=str, default="results.tsv", help="input tsv file")
parser.add_argument("--format", nargs='?', type=str, default="all", help="formula format")
parser.add_argument("--setting", nargs='?', type=str, default="comp", help="test setting")
args = parser.parse_args()

df = pd.read_csv(args.outdir+"/"+args.input, sep="\t")
#print("total {0} examples".format(len(df)))
if args.setting == "comp":
    #todo: change query setting according to the primitive quantifier
    train1 = df.query('depth==0 and \
               sentence.str.contains("one ")')
    train2 = df.query('depth==0 and \
               not sentence.str.contains("one ") and \
               tags.str.contains("adjective:no") and \
               tags.str.contains("adverb:no") and \
               tags.str.contains("conjunction:no") and \
               tags.str.contains("disjunction:no")')
    test = df.query('depth==0 and \
             not sentence.str.contains("one ") and \
             (tags.str.contains("adjective:yes") or \
              tags.str.contains("adverb:yes") or \
              tags.str.contains("conjunction:yes") or \
              tags.str.contains("disjunction:yes"))')
    tr = pd.concat([train1, train2], axis=0)
    train = tr.sample(frac=1)
    
    if args.format == "fol":
        train.drop(['vf', 'drs', 'vfpol', 'folpol'], axis=1).to_csv(args.outdir+"/"+"train_fol.tsv", sep="\t", index=False)
        test.drop(['vf', 'drs', 'vfpol', 'folpol'], axis=1).to_csv(args.outdir+"/"+"test_fol.tsv", sep="\t", index=False)
    elif args.format == "free":
        train.drop(['fol', 'drs', 'vfpol', 'folpol'], axis=1).rename(columns={'vf':'fol'}).to_csv(args.outdir+"/"+"train_free.tsv", sep="\t", index=False)
        test.drop(['fol', 'drs', 'vfpol', 'folpol'], axis=1).rename(columns={'vf':'fol'}).to_csv(args.outdir+"/"+"test_free.tsv", sep="\t", index=False)
    elif args.format == "drs":
        train.drop(['fol', 'vf', 'vfpol', 'folpol'], axis=1).rename(columns={'drs':'fol'}).to_csv(args.outdir+"/"+"train_drs.tsv", sep="\t", index=False)
        test.drop(['fol', 'vf', 'vfpol', 'folpol'], axis=1).rename(columns={'drs':'fol'}).to_csv(args.outdir+"/"+"test_drs.tsv", sep="\t", index=False)
    elif args.format == "all":
        train.drop(['vf', 'drs', 'vfpol', 'folpol'], axis=1).to_csv(args.outdir+"/"+"train_fol.tsv", sep="\t", index=False)
        test.drop(['vf', 'drs', 'vfpol', 'folpol'], axis=1).to_csv(args.outdir+"/"+"test_fol.tsv", sep="\t", index=False)
        train.drop(['fol', 'drs', 'vfpol', 'folpol'], axis=1).rename(columns={'vf':'fol'}).to_csv(args.outdir+"/"+"train_free.tsv", sep="\t", index=False)
        test.drop(['fol', 'drs', 'vfpol', 'folpol'], axis=1).rename(columns={'vf':'fol'}).to_csv(args.outdir+"/"+"test_free.tsv", sep="\t", index=False)
        train.drop(['fol', 'vf', 'vfpol', 'folpol'], axis=1).rename(columns={'drs':'fol'}).to_csv(args.outdir+"/"+"train_drs.tsv", sep="\t", index=False)
        test.drop(['fol', 'vf', 'vfpol', 'folpol'], axis=1).rename(columns={'drs':'fol'}).to_csv(args.outdir+"/"+"test_drs.tsv", sep="\t", index=False)
        train_auto =pd.DataFrame(index=[], columns=["id", "depth", "auto_sent1", "auto_sent2", "auto_sem1", "auto_sem2", "tags"])
        train_auto["id"] = train["id"]
        train_auto["depth"] = train["depth"]
        train_auto["auto_sent1"] = train["sentence"]
        train_auto["auto_sent2"] = train["sentence"]
        train_auto["auto_sem1"] = train["fol"]
        train_auto["auto_sem2"] = train["fol"]
        train_auto["tags"] = train["tags"]
        train_auto.to_csv(args.outdir+"/"+"train_auto.tsv", sep="\t", index=False)

elif args.setting == "depth":
    depth0 = df.query('depth==0', engine='python')
    depth1 = df.query('depth==1', engine='python')
    depth2 = df.query('depth==2', engine='python')
    depth3 = df.query('depth==3', engine='python')
    depth4 = df.query('depth==4', engine='python')
    tr = pd.concat([depth0, depth1], axis=0)
    train = tr.sample(frac=1)
    test = pd.concat([depth2, depth3, depth4], axis=0)
    if args.format == "fol":
        train.dropna(subset=['fol']).drop(['vf', 'drs', 'vfpol', 'folpol'], axis=1).to_csv(args.outdir+"/"+"train_fol.tsv", sep="\t", index=False)
        test.dropna(subset=['fol']).drop(['vf', 'drs', 'vfpol', 'folpol'], axis=1).to_csv(args.outdir+"/"+"test_fol.tsv", sep="\t", index=False)
    elif args.format == "free":
        train.dropna(subset=['fol']).drop(['fol', 'drs', 'vfpol', 'folpol'], axis=1).rename(columns={'vf':'fol'}).to_csv(args.outdir+"/"+"train_free.tsv", sep="\t", index=False)
        test.dropna(subset=['fol']).drop(['fol', 'drs', 'vfpol', 'folpol'], axis=1).rename(columns={'vf':'fol'}).to_csv(args.outdir+"/"+"test_free.tsv", sep="\t", index=False)
    elif args.format == "drs":
        train.dropna(subset=['fol']).drop(['fol', 'vf', 'vfpol', 'folpol'], axis=1).rename(columns={'drs':'fol'}).to_csv(args.outdir+"/"+"train_drs.tsv", sep="\t", index=False)
        test.dropna(subset=['fol']).drop(['fol', 'vf', 'vfpol', 'folpol'], axis=1).rename(columns={'drs':'fol'}).to_csv(args.outdir+"/"+"test_drs.tsv", sep="\t", index=False)
    elif args.format == "all":
        train.dropna(subset=['fol']).drop(['vf', 'drs', 'vfpol', 'folpol'], axis=1).to_csv(args.outdir+"/"+"train_fol.tsv", sep="\t", index=False)
        test.dropna(subset=['fol']).drop(['vf', 'drs', 'vfpol', 'folpol'], axis=1).to_csv(args.outdir+"/"+"test_fol.tsv", sep="\t", index=False)
        train.dropna(subset=['fol']).drop(['fol', 'drs', 'vfpol', 'folpol'], axis=1).rename(columns={'vf':'fol'}).to_csv(args.outdir+"/"+"train_free.tsv", sep="\t", index=False)
        test.dropna(subset=['fol']).drop(['fol', 'drs', 'vfpol', 'folpol'], axis=1).rename(columns={'vf':'fol'}).to_csv(args.outdir+"/"+"test_free.tsv", sep="\t", index=False)
        train.dropna(subset=['fol']).drop(['fol', 'vf', 'vfpol', 'folpol'], axis=1).rename(columns={'drs':'fol'}).to_csv(args.outdir+"/"+"train_drs.tsv", sep="\t", index=False)
        test.dropna(subset=['fol']).drop(['fol', 'vf', 'vfpol', 'folpol'], axis=1).rename(columns={'drs':'fol'}).to_csv(args.outdir+"/"+"test_drs.tsv", sep="\t", index=False)
        train_auto =pd.DataFrame(index=[], columns=["id", "depth", "auto_sent1", "auto_sent2", "auto_sem1", "auto_sem2", "tags"])
        train_auto["id"] = train["id"]
        train_auto["depth"] = train["depth"]
        train_auto["auto_sent1"] = train["sentence"]
        train_auto["auto_sent2"] = train["sentence"]
        train_auto["auto_sem1"] = train["fol"]
        train_auto["auto_sem2"] = train["fol"]
        train_auto["tags"] = train["tags"]
        train_auto.dropna(subset=['auto_sem1']).to_csv(args.outdir+"/"+"train_auto.tsv", sep="\t", index=False)

