# SyGNS
- repository for our ACL-IJCNLP2021 paper "SyGNS: A Systematic Generalization Testbed Based on Natural Language Semantics"

## Install Tools
```
$ sudo apt-get -y install swi-prolog
$ ./install.sh
```

## Dataset Creation
```
$ cd semparse
$ ./run_sp.sh 50000 50000 tmp
```
`tmp` is a sample directory for generated schemes and `tmp_results` is a sample directory for generated data.

## Train and Test for Systematicity (exact matching)
format: fol (FOL formulas), clf (DRS clausal forms), free (Variable-free formulas)
```
$ python split_traintest.py --setting "comp"
$ python seq2fol_transformer.py --format="fol" --train=$DATA_DIR/train_fol.tsv --test=$DATA_DIR/test_fol.tsv --output_dir=results --maxlength=100
$ python evaluate.py --outdir results --format="fol"
```

## Train and Test for Productivity (exact matching)
format: fol (FOL formulas), clf (DRS clausal forms), free (Variable-free formulas)
```
$ python split_traintest.py --setting "depth"
$ python seq2fol_transformer.py --format="fol" --train=$DATA_DIR/train_fol.tsv --test=$DATA_DIR/test_fol.tsv --output_dir=results --maxlength=300
$ python evaluate.py --outdir results --setting depth --format="fol"
```

## Evaluation (Proof by vampire)
- input 1: tsv file
```
$ cd semparse
$ python deformation_tsv.py --input results/prediction.tsv --format tptp
$ ./eval_proof.sh results/prediction.tsv tptp results
```

## Evaluation (Compute polarity)
- input 1: tsv file
```
$ cd semparse
$ python deformation_tsv.py --input results/prediction.tsv --format pol_eval
$ python evaluate.py --outdir results --setting polarity
```

## Evaluation (DRS F-score)
- input: input/output folder
```
$ cd semparse
$ ./fol2drs_for_eval.sh results
```

## Citation
If you use this code in any published research, please cite the following:
* Hitomi Yanaka, Koji Mineshima, Kentaro Inui, [SyGNS: A Systematic Generalization Testbed Based on Natural Language Semantics]() [arXiv](), Findings of the Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP2021), 2021.

```
@InProceedings{yanaka-EtAl:2021:acl,
  author    = {Yanaka, Hitomi and Mineshima, Koji and Inui, Kentaro},
  title     = {SyGNS: A Systematic Generalization Testbed Based on Natural Language Semantics},
  booktitle = {Findings of the Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP2021)},
  year      = {2021},
}
```

## Contact
For questions and usage issues, please contact hyanaka@is.s.u-tokyo.ac.jp .

## License
MIT License