
import yaml
import glob
import os.path
import argparse
import itertools
import math
import random
import collections


def load_vocab(file):
    with open(file, 'r', encoding='utf-8') as infile:
        loaded = yaml.load(infile, Loader=yaml.SafeLoader)
    if not loaded:
        raise ValueError("couldn't load file: " + file)
    return loaded


def schematize(sentence, cat):
    count = 0
    sent = []
    for word in sentence.split(' '):
        if word == cat:
            var = '{0[' + str(count) + ']}'
            word = word.replace(cat, var)
            count += 1
        sent.append(word)
    out = ' '.join(sent)
    return out


def instantiate(cat, items, sentences, depth):
    output = []
    for sentence in sentences:
        num_occur = sentence.count(cat)
        if num_occur == 0:
            output.append(sentence)
        else:
            if len(items) < num_occur:
                items = items * num_occur
                perms = list(itertools.permutations(items, num_occur))
            else:
                perms = list(itertools.permutations(items, num_occur))

            if depth == 'depth0':
                rate = 4 / 5
            elif depth == 'depth1':
                rate = 1 / 5
            elif depth == 'depth2':
                rate = 1 / 35
            elif depth == 'depth3':
                rate = 1 / 700
            elif depth == 'depth4':
                rate = 1 / 3000
            else:
                rate = 1 / len(perms)

            par = math.ceil(len(perms) * rate)
            perms = random.sample(perms, k=par)
            for perm in perms:
                res = schematize(sentence, cat).format(perm)
                output.append(res)
    return output


def perm_count(n, m):
    return math.factorial(n) // math.factorial(n - m)


def estimate(sentences, vocabs):
    estimated_num = 0
    data = collections.Counter()
    for sentence in sentences:
        sum = 1
        for vocab in vocabs:
            cat = vocab['category']
            occur = sentence.count(cat)
            size = len(vocab['surf'])
            sum = sum * pow(size, occur)
            data.update([(cat, occur)])
        estimated_num += sum
    return estimated_num, data


def main():
    parser = argparse.ArgumentParser('')
    parser.add_argument('DIRNAME',
                        help='the directory name for \
                              sentence schema', type=str)
    parser.add_argument('TOTAL',
                        help='the total number of sentences \
                              generated for each depth', type=int)

    args = parser.parse_args()

    dir = args.DIRNAME
    res_dir = dir + "_results"
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    vocabs = load_vocab('vocab.yaml')
    total = args.TOTAL

    for file in sorted(glob.glob(dir + "/*.scheme.txt")):
        basename = os.path.basename(file)
        print('Processing {0}'.format(basename))
        depth = basename.split('.')[0]
        resname = basename.replace('scheme.', '')
        fout = res_dir + '/' + resname

        if depth == 'depth0':
            num = total
        elif depth == 'depth1':
            num = total
        else:
            #num = total // 10
            num = total

        with open(file) as f:
            sentences = [s.rstrip() for s in f.readlines()]

            _, data = estimate(sentences, vocabs)
            print('number of schema: {0}'.format(len(sentences)))
 
            for vocab in vocabs:
                cat = vocab['category']
                items = vocab['surf']
                sentences = instantiate(cat, items, sentences, depth)
            max_num = len(sentences)
            print('max number of sentences: {:,}'.format(max_num))
            print('************************')
            sentences = random.sample(sentences, k=num)
            output = '\n'.join(sentences) + '\n'
            with open(fout, 'a') as f:
                f.write(output)


if __name__ == '__main__':
    main()
