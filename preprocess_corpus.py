# Copyright (c) 2021-present, Pascal Tikeng, MILA.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# python preprocess_corpus.py -d $data_file -st 1,2,3 -o /content -ma 100 -mi 2 -n 1000 -s False -r 0

#! pip install clean-text

from cleantext import clean

import io
import os
import unicodedata
import re
import random
import string
import tqdm
import argparse 

import pandas as pd
from sklearn.model_selection import train_test_split

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import ntpath

def path_leaf(path):
    # https://stackoverflow.com/a/8384788/11814682
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def cleanhtml(raw_html):
    # https://stackoverflow.com/a/12982689/11814682
    #cleanr = re.compile('<.*?>')
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def replace_by_space(text : str, tokens : list):
    #return text.translate(str.maketrans({t : "" for t in tokens}))
    for t in tokens :
        text = text.replace(t, "")
    return text

def cleanpunct(w, excapt=[]):
    # https://stackoverflow.com/a/34922745/11814682
    # replace punctuation with space : accept one or more copies of punctuation + plus zero or more copies of a space
    # "pot......kettle.....black." becomes "pot kettle black"
    # "mario...im-terrified-of-black-rioters-cuomo." becomes "mario im terrified of black rioters cuomo."
    return re.sub(r"""[%s]+ \ *"""%replace_by_space(string.punctuation, excapt)," ", w, flags=re.VERBOSE)
    #return re.sub(r"[,.;@#?!&$]+\ *", " ", w)
    
def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"<u\+\w+><u\+\w+>", "", w) # <u+00a0><u+00af>, <u+00a0><u+00b0>, ...
    w = cleanhtml(w)
    #w = cleanpunct(w, excapt = ["'", '"'])
    w = clean(w,
                fix_unicode=True,               # fix various unicode errors
                to_ascii=True,                  # transliterate to closest ASCII representation
                lower=False,                     # lowercase text
                no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
                no_urls=True,                  # replace all URLs with a special token
                no_emails=True,                # replace all email addresses with a special token
                no_phone_numbers=True,         # replace all phone numbers with a special token
                no_numbers=True,               # replace all numbers with a special token
                no_digits=True,                # replace all digits with a special token
                no_currency_symbols=True,      # replace all currency symbols with a special token
                no_punct=False,                 # remove punctuations
                replace_with_punct="",          # instead of removing punctuations you may replace them
                replace_with_url=" URL ",
                replace_with_email=" EMAIL ",
                replace_with_phone_number=" PHONE ",
                replace_with_number=" NUMBER ",
                replace_with_digit=" DIGIT ",
                replace_with_currency_symbol=" CUR ",
                lang="en"                       
            )
    w = w.strip("'").strip('"').replace("\n", " ").strip(" ")
    #w = cleanpunct(w, excapt = ["'", '"', "[", "]"])
    w = cleanpunct(w, excapt = ["'", '"'])
    return w

def is_valid(s):
    return len(cleanpunct(s).replace(" ", "")) != 0 and len(s.split(" "))> 1

def preprocess_paragraph(p, max_len):
    s = preprocess_sentence(p)
    l = len(s.split(" "))
    if l > max_len :
        result = [preprocess_sentence(s) for s in nltk.sent_tokenize(p)]
        result = [s for s in result if is_valid(s)]
        return  result
    else :
        return [s]

def preprocess_corpus(corpus, max_len):
    result = []
    #for p in tqdm.notebook.tqdm(corpus, desc="preprocess_corpus") :
    #for p in tqdm.tqdm_notebook(corpus, desc="preprocess_corpus") :
    for p in tqdm.tqdm(corpus, desc="preprocess_corpus") :    
        result.extend(preprocess_paragraph(p, max_len))
    return result

def n_option(l):
    assert l > 1
    if l%2 == 0:
        n = (l-2)//2 if (l//2)%2 == 0 else l//2
        start = (l-1)//2 - n//2 + 1
    else :
        n = (l-1)//2 if ((l-1)//2)%2 == 0 else (l+1)//2
        start = l//2 - n//2+1
    return n, start, start + n-1 # n, start, end

def process(x):
    x = x.replace("\t", " ")
    x = x.split(" ")  # nltk.word_tokenize(x)
    assert len(x) > 1
    _, start, end = n_option(len(x))
    l = random.randint(start, end)
    return " ".join(x[:l] + ["\t"] + x[l:])

def good_corpus(data, random_seed = 0) :
    random.seed(random_seed)
    #return [process(x).strip() + "\n" for x in data if x != "\n" and is_valid(x)]
    corpus = []
    
    #for x in tqdm.notebook.tqdm(data, desc="good_corpus") :
    #for x in tqdm.tqdm_notebook(data, desc="good_corpus") :
    for x in tqdm.tqdm(data, desc="good_corpus") :
        if x != "\n" and is_valid(x) :
            corpus.append(process(x).strip() + "\n")
    return corpus

def write_corpus(corpus, file_path, version = 1, random_seed = 0) :
    assert version in [1, 2]
    corpus = [s.strip()+"\n" for s in corpus]
    with open(file_path, "w") as f :
        if version == 2 :
            f.writelines(good_corpus(corpus, random_seed))
        else :
            f.writelines(corpus)

def stat(corpus):
    stat = [len(s.split(" ")) for s in corpus]
    N = len(corpus)
    mean_, max_, min_ = sum(stat)/N, max(stat), min(stat)
    print("len : %d, mean : %d, min : %d, max :%d"%(N, mean_, min_, max_))


FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")

if __name__ == '__main__':
    # python preprocess_corpus.py -d $data_file -st 1,2,3 -o /content -ma 100 -mi 2 -n 1000 -s False -r 0
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_file", required=True, type=str, help="input file")
    parser.add_argument("-st", "--steps", default="1:2,2:1-2,3:1-2", type=str, help="steps")
    parser.add_argument("-o", "--output_path", required=True, type=str)
    parser.add_argument("-ma", "--max_len", required=True, type=int)
    parser.add_argument("-ma3", "--max_len_step3", default = float("inf"), type=float)
    parser.add_argument("-mi", "--min_len", required=True, type=int)
    parser.add_argument("-n", "--n_samples", default=None, type=int)
    parser.add_argument("-s", '--shuffle', type=bool_flag, default = False)
    parser.add_argument("-r", "--random_seed", type=int, default=0)
   
    args = parser.parse_args()

    assert os.path.isfile(args.data_file)
    assert args.steps
    steps = { int(i.split(":")[0]) : [int(k) for k in (i.split(":")[1].split('-') if len(i.split(":")) == 2 else [])]  for i in args.steps.split(",")}
    steps = { k : steps[k] for k in sorted(steps.keys()) }
    for k, v in steps.items():
        assert k in [1, 2, 3] and all([i in [1, 2] for i in v])

    assert args.max_len > 0 and args.min_len > 0 and args.max_len_step3 > 0
    assert args.n_samples is None or args.n_samples > 0

    if not os.path.exists(args.output_path) :
        os.mkdir(args.output_path)

    filename, file_extension = os.path.splitext(path_leaf(args.data_file))

    random.seed(args.random_seed)
    print("===============")
    with open(args.data_file) as f:
        corpus = f.readlines()

    if args.shuffle :
        random.shuffle(corpus)

    corpus = corpus[:args.n_samples]
    print("stat corpus")
    stat(corpus)

    for i in steps.keys() :
        if i == 1 :
            for v in steps[i] :
                write_corpus(corpus, os.path.join(args.output_path, "%s%dv%d.%s"%(filename, i, v,'txt')), v, args.random_seed)
    
        if i == 2 :
            print("\n ================= corpus2")
            corpus = preprocess_corpus(corpus, max_len = args.max_len)
            print("stat corpus2")
            stat(corpus)
            for v in steps[i] :
                write_corpus(corpus, os.path.join(args.output_path, "%s%dv%d.%s"%(filename, i, v,'txt')), v, args.random_seed)

        if i == 3 :
            print("\n ================= corpus 3")
            min_len = args.min_len
            max_len = args.max_len_step3
            corpus3 = []
            #for s in tqdm.notebook.tqdm(corpus2, desc="build corpus 3") :
            #for s in tqdm.tqdm_notebook(corpus2, desc="build corpus 3") :
            for s in tqdm.tqdm(corpus, desc="build corpus 3") :
                l = len(s.strip().replace("\t", " ").split(" ")) 
                if min_len <= l <= max_len :
                    #print(s)
                    corpus3.append(s)
                else :
                    #print(s)
                    pass

            corpus3 = [s.strip()+"\n" for s in corpus3]
            corpus3[-1] = corpus3[-1].strip()

            print('remove : %d'%(len(corpus)-len(corpus3)))
            corpus = corpus3
            print("stat corpus3")
            stat(corpus)
            for v in steps[i] :
                write_corpus(corpus, os.path.join(args.output_path, "%s%dv%d.%s"%(filename, i, v,'txt')), v, args.random_seed)

"""
text = "I am a black man. An you? You, yes."
sent_text = nltk.sent_tokenize(text) # this gives us a list of sentences
print(sent_text)
# now loop over each sentence and tokenize it separately
for sentence in sent_text:
    tokenized_text = nltk.word_tokenize(sentence)
    print("tokenized_text", tokenized_text)
    tagged = nltk.pos_tag(tokenized_text)
    print("tagged",tagged)

print(string.punctuation)
"""