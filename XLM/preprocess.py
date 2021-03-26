#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


"""
Example: python data/vocab.txt data/train.txt special_token1,...
vocab.txt: 1stline=word, 2ndline=count
"""

import os
import sys

from src.logger import create_logger
from src.data.dictionary import Dictionary


if __name__ == '__main__':

    logger = create_logger(None, 0)

    voc_path = sys.argv[1]
    txt_path = sys.argv[2]
    bin_path = sys.argv[2] + '.pth'
    if len(sys.argv) > 3 :
        special_tokens = sys.argv[3].split(",")
    else :
        special_tokens = []
        
    if True :
        # bias corpus
        URL="URL"
        EMAIL="EMAIL"
        PHONE_NUMBER="PHONE"
        NUMBER="NUMBER"
        DIGIT="DIGIT"
        CUR="CUR" # currency_symbol
        st = [URL, EMAIL, PHONE_NUMBER, NUMBER,DIGIT,CUR]
        st = [s.lower() for s in st]
        special_tokens.extend(st)
        
    assert os.path.isfile(voc_path)
    assert os.path.isfile(txt_path)

    dico = Dictionary.read_vocab(voc_path, special_tokens)
    logger.info("")

    data = Dictionary.index_data(txt_path, bin_path, dico)
    logger.info("%i words (%i unique) in %i sentences." % (
        len(data['sentences']) - len(data['positions']),
        len(data['dico']),
        len(data['positions'])
    ))
    if len(data['unk_words']) > 0:
        logger.info("%i unknown words (%i unique), covering %.2f%% of the data." % (
            sum(data['unk_words'].values()),
            len(data['unk_words']),
            sum(data['unk_words'].values()) * 100. / (len(data['sentences']) - len(data['positions']))
        ))
        if len(data['unk_words']) < 30:
            for w, c in sorted(data['unk_words'].items(), key=lambda x: x[1])[::-1]:
                logger.info("%s: %i" % (w, c))
