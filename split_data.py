# Copyright (c) 2021-present, Pascal Tikeng, MILA.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

if __name__ == '__main__':
    import argparse, os, pandas as pd
    from sklearn.model_selection import train_test_split
    import ntpath

    def path_leaf(path):
        # https://stackoverflow.com/a/8384788/11814682
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_file", required=True, type=str)
    parser.add_argument("-o", "--output_path", required=True, type=str)
    parser.add_argument("-v", "--val_size", type=float, default=0.2)
    parser.add_argument("-r", "--random_seed", type=int, default=0)
    parser.add_argument("-t", '--task', default='pretrain', const='pretrain', nargs='?',
                                  choices=['pretrain','classification'], help='')
    args = parser.parse_args()

    assert os.path.isfile(args.data_file)
    if not os.path.exists(args.output_path) :
        os.mkdir(args.output_path)
    assert args.val_size > 0

    filename, file_extension = os.path.splitext(path_leaf(args.data_file))
    if args.task == "pretrain" :
        assert file_extension == ".txt"
        with open(args.data_file, "r") as f :
            corpus = f.readlines()
        corpus = [s.strip()+"\n" for s in corpus]
        train_corpus, val_corpus = train_test_split(corpus, 
                                            test_size = args.val_size, 
                                            random_state=args.random_seed)
        train_corpus[-1] = train_corpus[-1].strip()
        val_corpus[-1] = val_corpus[-1].strip()
        with open(os.path.join(args.output_path, "%s_train.txt"%filename), "w") as f :
            f.writelines(train_corpus)
        with open(os.path.join(args.output_path, "%s_val.txt"%filename), "w") as f :
            f.writelines(val_corpus)
        print("save %s and %s in %s"%("%s_train.txt"%filename, "%s_val.txt"%filename, args.output_path))
    else :
        assert file_extension == ".csv"
        data_frame = pd.read_csv(args.data_file)
        train_data_frame, val_data_frame = train_test_split(data_frame, 
                                            test_size = args.val_size, 
                                            random_state=args.random_seed)
        train_data_frame.to_csv(os.path.join(args.output_path, "%s_train.csv"%filename))
        val_data_frame.to_csv(os.path.join(args.output_path, "%s_val.csv"%filename))
        print("save %s and %s in %s"%("%s_train.csv"%filename, "%s_val.csv"%filename, args.output_path))