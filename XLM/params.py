import argparse 
import os
import json
from src.utils import bool_flag
from src.model.memory import HashingMemory

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Experiment name")
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")

    # float16 / AMP API
    parser.add_argument("--fp16", type=bool_flag, default=False,
                        help="Run model with float16")
    parser.add_argument("--amp", type=int, default=-1,
                        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.")

    # only use an encoder (use a specific decoder for machine translation)
    parser.add_argument("--encoder_only", type=bool_flag, default=True,
                        help="Only use an encoder")

    # model parameters
    parser.add_argument("--emb_dim", type=int, default=512,
                        help="Embedding layer size")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of Transformer layers")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of Transformer heads")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0,
                        help="Dropout in the attention layer")
    parser.add_argument("--gelu_activation", type=bool_flag, default=False,
                        help="Use a GELU activation instead of ReLU")
    parser.add_argument("--share_inout_emb", type=bool_flag, default=True,
                        help="Share input and output embeddings")
    parser.add_argument("--sinusoidal_embeddings", type=bool_flag, default=False,
                        help="Use sinusoidal embeddings")
    parser.add_argument("--use_lang_emb", type=bool_flag, default=True,
                        help="Use language embedding")

    # memory parameters
    parser.add_argument("--use_memory", type=bool_flag, default=False,
                        help="Use an external memory")
    if parser.parse_known_args()[0].use_memory:
        HashingMemory.register_args(parser)
        parser.add_argument("--mem_enc_positions", type=str, default="",
                            help="Memory positions in the encoder ('4' for inside layer 4, '7,10+' for inside layer 7 and after layer 10)")
        parser.add_argument("--mem_dec_positions", type=str, default="",
                            help="Memory positions in the decoder. Same syntax as `mem_enc_positions`.")

    # adaptive softmax
    parser.add_argument("--asm", type=bool_flag, default=False,
                        help="Use adaptive softmax")
    if parser.parse_known_args()[0].asm:
        parser.add_argument("--asm_cutoffs", type=str, default="8000,20000",
                            help="Adaptive softmax cutoffs")
        parser.add_argument("--asm_div_value", type=float, default=4,
                            help="Adaptive softmax cluster sizes ratio")

    # causal language modeling task parameters
    parser.add_argument("--context_size", type=int, default=0,
                        help="Context size (0 means that the first elements in sequences won't have any context)")

    # masked language modeling task parameters
    parser.add_argument("--word_pred", type=float, default=0.15,
                        help="Fraction of words for which we need to make a prediction")
    parser.add_argument("--sample_alpha", type=float, default=0,
                        help="Exponent for transforming word counts to probabilities (~word2vec sampling)")
    parser.add_argument("--word_mask_keep_rand", type=str, default="0.8,0.1,0.1",
                        help="Fraction of words to mask out / keep / randomize, among the words to predict")

    # input sentence noise
    parser.add_argument("--word_shuffle", type=float, default=0,
                        help="Randomly shuffle input words (0 to disable)")
    parser.add_argument("--word_dropout", type=float, default=0,
                        help="Randomly dropout input words (0 to disable)")
    parser.add_argument("--word_blank", type=float, default=0,
                        help="Randomly blank input words (0 to disable)")

    # data
    parser.add_argument("--data_path", type=str, default="",
                        help="Data path")
    parser.add_argument("--lgs", type=str, default="",
                        help="Languages (lg1-lg2-lg3 .. ex: en-fr-es-de)")
    parser.add_argument("--max_vocab", type=int, default=-1,
                        help="Maximum vocabulary size (-1 to disable)")
    parser.add_argument("--min_count", type=int, default=0,
                        help="Minimum vocabulary count")
    parser.add_argument("--lg_sampling_factor", type=float, default=-1,
                        help="Language sampling factor")

    # batch parameters
    parser.add_argument("--bptt", type=int, default=256,
                        help="Sequence length")
    parser.add_argument("--max_len", type=int, default=100,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--group_by_size", type=bool_flag, default=True,
                        help="Sort sentences by size during the training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of sentences per batch")
    parser.add_argument("--max_batch_size", type=int, default=0,
                        help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")
    parser.add_argument("--tokens_per_batch", type=int, default=-1,
                        help="Number of tokens per batch")

    # training parameters
    parser.add_argument("--split_data", type=bool_flag, default=False,
                        help="Split data across workers of a same node")
    parser.add_argument("--optimizer", type=str, default="adam,lr=0.0001",
                        help="Optimizer : Adam / AdamInverseSqrtWithWarmup / AdamCosineWithWarmup / \
                              Adadelta / Adagrad / Adamax / ASGD / SGD / RMSprop / Adam / Rprop)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=100000,
                        help="Epoch size / evaluation frequency (-1 for parallel data size)")
    parser.add_argument("--max_epoch", type=int, default=100000,
                        help="Maximum epoch size")
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--validation_metrics", type=str, default="",
                        help="Validation metrics")
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Accumulate model gradients over N iterations (N times larger batch sizes)")

    # training coefficients
    parser.add_argument("--lambda_mlm", type=str, default="1",
                        help="Prediction coefficient (MLM)")
    parser.add_argument("--lambda_clm", type=str, default="1",
                        help="Causal coefficient (LM)")
    parser.add_argument("--lambda_pc", type=str, default="1",
                        help="PC coefficient")
    parser.add_argument("--lambda_ae", type=str, default="1",
                        help="AE coefficient")
    parser.add_argument("--lambda_mt", type=str, default="1",
                        help="MT coefficient")
    parser.add_argument("--lambda_bt", type=str, default="1",
                        help="BT coefficient")

    # training steps
    parser.add_argument("--clm_steps", type=str, default="",
                        help="Causal prediction steps (CLM)")
    parser.add_argument("--mlm_steps", type=str, default="",
                        help="Masked prediction steps (MLM / TLM)")
    parser.add_argument("--mt_steps", type=str, default="",
                        help="Machine translation steps")
    parser.add_argument("--ae_steps", type=str, default="",
                        help="Denoising auto-encoder steps")
    parser.add_argument("--bt_steps", type=str, default="",
                        help="Back-translation steps")
    parser.add_argument("--pc_steps", type=str, default="",
                        help="Parallel classification steps")

    # reload pretrained embeddings / pretrained model / checkpoint
    parser.add_argument("--reload_emb", type=str, default="",
                        help="Reload pretrained word embeddings")
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pretrained model")
    parser.add_argument("--reload_checkpoint", type=str, default="",
                        help="Reload a checkpoint")

    # beam search (for MT only)
    parser.add_argument("--beam_size", type=int, default=1,
                        help="Beam size, default = 1 (greedy decoding)")
    parser.add_argument("--length_penalty", type=float, default=1,
                        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.")
    parser.add_argument("--early_stopping", type=bool_flag, default=False,
                        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.")

    # evaluation
    parser.add_argument("--eval_bleu", type=bool_flag, default=False,
                        help="Evaluate BLEU score during MT training")
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")

    # debug
    parser.add_argument("--debug_train", type=bool_flag, default=False,
                        help="Use valid sets for train sets (faster loading)")
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Debug multi-GPU / multi-node within a SLURM job")
    parser.add_argument("--debug", help="Enable all debug flags",
                        action="store_true")

    # multi-gpu / multi-node
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")

    # our
    # These three parameters will always be rounded to an integer number of batches, so don't be surprised if you see different values than the ones provided.
    parser.add_argument("--train_n_samples", type=int, default=-1, 
                        help="Just consider train_n_sample train data")
    parser.add_argument("--valid_n_samples", type=int, default=-1, 
                        help="Just consider valid_n_sample validation data")
    parser.add_argument("--test_n_samples", type=int, default=-1, 
                        help="Just consider test_n_sample test data for")
    parser.add_argument("--remove_long_sentences_train", type=bool_flag, default=False, 
                        help="remove long sentences in train dataset")
    parser.add_argument("--remove_long_sentences_valid", type=bool_flag, default=False, 
                        help="remove long sentences in valid dataset")
    parser.add_argument("--remove_long_sentences_test", type=bool_flag, default=False, 
                        help="remove long sentences in test dataset")
    
    parser.add_argument("--same_data_path", type=bool_flag, default=True, 
                        help="In the case of metalearning, this parameter, when passed to False, the data are" \
                            "searched for each task in a folder with the name of the task and located in data_path otherwise all the data are searched in data_path.")
    
    parser.add_argument("--config_file", type=str, default="", 
                        help="")

    parser.add_argument("--log_file_prefix", type=str, default="", 
                        help="Log file prefix. Name of the language to be evaluated in the case of the" \
                              "evaluation of one LM on another.")

    parser.add_argument("--aggregation_metrics", type=str, default="", 
                        help="name_metric1=mean(m1,m2,...);name_metric2=sum(m4,m5,...);...")

    parser.add_argument("--eval_tasks", type=str, default="", 
                        help="During metalearning we need tasks on which to refine and evaluate the model after each epoch." \
                              "task_name:train_n_samples,..."
                            )    
    # TIM
    parser.add_argument("--tim_layers_pos", type=str, default="", help="tim layers position : 0,1,5 for example")
    parser.add_argument("--use_group_comm", type=bool_flag, default=True)
    parser.add_argument("--use_mine", type=bool_flag, default=False)
    if parser.parse_known_args()[0].tim_layers_pos :
        # Transformers with Independent Mechanisms (TIM) model parameters
        parser.add_argument("--n_s", type=int, default=2, help="number of mechanisms")
        parser.add_argument("--H", type=int, default=8, help="number of heads for self-attention")
        parser.add_argument("--H_c", type=int, default=8, help="number of heads for inter-mechanism attention")
        parser.add_argument("--custom_mha", type=bool_flag, default=False)
        #if parser.parse_known_args()[0].custom_mha:
        parser.add_argument("--d_k", type=int, default=512, help="key dimension")
        parser.add_argument("--d_v", type=int, default=512, help="value dimension")
    
    
    parser.add_argument("--dim_feedforward", type=int, default=512*4, 
                        help="Dimension of Intermediate Layers in Positionwise Feedforward Net")

    parser.add_argument("--log_interval", type=int, default=-1, 
                        help="Interval (number of steps) between two displays : batch_size by default")
    parser.add_argument("--device", type=str, default="", help="cpu/cuda")
    parser.add_argument("--random_seed", type=int, default=0, help="random seed for reproductibility")
    

    return parser

config_dic = {
    # main parameters
    "dump_path":[str, "./dumped/"],
    "exp_name":[str, ""],
    "save_periodic":[int, 0],
    "exp_id":[str, ""],
    # AMP API
    "fp16":[bool, False],
    "amp":[int, -1],
    # only use an encoder (use a specific decoder for machine translation)
    "encoder_only":[bool, True],
    # model parameters
    "emb_dim":[int, 512], 
    "n_layers":[int, 4],
    "n_heads":[int, 8],
    "dropout":[float, 0],
    "attention_dropout":[float, 0],
    "gelu_activation":[bool, False], 
    "share_inout_emb":[bool, True], 
    "sinusoidal_embeddings":[bool, False],
    "use_lang_emb":[bool, True],
    # memory parameters
    "use_memory":[bool, False],
    "mem_enc_positions":[str, ""],
    "mem_dec_positions":[str, ""],
    # adaptive softmax
    "asm":[bool, False],
    "asm_cutoffs":[str, "8000,20000"],
    "asm_div_value":[float, 4],
    # causal language modeling task parameters
    "context_size":[int, 0],
    # masked language modeling task parameters
    "word_pred":[float, 0.15],
    "sample_alpha":[float, 0],
    "word_mask_keep_rand":[str, "0.8,0.1,0.1"],
    # input sentence noise
    "word_shuffle":[float, 0], 
    "word_dropout":[float, 0],
    "word_blank":[float, 0],
    # data
    "data_path":[str, ""],
    "lgs":[str, ""],
    "max_vocab":[int, -1],
    "min_count":[int, 0],
    "lg_sampling_factor":[float, -1],
    # batch parameters
    "bptt":[int, 256],
    "max_len":[int, 100],
    "group_by_size":[bool, True],
    "batch_size":[int, 32],
    "max_batch_size":[int, 0],
    "tokens_per_batch":[int, -1],
    # training parameters
    "split_data":[bool, False],
    "optimizer":[str, "adam,lr=0.0001"],
    "clip_grad_norm":[float, 5],
    "epoch_size":[int, 100000],
    "max_epoch":[int, 100000],
    "stopping_criterion":[str, ""],
    "validation_metrics":[str, ""],
    "accumulate_gradients":[int, 1],
    # training coefficients
    "lambda_mlm":[str, "1"],
    "lambda_clm":[str, "1"],
    "lambda_pc":[str, "1"], 
    "lambda_ae":[str, "1"],
    "lambda_mt":[str, "1"],
    "lambda_bt":[str, "1"],
    # training steps
    "clm_steps":[str, ""],
    "mlm_steps":[str, ""], 
    "mt_steps":[str, ""],
    "ae_steps":[str, ""], 
    "bt_steps":[str, ""], 
    "pc_steps":[str, ""],
    # reload pretrained embeddings / pretrained model / checkpoint, 1],
    "reload_emb":[str, ""],
    "reload_model":[str, ""],
    "reload_checkpoint":[str, ""],
    # beam search (for MT only)
    "beam_size":[int, 1],
    "length_penalty":[float, 1],
    "early_stopping":[bool, False],
    # evaluation
    "eval_bleu":[bool, False],
    "eval_only":[bool, False],
    # debug
    "debug_train":[bool, False],
    "debug_slurm":[bool, False],
    ###"debug":?
    # multi-gpu / multi-node
    "local_rank":[int, -1],
    "master_port":[int, -1],
    # our
    "train_n_samples":[int, -1],
    "valid_n_samples":[int, -1],
    "test_n_samples":[int, -1],
    "remove_long_sentences_train":[bool, False],
    "remove_long_sentences_valid":[bool, False],
    "remove_long_sentences_test":[bool, False],
    "same_data_path":[bool, True],
    #"config_file":[str, ""],
    #"log_file_prefix":[str, ""],
    "aggregation_metrics":[str, ""],
    "eval_tasks":[str, ""],

    "tim_layers_pos" : [str, ""],
    "use_group_comm":[bool, True],
    "n_s" : [int, 2],
    "H" : [int, 8],
    "H_c" : [int, 8],
    "custom_mha" : [bool, False],
    "d_k" : [int, 512], 
    "d_v" : [int, 512], 
    "dim_feedforward" : [int, 512*4],

    "log_interval":[int, -1],
    "device" : [str, ""],
    "random_seed": [int, 0],
    
    ######################## classify.py
    "train_data_file": [str, ""], 
    "val_data_file": [str, ""],
    #"model_path": [str, ""],
    "freeze_transformer":[bool, True],
    "version": [int, 1],
    "log_softmax": [bool, True],
    "shuffle":[bool, False],
    "min_len":[int, 1],
    "debug_num":[int, 0],
    "hidden_dim":[int, -1], 
    "gru_n_layers":[int, 1], 
    "bidirectional":[bool, False],
    "topK":[int, 3]
}

def from_config_file(params):
    if os.path.isfile(params.config_file):
        with open(params.config_file) as json_data:
            data_dict = json.load(json_data)
            for key, value in data_dict.items():
                conf = config_dic.get(key, "__key_error__")
                if conf != "__key_error__":   
                    if value == "False":
                        value = False
                    elif value == "True" :
                        value = True
                    """
                    try :
                        setattr(params, key, conf[0](value))
                    except :
                        setattr(params, key, value)
                    """
                    # Allow to overwrite the parameters of the json configuration file.
                    try :
                        value = conf[0](value)
                    except :
                        pass
                    if getattr(params, key, conf[1]) == conf[1] :
                        setattr(params, key, value)

    return params

def three_point(objectif, lgs, name) :
    if objectif == "..." :
        result = ""
        if name == "clm" : 
            langs = lgs.split("-")
            result = langs[0]
            for lg in langs[1:] :
                result = result+","+lg
        if name == "mlm" :
            langs = lgs.split("-")
            result = langs[0]
            for lg in langs[1:] :
                result = result+","+lg
            l = len(langs)
            for i in range(l-1):
                for j in range(i+1, l):
                    li = langs[i]
                    lj = langs[j]
                    result = result+","+li+"-"+lj
        elif name == "mt" :
          langs = lgs.split("-")
          l = len(langs)
          for i in range(l-1):
            for j in range(i+1, l):
              li = langs[i]
              lj = langs[j]
              result = result+","+li+"-"+lj
              result = result+","+lj+"-"+li
          if result.startswith(","):
            result = result[1:]
        elif name == "ae" :
          langs = lgs.split("-")
          result = langs[0]
          for lg in langs[1:] :
            result = result+","+lg
        elif name == "bt" :
          langs = lgs.split("-")
          l = len(langs)
          for i in range(l-1):
            for j in range(i+1, l):
              li = langs[i]
              lj = langs[j]
              result = result+","+li+"-"+lj+"-"+li
              result = result+","+lj+"-"+li+"-"+lj
          if result.startswith(","):
            result = result[1:]
        elif name == "pc" : 
            langs = lgs.split("-")
            l = len(langs)
            for i in range(l-1):
                for j in range(i+1, l):
                    li = langs[i]
                    lj = langs[j]
                    result = result+","+li+"-"+lj
                    result = result+","+lj+"-"+li
            if result.startswith(","):
                result = result[1:]
        return result
    else :      
        return objectif