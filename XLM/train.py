# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import json
import random

# our
import copy
import gc
import json
import os
prime_string = "_prime"

from src.slurm import init_signal_handler, init_distributed_mode
from src.data.loader import check_data_params, load_data
from src.utils import initialize_exp, set_sampling_probs, shuf_order, set_seeds
from src.model import check_model_params, build_model
from src.model.memory import HashingMemory
from src.trainer import SingleTrainer, EncDecTrainer
from src.evaluation.evaluator import SingleEvaluator, EncDecEvaluator

from params import get_parser, from_config_file, three_point

def one_epoch(trainer, params, eval_task = None) :
    """Makes a training epoch."""

    if not params.meta_learning :
        trainer.n_sentences = 0
        trainer.stats['progress'] = 0
        while trainer.n_sentences < trainer.epoch_size :
            # CLM steps
            for lang1, lang2 in shuf_order(params.clm_steps, params):
                trainer.clm_step(lang1, lang2, params.lambda_clm)
            
            # MLM steps (also includes TLM if lang2 is not None)
            for lang1, lang2 in shuf_order(params.mlm_steps, params):
                trainer.mlm_step(lang1, lang2, params.lambda_mlm)

            # parallel classification steps
            for lang1, lang2 in shuf_order(params.pc_steps, params):
                trainer.pc_step(lang1, lang2, params.lambda_pc)

            # denoising auto-encoder steps
            for lang in shuf_order(params.ae_steps):
                trainer.mt_step(lang, lang, params.lambda_ae)

            # machine translation steps
            for lang1, lang2 in shuf_order(params.mt_steps, params):
                trainer.mt_step(lang1, lang2, params.lambda_mt)

            # back-translation steps
            for lang1, lang2, lang3 in shuf_order(params.bt_steps):
                trainer.bt_step(lang1, lang2, lang3, params.lambda_bt)

            trainer.iter()
            
            trainer.stats['progress'] = min(int((trainer.n_sentences+1)/trainer.epoch_size*100), 100) 
            
    else :
        # our
        trainer.n_sentences = {}
        """
        Here we build language lists for each of our meta-taks. Indeed, for two language lists l1 and l2, 
        the objective will be done with l1[i] and l2[i] respectively, this for each index i of the two lists. 
        """
        lang1_dic, lang2_dic, lang3_dic = {}, {}, {}
        """
        In the case of meta-learning, we have a (meta-)data dictionary for each (meta-)task, 
        so the keys are the languages conserved by the task. 
        """
        data_keys_dic = {}
                 
        # equivalent to "for task in list of task" in the original algorithm,  except here we prepare all the tasks beforehand.
        for lgs in params.meta_params.keys() :
            if eval_task :
                trainer.n_sentences[lgs] = trainer.epoch_size if lgs != eval_task else 0
            else :
                trainer.n_sentences[lgs] = 0 if lgs not in params.eval_tasks else trainer.epoch_size
            
            trainer.stats[lgs]['progress'] = 0
            # CLM
            try :
                lang1_dic['clm_step']
            except KeyError :
                lang1_dic['clm_step'], lang2_dic['clm_step'], data_keys_dic['clm_step'] = [], [], []
            for lang1, lang2 in shuf_order(params.meta_params[lgs].clm_steps, params):
                lang1_dic['clm_step'].append(lang1)
                lang2_dic['clm_step'].append(lang2)
                data_keys_dic['clm_step'].append(lgs)
                    
            # MLM  
            try :
                lang1_dic['mlm_step']
            except KeyError :
                lang1_dic['mlm_step'], lang2_dic['mlm_step'], data_keys_dic['mlm_step'] = [], [], []
            for lang1, lang2 in shuf_order(params.meta_params[lgs].mlm_steps, params):
                lang1_dic['mlm_step'].append(lang1)
                lang2_dic['mlm_step'].append(lang2)
                data_keys_dic['mlm_step'].append(lgs)
                           
            # parallel classification
            try :
                lang1_dic['pc_step']
            except KeyError :
                lang1_dic['pc_step'], lang2_dic['pc_step'], data_keys_dic['pc_step'] = [], [], []
            for lang1, lang2 in shuf_order(params.meta_params[lgs].pc_steps, params):
                lang1_dic['pc_step'].append(lang1)
                lang2_dic['pc_step'].append(lang2)
                data_keys_dic['pc_step'].append(lgs)
                        
            # denoising auto-encoder
            try :
                lang1_dic['ae_step']
            except KeyError :
                lang1_dic['ae_step'], data_keys_dic['ae_step'] = [], []
            for lang1 in shuf_order(params.meta_params[lgs].ae_steps):
                lang1_dic['ae_step'].append(lang1)
                data_keys_dic['ae_step'].append(lgs)
                     
            # machine translation 
            try :
                lang1_dic['mt_step']
            except KeyError :
                lang1_dic['mt_step'], lang2_dic['mt_step'], data_keys_dic['mt_step'] = [], [], []
            for lang1, lang2 in shuf_order(params.meta_params[lgs].mt_steps, params):
                lang1_dic['mt_step'].append(lang1)
                lang2_dic['mt_step'].append(lang2)
                data_keys_dic['mt_step'].append(lgs)
                   
            # back-translation
            try :
                lang1_dic['bt_step']
            except KeyError :
                lang1_dic['bt_step'], lang2_dic['bt_step'], lang3_dic['bt_step'], data_keys_dic['bt_step'] = [], [], [], []
            for lang1, lang2, lang3 in shuf_order(params.meta_params[lgs].bt_steps):
                lang1_dic['bt_step'].append(lang1)
                lang2_dic['bt_step'].append(lang2) 
                lang3_dic['bt_step'].append(lang3)
                data_keys_dic['bt_step'].append(lgs)
                        
        flag = True
                
        # equivalent to "while not done do" in the original algorithm
        while flag :
                        
            # CLM steps
            a = trainer.clm_step(lang1_dic['clm_step'] , lang2_dic['clm_step'], params.lambda_clm, data_keys_dic['clm_step'])
                    
            # MLM steps (also includes TLM if lang2 is not None) 
            b = trainer.mlm_step(lang1_dic['mlm_step'] , lang2_dic['mlm_step'], params.lambda_mlm, data_keys_dic['mlm_step']) 
                   
            # parallel classification steps
            c = trainer.pc_step(lang1_dic['pc_step'] , lang2_dic['pc_step'], params.lambda_pc, data_keys_dic['pc_step']) 
                    
            if isinstance(trainer, EncDecTrainer) :
           
                # denoising auto-encoder steps
                d = trainer.mt_step(lang1_dic['ae_step'] , lang1_dic['ae_step'], params.lambda_ae, data_keys_dic['ae_step']) 
                
                # machine translation steps    
                e = trainer.mt_step(lang1_dic['mt_step'] , lang2_dic['mt_step'], params.lambda_mt, data_keys_dic['mt_step']) 

                # back-translation steps
                f = trainer.bt_step(lang1_dic['bt_step'] , lang2_dic['bt_step'], lang3_dic['bt_step'], params.lambda_bt, data_keys_dic['bt_step'])    
                    
                # todo : do things better
                if (not a) and (not b) and (not c) and (not d) and (not e) and (not f) :
                    flag = False # End of epoch
                else :
                    flag = True
            else :
                # todo : do things better
                if (not a) and (not b) and (not c) :
                    flag = False # End of epoch
                else :
                    flag = True
                    
            trainer.iter()  
            for lgs in params.meta_params.keys() :
                trainer.stats[lgs]['progress'] = min(int((trainer.n_sentences[lgs]+1)/trainer.epoch_size*100), 100) 

def end_of_epoch(trainer, evaluator, params, logger, eval_task = None):
    # evaluate perplexity
    scores = evaluator.run_all_evals(trainer)
        
    # print / JSON log
    if not params.meta_learning :
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
    else :
        for lgs in params.meta_params.keys() :
            if not lgs.endswith(prime_string) :
                logger.info("============ task : %s " % lgs)
                for k, v in scores[lgs].items():
                    if k != "epoch":
                        logger.info("%s -> %.6f" % (k, v))

        if not eval_task :
            logger.info("============ all")
            for k, v in scores.items():
                if not (k in (list(params.meta_params.keys())+['epoch'])) :
                    logger.info("%s -> %.6f" % (k, v))
                
    if params.is_master:
        logger.info("__log__:%s" % json.dumps( scores[eval_task] if eval_task else scores))

    # end of epoch
    if not params.eval_only:
        trainer.save_best_model(scores)
        trainer.save_periodic()
        trainer.end_epoch(scores)

def main(params):
    
    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)
    
    # initialize the experiment
    meta_params = copy.deepcopy(params).meta_params
    params.meta_params = "..." # to long to be log
    logger = initialize_exp(params)
    params.meta_params = meta_params

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler()

    # load data
    data = load_data(params)
    if params.epoch_size == - 1 and params.max_train_data_size != 0 :
        params.epoch_size = params.max_train_data_size
    
    # todo : good params.n_words (We take the one from the first task have this parameter for the moment.)
    """
    But we think that if all the task data are based on the same vocabulary, all these parameters will be the same, 
    and therefore no problem if we choose one at random.
    """
    p = params.meta_params[data['key']]

    # build model
    if params.encoder_only:
        model = build_model(params = p, dico = data['dico'])
    else:
        encoder, decoder = build_model(params = p, dico = data['dico'])
        
    # todo : good pad_index and eos_index and ... (I'll take the one from the first task for the moment.)
    """
    But we think that if all the task data are based on the same vocabulary, all these parameters will be the same, 
    and therefore no problem if we choose one at random.
    """
    params.n_words = p.n_words
    params.bos_index = p.bos_index
    params.eos_index = p.eos_index
    params.pad_index = p.pad_index
    params.unk_index = p.unk_index
    params.mask_index = p.mask_index
    #print("p.n_words, p.bos_index, p.eos_index, p.pad_index, p.unk_index, p.mask_index")
    #print(p.n_words, p.bos_index, p.eos_index, p.pad_index, p.unk_index, p.mask_index)

    # build trainer, reload potential checkpoints / build evaluator
    params_tmp = copy.deepcopy(params)
    if params.encoder_only:
        trainer = SingleTrainer(model, data, params)
        evaluator = SingleEvaluator(trainer, data, params)
    else:
        trainer = EncDecTrainer(encoder, decoder, data, params)
        evaluator = EncDecEvaluator(trainer, data, params)

    if params.eval_tasks:
        logger.info("============ Evaluation task ============")
        eval_trainers = {}
        for eval_task in params.eval_tasks :
            logger.info("============ %s ============" % eval_task)
            eval_trainers[eval_task] = {}
            p = copy.deepcopy(params_tmp)
            if params.encoder_only:
                eval_trainers[eval_task]["trainer"] = SingleTrainer(copy.deepcopy(model), data, p)
                eval_trainers[eval_task]["evaluator"] = SingleEvaluator(eval_trainers[eval_task]["trainer"], data, p)
            else:
                eval_trainers[eval_task]["trainer"] = EncDecTrainer(copy.deepcopy(encoder), copy.deepcopy(decoder), data, p)
                eval_trainers[eval_task]["evaluator"] = EncDecEvaluator(eval_trainers[eval_task]["trainer"], data, p)
            
            
            p.dump_path = os.path.join(p.dump_path, eval_task)
            if not os.path.exists(p.dump_path):
                os.makedirs(p.dump_path)
            p.meta_params = {eval_task : params.meta_params[eval_task]}
            eval_trainers[eval_task]["params"] = p

    # evaluation
    if params.eval_only:
        end_of_epoch(trainer = trainer, evaluator = evaluator, params = params, logger = logger)
        if params.eval_tasks:
            logger.info("============ Evaluation task ============")
            for eval_task in params.eval_tasks :
                logger.info("============ %s ============" % eval_task)
                end_of_epoch(
                            trainer = eval_trainers[eval_task]['trainer'], 
                            evaluator = eval_trainers[eval_task]['evaluator'], 
                            params = eval_trainers[eval_task]["params"], 
                            logger = logger,
                            eval_task = eval_task 
                        )
        exit()

    # set sampling probabilities for training
    set_sampling_probs(data, params)
    
    # language model training
    for _ in range(params.max_epoch):
        
        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        one_epoch(trainer, params)

        if params.eval_tasks:
            logger.info("============ Evaluation task ============")
            for eval_task in params.eval_tasks :
                logger.info("============ %s ============" % eval_task)
                if params.encoder_only:
                    eval_trainers[eval_task]['trainer'].model = copy.deepcopy(trainer.model)
                else :
                    eval_trainers[eval_task]['trainer'].encoder = copy.deepcopy(trainer.encoder)
                    eval_trainers[eval_task]['trainer'].decoder = copy.deepcopy(trainer.decoder)
                one_epoch(eval_trainers[eval_task]['trainer'], eval_trainers[eval_task]["params"], eval_task = eval_task)
            
        logger.info("============ End of epoch %i ============" % trainer.epoch)

        end_of_epoch(trainer = trainer, evaluator = evaluator, params = params, logger = logger)

        if params.eval_tasks:
            logger.info("============ Evaluation task ============")
            for eval_task in params.eval_tasks :
                end_of_epoch(
                            trainer = eval_trainers[eval_task]['trainer'], 
                            evaluator = eval_trainers[eval_task]['evaluator'], 
                            params = eval_trainers[eval_task]["params"], 
                            logger = logger,
                            eval_task = eval_task 
                        )

        # our
        logger.info("============ garbage collector collecting %d ..." % gc.collect())
        
    
def check_meta_learning_params(params) :
    """
    This method basically verifies if there is a meta-task that is not present in any objective (clm, mlm, pc, mt, ae, bt)
    """
    for _, clm, mlm, pc, mt, ae, bt in zip(params.langs, params.clm_steps, params.mlm_steps, params.pc_steps, params.mt_steps, params.ae_steps, params.bt_steps) :       
        assert not all([objectif == [] for objectif in [clm, mlm, pc, mt, ae, bt]]), "Every task must be present in some of objectif" 
                                    
if __name__ == '__main__':

    # generate parser / parse parameters
    params = get_parser().parse_args()

    params = from_config_file(params)

    set_seeds(params.random_seed)

    if params.device not in ["cpu", "cuda"] :
        params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else :
        params.device = torch.device(params.device)
    

    # debug mode
    if params.debug:
        params.exp_name = 'debug'
        params.exp_id = 'debug_%08i' % random.randint(0, 100000000)
        params.debug_slurm = True
        params.debug_train = True
    
    params.n_samples={}
    params.n_samples['train'] = params.train_n_samples
    params.n_samples['valid'] = params.valid_n_samples
    params.n_samples['test'] = params.test_n_samples

    params.remove_long_sentences = {}
    params.remove_long_sentences['train'] = params.remove_long_sentences_train
    params.remove_long_sentences['valid'] = params.remove_long_sentences_valid
    params.remove_long_sentences['test'] = params.remove_long_sentences_test

    
    # Check to see if we need to do metalearning.
    params.meta_learning = False
    
    meta_lgs = params.lgs.split("|")
    
    params.meta_params = {}
    params.n_task = len(meta_lgs)
    
    meta_tmp = ["" for _ in range(params.n_task)]
    
    meta_clm = []
    if params.clm_steps == "" :
        meta_clm = meta_tmp
    else :
        meta_clm = params.clm_steps.split("|")
        
    meta_mlm = []
    if params.mlm_steps == "" :
        meta_mlm = meta_tmp
    else :
        meta_mlm = params.mlm_steps.split("|")
        
    meta_pc = []
    if params.pc_steps == "" :
        meta_pc = meta_tmp
    else :
        meta_pc = params.pc_steps.split("|")
        
    meta_mt = []
    if params.mt_steps == "" :
        meta_mt = meta_tmp
    else :
        meta_mt = params.mt_steps.split("|")
       
    meta_ae = []
    if params.ae_steps == "" :
        meta_ae = meta_tmp
    else :
        meta_ae = params.ae_steps.split("|")
        
    meta_bt = []
    if params.bt_steps == "" :
        meta_bt = meta_tmp
    else :
        meta_bt = params.bt_steps.split("|")
    
    langs, clms, mlms, pcs, mts, aes, bts = [], [], [], [], [], [], []
    
    if params.n_task != 1 :
        params.meta_learning = True
        
    # check parameters
    for meta_objectif in [meta_clm, meta_mlm, meta_pc, meta_mt, meta_ae, meta_bt] :
        assert len(meta_objectif) == params.n_task, "If you pass an objective parameter for a meta-task, do the same for all the other tasks (space if no objective)."
       
    data_path = params.data_path
    for lgs, clm, mlm, pc, mt, ae, bt in zip(meta_lgs, meta_clm, meta_mlm, meta_pc, meta_mt, meta_ae, meta_bt) :
        
        params.lgs = lgs 
        params.clm_steps = three_point(objectif = clm, lgs = lgs, name="clm")
        params.mlm_steps = three_point(objectif = mlm, lgs = lgs, name="mlm")
        params.pc_steps = three_point(objectif = pc, lgs = lgs, name="pc")
        params.mt_steps = three_point(objectif = mt, lgs = lgs, name="mt")
        params.ae_steps = three_point(objectif = ae, lgs = lgs, name="ae") 
        params.bt_steps = three_point(objectif = bt, lgs = lgs, name="bt")
        
        if params.meta_learning and not params.same_data_path:
            params.data_path = data_path+"/"+lgs

        check_data_params(params)
        check_model_params(params)    
        
        try :
            params.meta_params[lgs]
            params.meta_params[lgs + prime_string] = copy.deepcopy(params)
        except KeyError :
            params.meta_params[lgs] = copy.deepcopy(params)
        
        langs.append(params.langs)
        clms.append(params.clm_steps)
        mlms.append(params.mlm_steps)
        pcs.append(params.pc_steps)
        mts.append(params.mt_steps)
        aes.append(params.ae_steps)
        bts.append(params.bt_steps)
        
    if params.meta_learning :
        params.langs = langs
        params.clm_steps = clms
        params.mlm_steps = mlms
        params.pc_steps = pcs
        params.mt_steps = mts
        params.ae_steps = aes
        params.bt_steps = bts
        # our
        check_meta_learning_params(params)

        if params.eval_tasks :
            eval_tasks_dico = {}
            for eval_task in params.eval_tasks.split(","):
                eval_task = eval_task.split(":")
                assert eval_task[0] in params.meta_params.keys() 
                eval_tasks_dico[eval_task[0]] = {}
                eval_tasks_dico[eval_task[0]]["train"] = int(eval_task[1])
                #eval_tasks_dico[eval_task[0]]["test"] = eval_task[1]
                #eval_tasks_dico[eval_task[0]]["valid"] = eval_task[1]
            params.eval_tasks = eval_tasks_dico
        else :
            params.eval_tasks = {}
    else :
        params.eval_tasks = {}
        
    params.lgs = meta_lgs
    params.data_path = data_path

    # run experiment
    main(params)
