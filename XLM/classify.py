import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
#from optim import ScheduledOptim
import os
import numpy as np
from sklearn.metrics import f1_score, classification_report, accuracy_score

from src.slurm import init_signal_handler, init_distributed_mode
from src.utils import initialize_exp, set_seeds
from src.bias_classification import load_dataset, BertClassifier, Trainer, label_dict
from src.optim import get_optimizer
from src.utils import bool_flag

from params import get_parser, from_config_file

def get_acc(pred, label):
    arr = (np.array(pred) == np.array(label)).astype(float)
    if arr.size != 0: # check NaN 
        return arr.mean()*100
    return 0

def f1_score_func(pred, label):
    # https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
    return f1_score(pred, label, average='weighted', labels=np.unique(pred))*100

def top_k(logits, y, k : int = 1):
    """
    logits : (bs, n_labels)
    y : (bs,)
    """
    labels_dim = 1
    assert 1 <= k <= logits.size(labels_dim)
    
    k_labels = torch.topk(input = logits, k = k, dim=labels_dim, largest=True, sorted=True)[1]

    # True (#0) if `expected label` in k_labels, False (0) if not
    a = ~torch.prod(input = torch.abs(y.unsqueeze(labels_dim) - k_labels), dim=labels_dim).to(torch.bool)
    
    # These two approaches are equivalent
    if False :
        y_pred = torch.empty_like(y)
        for i in range(y.size(0)):
            if a[i] :
                y_pred[i] = y[i]
            else :
                y_pred[i] = k_labels[i][0]
        #correct = a.to(torch.int8).numpy()
    else :
        a = a.to(torch.int8)
        y_pred = a * y + (1-a) * k_labels[:,0]
        #correct = a.numpy()
    
    y_pred = y_pred.cpu().numpy()
    #f1 = f1_score(y_pred, y, average='weighted')*100
    f1 = f1_score_func(y_pred, y)
    #acc = sum(correct)/len(correct)*100
    #acc = accuracy_score(y_pred, y)*100
    acc = get_acc(y_pred, y)
    #print("============")
    #print(logits.max(1)[1], y_pred)
    #print(logits.shape, y.shape)
    #print(logits, y, k, k_labels, y_pred, a)
    return acc, f1, y_pred

def get_loss(model, batch, params): 
    (x, lengths, langs), y1, y2 = batch
        
    #if params.n_langs > 1 :
    if False :
        langs = langs.to(params.device)
    else : 
        langs = None
    logits, loss = model(x.to(params.device), lengths.to(params.device), y=y1.to(params.device), positions=None, langs=langs)
        
    stats = {}
    n_words = lengths.sum().item()
    stats['n_words'] = n_words
    #stats["xe_loss"] = loss.item() * 6
    stats["avg_loss"] = loss.item()
    stats["loss"] = loss.item()
        
    logits = F.softmax(logits, dim = -1)
    stats["label_pred"] = logits.max(1)[1].view(-1).detach().cpu().numpy()
    stats["label_id"] = y2.view(-1).numpy()
        
    for k in range(1, params.topK+1):
        k_acc, k_f1, y_pred = top_k(logits = logits.detach().cpu(), y=y2, k=k)
        stats["top%d_avg_acc"%k] = k_acc
        stats["top%d_avg_f1_score"%k] = k_f1
        stats["top%d_acc"%k] = k_acc
        stats["top%d_f1_score"%k] = k_f1
        stats["top%d_label_pred"] = y_pred
        
    #if params.version == 2 :
    #    stats["q_c"] = logits.detach().cpu()#.numpy()
    #    stats["p_c"] = y1.detach().cpu()#.numpy()
        
    acc = get_acc(stats["label_pred"], stats["label_id"])
    stats["avg_acc"] = acc
    stats["acc"] = acc
        
    f1 = f1_score_func(stats["label_pred"], stats["label_id"])
    stats["avg_f1_score"] = f1
    stats["f1_score"] = f1
        
    return loss, stats

def end_of_epoch(stats_list, val_first = True):
    scores = {}
    for prefix, total_stats in zip(["val", "train"] if val_first else ["train","val"], stats_list):
        loss = []
        label_pred = []
        label_ids = []
            
        label_pred_topK = { k : [] for k in range(1, params.topK+1)}
            
        for stats in total_stats :
            label_pred.extend(stats["label_pred"])
            label_ids.extend(stats["label_id"])
            loss.append(stats['loss'])
                
            for k in range(1, params.topK+1):
                label_pred_topK[k].extend(stats["top%d_label_pred"])

        scores["%s_acc"%prefix] = get_acc(label_pred, label_ids) #accuracy_score(label_pred, label_ids)*100
        scores["%s_f1_score_weighted"%prefix] = f1_score_func(label_pred, label_ids)
        
        print(label_pred, label_ids)
        for k in range(1, params.topK+1):
            print(k, label_pred_topK[k], label_ids)
            scores["top%d_%s_acc"%(k, prefix)] = get_acc(label_pred_topK[k], label_ids)
            scores["top%d_%s_f1_score_weighted"%(k, prefix)] = f1_score_func(label_pred_topK[k], label_ids)
            
        #report = classification_report(y_true = label_ids, y_pred = label_pred, labels=label_dict.values(), 
        #    target_names=label_dict.keys(), sample_weight=None, digits=4, output_dict=True, zero_division='warn')
        report = classification_report(y_true = label_ids, y_pred = label_pred, digits=4, output_dict=True)
        m_v = {'precision': [], 'recall': [], 'f1-score':[]}
        if True :
            for k in report :
                if k in label_dict.keys() :
                    v = report.get(k, None)
                    for k1 in m_v.keys():
                        m_v[k1] = m_v.get(k1, []) + [v[k1]]
                    scores["%s_class_%s"%(prefix, k)] = v
                else :
                    scores["%s_%s"%(prefix, k)] = report.get(k, None)
            
            # AP_c and MAP
            for k in m_v.keys():
                v = m_v.get(k, [0.0])
                scores["%s_mean_average_%s"%(prefix, k)] = sum(v) / len(v)
                        
        l = len(loss)
        l = l if l != 0 else 1
        scores["%s_loss"%prefix] = sum(loss) / l
        
    return scores

def main(params):

    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)
    
    # initialize the experiment
    logger = initialize_exp(params)

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler()
            
    model = BertClassifier(n_labels = 6, params = params, logger = logger)
    model = model.to(params.device)
            
    train_dataset, val_dataset = load_dataset(params, logger, model.dico)
    #logger.info(model.dico.word2id)
    #exit()
    
    # optimizers
    if False :
        lr= 1e-4
        betas=(0.9, 0.999) 
        weight_decay=0.01
        optimizer_p = Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        #optimizer_e = get_optimizer(model.model.parameters(), params.optimizer)
    else :
        #params.optimizer_e = params.optimizer
        params.optimizer_p = params.optimizer
        if not params.freeze_transformer :
            #optimizer_e = get_optimizer(model.model.parameters(), params.optimizer_e)
            optimizer_p = get_optimizer(model.model.pred_layer.parameters(), params.optimizer_p)
        else :
            #optimizer_e =  None
            optimizer_p = get_optimizer(model.model.pred_layer.parameters(), params.optimizer_p)
        
    trainer = Trainer(params, model, optimizer_p, train_dataset, val_dataset, logger)
    
    logger.info("")
    if not params.eval_only :
        trainer.train(get_loss, end_of_epoch)
    else :
        trainer.eval(get_loss, end_of_epoch)
        
if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    
    parser.add_argument("--freeze_transformer", type=bool_flag, default=True, 
                        help="freeze the transformer encoder part of the model")
    parser.add_argument('--version', default=1, const=1, nargs='?',
                        choices=[1, 2], 
                        help=  '1 : averaging the labels with the confidence scores as weights (might be noisy) \
                                2 : computed the coefficient of variation (CV) among annotators for \
                                    each sample in the dataset \
                                    see bias_classification_loss.py for more informations about v2')
    
    #if parser.parse_known_args()[0].version == 2:
    parser.add_argument("--log_softmax", type=bool_flag, default=True, 
                        help="use log_softmax in the loss function instead of log")

    parser.add_argument("--train_data_file", type=str, default="", help="file (.csv) containing the data")
    parser.add_argument("--val_data_file", type=str, default="", help="file (.csv) containing the data")
    
    parser.add_argument("--shuffle", type=bool_flag, default=False, help="shuffle Dataset")
    #parser.add_argument("--group_by_size", type=bool_flag, default=True, help="Sort sentences by size during the training")
    
    parser.add_argument("--codes", type=str, required=True, help="path of bpe code")
    
    parser.add_argument("--min_len", type=int, default=1, 
                        help="minimun sentence length before bpe in training set")
    
    parser.add_argument('--debug_num', default=0, const=0, nargs='?',
                        choices=[0, 1, 2], 
                        help=  '0 : Transformer + Linear \
                                1 : Transformer + Linear + Tanh + Dropout + Linear \
                                2 : Transformer + GRU + Dropout + Linear')
    #if parser.parse_known_args()[0].debug_num in [0, 2] :
    parser.add_argument("--hidden_dim", type=int, default=-1, 
                        help="hidden dimension of classifier")
    
    # GRU
    #if parser.parse_known_args()[0].debug_num == 2 :
    parser.add_argument("--gru_n_layers", type=int, default=1, 
                        help="number of layers, GRU")
    parser.add_argument("--bidirectional", type=bool_flag, default=False, help="bidirectional GRU or not")
    
    parser.add_argument('--topK', default=3, const=3, nargs='?',
                        choices=[1, 2, 3, 4, 5, 6], 
                        help="topK")
    """
    if not os.path.isfile(from_config_file(parser.parse_known_args()[0]).reload_model) :
        parser.add_argument("--model_path", type=str, default="", help="Model path")
        params = parser.parse_args()
    else :
        params = parser.parse_args()
        params.model_path = params.reload_model
    """
    params = parser.parse_args()
    params = from_config_file(params)
    params.model_path = params.reload_model

    set_seeds(params.random_seed)

    if params.device not in ["cpu", "cuda"] :
        params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else :
        params.device = torch.device(params.device)

    # check parameters
    assert os.path.isfile(params.model_path)
    assert os.path.isfile(params.codes)
    
    # run experiment
    main(params)