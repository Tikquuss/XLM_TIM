import torch
import torch.nn as nn
from torch.optim import Adam
#from optim import ScheduledOptim
import os
import numpy as np

from src.slurm import init_signal_handler, init_distributed_mode
from src.utils import initialize_exp, set_seeds
from src.bias_classification import BiasClassificationLoss, BiasClassificationDataset, BertClassifier, Trainer
from src.utils import AttrDict
from src.data.dictionary import Dictionary
from src.model.transformer import TransformerModel
from src.utils import bool_flag

from params import get_parser, from_config_file

def get_acc(pred, label):
    arr = (np.array(pred) == np.array(label)).astype(float)
    if arr.size != 0: # check NaN 
        return arr.mean()*100
    return 0


def main(params):

    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)
    
    # initialize the experiment
    logger = initialize_exp(params)

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler()

    logger.warning("Reload transformer model path from %s"%params.model_path)
    reloaded = torch.load(params.model_path, map_location=params.device)
    model_params = AttrDict(reloaded['params'])
    logger.info("Supported languages: %s" % ", ".join(model_params.lang2id.keys()))

    # update dictionary parameters
    for name in ['n_words', 'bos_index', 'eos_index', 'pad_index', 'unk_index', 'mask_index']:
        setattr(params, name, getattr(model_params, name))

    # build dictionary / build encoder / build decoder / reload weights
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    encoder = TransformerModel(model_params, dico, is_encoder=True, with_output=True).to(params.device)
    encoder.load_state_dict(reloaded['model'])
    
    if params.freeze_transformer :
        for param in encoder.parameters():
            param.requires_grad = False
            
    model = BertClassifier(bert = encoder, n_labels = 6, dropout=0.1, debug_num = 0)
    
    logger.info(model)
            
    params.lang = params.lgs
    params.lang_id = model_params.lang2id[params.lang]
    
    params.train_n_samples = None if params.train_n_samples==-1 else params.train_n_samples
    params.valid_n_samples = None if params.valid_n_samples==-1 else params.valid_n_samples
    
    if not params.eval_only :
        logger.info("Loading data from %s ..."%params.train_data_file)
        train_dataset = BiasClassificationDataset(params.train_data_file, params, dico, logger, params.train_n_samples)
        setattr(params, "train_num_step", len(train_dataset))
        setattr(params, "train_num_data", train_dataset.n_samples)
    else :
        train_dataset = None
    
    logger.info("Loading data from %s ..."%params.val_data_file)
    val_dataset = BiasClassificationDataset(params.val_data_file, params, dico, logger, params.valid_n_samples)

    logger.info("============ Data summary")
    if not params.eval_only :
        logger.info("train : %d"%train_dataset.n_samples)
    logger.info("valid : %d"%val_dataset.n_samples)
    logger.info("")
    
    if params.version == 2 :
        # If a softmax is applied to the model output, 
        # log_softmax is no longer required for the cross-entropy operation (log will be sufficient).
        if params.softmax :
            assert not params.log_softmax, "Softmax is applied twice"
        criterion = BiasClassificationLoss(softmax = params.log_softmax).to(params.device)
    else :
        criterion = nn.CrossEntropyLoss().to(params.device)
        #criterion = nn.BCEWithLogitsLoss().to(params.device)

    lr= 1e-4
    betas=(0.9, 0.999) 
    weight_decay=0.01
    optimizer = Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    #warmup_steps=10000
    #optim_schedule = ScheduledOptim(optimizer, mode.transformer.d_model, n_warmup_steps=warmup_steps)
    # TODO
    
    def get_loss(model, batch): # make sure loss is a scalar tensor
        (x, lengths, langs), y = batch
        
        #if params.n_langs > 1 :
        if False :
            langs = langs.to(params.device)
        else : 
            langs = None
            
        logits = model(x.to(params.device), lengths.to(params.device), langs)
        
        loss = criterion(logits, y) 
        
        stats = {}
        n_words = x.size(0) * x.size(1)
        stats['n_words'] = n_words
        stats["loss"] = loss.item()
        if params.version == 2 :
            stats["q_c"] = logits.detach().cpu()#.numpy()
            stats["p_c"] = y.detach().cpu()#.numpy()
            stats["label_pred"] = logits.max(1)[1].view(-1).cpu().numpy()
            stats["label_id"] = y.max(1)[1].view(-1).cpu().numpy()
        else :
            stats["label_pred"] = logits.max(1)[1].view(-1).cpu().numpy()
            stats["label_id"] = y.view(-1).cpu().numpy()

        return loss, stats
    
    trainer = Trainer(params, model, optimizer, train_dataset, val_dataset, logger)
    
    def end_of_epoch(stats_list):
        scores = {}
        for prefix, total_stats in zip(["val", "train"], stats_list):
            loss = 0
            label_pred = []
            label_ids = []
            for stats in total_stats :
                label_pred.extend(stats["label_pred"])
                label_ids.extend(stats["label_id"])
                loss += stats['loss']

            scores["%s_acc"%prefix] = get_acc(label_pred, label_ids) 
            if params.version == 2 :
                # TODO : AP_c and MAP using stats["q_c"] and stats["p_c"] : ask Olawale | Dianbo
                pass

            scores["%s_loss"%prefix] = loss / len(total_stats)

        return scores
    
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
    parser.add_argument("--softmax", type=bool_flag, default=False, 
                                help="use softmax as last layer of bert model")
    parser.add_argument('--version', default=1, const=1, nargs='?',
                        choices=[1, 2], 
                        help='1 : averaging the labels with the confidence scores as weights (might be noisy) \
                              2 : computed the coefficient of variation CV among in the dataset \
                              see bias_classification_loss.py for more informations about v2')
    
    #if parser.parse_known_args()[0].version == 2:
    parser.add_argument("--log_softmax", type=bool_flag, default=False, 
                        help="use log_softmax in the loss function instead of log")

    parser.add_argument("--train_data_file", type=str, default="", help="file (.csv) containing the data")
    parser.add_argument("--val_data_file", type=str, default="", help="file (.csv) containing the data")
    
    parser.add_argument("--shuffle", type=bool_flag, default=False, help="shuffle Dataset")
    #parser.add_argument("--group_by_size", type=bool_flag, default=True, help="Sort sentences by size during the training")
    
    parser.add_argument("--codes", type=str, required=True, help="path of bpe code")
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

    # run experiment
    main(params)