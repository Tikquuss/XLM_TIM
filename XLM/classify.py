import torch
import os

from src.slurm import init_signal_handler, init_distributed_mode
from src.utils import initialize_exp, set_seeds
from src.bias_classification import BiasClassificationLoss, BiasClassificationDataset, BertClassifier
from src.utils import AttrDict
from src.data.dictionary import Dictionary
from src.model.transformer import TransformerModel
from src.utils import bool_flag

from params import get_parser, from_config_file

def main(params):

    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)
    
    # initialize the experiment
    logger = initialize_exp(params)

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler()

    reloaded = torch.load(params.model_path)
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
            
    model = BertClassifier(bert = encoder, n_labels = 2, dropout=0.1, debug_num = 0)
            
    params.lang = params.lgs
    params.id = model_params.lang2id[params.lang]

    # TODO

if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    
    parser.add_argument("--freeze_transformer", type=bool_flag, default=True, 
                        help="freeze the transformer encoder part of the model")
    
    if not os.path.isfile(parser.parse_known_args()[0].reload_model) :
        parser.add_argument("--model_path", type=str, default="", help="Model path")
        params = parser.parse_args()
    else :
        params = parser.parse_args()
        params.model_path = params.reload_model
        
    params = from_config_file(params)

    set_seeds(params.random_seed)

    if params.device not in ["cpu", "cuda"] :
        params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else :
        params.device = torch.device(params.device)

    # check parameters
    assert os.path.isfile(params.model_path)

    # run experiment
    main(params)