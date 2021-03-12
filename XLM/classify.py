import torch

from src.slurm import init_signal_handler, init_distributed_mode
from src.utils import initialize_exp, set_seeds
from src.bias_classification import BiasClassificationLoss, BiasClassificationDataset

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
    encoder.load_state_dict(reloaded['encoder'])
    params.id = model_params.lang2id[params.lang]

    # TODO

if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()

    # model / output paths
    parser.add_argument("--model_path", type=str, default="", help="Model path")
    params = parser.parse_args()
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