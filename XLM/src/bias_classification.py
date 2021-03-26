from torch.utils.data import Dataset
from torch import Tensor
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

import os
import json
from tqdm import tqdm
import itertools
import gc
import time
from collections import OrderedDict
import numpy as np
import pandas as pd
import random

from .utils import truncate, AttrDict 
from .data.dictionary import Dictionary
from .model.transformer import TransformerModel
from .trainer import Trainer as MainTrainer

#git clone https://github.com/NVIDIA/apex
#pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
#import apex

# bias corpus
URL="URL"
EMAIL="EMAIL"
PHONE_NUMBER="PHONE"
NUMBER="NUMBER"
DIGIT="DIGIT"
CUR="CUR" # currency_symbol
special_tokens = []
st = [URL, EMAIL, PHONE_NUMBER, NUMBER,DIGIT,CUR]
st = [s.lower() for s in st]
special_tokens.extend(st)
dico_rest=4+len(special_tokens)

label_dict = {"0":0, "1":1, "2":2, "3" : 3, "4" : 4, "5" : 5}

class GRU(nn.Module):
    def __init__(self, d_model, params):
        super().__init__()
        self.drop = nn.Dropout(params.dropout if not params.freeze_transformer else 0)
        self.rnn = nn.GRU(d_model, params.hidden_dim, num_layers = params.n_layers, 
                            bidirectional = params.bidirectional, batch_first = True,
                            dropout = 0 if params.n_layers < 2 else params.dropout)
    def forward(self, x):
        x = x.transpose(0, 1) # (batch_size, input_seq_len, d_model)
        _, x = self.rnn(self.drop(x)) # [n layers * n directions, batch size, emb dim]
        if self.rnn.bidirectional:
            return torch.cat((x[-2,:,:], x[-1,:,:]), dim = 1)
        else:
            return x[-1,:,:]   
        
class PredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    BERT model for token-level classification
    debug_num = 0 : Linear/AdaptiveLogSoftmaxWithLoss
    debug_num = 1 : Linear + Tanh + Dropout + Linear/AdaptiveLogSoftmaxWithLoss
    debug_num = 2 : GRU + Dropout + Linear/AdaptiveLogSoftmaxWithLoss
    """
    def __init__(self, d_model, n_labels, params):
        super().__init__()
        self.asm = params.asm
        self.n_labels = n_labels
        self.debug_num = params.debug_num
        if self.debug_num == 0 :
            net = [nn.Dropout(params.dropout if not params.freeze_transformer else 0)]
            if params.asm is False:
                net.append(nn.Linear(d_model, n_labels))
            else :
                in_features=d_model
        elif self.debug_num == 1 :
            net = [
                nn.Dropout(params.dropout if not params.freeze_transformer else 0),
                nn.Linear(d_model, params.hidden_dim), 
                nn.Tanh(),
                nn.Dropout(params.dropout)
            ]
            if params.asm is False:
                net.append(nn.Linear(params.hidden_dim, n_labels))
            else :
                in_features=params.hidden_dim
        elif self.debug_num == 2 :
            net = [
                nn.Dropout(params.dropout if not params.freeze_transformer else 0),
                GRU(d_model, params),
                nn.Dropout(params.dropout if params.n_layers < 2 else 0)
            ]
            if params.asm is False:
                net.append(nn.Linear(params.hidden_dim * 2 if params.bidirectional else params.hidden_dim, 
                                        n_labels))
            else :
                in_features=params.hidden_dim * 2 if params.bidirectional else params.hidden_dim
        else :
            raise NotImplementedError("debug_num = %d not found"%(self.debug_num))
        
        self.proj = nn.Sequential(*net)
        
        if params.asm is True:
            self.classifier = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=in_features,
                n_classes=n_labels,
                cutoffs=params.asm_cutoffs,
                div_value=params.asm_div_value,
                head_bias=True,  # default is False
            )
        
        else :
            if params.version == 2 :
                if False :
                    self.criterion = BiasClassificationLoss(softmax = params.log_softmax).to(params.device)
                else :
                    self.criterion = nn.BCEWithLogitsLoss().to(params.device)
            else :
                self.criterion = nn.CrossEntropyLoss().to(params.device)

    def forward(self, x, y, get_scores=False):
        """
        Compute the loss, and optionally the scores.
        """   
        x = self.proj(x)
        if self.asm is False:
            scores = x.view(-1, self.n_labels)
            loss = self.criterion(scores, y)
        else:
            _, loss = self.classifier(x, y)
            scores = self.classifier.log_prob(x) if get_scores else None

        return scores, loss

    def get_scores(self, x):
        """
        Compute scores.
        """    
        x = self.proj(x)
        assert x.dim() == 2
        return self.classifier.log_prob(x) if self.asm else x

## Bert for classification
class BertClassifier(nn.Module):
    """BERT model for token-level classification
    debug_num = 0 : Transformer + Linear 
    debug_num = 1 : Transformer + Linear + Tanh + Dropout + Linear
    debug_num = 2 : Transformer + GRU + Dropout + Linear'
    """
    def __init__(self, n_labels, params, logger):
        super().__init__()
        
        logger.warning("Reload transformer model path from %s"%params.model_path)
        reloaded = torch.load(params.model_path, map_location=params.device)
        model_params = AttrDict(reloaded['params'])
        logger.info("Supported languages: %s" % ", ".join(model_params.lang2id.keys()))

        # update dictionary parameters
        for name in ['n_words', 'bos_index', 'eos_index', 'pad_index', 'unk_index', 'mask_index']:
            setattr(params, name, getattr(model_params, name))

        # build dictionary / build encoder / build decoder / reload weights
        dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'], rest=dico_rest)
        model = TransformerModel(model_params, dico, is_encoder=True, with_output=True).to(params.device)
        state_dict = reloaded['model']
        # handle models from multi-GPU checkpoints
        if 'checkpoint' in params.model_path:
            state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        
        params.max_batch_size = 0
    
        self.model = model
        if params.freeze_transformer :
            for param in self.model.parameters():
                param.requires_grad = False
            #self.model.eval()
            
        d_model = model.dim
        params.hidden_dim = d_model if params.hidden_dim == -1 else params.hidden_dim
        self.model.pred_layer = PredLayer(d_model, n_labels, params)
        for param in self.model.pred_layer.parameters():
            param.requires_grad = True
        #self.model.pred_layer.train()
        
        logger.info(self.model)
                    
        self.freeze_transformer = params.freeze_transformer
        params.lang = params.lgs
        params.lang_id = model_params.lang2id[params.lang]
        self.dico = dico
        self.debug_num = params.debug_num

    def forward(self, x, lengths, y, positions=None, langs=None, get_scores = True):
        """
        Inputs:
            `x`        : LongTensor of shape (slen, bs)
            `lengths`  : LongTensor of shape (bs,)
        """
        slen, bs = x.size()
        assert lengths.size(0) == bs and lengths.max().item() == slen
        
        if self.freeze_transformer :
            with torch.no_grad():
                h = self.model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)#.contiguous() # (seq_len, batch_size, d_model)
        else :
            h = self.model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)#.contiguous() # (seq_len, batch_size, d_model)
        
        return self.model.pred_layer(h[0] if self.debug_num != 2 else h, y, get_scores = get_scores)

    def parameters(self):
        return self.model.pred_layer.parameters() if self.freeze_transformer else self.model.parameters()

# Below is one way to bpe-ize sentences
#codes = "" # path to the codes of the model
def to_bpe(sentences, codes : str, logger, fastbpe = os.path.join(os.getcwd(), 'tools/fastBPE/fast')):
    """Sentences have to be in the BPE format, i.e. tokenized sentences on which you applied fastBPE.
    https://github.com/facebookresearch/XLM/blob/master/generate-embeddings.ipynb"""
    # write sentences to tmp file
    tmp_file1 = './tmp_sentences.txt'
    tmp_file2 = './tmp_sentences.bpe'
    with open(tmp_file1, 'w') as fwrite:
        for sent in sentences:
            fwrite.write(sent + '\n')
    
    # apply bpe to tmp file
    os.system('%s applybpe %s %s %s' % (fastbpe, tmp_file2, tmp_file1, codes))
    
    # load bpe-ized sentences
    sentences_bpe = []
    with open(tmp_file2) as f:
        for line in f:
            sentences_bpe.append(line.rstrip())
    
    logger.info("Delete %s and %s"%(tmp_file1, tmp_file2))
    os.remove(tmp_file1)
    os.remove(tmp_file2)
    
    return sentences_bpe

class BiasClassificationDataset(Dataset):
    """ Dataset class for Bias Classification"""
    labels = (0, 1, 2, 3, 4, 5)
    def __init__(self, file, params, dico, logger, n_samples = None, min_len=1):
        assert params.version in [1, 2]
        self.params = params
        self.dico = dico
        self.n_samples = n_samples
        self.shuffle = params.shuffle
        self.group_by_size = params.group_by_size
        self.version = params.version
        self.in_memory = True

        data = [inst for inst in self.get_instances(pd.read_csv(file))]
        sentences, labels1, labels2 = zip(*data)
        
        # remove short sentence
        l = len(sentences)
        sentences = [sent for sent in sentences if len(sent.split(" ")) >= min_len]
        logger.info('Remove %d sentences of length < %d' % (l - len(sentences), min_len))
        
        # lower
        logger.info("Do lower...")
        sentences = [s.lower() for s in sentences]
        
        # bpe-ize sentences
        sentences = to_bpe(sentences, codes=params.codes, logger = logger)
        
        # check how many tokens are OOV
        n_w = len([w for w in ' '.join(sentences).split()])
        n_oov = len([w for w in ' '.join(sentences).split() if w not in dico.word2id])
        logger.info('Number of out-of-vocab words: %s/%s' % (n_oov, n_w))
        
        data = list(zip(sentences, labels1, labels2))

        self.n_samples = len(data)
        self.batch_size = self.n_samples if self.params.batch_size > self.n_samples else self.params.batch_size
        
        if self.in_memory :
            i = 0
            tmp = []
            while self.n_samples > i :
                i += self.batch_size
                x, y1, y2 = zip(*data[i-self.batch_size:i])
                tmp.append((self.to_tensor(x), torch.stack(y1), torch.stack(y2)))
            self.data = tmp
        else :
            self.data = data
        
    def __len__(self):
        return self.n_samples // self.batch_size

    def __getitem__(self, index):
        if not self.in_memory :
            x, y1, y2 = self.data[index]
            return self.to_tensor(x), y1, y2
        else :
            return self.data[index]
    
    def get_instances(self, df):
        columns = list(df.columns[1:]) # excapt "'Unnamed: 0'"
        rows = df.iterrows()
        if self.shuffle :
            random.shuffle(rows)
        if self.n_samples :
            rows = list(rows)[:self.n_samples]
        if self.group_by_size :
            rows = sorted(rows, key = lambda x : len(x[1]["content"].split()), reverse=False)
        for row in rows : 
            row = row[1]
            text = row["content"]
            b = [row['answerForQ1.Worker1'], row['answerForQ1.Worker2'], row['answerForQ1.Worker3']]
            c = [row['answerForQ2.Worker1'], row['answerForQ2.Worker2'], row['answerForQ2.Worker3']]
            
            s = sum(c)
            s = 1 if s == 0 else s
            if self.version == 1 :
                label = sum([ label * conf for label, conf in  zip(b, c) ])// s
                label = torch.tensor(label, dtype=torch.long)
                yield text, label, label
            elif self.version == 2 : 
                p_c = [0]*6
                for (b_i, c_i) in zip(b, c) :
                    p_c[b_i] += c_i/s
                label = b[np.argmax(a = c)] # the class with the maximum confidence score was used as the target
                yield text, torch.tensor(p_c, dtype=torch.float), torch.tensor(label, dtype=torch.long)
                
    def __iter__(self): # iterator to load data
        if not self.in_memory :
            i = 0
            while self.n_samples > i :
                i += self.batch_size
                x, y1, y2 = zip(*self.data[i-self.batch_size:i])
                yield self.to_tensor(x), torch.stack(y1), torch.stack(y2)
        else :
            for batch in self.data :
                yield batch
            
    def to_tensor(self, sentences):
        if type(sentences) == str :
            sentences = [sentences]
        else :
            assert type(sentences) in [list, tuple]
        
        # These two approaches are equivalent
        if False :
            word_ids = [torch.LongTensor([self.dico.index(w) for w in s.strip().split()]) for s in sentences]
            lengths = torch.LongTensor([len(s) + 2 for s in word_ids])
            batch = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.params.pad_index)
            batch[0] = self.params.eos_index
            for j, s in enumerate(word_ids):
                if lengths[j] > 2:  # if sentence not empty
                    batch[1:lengths[j] - 1, j].copy_(s)
                batch[lengths[j] - 1, j] = self.params.eos_index
            langs = batch.clone().fill_(self.params.lang_id)
            batch, lengths = truncate(batch, lengths, self.params.max_len, self.params.eos_index)
            return batch, lengths, langs
        else :
            # add </s> sentence delimiters
            sentences = [(('</s> %s </s>' % sent.strip()).split()) for sent in sentences]
            bs = len(sentences)
            slen = max([len(sent) for sent in sentences])
            word_ids = torch.LongTensor(slen, bs).fill_(self.params.pad_index)
            for i in range(bs):
                sent = torch.LongTensor([self.dico.index(w) for w in sentences[i]])
                #a = self.dico.ids_to_words(sent.numpy())
                #print(sent, sentences[i], a)
                word_ids[:len(sent), i] = sent
                
            lengths = torch.LongTensor([len(sent) for sent in sentences])
            # NOTE: No more language id (removed it in a later version)
            # langs = torch.LongTensor([params.lang2id[lang] for _, lang in sentences]).unsqueeze(0).expand(slen, bs) if params.n_langs > 1 else None
            langs = None
            word_ids, lengths = truncate(word_ids, lengths, self.params.max_len, self.params.eos_index)
            return word_ids, lengths, langs
        
def load_dataset(params, logger, dico) :
    params.train_n_samples = None if params.train_n_samples==-1 else params.train_n_samples
    params.valid_n_samples = None if params.valid_n_samples==-1 else params.valid_n_samples
    
    if not params.eval_only :
        logger.info("Loading data from %s ..."%params.train_data_file)
        train_dataset = BiasClassificationDataset(params.train_data_file, params, dico, 
                                                logger, params.train_n_samples, min_len = params.min_len)
        setattr(params, "train_num_step", len(train_dataset))
        setattr(params, "train_num_data", train_dataset.n_samples)
    else :
        train_dataset = None
    
    logger.info("")
    logger.info("Loading data from %s ..."%params.val_data_file)
    val_dataset = BiasClassificationDataset(params.val_data_file, params, dico, logger, params.valid_n_samples)

    logger.info("")
    logger.info("============ Data summary")
    if not params.eval_only :
        logger.info("train : %d"%train_dataset.n_samples)
    logger.info("valid : %d"%val_dataset.n_samples)
    logger.info("")
    
    return train_dataset, val_dataset

""" Training Class"""
#possib = ["%s_%s_%s"%(i, j, k) for i, j, k in itertools.product(["train", "val"], ["mlm", "nsp"], ["ppl", "acc", "loss"])]
possib = []
possib.extend(["%s_%s"%(i, j) for i, j in itertools.product(["train", "val"], ["f1_score_weighted", "acc", "loss"])])
tmp_type = lambda name : "ppl" in name or "loss" in name

class Trainer(object):
    """Training Helper Class"""
    def __init__(self, params, model, optimizer, train_data_iter, val_data_iter, logger):
        self.params = params
        self.model = model
        self.optimizer = optimizer # optim

        # iterator to load data
        self.train_data_iter = train_data_iter 
        self.val_data_iter = val_data_iter 

        self.device = params.device # device name
        self.logger = logger

        # epoch / iteration size
        self.epoch_size = self.params.epoch_size
        if self.epoch_size == -1 and not params.eval_only:
            self.epoch_size = self.params.train_num_data
        assert self.epoch_size > 0 or params.eval_only

        # validation metrics
        self.metrics = []
        metrics = [m for m in self.params.validation_metrics.split(',') if m != '']
        for i in range(len(metrics)) :
            if tmp_type(metrics[i]) :
                metrics[i] = '_%s'%metrics[i]
        for m in metrics:
            m = (m[1:], False) if m[0] == '_' else (m, True)
            assert m[0] in possib
            self.metrics.append(m)
        self.best_metrics = {metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics}

        # stopping criterion used for early stopping
        if self.params.stopping_criterion != '':
            split = self.params.stopping_criterion.split(',')
            assert len(split) == 2 and split[1].isdigit()
            assert split[0] in possib
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0

            if tmp_type(split[0]) :
                split[0] = '_%s'%split[0]

            if split[0][0] == '_':
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_criterion = None


        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sentences = 0
        self.stats = OrderedDict([('processed_s', 0), ('processed_w', 0)])
        self.last_time = time.time()

        self.log_interval = self.params.log_interval
        if self.log_interval == -1 and not params.eval_only:
            self.log_interval = self.params.batch_size
        assert self.log_interval > 0 or params.eval_only

        if params.reload_checkpoint :
            self.load_checkpoint(checkpoint_path = params.reload_checkpoint)        

        self.checkpoint_path = os.path.join(params.dump_path, "checkpoint.pth")
        if os.path.isfile(self.checkpoint_path) :
            self.load_checkpoint()
    
        nb_p = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f'Found {nb_p:,} trainable parameters in model.\n')
        
        # float16 / distributed (no AMP)
        assert params.amp >= 1 or not params.fp16
        #assert params.amp >= 0 or params.accumulate_gradients == 1
        self.model = self.model.to(self.device)
        if params.multi_gpu and params.amp == -1:
            self.logger.info("Using nn.parallel.DistributedDataParallel ...")
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[params.local_rank], output_device=params.local_rank, broadcast_buffers=True)
        
        if params.amp >= 0:
            self.init_amp()
            if params.multi_gpu:
                self.logger.info("Using apex.parallel.DistributedDataParallel ...")
                import apex
                self.model = apex.parallel.DistributedDataParallel(self.model, delay_allreduce=True)

    def init_amp(self):
        """
        Initialize AMP optimizer.
        """
        assert self.params.amp == 0 and self.params.fp16 is False or self.params.amp in [1, 2, 3] and self.params.fp16 is True
        
        # Allow Amp to perform casts as required by the opt_level : https://nvidia.github.io/apex/amp.html
        import apex
        self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer, opt_level='O%i' % self.params.amp)

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1

    def optimize(self, loss, retain_graph=False):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            self.logger.warning("NaN detected")
            # exit()

        # regular optimization
        if self.params.amp == -1:
            if self.params.accumulate_gradients == 1 :
                self.optimizer.zero_grad()
                loss.backward(retain_graph=retain_graph)
                
                if self.params.clip_grad_norm > 0:
                    # norm_check_a = (sum([p.grad.norm(p=2).item() ** 2 for p in self.model.parameters()])) ** 0.5
                    clip_grad_norm_(self.model.parameters(), self.params.clip_grad_norm)
                    # norm_check_b = (sum([p.grad.norm(p=2).item() ** 2 for p in self.model.parameters()])) ** 0.5
                    # self.logger.info(norm_check_a, norm_check_b)
                
                self.optimizer.step()
            else : # accumulate gradient if need
                # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20
                if self.n_iter % self.params.accumulate_gradients == 0:
                    loss.backward(retain_graph=retain_graph)
                    if self.params.clip_grad_norm > 0:
                        # norm_check_a = (sum([p.grad.norm(p=2).item() ** 2 for p in self.model.parameters()])) ** 0.5
                        clip_grad_norm_(self.model.parameters(), self.params.clip_grad_norm)
                        # norm_check_b = (sum([p.grad.norm(p=2).item() ** 2 for p in self.model.parameters()])) ** 0.5
                        # self.logger.info(norm_check_a, norm_check_b)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                else :
                    loss.backward(retain_graph=retain_graph)


        # AMP optimization
        else:
            import apex
            if self.n_iter % self.params.accumulate_gradients == 0:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward(retain_graph=retain_graph)
                
                if self.params.clip_grad_norm > 0:
                    # norm_check_a = (sum([p.grad.norm(p=2).item() ** 2 for p in self.model.parameters()])) ** 0.5
                    clip_grad_norm_(apex.amp.master_params(self.optimizer), self.params.clip_grad_norm)
                    # norm_check_b = (sum([p.grad.norm(p=2).item() ** 2 for p in self.model.parameters()])) ** 0.5
                    # self.logger.info(norm_check_a, norm_check_b)

                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                with apex.amp.scale_loss(loss, optimizer, delay_unscale=True) as scaled_loss:
                    scaled_loss.backward(retain_graph=retain_graph)

    def plot_score(self, scores):
        for key, value in scores.items():
            try :
                self.logger.info("%s -> %.6f" % (key, value))
            except TypeError: #must be real number, not dict
                self.logger.info("%s -> %s" % (key, value))
        if self.params.is_master:
            self.logger.info("__log__:%s" % json.dumps(scores))

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            if metric not in scores:
                self.logger.warning("Metric \"%s\" not found in scores!" % metric)
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                self.logger.info('New best score for %s: %.6f' % (metric, scores[metric]))
                self.save_checkpoint('best_%s' % metric, include_optimizer=False)
 
    def save_checkpoint(self, name, include_optimizer = True):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return

        checkpoint_path = os.path.join(self.params.dump_path, '%s.pth' % name)
        self.logger.info("Saving %s to %s ..." % (name, checkpoint_path))

        data = {
            "model" : self.model.state_dict(), 
            "params": {k: v for k, v in self.params.__dict__.items()},
            'epoch': self.epoch,
            'n_total_iter': self.n_total_iter,
            'best_metrics': self.best_metrics,
            'best_criterion': self.best_criterion,
        }

        if include_optimizer:
            self.logger.warning(f"Saving optimizer ...")
            data['optimizer'] = self.optimizer.state_dict()

        torch.save(data, checkpoint_path)

    def load_checkpoint(self, checkpoint_path = None):
        """
        Reload a checkpoint if we find one.
        """
        """
        checkpoint_path = self.checkpoint_path
        if not os.path.isfile(checkpoint_path):
            if self.params.reload_checkpoint == '':
                return
            else:
                checkpoint_path = self.params.reload_checkpoint
                assert os.path.isfile(checkpoint_path)
        """
        checkpoint_path = self.checkpoint_path if checkpoint_path is None else checkpoint_path

        reloading_checkpoint_condition = not self.params.eval_only or (self.params.eval_only and not self.params.reload_model)  

        if reloading_checkpoint_condition : 
            if self.params.eval_only :
                self.logger.warning("You started the evaluation without specifying the model to be used for the evaluation, so the last checkpoint found will be loaded.")
            self.logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")

        assert os.path.isfile(checkpoint_path)
        data = torch.load(checkpoint_path, map_location='cpu')
        # reload model parameters
        self.model.load_state_dict(data["model"])

        if not self.params.eval_only :
            # reload optimizer
            if reloading_checkpoint_condition :
                if False:  # AMP checkpoint reloading is buggy, we cannot do that - TODO: fix - https://github.com/NVIDIA/apex/issues/250
                    self.logger.warning(f"Reloading checkpoint optimizer ...")
                    self.optimizer.load_state_dict(data['optimizer'])
                else:  # instead, we only reload current iterations / learning rates
                    self.logger.warning(f"Not reloading checkpoint optimizer.")
                    for group_id, param_group in enumerate(self.optimizer.param_groups):
                        if 'num_updates' not in param_group:
                            self.logger.warning(f"No 'num_updates' for optimizer.")
                            continue
                        self.logger.warning(f"Reloading 'num_updates' and 'lr' for optimizer.")
                        param_group['num_updates'] = data['optimizer']['param_groups'][group_id]['num_updates']
                        param_group['lr'] = self.optimizer.get_lr_for_step(param_group['num_updates'])

            # reload main metrics
            self.epoch = data['epoch'] + 1
            self.n_total_iter = data['n_total_iter']
            self.best_metrics = data['best_metrics']
            self.best_criterion = data['best_criterion']
            if reloading_checkpoint_condition :
                self.logger.warning(f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ...")
            else :
                self.logger.warning(f"Parameters reloaded. Epoch {self.epoch} / iteration {self.n_total_iter} ...")

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if not self.params.is_master:
            return
        if self.params.save_periodic > 0 and self.epoch % self.params.save_periodic == 0:
            self.save_checkpoint('periodic_%i' % self.epoch, include_optimizer=False)

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None and (self.params.is_master or not False):
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_criterion:
                self.best_criterion = scores[metric]
                self.logger.info("New best validation score: %f" % self.best_criterion)
                self.decrease_counts = 0
            else:
                self.logger.info("Not a better validation score (%i / %i)."
                            % (self.decrease_counts, self.decrease_counts_max))
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                self.logger.info("Stopping criterion has been below its best value for more "
                            "than %i epochs. Ending the experiment..." % self.decrease_counts_max)
                if self.params.multi_gpu and 'SLURM_JOB_ID' in os.environ:
                    os.system('scancel ' + os.environ['SLURM_JOB_ID'])
                exit()
        self.save_checkpoint("checkpoint", include_optimizer=True)
        self.epoch += 1
        
    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % self.log_interval != 0:
            return

        s_iter = "%7i - " % self.n_total_iter
        s_stat = ' || '.join([
            '{}: {:7.4f}'.format(k, np.mean(v)) for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        ])
        for k in self.stats.keys():
            #if type(self.stats[k]) is list:
            #    del self.stats[k][:]
            if ("loss" in k or "acc" in k or 'f1_score' in k) and (not "avg" in k) :
                self.stats[k] = []

        # learning rates
        s_lr = ""
        s_lr = s_lr + (" - LR: ") + " / ".join("{:.4e}".format(group['lr']) for group in self.optimizer.param_groups)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} sent/s - {:8.2f} words/s - ".format(
            self.stats['processed_s'] * 1.0 / diff,
            self.stats['processed_w'] * 1.0 / diff
        )
        self.stats['processed_s'] = 0
        self.stats['processed_w'] = 0
        self.last_time = new_time

        progress = str(self.stats['progress'])+"% -"
        # log progress + speed + stats + learning rate
        self.logger.info("")
        self.logger.info(s_iter + progress + s_speed + s_stat + s_lr)

    def train_step(self, get_loss):
        self.model.train() # train mode
        total_stats = []

        for i, batch in enumerate(self.train_data_iter):

            # forward / loss
            loss, stats = get_loss(self.model, batch, self.params)
            #loss = loss.mean() # mean() for Data Parallelism
            
            # optimize
            self.optimize(loss)

            total_stats.append(stats)

            # number of processed sentences / words
            self.n_sentences += self.params.batch_size
            self.stats['processed_s'] += self.params.batch_size
            self.stats['processed_w'] += stats['n_words']
            self.stats['progress'] = min(int(((i+1)/self.params.train_num_step)*100), 100) 

            for name in stats.keys() :
                if "loss" in name or 'acc' in name or "f1_score" in name:
                    self.stats[name] = self.stats.get(name, []) + [stats[name]]

            self.iter()
            self.print_stats()
    
            if self.epoch_size < self.n_sentences :
                break

        return total_stats

    def eval_step(self, get_loss):
        self.model.eval() # eval mode
        total_stats = []
        with torch.no_grad(): 
            for batch in tqdm(self.val_data_iter, desc='val'):
                _, stats = get_loss(self.model, batch, self.params) 
                total_stats.append(stats)
        return total_stats
    
    def train(self, get_loss, end_of_epoch):
        """ Train Loop """

        for _ in range(self.params.max_epoch):
            
            self.logger.info("============ Starting epoch %i ... ============" % self.epoch)
            self.n_sentences = 0
            for k in self.stats.keys():
                if "avg" in k :
                    self.stats[k] = []
            train_stats = self.train_step(get_loss)
            self.logger.info("============ End of epoch %i ============" % self.epoch)

            val_stats = self.eval_step(get_loss)

            scores = end_of_epoch([val_stats, train_stats])
            self.plot_score(scores)

            # end of epoch
            self.save_best_model(scores)
            self.save_periodic()
            self.end_epoch(scores)

    def eval(self, get_loss, end_of_epoch):
        """ Eval Loop """
        val_stats = self.eval_step(get_loss)
        scores = end_of_epoch([val_stats])
        self.plot_score(scores)

#eps = torch.finfo(torch.float32).eps # 1.1920928955078125e-07
#eps = 1e-20 # TODO : search for the smallest `eps` number under pytorch such as `torch.log(eps) != -inf`

def bias_classification_loss(q_c: Tensor, p_c: Tensor, reduction : str = "mean", softmax = False) -> Tensor:
    r"""assume p_c, q_c is (batch_size, num_of_classes)

    We have implemented this version to be able to call it directly as torch.nn.functional.some_loss_
    
    mean has been used by default for reduction in Olawale | Dianbo | Yoshua paper.
    They used log instead of log_softmax, but some q_c entries can be null if a softmax 
    is not applied to the output of the model beforehand, resulting in an infinite loss (nan).

    \begin{equation}
        L_{model} = \frac{1}{N} \sum_{i=1}^{N} CE\bigg(p\big(x_i\big),q\big(x_i\big)\bigg)
    \end{equation}
    where $CE(p(x_i), q(x_i))$ is the cross entropy between $p(x_i)$ and $q(x_i)$ for the $ith$ sample, and $N$ is the size of the dataset.
    
    \begin{equation}
        CE(p,q) = -\sum_{i=1}^{c}p_c(x)\log(q_c(x))
    \end{equation}
    
    $q_c(x)$ is the predicted probability of sample $x$ in class $c$, equivalently, the output probabilities from the model.
    $p_c(x)$ is the probability of sample $x$ in class $c$, equivalently, $p_c(x)$ is a $c-length$ vector with entries such that $\sum_{i=1}^{c}p_c(x)=1$. The entries of $p_c(x)$ are the normalized confidence scores of the annotators with index given by the respective voted class. As an example, for this sample with $S=(b, c) = ([4, 3, 2], [4, 3, 5])$, the bias scores of the $3$ different annotators with their confidence level is represented with an array of tuples,  $S$,where each tuple,  $(b_i,c_i)$ is the bias score $b_i$ with the associated confidence score, $c_i$ by annotator $i$. To calculate $p_c(S)$, we first normalize the confidence scores across the $3$ different annotators such that $\sum_{i=1}^{3}c_i=1$. The resulting $p_c(x)$ for the entry, is :
    
    \begin{align*}
      S &= \bigg[ (4,4), (3,3), (2,5) \bigg] \\
      S_{normalized} &=  \bigg[ (4,4/12= 0.3333), (3, 3/12=0.25), (2,5/12=0.4167) \bigg] \\
      p_c(S) &= [ 0., 0., 0.4167, 0.25, 0.3333, 0. ]
    \end{align*}  

    >>> p_c = torch.tensor([[0, 0, 0.4166666666666667, 0.25, 0.3333333333333333, 0],
                            [0, 0, 0.4166666666666667, 0.25, 0.3333333333333333, 0]])

    >>> q_c = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.3, 0.3],
                            [0.5, 0.2, 0, 0.1, 0.2, 0]])

    >>> bias_classification_loss(q_c, p_c, softmax=True) 
        tensor(1.8391)

    >>> p_c = torch.tensor([[0.4, 0.6], [0.1, 0.9]])
    >>> q_c = torch.tensor([[0.5, 0.5], [0.6, 0.4]])

    >>> bias_classification_loss(q_c, p_c, softmax=False) 
        tensor(0.7844)
    """
    assert reduction in ["mean", "sum", "none"]
    #assert torch.equal(torch.sum(p_c, dim = 1), torch.ones(bach_size, dtype=p_c.dtype))
    #assert torch.equal(torch.sum(q_c, dim = 1), torch.ones(bach_size, dtype=q_c.dtype))
    
    if softmax :
        CE = torch.sum(- p_c * F.log_softmax(q_c, dim = 1), dim = 1) # batch_size
    else :
        CE = torch.sum(- p_c * torch.log(q_c + torch.finfo(q_c.dtype).eps), dim = 1) # batch_size
    if reduction == "none" :
        return CE
    elif reduction == "mean" :
        return torch.mean(CE)
    elif reduction == "sum" :
        return torch.sum(CE)

class BiasClassificationLoss(nn.Module):
    r"""We have implemented this version in order to be able to do .to(devise) on the loss function, motivated by 
    the basic loss functions in pytorch : https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
    
    mean has been used by default for reduction in Olawale | Dianbo | Yoshua paper.
    They used log instead of log_softmax, but some q_c entries can be null if a softmax 
    is not applied to the output of the model beforehand, resulting in an infinite loss (nan).

    \begin{equation}
        L_{model} = \frac{1}{N} \sum_{i=1}^{N} CE\bigg(p\big(x_i\big),q\big(x_i\big)\bigg)
    \end{equation}
    where $CE(p(x_i), q(x_i))$ is the cross entropy between $p(x_i)$ and $q(x_i)$ for the $ith$ sample, and $N$ is the size of the dataset.
    
    \begin{equation}
        CE(p,q) = -\sum_{i=1}^{c}p_c(x)\log(q_c(x))
    \end{equation}
    
    $q_c(x)$ is the predicted probability of sample $x$ in class $c$, equivalently, the output probabilities from the model.
    $p_c(x)$ is the probability of sample $x$ in class $c$, equivalently, $p_c(x)$ is a $c-length$ vector with entries such that $\sum_{i=1}^{c}p_c(x)=1$. The entries of $p_c(x)$ are the normalized confidence scores of the annotators with index given by the respective voted class. As an example, for this sample with $S=(b, c) = ([4, 3, 2], [4, 3, 5])$, the bias scores of the $3$ different annotators with their confidence level is represented with an array of tuples,  $S$,where each tuple,  $(b_i,c_i)$ is the bias score $b_i$ with the associated confidence score, $c_i$ by annotator $i$. To calculate $p_c(S)$, we first normalize the confidence scores across the $3$ different annotators such that $\sum_{i=1}^{3}c_i=1$. The resulting $p_c(x)$ for the entry, is :
    
    \begin{align*}
      S &= \bigg[ (4,4), (3,3), (2,5) \bigg] \\
      S_{normalized} &=  \bigg[ (4,4/12= 0.3333), (3, 3/12=0.25), (2,5/12=0.4167) \bigg] \\
      p_c(S) &= [ 0., 0., 0.4167, 0.25, 0.3333, 0. ]
    \end{align*}  

    >>> p_c = torch.tensor([[0, 0, 0.4166666666666667, 0.25, 0.3333333333333333, 0],
                            [0, 0, 0.4166666666666667, 0.25, 0.3333333333333333, 0]])

    >>> q_c = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.3, 0.3],
                            [0.5, 0.2, 0, 0.1, 0.2, 0]])

    >>> BiasClassificationLoss(softmax=True)(q_c, p_c) 
        tensor(1.8391)

    >>> p_c = torch.tensor([[0.4, 0.6], [0.1, 0.9]])
    >>> q_c = torch.tensor([[0.5, 0.5], [0.6, 0.4]])

    >>> BiasClassificationLoss(softmax=False)(q_c, p_c) 
        tensor(0.7844)
    """
    def __init__(self, reduction: str = 'mean', softmax = False) -> None:
        super(BiasClassificationLoss, self).__init__()
        assert reduction in ["mean", "sum", "none"]
        self.reduction = reduction
        self.softmax = softmax
    
    def forward(self, q_c: Tensor, p_c: Tensor) -> Tensor:
        """assume p_c, q_c is (batch_size, num_of_classes)"""
        #assert torch.equal(torch.sum(p_c, dim = 1), torch.ones(bach_size, dtype=p_c.dtype))
        #assert torch.equal(torch.sum(q_c, dim = 1), torch.ones(bach_size, dtype=q_c.dtype))
        if self.softmax :
            CE = torch.sum(- p_c * F.log_softmax(q_c, dim = 1), dim = 1) # batch_size
        else :
            CE = torch.sum(- p_c * torch.log(q_c + torch.finfo(q_c.dtype).eps), dim = 1) # batch_size
        if self.reduction == "none" :
            return CE
        elif self.reduction == "mean" :
            return torch.mean(CE)
        elif self.reduction == "sum" :
            return torch.sum(CE)