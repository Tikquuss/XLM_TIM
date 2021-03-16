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

#git clone https://github.com/NVIDIA/apex
#pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
#import apex

import pandas as pd
import random

#eps = torch.finfo(torch.float32).eps # 1.1920928955078125e-07
#eps = 1e-20 # TODO : search for the smallest `eps` number under pytorch such as `torch.log(eps) != -inf`

## Bert for classification
class BertClassifier(nn.Module):
    """BERT model for token-level classification"""
    def __init__(self, bert, n_labels, dropout=0.1, debug_num = 0):
        super().__init__()
        self.bert = bert
        d_model = bert.dim

        self.debug_num = debug_num
        if self.debug_num == 0 :
            self.fc = nn.Linear(d_model, d_model)
            self.activ = nn.Tanh()
            self.drop = nn.Dropout(dropout)
            self.classifier = nn.Linear(d_model, n_labels)
        else :
            self.classifier = nn.Linear(d_model, n_labels)

    def forward(self, batch, lengths, langs):
        h = self.bert('fwd', x=batch, lengths=lengths, langs=langs, causal=False) # (seq_len, batch_size, d_model)
        h = h.transpose(0, 1) # (batch_size, input_seq_len, d_model)
        # [CLS] : The final hidden state corresponding to this token is used as the aggregate 
        # sequence representation for classification
        # BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (page : TODO)
        C = h[:, 0] #or h[:,0,:] # (batch_size, d_model)
        if self.debug_num == 0 :
            pooled_h = self.activ(self.fc(C)) # (batch_size, d_model)
            logits = self.classifier(self.drop(pooled_h)) # (batch_size, n_labels)
        else :
            logits = self.classifier(C) # (batch_size, n_labels)

        return logits

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

class BiasClassificationDataset(Dataset):
    """ Dataset class for Bias Classification"""
    labels = (0, 1, 2, 3, 4, 5)
    def __init__(self, file, params, dico, n_samples = None):
        assert params.version in [1, 2]
        self.params = params
        self.dico = dico
        self.n_samples = n_samples
        self.shuffle = params.shuffle
        self.group_by_size = params.group_by_size
        self.version = params.version

        self.data = [inst for inst in self.get_instances(pd.read_csv(file))]
        """
        if params.shuffle :
            self.data = random.shuffle(list(self.data))
        if n_samples is not None :
            self.data = self.data[:n_samples]
        if params.group_by_size :
            self.data.sort(reverse=False, key = lambda x : len(x[0].split(" ")))
        """
        self.n_samples = len(self.data)
        self.batch_size = self.n_samples if self.params.batch_size > self.n_samples else self.params.batch_size
        
    def __len__(self):
        return self.n_samples // self.batch_size

    def __getitem__(self, index):
        x, y = self.data[index]
        return self.to_tensor(x), y
    
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
            line = [row[1][col] for col in columns] # text, label1, label2, label3, conf1, conf2, conf3

            if False :
                text = line[1]
                b, c = line[2:5], line[5:8]
            else :
                text = line[0]
                b, c = line[1:4], line[4:7]
            s = sum(c)
            s = 1 if s == 0 else s
            if self.version == 1 :
                label = sum([ label * conf for label, conf in  zip(b, c) ])// s
                yield text, torch.tensor(label, dtype=torch.long)
                #return text, torch.tensor(label, dtype=torch.long)
            elif self.version == 2 : 
                p_c = [0]*6
                for (b_i, c_i) in zip(b, c) :
                    p_c[b_i] += c_i/s

                yield text, torch.tensor(p_c, dtype=torch.float)
                #return text, torch.tensor(p_c, dtype=torch.float)
    
    def __iter__(self): # iterator to load data
        i = 0
        data = list(self.data) # TypeError: 'generator' object is not subscriptable
        while self.n_samples > i :
            i += self.batch_size
            x, y = zip(*data[i-self.batch_size:i])

            yield self.to_tensor(x), torch.stack(y)
            
    def to_tensor(self, sentences):
        if type(sentences) == str :
            sentences = [sentences]
        else :
            assert type(sentences) in [list, tuple]
        
        word_ids = [torch.LongTensor([self.dico.index(w) for w in s.strip().split()]) for s in sentences]
        lengths = torch.LongTensor([len(s) + 2 for s in word_ids])
        batch = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.params.pad_index)
        batch[0] = self.params.eos_index
        for j, s in enumerate(word_ids):
            if lengths[j] > 2:  # if sentence not empty
                batch[1:lengths[j] - 1, j].copy_(s)
            batch[lengths[j] - 1, j] = self.params.eos_index
        langs = batch.clone().fill_(self.params.lang_id)
        
        return batch, lengths, langs


""" Training Class"""
possib = ["%s_%s_%s"%(i, j, k) for i, j, k in itertools.product(["train", "val"], ["mlm", "nsp"], ["ppl", "acc", "loss"])]
possib.extend(["%s_%s"%(i, j) for i, j in itertools.product(["train", "val"], ["ppl", "acc", "loss"])])
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

        """
        # models and checkpoints    
        if params.reload_transformer :
            logger.warning("Reload transformer model path from %s"%params.reload_transformer)
            assert os.path.isfile(params.reload_transformer)
            self.load(pretrain_file = params.reload_transformer)

        if params.reload_model :
            logger.warning("Reload model from %s"%params.reload_model)
            assert os.path.isfile(params.reload_model)
            self.load(model_file = params.reload_model)

        if params.reload_checkpoint :
            self.load_checkpoint(checkpoint_path = params.reload_checkpoint)        

        self.checkpoint_path = os.path.join(params.dump_path, "checkpoint.pth")
        if os.path.isfile(self.checkpoint_path) :
            self.load_checkpoint()

        if params.freeze_transformer :
            for param in self.model.transformer.parameters():
                param.requires_grad = False
        """
        nb_p = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f'Found {nb_p:,} trainable parameters in model.\n')
        
        # float16 / distributed (no AMP)
        assert params.amp >= 1 or not params.fp16
        #assert params.amp >= 0 or params.accumulate_gradients == 1
        self.model = self.model.to(self.device)
        if params.multi_gpu and params.amp == -1:
            #self.logger.info("Using nn.DataParallel ...")
            #self.model = nn.DataParallel(self.model)
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
            self.logger.info("%s -> %.6f" % (key, value))
        if self.params.is_master:
            self.logger.info("__log__:%s" % json.dumps(scores))

    def load(self, model_file = None, pretrain_file = None):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file and os.path.isfile(model_file):
            #self.logger.info('Loading the model from', model_file)
            data = torch.load(model_file, map_location='cpu')
            if type(data) == dict :
                data = data["model"]
            self.model.load_state_dict(data)

        elif pretrain_file and os.path.isfile(pretrain_file): # use pretrained transformer
            #self.logger.info('Loading the pretrained model from', pretrain_file)
            if pretrain_file.endswith('.pth'): # pretrain model file in pytorch
                data = torch.load(pretrain_file, map_location='cpu')
                if type(data) == dict :
                    data = data["model"]
                self.model.transformer.load_state_dict(
                    {key[12:]: # remove 'transformer.' (in 'transformer.embedding.norm.bias' for example)
                        value
                        for key, value in data.items()
                        if key.startswith('transformer')} # load only transformer parts
                ) 

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
#############################################################
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
            if type(self.stats[k]) is list:
                del self.stats[k][:]
            if "loss" in k :
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
        #iter_bar = tqdm(self.train_data_iter, desc='Iter (train loss=X.XXX)')

        for i, batch in enumerate(self.train_data_iter):

            # cuda
            #batch = [t.to(self.device) for t in batch]

            # forward / loss
            loss, stats = get_loss(self.model, batch)
            loss = loss.mean() # mean() for Data Parallelism

            # optimize
            self.optimize(loss)

            total_stats.append(stats)
            #iter_bar.set_description('Iter (train loss=%5.3f)'%loss.item())

            # number of processed sentences / words
            self.n_sentences += self.params.batch_size
            # todo
            self.stats['processed_s'] += self.params.batch_size
            self.stats['processed_w'] += stats['n_words']
            self.stats['progress'] = min(int(((i+1)/self.params.train_num_step)*100), 100) 

            for name in stats.keys() :
                if "loss" in name :
                    self.stats[name] = self.stats.get(name, []) + [stats[name]]

            self.iter()
            self.print_stats()
    
            if self.epoch_size < self.n_sentences :
                break

        return total_stats

    def eval_step(self, get_loss):
        self.model.eval() # eval mode
        total_stats = []
        #iter_bar = tqdm(self.val_data_iter, desc='Iter (val loss=X.XXX)')
        #for i, batch in enumerate(iter_bar):
        for batch in tqdm(self.val_data_iter, desc='val'):
            # cuda
            #batch = [t.to(self.device) for t in batch]

            # forward / loss
            loss, stats = get_loss(self.model, batch)
            loss = loss.mean() # mean() for Data Parallelism

            total_stats.append(stats)
            #iter_bar.set_description('Iter (val loss=%5.3f)'%loss.item())

        return total_stats
    
    def train(self, get_loss, end_of_epoch):
        """ Train Loop """

        for _ in range(self.params.max_epoch):
            
            self.logger.info("============ Starting epoch %i ... ============" % self.epoch)
            self.n_sentences = 0
            train_stats = self.train_step(get_loss)
            self.logger.info("============ End of epoch %i ============" % self.epoch)

            val_stats = self.eval_step(get_loss)

            scores = end_of_epoch([train_stats, val_stats])
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