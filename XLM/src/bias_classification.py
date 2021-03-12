from torch.utils.data import Dataset
from torch import Tensor
import torch
import torch.nn.functional as F
import torch.nn as nn

#eps = torch.finfo(torch.float32).eps # 1.1920928955078125e-07
#eps = 1e-20 # TODO : search for the smallest `eps` number under pytorch such as `torch.log(eps) != -inf`

## Bert for classification
class BertClassifier(nn.Module):
    """BERT model for token-level classification"""
    def __init__(self, bert, n_labels, dropout=0.1, debug_num = 0):
        super().__init__()
        self.transformer = bert
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
        h = self.bert('fwd', x=batch, lengths=lengths, langs=langs, causal=False)
        h = h.transpose(0, 1)
        # h : (batch_size, input_seq_len, d_model)
        # [CLS] : The final hidden state corresponding to this token is used as the aggregate 
        # sequence representation for classification
        # BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (page : todo)
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
    def __init__(self, file, pipeline=[], n_samples = None, shuffle=False, group_by_size = True, version = 1):
        self.n_samples = n_samples
        self.shuffle = shuffle
        self.group_by_size = group_by_size
        assert version in [1, 2]
        self.version = version

        self.tensors = []    
        for instance in self.get_instances(pd.read_csv(file)): # instance : tuple of fields
                for proc in pipeline: # a bunch of pre-processing
                    instance = proc(instance)
                self.tensors.append(instance)
    
    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, index):
        return self.tensors[index]

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
            text_a = line[1]
            b, c = line[2:5], line[5:8]
            s = sum(c)
            s = 1 if s == 0 else s
            if self.version == 1 :
                label = sum([ label * conf for label, conf in  zip(b, c) ])// s
                yield label, text
            elif self.version == 2 : 
                p_c = [0]*6
                for (b_i, c_i) in zip(b, c) :
                    p_c[b_i] += c_i/s

                yield p_c, text


