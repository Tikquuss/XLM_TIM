import math
import json
import copy
from typing import Callable, List, Optional, Tuple, Any
import math
import numpy as np
import warnings

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Dropout, Linear, LayerNorm, Sequential
# The disadvantage of their method (MultiheadAttention) is that they impose d_embed=dk=dv.
#from torch.nn import MultiheadAttention
from torch.nn.init import xavier_uniform_
from torch.autograd import Variable

#from .transformer import MultiHeadAttention as MHA_dv_eq_dk_eq_dim_div_h #  custom_mha


"""
Note that we cannot use -inf (float("-inf") | -np.inf | ...) here, because at some edge cases, the attention 
weight (before softmax) for some padded element in query will become -inf, which results in NaN in model 
parameters and the final loss value (Thanks https://github.com/pytorch/fairseq/blob/master/fairseq/modules/transformer_layer.py#L122 for 
this remark)
"""
neg_inf = -1e18 # float("-inf")

def custom_rand(shape : tuple, a = 0, b = 1., random_seed = 0, requires_grad = False) :
    """generate a random tensor of shape `shape` fill with number in range (a, b)"""
    torch.manual_seed(random_seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    return (b - a) * torch.rand(shape).requires_grad_(requires_grad) + b 

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for _ in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        #return lambda x : x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        return F.gelu
    elif activation == "prelu" :
        a = 0.01
        def f(x):
            # return torch.relu(x) + a * (x - abs(x))*0.5
            return F.prelu(input = x, weight = torch.tensor(a))
        return f
    elif activation == "elu" :
        a = 1.
        def f(x):
            return F.elu(input = x, alpha = torch.tensor(a))
        return f

    raise RuntimeError("activation should be relu/gelu/prelu/elu, not {}".format(activation))


## Scaled dot product attention
"""
I commented this part well so that it could be useful to other people, because I had a little trouble 
implementing it correctly (an error in the mask can completely harm the performance of the model).
"""
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, 
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None, pad_is_first = False):
        """
        query : (batch_size, len_q, d_k) 
        key : (batch_size, len_k, d_k)
        value : (batch_size, len_k, d_v)
        attn_mask : (len_q, len_k) or (batch_size, len_q, len_k)
        key_padding_mask : (batch_size, len_q) or (batch_size, len_q, 1)

        return : (batch_size, len_q, d_v), (batch_size, num_heads, len_q, len_k)

        key_padding_mask (float16 | float32 | float64 | uint8 | bool Tensor, optional): mask to exclude
                keys that are pads, where padding elements are indicated by _.
    
        attn_mask (float16 | float32 | float64 | uint8 | bool Tensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).

        We take a case where `(batch_size, len_q) = (2, 3)` (`2` source sentences of length `3` each) where the last two 
        tokens of the first sentence are equal to the padding token, as well as for the last token of the second sentence of 
        the batch. `src = [[x11, pad_index, pad_index], [x21, x22, pad_index]]`. 
        We consider the padding mask here, and give some examples of its shape. But this is still valid for the attention 
        mask with respect to the mask values.
        
        * if `pad_is_first == True` (deprecated), then :
            - if key_padding_mask is of type :
                - `bool`, it must be in the form `[[True, False, False], [True, True, False]]` (False for pad_index and True elsewhere)
                    `attn = attn.masked_fill(~key_padding_mask, neg_inf)` # (~ in front of key_padding_mask for not)

                - `uint8` : `[[1, 0, 0], [1, 1, 0]]` (0 for pad_index and 1 elsewhere)
                    `attn = attn.masked_fill(key_padding_mask == 0, neg_inf)`
                    # or
                    `attn = attn.masked_fill(~key_padding_mask.to(torch.bool), neg_inf)` # (~ in front of key_padding_mask for not)
        
                - `float16 | float32 | float64` : `[[1, -inf, -inf], [1, 1, -inf]]` (-inf for pad_index and 1 other)
                      `attn = attn * key_padding_mask`

        * if `pad_is_first == False`, then :
            - if key_padding_mask is of type :
                - `bool`, it must be in the form `[[False, True, True], [False, False, False]]` (True for pad_index and False elsewhere)
                    `attn = attn.masked_fill(key_padding_mask, neg_inf)`

                - `uint8` : `[[0, 1, 1], [0, 0, 1]]` (0 for pad_index and 1 elsewhere)
                    `attn = attn.masked_fill(key_padding_mask == 1, neg_inf)`
                    # or
                    `attn = attn.masked_fill(key_padding_mask.to(torch.bool), neg_inf)` 
              
                - `float16 | float32 | float64` : `[[0, -inf, -inf], [0, 0, -inf]]` (-inf for pad_index and 0 else)
                    `attn = attn + key_padding_mask`

          Note that we cannot use -inf (float("-inf") | -np.inf | ...) here, because at some edge cases, the attention 
          weight (before softmax) for some padded element in query will become -inf, which results in NaN in model 
          parameters (Thanks https://github.com/pytorch/fairseq/blob/master/fairseq/modules/transformer_layer.py#L122 for 
          this remark)
        """
        batch_size, len_q, _ = q.shape
        len_k = k.size(1)
             
        attn = torch.bmm(q, k.transpose(1, 2)) # (batch_size, len_q, len_k)
        attn = attn / self.temperature # (batch_size, len_q, len_k)

        # attention mask : prevents the attention from looking forward in time
        if attn_mask is not None:
            # Thanks https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L4715 for this check
            assert (
                attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
            ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(attn_mask.dtype)

            if attn_mask.dim() == 2: # (len_q, len_k)
                attn_mask = attn_mask.unsqueeze(0) # (1, len_q, len_k)
                if list(attn_mask.size()) != [1, len_q, len_k]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [batch_size, len_q, len_k]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
                    #raise NotImplementedError("The size of the 3D attn_mask is not supported")
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))

            if pad_is_first :
                warnings.warn("`pad_is_first == True` is deprecated for attn_mask.")
                if attn_mask.dtype == torch.bool:
                    attn = attn.masked_fill(~attn_mask.bool(), neg_inf) # (batch_size, len_q, len_k)
                elif attn_mask.dtype == torch.uint8:
                    warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                    attn = attn.masked_fill(attn_mask == 0, neg_inf) # (batch_size, len_q, len_k)
                    #attn = attn.masked_fill(~attn_mask.to(torch.bool), neg_inf) # (batch_size, len_q, len_k)
                else: # float
                    attn *= attn_mask.to(attn.dtype) # (batch_size, len_q, len_k)
            else :
                if attn_mask.dtype == torch.bool:
                    attn = attn.masked_fill(attn_mask.bool(), neg_inf) # (batch_size, len_q, len_k)
                elif attn_mask.dtype == torch.uint8:
                    warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                    attn = attn.masked_fill(attn_mask == 1, neg_inf) # (batch_size, len_q, len_k)
                    #attn = attn.masked_fill(attn_mask.to(torch.bool), neg_inf) # (batch_size, len_q, len_k)
                else: # float
                    attn += attn_mask.to(attn.dtype) # (batch_size, len_q, len_k)
                
        # Masking to ignore padding (query side)
        if key_padding_mask is not None:
            # Thanks https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L4739 for this check
            assert (
                key_padding_mask.dtype == torch.float32
                or key_padding_mask.dtype == torch.float64
                or key_padding_mask.dtype == torch.float16
                or key_padding_mask.dtype == torch.uint8
                or key_padding_mask.dtype == torch.bool
            ), "Only float, byte, and bool types are supported for key_padding_mask, not {}".format(key_padding_mask.dtype)
        
            assert key_padding_mask.size(0) == batch_size
            assert key_padding_mask.size(1) == len_q

            if key_padding_mask.dim() == 2: # (batch_size, len_q) 
                key_padding_mask = key_padding_mask.unsqueeze(-1) # (batch_size, len_q, 1)
            elif key_padding_mask.dim() == 3:
                if key_padding_mask.size(2) != 1 : # (batch_size, len_q, 1)
                    raise RuntimeError("The size of the 3D key_padding_mask is not correct.")
                    #raise NotImplementedError("The size of the 3D key_padding_mask is not supported.")
            else:
                raise RuntimeError("key_padding_mask's dimension {} is not supported".format(key_padding_mask.dim()))

            if pad_is_first :
                warnings.warn("`pad_is_first = True` is deprecated for key_padding_mask.")
                if key_padding_mask.dtype == torch.bool:
                    attn = attn.masked_fill(~key_padding_mask.bool(), neg_inf) # (batch_size, len_q, len_k)
                elif key_padding_mask.dtype == torch.uint8:
                    warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                    attn = attn.masked_fill(key_padding_mask == 0, neg_inf) # (batch_size, len_q, len_k)
                    #attn = attn.masked_fill(~key_padding_mask.to(torch.bool), neg_inf) # (batch_size, len_q, len_k)
                else: # float
                    attn *= key_padding_mask.to(attn.dtype) # (batch_size, len_q, len_k)
            else :
                if key_padding_mask.dtype == torch.bool:
                    attn = attn.masked_fill(key_padding_mask.bool(), neg_inf) # (batch_size, len_q, len_k)
                elif key_padding_mask.dtype == torch.uint8:
                    warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                    attn = attn.masked_fill(key_padding_mask == 1, neg_inf) # (batch_size, len_q, len_k)
                    #attn = attn.masked_fill(key_padding_mask.to(torch.bool), neg_inf) # (batch_size, len_q, len_k)
                else: # float
                    attn += key_padding_mask.to(attn.dtype) # (batch_size, len_q, len_k)

        attn = self.softmax(attn) # (batch_size, len_q, len_k)
        attn = self.dropout(attn) # (batch_size, len_q, len_k)
        output = torch.bmm(attn, v) # (batch_size, len_q, d_v)

        return output, attn

## Multi-head attention
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, num_heads, d_k, d_v, dropout=0.1, debug_num = 0):
        super(MultiHeadAttention, self).__init__()
        self.debug_num = debug_num # Just a difference in implementation details (0 is fast and more parallelisable)
        self.num_heads = num_heads
        if debug_num == 0 :
            self.d_k = d_k
            self.d_v = d_v

            self.W_Q = nn.Linear(d_model, num_heads * d_k)
            self.W_K = nn.Linear(d_model, num_heads * d_k)
            self.W_V = nn.Linear(d_model, num_heads * d_v)
        
        elif debug_num == 1 :
            self.W_Q = _get_clones(nn.Linear(d_model, d_k), num_heads)
            self.W_K = _get_clones(nn.Linear(d_model, d_k), num_heads)
            self.W_V = _get_clones(nn.Linear(d_model, d_v), num_heads)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5),
                                                      attn_dropout=dropout)
        self.W_0 = nn.Linear(num_heads * d_v, d_model)

        # We delegate this to the encoder and decoder layer, just like the norm layer.
        #self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, 
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None):
        """
        query : (batch_size, len_q, d_model)
        key : (batch_size, len_k, d_model)
        value : (batch_size, len_v, d_model)
        attn_mask : (len_q, len_k) or (batch_size, len_q, len_k)
        key_padding_mask : (batch_size, len_q) or (batch_size, len_q, 1)

        return : # (batch_size, len_q, d_model), (batch_size, num_heads, len_q, len_k)

        key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, where padding elements are indicated by 1s.
    
        attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
        """
        if self.debug_num == 0 :
            d_k, d_v, num_heads = self.d_k, self.d_v, self.num_heads

            batch_size, len_q, _ = q.size()
            batch_size, len_k, _ = k.size()
            batch_size, len_v, _ = v.size()

            q = self.W_Q(q).view(batch_size, len_q, num_heads, d_k)
            k = self.W_K(k).view(batch_size, len_k, num_heads, d_k)
            v = self.W_V(v).view(batch_size, len_v, num_heads, d_v)

            q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (num_heads*batch_size) x len_q x d_k
            k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (num_heads*batch_size) x len_k x d_k
            v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (num_heads*batch_size) x len_v x d_v

            # Repeat masks num_heads times
            if attn_mask is not None:
                attn_mask = attn_mask.repeat(num_heads, 1, 1)  # (num_heads*batch_size) x len_q x len_k
            if key_padding_mask is not None:
                # todo ?
                key_padding_mask = key_padding_mask.unsqueeze(-1) #.repeat(1, 1, len_q)
                key_padding_mask = key_padding_mask.repeat(num_heads, 1, 1)

            output, attn = self.attention(q, k, v, key_padding_mask, attn_mask)

            output = output.view(num_heads, batch_size, len_q, d_v) # num_heads x batch_size x len_q x d_v
            output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, len_q, -1)  # batch_size x len_q x (num_heads*d_v)

            output = self.W_0(output) # batch_size x len_q x d_model
            #output = self.dropout(output) # batch_size x len_q x d_model
            
            # attn : # (batch_size * num_heads) x len_q x len_k
            attn = attn.contiguous().view(batch_size, num_heads, len_q, len_k) # batch_size x num_heads x len_q x len_k

            return output, attn

        elif self.debug_num == 1 :
            # Same mask applied to all heads.
            if attn_mask is not None:
                #attn_mask = attn_mask.unsqueeze(1)
                pass
            if key_padding_mask is not None :
                key_padding_mask = key_padding_mask.unsqueeze(-1) 
            
            heads = [ 
                    self.attention(W_Q_i(q), W_K_i(k), W_V_i(v), key_padding_mask, attn_mask) # (batch_size, len_q, d_v), (batch_size, len_q, len_k)
                    for W_Q_i, W_K_i, W_V_i in zip(self.W_Q, self.W_K, self.W_V)
                    ]
            output, attn = zip(*heads) # (batch_size, len_q, d_v)*num_heads, (batch_size, len_q, len_k)*num_heads
            
            output = torch.cat(output, dim=-1) # batch_size x len_q x (num_heads*d_v)
            output = self.W_0(output) # batch_size x len_q x d_model
            #output = self.dropout(output) # batch_size x len_q x d_model
            
            attn = torch.stack(attn, dim=1) # batch_size x num_heads x len_q x len_k

            return output, attn
        
## **TIM (as Encoder Layer)**

### Group Linear Layer
def GroupLinear(inputs : Tensor, layers : ModuleList, activation = None) -> Tensor :
    """
    Args:
        inputs  : (ns, batch_size, _, d_in) 
        layers : (ns, d_in, d_out)
        activation : callable

    return (ns, batch_size, _, d_out)
    """
    assert len(layers) == len(inputs)
    if activation is not None:
        return [activation(layer(x)) for layer, x in zip(layers, inputs)]
    else :
        return [layer(x) for layer, x in zip(layers, inputs)]

### Mechanism Competition
class MechanismCompetitionLayer(Module):
    """Compute Mechanism Competition"""
    def __init__(self, d_mech, n_s, bias = True) :
        """
        d_mech : single mechanism dimension
        n_s : number of mechanisms
        """
        super(MechanismCompetitionLayer, self).__init__()
        self.W_c = _get_clones(module = nn.Linear(d_mech, 1, bias = bias), N = n_s)
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, h : Tensor) -> Tensor:
        """
        h : (ns, batch_size, seq_len, d_mech)
        """
        c = GroupLinear(inputs = h, layers = self.W_c) # (batch_size, seq_len, 1)*ns
        c = torch.stack(c) # (ns, batch_size, seq_len, 1)
        c = self.softmax(input = c) # (ns, batch_size, seq_len, 1)
    
        return c
    
### Mechanism-wise self-attention sub-layer
class MechanismWiseSelfAttentionSubLayer(Module):
    """Mechanism-wise self-attention sub-layer"""
    def __init__(self, d_mech, H, n_s, d_k, d_v, dropout=0.1, custom_mha : callable = None) :
        super(MechanismWiseSelfAttentionSubLayer, self).__init__()
        self.is_custom_mha = custom_mha is not None
        if not self.is_custom_mha :
            self.attn = _get_clones(module = MultiHeadAttention(d_mech, H, d_k, d_v, dropout), N = n_s)
        else :
            assert d_v == d_k == d_mech // H
            self.attn = _get_clones(module = custom_mha(H, d_mech, dropout=dropout), N = n_s)
            self.H=H

        self.norm = LayerNorm(d_mech)
    
    def forward(self, h : Tensor, c : Tensor, 
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
            h : input hidden representation  
            c : Mechanism Competition
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            h : (n_s, batch_size, seq_len, d_mech)
            c : (n_s, batch_size, seq_len, 1)
            src_mask : (seq_len, seq_len) or (batch_size, seq_len, seq_len)
            src_key_padding_mask : (batch_size, seq_len)

            return : (n_s, batch_size, seq_len, d_mech), (n_s, batch_size, H, seq_len, seq_len)
        """
        if not self.is_custom_mha :
            M = [
                layer(q = x, k = x, v = x, 
                      key_padding_mask = key_padding_mask, attn_mask = attn_mask) 
                      # (batch_size, seq_len, d_mech), (batch_size, H, seq_len, seq_len)
                for layer, x in zip(self.attn, h) ] 
            M, attn = zip(*M) # (batch_size, seq_len, d_mech)*n_s, (batch_size, H, seq_len, seq_len)*n_s
            attn = torch.stack(attn) # (n_s, batch_size, H, seq_len, seq_len)
        else :
            M = [
                layer(x, mask = key_padding_mask)  # (batch_size, seq_len, d_mech)
                for layer, x in zip(self.attn, h) ]

            n_s, batch_size, seq_len, _ = h.shape
            attn = torch.empty(n_s, batch_size, self.H, seq_len, seq_len)

        h = [self.norm(h_i + c_i * M_i) for h_i, c_i, M_i in zip(h, c, M)] # (batch_size, seq_len, d_mech)*n_s
        h = torch.stack(h) # (n_s, batch_size, seq_len, d_mech)

        return h, attn

### Inter-mechanism Attention Sub-Layer
class InterMechanismAttentionSubLayer(Module):
    """Inter-mechanism Attention Sub-Layer"""
    def __init__(self, d_mech, H_c, n_s, d_k, d_v, dropout=0.1, custom_mha : callable = None) :
        """"""
        super(InterMechanismAttentionSubLayer, self).__init__()
        self.is_custom_mha = custom_mha is not None
        if not self.is_custom_mha :
            self.attn = _get_clones(module = MultiHeadAttention(d_mech, H_c, d_k, d_v, dropout), N = n_s)
        else :
            assert d_v == d_k == d_mech // H_c
            self.attn = _get_clones(module =  custom_mha(H_c, d_mech, dropout=dropout), N = n_s)
            self.H_c=H_c

        self.norm = LayerNorm(d_mech)
 
    def forward(self, h : Tensor, 
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
            h : input hidden representation  
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
 
        Shape:
        
            h : (n_s, batch_size, seq_len, d_mech)
            src_mask : (seq_len, seq_len) or (batch_size, seq_len, seq_len)
            src_key_padding_mask : (batch_size, seq_len)
 
            return : # (n_s, batch_size, seq_len, d_mech), (n_s, batch_size, H_c, seq_len, seq_len)
        """


        if not self.is_custom_mha  :
            M = [
                layer(q = x, k = x, v = x, 
                      key_padding_mask = key_padding_mask, attn_mask = attn_mask) 
                  # (batch_size, seq_len, d_mech), (batch_size, H_c, seq_len, seq_len)
                for layer, x in zip(self.attn, h) ] 
            
            M, attn = zip(*M) # (batch_size, seq_len, d_mech)*n_s, (batch_size, H_c, seq_len, seq_len)*n_s
            attn = torch.stack(attn) # (n_s, batch_size, H_c, seq_len, seq_len)
        else :
            M = [
                layer(x, mask = key_padding_mask)  # (batch_size, seq_len, d_mech), (batch_size, H, seq_len, seq_len)
                for layer, x in zip(self.attn, h) ]

            n_s, batch_size, seq_len, _ = h.shape
            attn = torch.empty(n_s, batch_size, self.H_c, seq_len, seq_len)

        
        h = [self.norm(h_i + M_i) for h_i, M_i in zip(h,  M)] # (batch_size, seq_len, d_mech)*n_s
        h = torch.stack(h) # (n_s, batch_size, seq_len, d_mech)
 
        return h, attn
    
class MechanismWisePositionWiseFFNSubLayer(Module):
    def __init__(self, d_mech, n_s, dffn_m, bias = True) :
      
        super(MechanismWisePositionWiseFFNSubLayer, self).__init__()
        self.W_1 = _get_clones(module = nn.Linear(d_mech, dffn_m, bias = bias), N = n_s)
        self.W_2 = _get_clones(module = nn.Linear(dffn_m, d_mech, bias = bias), N = n_s)
        self.norm = LayerNorm(d_mech)
        self.activation = _get_activation_fn("relu")
    
    def forward(self, h):
        """
        h : (n_s, batch_size, seq_len, d_mech)
        """
        out = GroupLinear(h, self.W_1, activation=self.activation) # (n_s, batch_size, seq_len, dffn_m)
        out = GroupLinear(out, self.W_2) # (n_s,  batch_size, seq_len, d_mech)
        h = [self.norm(h_i + F_i) for h_i, F_i in zip(h, out)] # (batch_size, seq_len, d_mech)*n_s
        h = torch.stack(h) # (n_s, batch_size, seq_len, d_mech)
        return h
    
class TIM_EncoderLayer(Module):
    def __init__(self, d_model, d_ffn, n_s, d_k, d_v, H, H_c, dropout=0.1, activation="relu", bias = True,
                 custom_mha : callable = None) :
        """
        n_s : number of mechanisms
        d_k : key size
        d_v : value size
        H : number of heads for self-attention 
        H_c : number of heads for inter-mechanism attention

        is custom_mha == False, make sure d_k = d_v =  d_mesh // H = d_mesh // H_c
        d_mesh = d_model // H
        """
        super(TIM_EncoderLayer, self).__init__()

        assert d_model % n_s == 0
        assert d_ffn % n_s == 0

        self.n_s = n_s
        self.d_mech = d_model // n_s
        dffn_m = d_ffn // n_s 

        assert custom_mha is None or d_k == d_v == self.d_mech // H == self.d_mech // H_c
        
        # Mechanism Competition sub-layer
        self.competitionSubLayer = MechanismCompetitionLayer(self.d_mech, n_s, bias = bias) 
        
        # Mechanism-wise self-attention sub-layer
        self.attn1 = MechanismWiseSelfAttentionSubLayer(self.d_mech, H, n_s, d_k, d_v, dropout, custom_mha) 
        self.dropout1 = Dropout(dropout)

        # Inter-mechanism Attention Sub-Layer
        self.attn2 = InterMechanismAttentionSubLayer(self.d_mech, H_c, n_s, d_k, d_v, dropout,  custom_mha)
        self.dropout2 = Dropout(dropout)

        # Mechanism-wise, Position-Wise, FFN SubLayer
        self.ffn = MechanismWisePositionWiseFFNSubLayer(self.d_mech, n_s, dffn_m, bias)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, h : Tensor, 
                src_mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor] :
        """
            h : input hidden representation  
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
        
            h : (batch_size, seq_len, d_model), d_model = d_mech*n_s
            src_mask : (seq_len, seq_len) or (batch_size, seq_len, seq_len)
            src_key_padding_mask : (batch_size, seq_len)

            return : # (batch_size, seq_len, d_model), (batch_size, num_heads, input_dim, target_dim), -*-
        """
        assert self.d_mech * self.n_s == h.size(2)
        batch_size = h.size(0)
        seq_len = h.size(1)
         
        h = h.contiguous().view(batch_size, seq_len, self.d_mech, self.n_s) # (batch_size, seq_len, d_mech, n_s)
        h = h.permute(3, 0, 1, 2) # (n_s, batch_size, seq_len, d_mech)

        # Step 1: Compute Mechanism Competition
        c = self.competitionSubLayer(h) # (n_s, batch_size, seq_len, 1)

        # Step 2: Mechanism-wise self-attention sub-layer
        h, attn1 = self.attn1(h, c, src_key_padding_mask, src_mask ) # (n_s, batch_size, seq_len, d_mech), (n_s, batch_size, H, seq_len, seq_len)
        h = self.dropout1(h)

        # Step 4: Mechanism-wise, Position-Wise, FFN SubLayer
        h = self.ffn(h) # (n_s, batch_size, seq_len, d_mech)
        h = self.dropout3(h)

        # Step 3: Inter-mechanism Attention Sub-Layer
        h, attn2 = self.attn2(h, src_key_padding_mask, src_mask) # (n_s, batch_size, seq_len, d_mech), (n_s, batch_size, H_c, seq_len, seq_len)
        h = self.dropout2(h)

        # Step 4: Mechanism-wise, Position-Wise, FFN SubLayer
        #h = self.ffn(h) # (n_s, batch_size, seq_len, d_mech)
        #h = self.dropout3(h)

        h = h.permute(1, 2, 3, 0) # (batch_size, seq_len, d_mech, n_s)
        h = h.contiguous().view(batch_size, seq_len, self.d_mech * self.n_s) # (seq_len, batch_size, d_model)
        
        return h, attn1, attn2