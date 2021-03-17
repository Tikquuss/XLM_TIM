

import torch
import logging
import torch.nn as nn
import numpy as np
#from speech.data_io.data_io import length_to_mask

from .group_linear import GroupLinear

logger = logging.getLogger(__name__)


class MultiheadAttention(nn.Module):
    """ The class is a wrapper of MultiHead Attention for torch.nn.MultiHeadAttention.
    ref: https://pytorch.org/docs/stable/nn.html

    Arguements
    ----------
    num_heads : int
        parallel attention heads.
    dropout : float
        a Dropout layer on attn_output_weights. Default: 0.0.
    bias : bool
        add bias as module parameter. Default: True.
    add_bias_kv : bool
        add bias to the key and value sequences at dim=0.
    add_zero_attn : bool
        add a new batch of zeros to the key and value sequences at dim=1.
    kdim : int
        total number of features in key. Default: None.
    vdim : int
        total number of features in value. Default: None.
    """

    def __init__(
        self,
        nhead,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        nb=1,
    ):
        super().__init__()
        self.nhead = nhead
        self.dropout = dropout
        self.bias = bias
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        self.kdim = kdim
        self.vdim = vdim
        self.nb = nb

    def init_params(self, first_input):
        if len(first_input.shape) == 4:
            first_input = first_input.reshape(
                first_input.shape[0],
                first_input.shape[1],
                first_input.shape[2] * first_input.shape[3],
            )

        self.embed_dim = first_input.shape[-1] // self.nb

        if self.kdim is not None:
            self.kdim = self.kdim // self.nb

        if self.vdim is not None:
            self.vdim = self.vdim // self.nb

        self.att = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.nhead // self.nb,
            dropout=self.dropout,
            bias=self.bias,
            add_bias_kv=self.add_bias_kv,
            add_zero_attn=self.add_zero_attn,
            kdim=self.kdim,
            vdim=self.vdim,
        ).to(first_input.device)

    def forward(
        self,
        query,
        key,
        value,
        attn_mask=None,
        key_padding_mask=None,
        init_params=False,
    ):
        """
        Arguements
        ----------
        query: tensor
            (L, N, E)(L,N,E) where L is the target sequence length, N is the batch size, E is the embedding dimension.
        key: tensor
            (S, N, E)(S,N,E) , where S is the source sequence length, N is the batch size, E is the embedding dimension.
        value: tensor
            (S, N, E)(S,N,E) where S is the source sequence length, N is the batch size, E is the embedding dimension.
        key_padding_mask: tensor
            (N, S)(N,S) where N is the batch size, S is the source sequence length. If a ByteTensor is provided, the non-zero positions will be ignored while the position with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the value of True will be ignored while the position with the value of False will be unchanged.
        attn_mask: tensor
            2D mask (L, S)(L,S) where L is the target sequence length, S is the source sequence length. 3D mask (N*num_heads, L, S)(N∗num_heads,L,S) where N is the batch size, L is the target sequence length, S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend while the zero positions will be unchanged. If a BoolTensor is provided, positions with True is not allowed to attend while False values will be unchanged. If a FloatTensor is provided, it will be added to the attention weight.

        Outputs
        -------
        attn_output: tensor
            (L, N, E)(L,N,E) where L is the target sequence length, N is the batch size, E is the embedding dimension.
        attn_output_weights: tensor
            (N, L, S)(N,L,S) where N is the batch size, L is the target sequence length, S is the source sequence length.
        """
        if init_params:
            self.init_params(key)

        # give tensors of shape (time, batch, fea)
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)
        tq, bsz, _ = query.shape

        query = query.reshape(
            (
                query.shape[0],
                query.shape[1] * self.nb,
                query.shape[2] // self.nb,
            )
        )
        key = key.reshape(
            (key.shape[0], key.shape[1] * self.nb, key.shape[2] // self.nb)
        )
        value = value.reshape(
            (
                value.shape[0],
                value.shape[1] * self.nb,
                value.shape[2] // self.nb,
            )
        )

        if key_padding_mask is not None:
            key_padding_mask = (
                key_padding_mask.unsqueeze(1)
                .repeat(1, self.nb, 1)
                .reshape(
                    (
                        key_padding_mask.shape[0] * self.nb,
                        key_padding_mask.shape[1],
                    )
                )
            )

        output, attention = self.att(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )

        output = output.reshape((tq, bsz, output.shape[2] * self.nb))

        # reshape the output back to (batch, time, fea)
        output = output.permute(1, 0, 2)

        return output, attention


class PositionalwiseFeedForward(nn.Module):
    def __init__(self, d_ffn, nb=1, dropout=0.1, activation=nn.ReLU):
        """The class implements the positional-wise feadd forward module in “Attention Is All You Need”

        Arguements
        ----------
        d_ffn: int
            dimention of representation space of this positional-wise feadd forward module
        dropout: float
            dropout
        activation: torch class
            activation functions to be applied (Recommandation: ReLU, GELU)
        """
        super().__init__()
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.activation = activation
        self.nb = nb

    def init_params(self, first_input):
        self.input_size = first_input.shape[-1]

        self.ffn = nn.Sequential(
            GroupLinear(self.input_size, self.d_ffn, nb=self.nb),
            self.activation(),
            nn.Dropout(self.dropout),
            GroupLinear(self.d_ffn, self.input_size, nb=self.nb),
        ).to(first_input.device)

    def forward(self, x, init_params=False):
        if init_params:
            self.init_params(x)

        # give a tensor of shap (time, batch, fea)
        x = x.permute(1, 0, 2)

        x = self.ffn(x)

        # reshape the output back to (batch, time, fea)
        x = x.permute(1, 0, 2)

        return x

