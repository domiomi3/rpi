from __future__ import annotations

import copy

import torch

from typing import Optional, Union, Callable
from statistics import mean
from torch import Tensor, optim
from torch.nn import functional as F, Module, MultiheadAttention, ModuleList, Linear, LayerNorm, Conv1d, Dropout
from torch.nn.init import xavier_uniform_
from lightning import LightningModule, seed_everything
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAccuracy, BinaryAUROC


class ModelWrapper(LightningModule):
    def __init__(self, model, lr_init, weight_decay, seed):
        super().__init__()
        seed_everything(seed)
        self.model = model
        self.loss_metric = torch.nn.BCELoss()
        self.metrics = MetricCollection([
            BinaryPrecision(),
            BinaryRecall(),
            BinaryF1Score(),
            BinaryAccuracy(),
            BinaryAUROC()
        ])
        self.lr_init = lr_init
        self.weight_decay = weight_decay
        self.train_metrics = self.metrics.clone(prefix='train_')
        self.valid_metrics = self.metrics.clone(prefix='val_')
        self.valid_losses = []
        self.train_losses = []

    def forward(self, rna_embed, protein_embed):
        protein_embed = protein_embed.float()
        rna_embed = rna_embed.float()
        return self.model(rna_embed, protein_embed)

    def training_step(self, batch, batch_idx):
        rna_embed, protein_embed, y, _ = batch
        y = y.float()
        y_hat = self.forward(rna_embed, protein_embed)
        y_hat = y_hat.reshape(y_hat.shape[0])
        loss = self.loss_metric(y_hat, y)
        self.train_losses.append(loss.item())
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.train_metrics.update(y_hat, y)

        return loss

    def validation_step(self, batch, batch_idx):
        rna_embed, protein_embed, y, _ = batch
        y_hat = self(rna_embed, protein_embed)
        y_hat = y_hat.reshape(y_hat.shape[0])
        y = y.float()
        loss = self.loss_metric(y_hat, y)
        self.valid_losses.append(loss.item())

        self.log("val_loss", loss, on_step=True, on_epoch=False, logger=True, prog_bar=True)
        self.valid_metrics.update(y_hat, y)

    def on_validation_epoch_end(self) -> None:
        output = self.valid_metrics.compute()
        self.log_dict(output, on_step=False, on_epoch=True)
        # remember to reset metrics at the end of the epoch
        valid_loss = mean(self.valid_losses)
        self.log("valid_loss_epoch", valid_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.valid_losses = []
        self.valid_metrics.reset()

    def on_train_epoch_end(self) -> None:
        output = self.train_metrics.compute()
        self.log_dict(output)
        train_loss = mean(self.train_losses)
        self.log("train_loss_epoch", train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_losses = []
        self.train_metrics.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr_init, weight_decay=self.weight_decay)
        return optimizer

    def _shared_eval(self, batch):
        protein_embed, rna_embed, y = batch
        y = y.float()
        y_hat = self.forward(rna_embed, protein_embed)
        y_hat = y_hat.reshape(y_hat.shape[0])
        return (self.metric(y_hat, y),
                self.f1_metric(y_hat, y),
                self.precision_metric(y_hat, y),
                self.recall_metric(y_hat, y))


class RNAProteinInterAct(Module):
    r"""
    Code heavily inspired from https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu

        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
            other attention and feedforward operations, otherwise after. Default: ``False`` (after).
        embed_dim: dimensionality of input embeddings
        key_padding_mask: if ``True`` padded zeros are masked. Default: ``false``.
        device: device which runs computations.
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 embed_dim: int = 640,
                 key_padding_mask: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        

        encoder_layer = RNAProteinEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                               activation, layer_norm_eps, batch_first, norm_first,
                                               **factory_kwargs)

        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = RNAProteinEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.key_padding_mask = key_padding_mask
        self.batch_first = batch_first
        self.embed_dim = embed_dim

        self.activation = torch.relu

        self.linear_reduce_1 = Linear(embed_dim, d_model)
        self.linear_reduce_2 = Linear(embed_dim, d_model)

        self.linear1 = Linear(d_model, d_model // 2, **factory_kwargs)
        self.linear2 = Linear(d_model // 2, d_model // 4, **factory_kwargs)
        self.linear3 = Linear(d_model // 4, 1, **factory_kwargs)

    def forward(self, rna_embed: Tensor, protein_embed: Tensor,
                ) -> Tensor:
        r"""Forward path.

        Args:
            rna_embed: RNA embedding of a sequence (required).
            protein_embed: protein embedding of a sequence (required).
        """

        if not rna_embed.dim() == protein_embed.dim():
            raise RuntimeError("both embeddings must have identical dimensionality")
        is_batched = rna_embed.dim() == 3 and protein_embed.dim() == 3
        if not self.batch_first and rna_embed.size(1) != protein_embed.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and rna_embed.size(0) != protein_embed.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if protein_embed.size(-1) != self.embed_dim or rna_embed.size(-1) != self.embed_dim:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        rna_mask = None
        protein_mask = None
        if self.key_padding_mask:
            rna_mask = rna_embed[:, :, 0] == 0.0
            protein_mask = protein_embed[:, :, 0] == 0.0

        x_1 = self.activation(self.linear_reduce_1(rna_embed))

        x_2 = self.activation(self.linear_reduce_2(protein_embed))

        x_1, x_2 = self.encoder(x_1, x_2, rna_mask=rna_mask, protein_mask=protein_mask)

        x = torch.cat((x_1, x_2), 1)

        x = self.activation(self.linear1(x))

        x = self.activation(self.linear2(x))

        x = torch.sigmoid(self.linear3(x))

        return x[:, 0]
        # NOTE: try different tokens, e.g. mean (see below)
        # return x.mean(dim=1)

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device='cpu') -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)



class RNAProteinEncoder(Module):
    r"""Encoder for RNA-Protein interaction prediction

    Args:
        encoder_layer: an instance of the RNAProteinEncoder() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True):
        super().__init__()
        self.layers_1 = _get_clones(encoder_layer, num_layers)
        self.layers_2 = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # this attribute saves the value providedat object construction
        self.enable_nested_tensor = enable_nested_tensor
        # this attribute controls whether nested tensors are used
        self.use_nested_tensor = enable_nested_tensor

    def forward(
            self,
            rna_embed: Tensor,
            protein_embed: Tensor,
            mask: Optional[Tensor] = None,
            rna_mask: Optional[Tensor] = None,
            protein_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            rna_embed: RNA embedding of a sequence (required).
            protein_embed: protein embedding of a sequence (required).
            mask: the mask for the src sequence (optional).
            rna_mask: mask for padded zeros in rna embedding (optional).
            protein_mask: mask for padded zeros in protein embedding (optional).
        """

        # NOTE: as I am reading the docs correctly this is for masking the padded "tokens"
        output_1, output_2 = rna_embed, protein_embed

        for mod_1, mod_2 in zip(self.layers_1, self.layers_2):
            in_1, in_2 = output_1, output_2
            # Using two parallel Encoder Layers
            output_1 = mod_1(in_1, in_2, src_mask=mask,
                             key_padding_mask_1=rna_mask, key_padding_mask_2=protein_mask)
            output_2 = mod_2(in_2, in_1, src_mask=mask,
                             key_padding_mask_1=protein_mask, key_padding_mask_2=rna_mask)

        # if self.norm is not None:
        #    output = self.norm(output)

        return output_1, output_2


class RNAProteinEncoderLayer(Module):
    r"""Encoder Layer for RNA-Protein interaction prediction.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise, it's done after. Default: ``False`` (after).
        device: device which runs computations.
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                             **factory_kwargs)

        # Implementation of Feedforward model
        self.linear1_1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2_1 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm0_1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm1_1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm1_2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1_1 = Dropout(dropout)
        self.dropout2_1 = Dropout(dropout)
        self.dropout2_2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def forward(
            self,
            rna_embed: Tensor,
            protein_embed: Tensor,
            src_mask: Optional[Tensor] = None,
            key_padding_mask_1: Optional[Tensor] = None,
            key_padding_mask_2: Optional[Tensor] = None,
            is_causal: bool = False) -> tuple[Tensor, Tensor]:
        r"""Forward shape

        Args:
            rna_embed: RNA embedding of a sequence (required).
            protein_embed: protein embedding of a sequence (required) NOTE: rna_embed and protein_embed can be switched.
            src_mask: the mask for the src sequence (optional).
            key_padding_mask_1: the mask for the first input sequence (optional).
            key_padding_mask_2: the mask for the second input sequence (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
        """

        x_1 = rna_embed
        x_2 = protein_embed
        # First apply Self-Attention
        x_1 = x_1 + self._sa_block(self.norm0_1(x_1), src_mask, key_padding_mask_1, is_causal=is_causal)

        # Then Cross-Attention
        if self.norm_first:
            x_1 = x_1 + self._ca_block(self.norm1_1(x_1), self.norm1_2(x_2), src_mask, key_padding_mask_2,
                                       is_causal=is_causal)
            x_1 = x_1 + self._ff_block_1(self.norm2(x_1))
        else:
            raise NotImplementedError("Not supported for now")
            # x_1 = self.norm1_1(x_1 + self._ca_block(x_1, x_2, src_mask, key_padding_mask_2, is_causal=is_causal))
            # x_1 = self.norm2(x_1 + self._ff_block_1(x_1))

        return x_1

    # cross-attention block
    def _ca_block(self, x_1: Tensor,  # RNA-embedding
                  x_2: Tensor,  # protein-embedding
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.cross_attn(x_1, x_2, x_2,
                            # attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=True, is_causal=is_causal)[0]
        return self.dropout1_1(x)

    # feed forward block
    def _ff_block_1(self, x: Tensor) -> Tensor:
        x = self.linear2_1(self.dropout(self.activation(self.linear1_1(x))))
        return self.dropout2_1(x)

    def _sa_block(self, x_1: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x_1, x_1, x_1,
                           # attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True, is_causal=is_causal)[0]
        return self.dropout2_2(x)


class BaseLinear(Module):
    """
    Baseline model based on linear layers.
    IDEA: improve model with NAS & HPO :-)
    """
    def __init__(self, d_model, *args, **factory_kwargs):
        # Implementation of Feedforward model
        super().__init__(*args, **factory_kwargs)
        self.linear1 = Linear(d_model, d_model // 8, **factory_kwargs)
        # self.linear2 = Linear(d_model // 2, d_model // 4)
        # self.linear3 = Linear(d_model // 4, d_model // 8)
        self.linear4 = Linear(d_model // 8, d_model // 16)
        self.linear5 = Linear(d_model // 16, 1)
        self.activation = torch.relu

    def forward(self, x_1, x_2):
        x = torch.cat((x_1, x_2), 1)
        x = self.activation(self.linear1(x))
        # x = self.activation(self.linear2(x))
        # x = self.activation(self.linear3(x))
        x = self.activation(self.linear4(x))
        x = torch.sigmoid(self.linear5(x))
        return x.mean(dim=1)


class BaseCNN(Module):
    """
        Baseline model based on convolutional layers.
        IDEA: improve model with NAS & HPO :-)
        """
    def __init__(self, d_model, *args, **factory_kwargs):
        super().__init__(*args, **factory_kwargs)
        self.conv1 = Conv1d(d_model, d_model // 16, 1)
        self.linear1 = Linear(640, 640 // 8)
        self.linear2 = Linear(640 // 8, 1)
        self.activation = torch.relu

    def forward(self, x_1, x_2):
        x = torch.cat((x_1, x_2), 1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x.mean(dim=1)


def _get_clones(module, n):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for _ in range(n)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
