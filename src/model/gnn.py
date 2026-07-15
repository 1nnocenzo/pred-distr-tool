from __future__ import annotations

from typing import TYPE_CHECKING, Any
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch_geometric.nn import SAGEConv, SAGPooling, global_mean_pool, GraphNorm, global_max_pool, global_add_pool, GATConv
from torch_geometric.utils import dropout_edge

if TYPE_CHECKING:
    from collections.abc import Callable
    from torch_geometric.data import Data


class GraphConvolutionSage(nn.Module):
    """Graph convolutional stack (SAGE first layer, then SAGE) with optional bidirectional
    message passing and optional SAGPooling before graph readout."""

    def __init__(
        self,
        in_feats: int,
        hidden_dim: int,
        num_conv_wo_resnet: int,
        num_resnet_layers: int,
        *,
        conv_activation: Callable[..., torch.Tensor] = functional.leaky_relu,
        conv_act_kwargs: dict[str, Any] | None = None,
        dropout_p: float = 0.2,
        dropedge_p: float = 0.3,
        force_undirected: bool = False,
        bidirectional: bool = True,
        combine: str = "sum",  # "sum" keeps width, "cat" doubles width
        # --- SAGPooling knobs ---
        use_sag_pool: bool = False,
        sag_ratio: float = 0.5,          # fraction of nodes to keep per-graph
        sag_nonlinearity: Callable[..., torch.Tensor] = torch.tanh,
        readout: str = "mean",  # "mean" | "max" | "meanmax" | "meansum" | "meanmaxsum"
    ) -> None:
        super().__init__()
        if num_conv_wo_resnet < 1:
            raise ValueError("num_conv_wo_resnet must be at least 1")
        if combine not in ("sum", "cat"):
            raise ValueError("combine must be 'sum' or 'cat'")
        if not (0.0 < sag_ratio <= 1.0):
            raise ValueError("sag_ratio must be in (0, 1]")

        self.conv_activation = conv_activation
        self.conv_act_kwargs = conv_act_kwargs or {}
        self.dropedge_p = dropedge_p
        self.force_undirected = force_undirected
        self.bidirectional = bidirectional
        self.combine = combine
        self.use_sag_pool = use_sag_pool
        self.readout = readout

        # --- GRAPH ENCODER ---
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        # First layer GAT
        #self.convs.append(GATConv(in_feats, hidden_dim, heads=4, concat=True))

        #hidden_dim = hidden_dim * 4
        # First layer: SAGE
        self.convs.append(SAGEConv(in_feats, hidden_dim))
        base_out = hidden_dim
        out_dim = base_out if (not bidirectional or combine == "sum") else (base_out * 2)
        _readout_mult = {"mean": 1, "max": 1, "meanmax": 2, "meansum": 2, "meanmaxsum": 3}
        if readout not in _readout_mult:
            raise ValueError("readout must be 'mean' | 'max' | 'meanmax' | 'meansum' | 'meanmaxsum'")
        self.graph_emb_dim = out_dim * _readout_mult[readout]
        self.norms.append(GraphNorm(out_dim))

        # Subsequent layers: SAGE with fixed width == out_dim
        for _ in range(num_conv_wo_resnet - 1):
            self.convs.append(SAGEConv(out_dim, out_dim))
            self.norms.append(GraphNorm(out_dim))
        for _ in range(num_resnet_layers):
            self.convs.append(SAGEConv(out_dim, out_dim))
            self.norms.append(GraphNorm(out_dim))

        self.drop = nn.Dropout(dropout_p)
        # Start residuals after the initial non-residual stack
        self._residual_start = num_conv_wo_resnet
        self.out_dim = out_dim  # expose final node embedding width

        # --- SAGPooling layer (applied once, after all convs) ---
        # Uses SAGEConv internally for attention scoring to match the stack.
        if self.use_sag_pool:
            self.sag_pool = SAGPooling(
                in_channels=self.out_dim,
                ratio=sag_ratio,
                GNN=SAGEConv,
                nonlinearity=sag_nonlinearity,
            )
        else:
            self.sag_pool = None

        # Detect whether dropout_edge supports force_undirected (version-agnostic)
        sig = inspect.signature(dropout_edge)
        self._dropedge_has_force_undirected = "force_undirected" in sig.parameters

    def _apply_conv_bidir(self, conv, x, edge_index_do):
        """Apply conv forward/backward and fuse by configured combine mode."""
        x_f = conv(x, edge_index_do)
        if not self.bidirectional:
            return x_f
        x_b = conv(x, edge_index_do.flip(0))
        if self.combine == "sum":
            return (x_f + x_b) / 2
        else:  # "cat"
            return torch.cat([x_f, x_b], dim=-1)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, conv in enumerate(self.convs):
            # If you want edge dropout back, remove the triple quotes below.
            if self._dropedge_has_force_undirected:
                edge_index_do, _ = dropout_edge(
                    edge_index, p=self.dropedge_p, force_undirected=self.force_undirected, training=self.training
                )
            else:
                edge_index_do, _ = dropout_edge(edge_index, p=self.dropedge_p, training=self.training)
            #edge_index_do = edge_index  # dropedge disabled for debugging

            if self.combine == "cat" and self.bidirectional and i == 0:
                x_new = self._apply_conv_bidir(conv, x, edge_index_do)
            elif self.combine == "sum" and self.bidirectional:
                x_new = self._apply_conv_bidir(conv, x, edge_index_do)
            else:
                x_new = conv(x, edge_index_do)

            x_new = self.norms[i](x_new, batch=batch)
            x_new = self.conv_activation(x_new, **self.conv_act_kwargs)
            x_new = self.drop(x_new)

            x = x_new if i < self._residual_start else x + x_new

        # --- SAGPooling (hierarchical pooling before readout) ---
        if self.sag_pool is not None:
            # SAGPooling may also return edge_attr, perm, score; we ignore those here.
            x, edge_index, _, batch, _, _ = self.sag_pool(x, edge_index, batch=batch)

        if self.readout == "mean":
            return global_mean_pool(x, batch)
        elif self.readout == "max":
            return global_max_pool(x, batch)
        elif self.readout == "meanmax":
            return torch.cat([global_mean_pool(x, batch),
                              global_max_pool(x, batch)], dim=-1)
        elif self.readout == "meansum":
            return torch.cat([global_mean_pool(x, batch),
                              global_add_pool(x, batch)], dim=-1)
        elif self.readout == "meanmaxsum":
            return torch.cat([global_mean_pool(x, batch),
                              global_max_pool(x, batch),
                              global_add_pool(x, batch)], dim=-1)
        else:
            raise ValueError("readout must be 'mean' | 'max' | 'meanmax' | 'meansum' | 'meanmaxsum'")

        # Graph-level embedding
        #return global_mean_pool(x, batch)


class GNN(nn.Module):
    """GNN = Graph encoder (SAGE→SAGE [+ optional SAGPooling]) + MLP head."""

    def __init__(
        self,
        in_feats: int,
        hidden_dim: int,
        num_conv_wo_resnet: int,
        num_resnet_layers: int,
        mlp_units: list[int],
        *,
        conv_activation: Callable[..., torch.Tensor] = functional.leaky_relu,
        conv_act_kwargs: dict[str, Any] | None = None,
        mlp_activation: Callable[..., torch.Tensor] = functional.leaky_relu,
        mlp_act_kwargs: dict[str, Any] | None = None,
        classes: list[str] | None = None,
        dropout_p: float = 0.2,
        dropedge_p: float = 0.3,
        force_undirected: bool = False,
        bidirectional: bool = True,
        combine: str = "sum",
        output_dim: int = 1,
        # SAGPooling controls (forwarded)
        use_sag_pool: bool = False,
        sag_ratio: float = 0.5,
        sag_nonlinearity: Callable[..., torch.Tensor] = torch.tanh,
        readout: str = "mean",  # "mean" | "max" | "meanmax" | "meansum" | "meanmaxsum"
        num_features_manual: int = 0,
    ) -> None:
        super().__init__()

        # Graph encoder
        self.graph_conv = GraphConvolutionSage(
            in_feats=in_feats,
            hidden_dim=hidden_dim,
            num_conv_wo_resnet=num_conv_wo_resnet,
            num_resnet_layers=num_resnet_layers,
            conv_activation=conv_activation,
            conv_act_kwargs=conv_act_kwargs,
            dropout_p=dropout_p,
            dropedge_p=dropedge_p,
            force_undirected=force_undirected,
            bidirectional=bidirectional,
            combine=combine,
            use_sag_pool=use_sag_pool,
            sag_ratio=sag_ratio,
            sag_nonlinearity=sag_nonlinearity,
            readout=readout,


        )

        # MLP head
        self.mlp_activation = mlp_activation
        self.mlp_act_kwargs = mlp_act_kwargs or {}
        self.classes = classes
        self.num_features_manual = num_features_manual

        last_dim = self.graph_conv.graph_emb_dim + self.num_features_manual #self.graph_conv.out_dim if readout != "meanmax" else self.graph_conv.out_dim * 2
        #last_dim = self.graph_conv.out_dim  # robust to combine/bidir choices
        self.fcs = nn.ModuleList()
        for out_dim_ in mlp_units:
            self.fcs.append(nn.Linear(last_dim, out_dim_))
            last_dim = out_dim_
        self.out = nn.Linear(last_dim, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x = self.graph_conv(data)
        if self.num_features_manual > 0:
            z = data.feature_vec
            # concat x and z as unique vector
            x = torch.cat([x, z], dim=-1)
        for fc in self.fcs:
            if self.mlp_activation is not None:
                x = self.mlp_activation(fc(x), **self.mlp_act_kwargs)
            else:
                x = fc(x)
        return self.out(x)