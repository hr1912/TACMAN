import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from dgl import function as fn
from dgl.nn.pytorch.softmax import edge_softmax


class GraphAttentionLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 n_heads=4,
                 feat_drop=0.01,
                 attn_drop=0.6,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 attn_type='add',  # or 'add' as the original paper
                 heads_fuse='mean',  # 'flat' or 'mean'
                 ):
        super(GraphAttentionLayer, self).__init__()
        self._n_heads = n_heads
        self._in_src_dim, self._in_dst_dim = in_dim
        self._out_dim = out_dim

        ### weights for linear feature transform
        if isinstance(in_dim, tuple):
            ### asymmetric case
            self.fc_src = nn.Linear(
                self._in_src_dim, out_dim * n_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_dim, out_dim * n_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_dim, out_dim * n_heads, bias=False)
        ### weights for attention computation
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, n_heads, out_dim)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, n_heads, out_dim)))
        if residual:
            if self._in_dst_dim != out_dim:
                self.res_fc = nn.Linear(
                    self._in_dst_dim, n_heads * out_dim, bias=False)

        else:
            self.register_buffer('res_fc', None)

        self.leaky_relu = nn.LeakyReLU(negative_slope)  # for thresholding attentions
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.reset_parameters()

        self.activation = activation  # output
        self.attn_type = attn_type
        self._set_attn_fn()
        self.heads_fuse = heads_fuse
        self._set_fuse_fn()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:  # bipartite graph neural networks
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, g, feat, return_attn=False):
        g = g.local_var()
        ### feature linear transform
        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            feat_src = self.fc_src(h_src).view(-1, self._n_heads, self._out_dim)
            feat_dst = self.fc_dst(h_dst).view(-1, self._n_heads, self._out_dim)
        else:
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(
                -1, self._n_heads, self._out_dim)
        # compute attention score
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        if self.heads_fuse == 'mul':
            er /= np.sqrt(self._out_dim)
        g.srcdata.update({'ft': feat_src, 'el': el})
        g.dstdata.update({'er': er})
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        g.apply_edges(self.attn_fn)

        e = self.leaky_relu(g.edata.pop('e'))
        # compute softmax (normalized weights)
        g.edata['a'] = self.attn_drop(edge_softmax(g, e))
        # message passing
        g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
        rst = g.dstdata['ft']
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_dim)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)

        # handling multi-heads
        rst = self.fuse_heads(rst)
        if return_attn:
            return rst, g.edata['a']
        return rst

    def _set_attn_fn(self, ):
        if self.attn_type == 'mul':
            self.attn_fn = fn.u_mul_v('el', 'er', 'e')
        elif self.attn_type == 'add':
            # use the same attention as the GAT paper
            self.attn_fn = fn.u_add_v('el', 'er', 'e')
        else:
            raise ValueError('`attn_type` shoul be either "add" (paper) or "mul"')

    def _set_fuse_fn(self, ):
        # function handling multi-heads
        if self.heads_fuse is None:
            self.fuse_heads = lambda x: x
        elif self.heads_fuse == 'flat':
            self.fuse_heads = lambda x: x.flatten(1)  # then the dim_out is of H * D_out
        elif self.heads_fuse == 'mean':
            self.fuse_heads = lambda x: x.mean(1)  # then the dim_out is of D_out
        elif self.heads_fuse == 'max':
            self.fuse_heads = lambda x: torch.max(x, 1)[0]  # then the dim_out is of D_out


class Transformer(nn.Module):
    def __init__(self, n_channels, att_drop=0., act='none', num_heads=1):
        super(Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        assert self.n_channels % (self.num_heads * 4) == 0

        self.query = nn.Linear(self.n_channels, self.n_channels//4)
        self.key   = nn.Linear(self.n_channels, self.n_channels//4)
        self.value = nn.Linear(self.n_channels, self.n_channels)

        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif act == 'none':
            self.act = lambda x: x
        else:
            assert 0, f'Unrecognized activation function {act} for class Transformer'

    def reset_parameters(self):

        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.query.weight, gain=gain)
        nn.init.xavier_uniform_(self.key.weight, gain=gain)
        nn.init.xavier_uniform_(self.value.weight, gain=gain)
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)

    def forward(self, x, mask=None):
        B, M, C = x.size() # batchsize, num_metapaths, channels
        H = self.num_heads
        if mask is not None:
            assert mask.size() == torch.Size((B, M))

        f = self.query(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]
        g = self.key(x).view(B, M, H, -1).permute(0,2,3,1)   # [B, H, -1, M]
        h = self.value(x).view(B, M, H, -1).permute(0,2,1,3) # [B, H, M, -1]

        beta = F.softmax(self.act(f @ g / math.sqrt(f.size(-1))), dim=-1) # [B, H, M, M(normalized)]
        beta = self.att_drop(beta)
        if mask is not None:
            beta = beta * mask.view(B, 1, 1, M)
            beta = beta / (beta.sum(-1, keepdim=True) + 1e-12)

        o = self.gamma * (beta @ h) # [B, H, M, -1]
        return o.permute(0,2,1,3).reshape((B, M, C)) + x


class Conv1d1x1(nn.Module):
    def __init__(self, cin, cout, groups, bias=True, cformat='channel-first'):
        super(Conv1d1x1, self).__init__()
        self.cin = cin
        self.cout = cout
        self.groups = groups
        self.cformat = cformat
        if not bias:
            self.bias = None
        if self.groups == 1: # different keypoints share same kernel
            self.W = nn.Parameter(torch.randn(self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(1, self.cout))
        else:
            self.W = nn.Parameter(torch.randn(self.groups, self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.groups, self.cout))

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.W, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.groups == 1:
            if self.cformat == 'channel-first':
                return torch.einsum('bcm,mn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                return torch.einsum('bmc,mn->bnc', x, self.W) + self.bias.T
            else:
                assert False
        elif self.cformat == 'channel-first':
            return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias
        elif self.cformat == 'channel-last':
            return torch.einsum('bmc,cmn->bnc', x, self.W) + self.bias.T
        else:
            assert False
