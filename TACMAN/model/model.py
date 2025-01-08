from .baselayer import *


# with gat (between cell and gene)
class MYHGNN_GAT(nn.Module):
    def __init__(self, cell_size, gene_size,
                 nfeat, hidden, nclass,
                 num_feats1, num_feats2,
                 dropout, input_drop, att_drop,
                 act, residual=False,):
        # dropout: project layer dropout
        # input_drop: input dropout after embedding
        # att_drop: attention dropout
        super(MYHGNN_GAT, self).__init__()
        self.residual = residual
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)

        #  first to embeding cell and gene
        self.cell_embedings = nn.ParameterDict({})
        self.gene_embedings = nn.ParameterDict({})
        for k, v in cell_size.items():
            
            self.cell_embedings[k] = nn.Parameter(
                        torch.Tensor(v, nfeat).uniform_(-0.5, 0.5))

        for k, v in gene_size.items():
            
            self.gene_embedings[k] = nn.Parameter(
                        torch.Tensor(v, nfeat).uniform_(-0.5, 0.5))


        self.cell_project_layers = nn.Sequential(
            Conv1d1x1(nfeat, hidden, num_feats1, bias=True, cformat='channel-first'),
            nn.LayerNorm([num_feats1, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
            # Conv1d1x1(hidden, hidden, num_feats1, bias=True, cformat='channel-first'),
            # nn.LayerNorm([num_feats1, hidden]),
            # nn.PReLU(),
            # nn.Dropout(dropout),
        )

        self.gene_project_layers = nn.Sequential(
            Conv1d1x1(nfeat, hidden, num_feats2, bias=True, cformat='channel-first'),
            nn.LayerNorm([num_feats2, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
            # Conv1d1x1(hidden, hidden, num_feats2, bias=True, cformat='channel-first'),
            # nn.LayerNorm([num_feats2, hidden]),
            # nn.PReLU(),
            # nn.Dropout(dropout),
        )


        self.cell_semantic_aggr_layers = Transformer(hidden, att_drop, act)
        self.gene_semantic_aggr_layers = Transformer(hidden, att_drop, act)

        self.cell_concat_project_layer = nn.Sequential(
            nn.Linear(num_feats1 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.PReLU(),
            nn.Dropout(dropout),)
        self.gene_concat_project_layer = nn.Sequential(
            nn.Linear(num_feats2 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.PReLU(),
            nn.Dropout(dropout),)
        if self.residual:
            self.res_fc_cell = nn.Linear(nfeat, hidden, bias=True)
            self.res_fc_gene = nn.Linear(nfeat, hidden, bias=True)
        

        self.mix_layer = GraphAttentionLayer((hidden,hidden), hidden)
        self.classifier = nn.Linear(hidden, nclass)
        self.outnorm = nn.LayerNorm(nclass)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")

        for layer in self.cell_project_layers:
            if isinstance(layer, Conv1d1x1):
                layer.reset_parameters()
        for layer in self.gene_project_layers:
            if isinstance(layer, Conv1d1x1):
                layer.reset_parameters()

        for layer in self.cell_concat_project_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)

        for layer in self.gene_concat_project_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)


        if self.residual:
            nn.init.xavier_uniform_(self.res_fc_cell.weight, gain=gain)
            nn.init.xavier_uniform_(self.res_fc_gene.weight, gain=gain)

    def forward(self, feats_dict_cell, feats_dict_gene, g):
        feats_dict1 = {}
        feats_dict2 = {}   
        for k, v in feats_dict_cell.items():
            feats_dict1[k] = v@self.cell_embedings[k]
        for k, v in feats_dict_gene.items():
            feats_dict2[k] = v@self.gene_embedings[k]
        
        x = self.input_drop(torch.stack(list(feats_dict1.values()), dim=1))
        x = self.cell_project_layers(x)
        x = self.cell_semantic_aggr_layers(x)
        cell_embedding = self.cell_concat_project_layer(x.reshape(feats_dict1['C'].size(0), -1))
        if self.residual:
            tgt_feat1 = self.input_drop(feats_dict1['C'])  
            cell_embedding = cell_embedding + self.res_fc_cell(tgt_feat1)

        y = self.input_drop(torch.stack(list(feats_dict2.values()), dim=1))
        y = self.gene_project_layers(y)
        y = self.gene_semantic_aggr_layers(y)
        gene_embedding = self.gene_concat_project_layer(y.reshape(feats_dict2['GC'].size(0), -1))
        if self.residual:
            tgt_feat2 = self.input_drop(feats_dict2['GC'])
            gene_embedding = gene_embedding + self.res_fc_gene(tgt_feat2)

        cell = self.mix_layer(g['G', 'expressed', 'C'],(gene_embedding,cell_embedding))
        out = self.classifier(cell)
        out = self.outnorm(out)

        return out,cell_embedding,gene_embedding

class MYHGNN(nn.Module):
    def __init__(self, cell_size, nfeat, hidden, nclass,
                 num_feats1, dropout, input_drop, att_drop,
                 act, residual=True):
        super(MYHGNN, self).__init__()
        self.residual = residual
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)

        #  first to embeding cell and gene
        self.cell_embedings = nn.ParameterDict({})
        for k, v in cell_size.items():
            
            self.cell_embedings[k] = nn.Parameter(
                        torch.Tensor(v, nfeat).uniform_(-0.5, 0.5))


        self.cell_project_layers = nn.Sequential(
            Conv1d1x1(nfeat, hidden, num_feats1, bias=True, cformat='channel-first'),
            nn.LayerNorm([num_feats1, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
            Conv1d1x1(hidden, hidden, num_feats1, bias=True, cformat='channel-first'),
            nn.LayerNorm([num_feats1, hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
        )

        self.cell_semantic_aggr_layers = Transformer(hidden, att_drop, act)

        self.cell_concat_project_layer = nn.Sequential(
            nn.Linear(num_feats1 * hidden, hidden),
            nn.LayerNorm(hidden),
            nn.PReLU(),
            nn.Dropout(dropout),)

        if self.residual:
            self.res_fc = nn.Linear(nfeat, hidden, bias=True)
        

        self.classifier = nn.Linear(hidden, nclass)
        self.outnorm = nn.LayerNorm(nclass)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")

        for layer in self.cell_project_layers:
            if isinstance(layer, Conv1d1x1):
                layer.reset_parameters()


        for layer in self.cell_concat_project_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)

        if self.residual:
            nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
            
    def forward(self, feats_dict_cell):
        feats_dict1 = {}
        
        for k, v in feats_dict_cell.items():

            feats_dict1[k] = v@self.cell_embedings[k]

        
        x = self.input_drop(torch.stack(list(feats_dict1.values()), dim=1))
        x = self.cell_project_layers(x)
        x = self.cell_semantic_aggr_layers(x)
        cell_embedding = self.cell_concat_project_layer(x.reshape(feats_dict1['C'].size(0), -1))
        if self.residual:
            tgt_feat1 = self.input_drop(feats_dict1['C'])  
            cell_embedding = cell_embedding + self.res_fc(tgt_feat1)

        out = self.classifier(cell_embedding)
        out = self.outnorm(out)

        return out, cell_embedding