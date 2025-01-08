import time
import sys
import torch.nn as nn

# sys.path.append('/public/workspace/ruru_97/projects/schgnn/csMAHN')
from . import preprocess as pp
from .utility import *
# from .plot import *
from .loss import make_class_weights
from .eval import get_acc, get_F1_score, get_AMI
from ..model import MYHGNN


class Trainer():
    def __init__(self,
                 adatas,
                 g,
                 inter_net,
                 list_idx,
                 labels,
                 n_classes,
                 key_class,

                 cell_metapath=['C', 'CC', 'CCGGC', 'CGGC', 'CGGCC', 'CCGGCC'],
                 max_hops_path=6,

                 lr=0.01,
                 weight_decay=0.001,
                 threshold=0.9,

                 stages=[200, 200, 200],):

        self.cell_metapath = cell_metapath
        self.max_hops_path = max_hops_path

        self.adatas = adatas
        self.n_classes = n_classes
        self.key_class = key_class
        self.g = g
        self.list_idx = list_idx
        self.inter_net = inter_net
        self.labels = labels
        self.cluster_labels = adatas[1].obs['leiden'].tolist()
        self.reference_label, self.query_label = get_labels_from_adatas(self.adatas, self.key_class)
        self.encode_and_types = pp.get_typeEncoder_ref(self.reference_label)
        self.ref_nums = self.adatas[0].shape[0]

        self.stages = stages
        self.lr = lr
        self.weight_decay = weight_decay
        self.threshold = threshold

        self.propage_metapath(self.g, self.cell_metapath, self.max_hops_path)


        _log_names = (
            'time',
            'train_loss',
            'test_loss',
            'train_acc',
            'test_acc',
            'AMI',
            'microF1',
            'macroF1',
            'weightedF1',
        )

    def init_best(self):
        self.best_epoch = 0
        self.best_pred_acc = 0
        self.best_f1 = 0
        self.best_val_acc = 0
        self.best_ami = 0
        self.count = 0

    def init_log_list(self):
        self.train_loss_list = []
        self.val_loss_list = []
        self.train_acc_list = []
        self.val_acc_list = []
        self.pre_acc_list = []
        self.f1_list = []
        self.ami_list = []

    def propage_metapath(self,
                         g,
                         cell_metapath,
                         max_hops):
        self.feats_cell = {}
        max_hops = max(max([len(x) for x in cell_metapath]) + 1, max_hops)
        # compute k-hop feature
        g = hg_propagate(g, max_hops, echo=False)

        for k in cell_metapath:
            self.feats_cell[k] = g.nodes['C'].data[k]

        self.data_size_cell = {k: v.size(-1) for k, v in self.feats_cell.items()}

        self.g = clear_hg(g, echo=False)

    # def propage_metapath(g,
    #                       cell_metapath = ['C','CC','CCGGC','CGGC','CGGCC','CCGGCC'],
    #                       gene_metapath = ['GC', 'GCC', 'GGC'],
    #                       max_hops = 6,):
    #         feats_cell = {}
    #         feats_gene = {}
    #         extra_metapath = cell_metapath + gene_metapath
    #         max_hops = max(max([len(x) for x in extra_metapath]) + 1, max_hops)
    #         # compute k-hop feature
    #         g = hg_propagate(g, max_hops, echo=False)

    #         for k in cell_metapath:
    #             feats_cell[k] = g.nodes['C'].data[k]
    #         for k in gene_metapath:
    #             feats_gene[k] = g.nodes['G'].data[k]

    #         data_size_cell = {k: v.size(-1) for k, v in feats_cell.items()}
    #         data_size_gene = {k: v.size(-1) for k, v in feats_gene.items()}

    #         g = clear_hg(g, echo=False)
    #         feats = [feats_cell,feats_gene]
    #         data_size = [data_size_cell, data_size_gene]
    #         return feats, data_size, g

    def get_idx(self, list_idx, to_tensor=False):
        # process index
        train_idx, val_idx, pred_idx = list_idx

        train_node_nums = len(train_idx)
        valid_node_nums = len(val_idx)
        pred_node_nums = len(pred_idx)

        trainval_point = train_node_nums
        valpred_point = trainval_point + valid_node_nums
        total_num_nodes = valpred_point + pred_node_nums

        # tranform to tensor
        if to_tensor:
            train_idx, val_idx, pred_idx = [torch.tensor(idx) for idx in list_idx]

        return train_idx, val_idx, pred_idx, trainval_point, valpred_point, total_num_nodes

    def get_enhance_idx(self, stage, output, labels, train_idx, val_idx, pred_idx, valpred_point, total_num_nodes,
                        threshold, ):
        preds = output.argmax(dim=-1)
        predict_prob = output.softmax(dim=1)

        train_acc = get_acc(preds[train_idx], labels[train_idx])
        val_acc = get_acc(preds[val_idx], labels[val_idx])
        pred_acc = get_acc(preds[pred_idx], labels[pred_idx])

        print(f'Stage {stage - 1} history model:\n\t' \
              + f'Train acc {train_acc * 100:.4f} Val acc {val_acc * 100:.4f} Pred acc {pred_acc * 100:.4f}')

        confident_mask = predict_prob.max(1)[0] > threshold
        pred_enhance_offset = np.where(confident_mask[valpred_point:total_num_nodes])[0]
        pred_enhance_idx = pred_enhance_offset + valpred_point
        enhance_idx = pred_enhance_idx.tolist()

        print(f'Stage: {stage}, threshold {threshold}')
        pred_confident_level = (predict_prob[pred_enhance_idx].argmax(1) == labels[pred_enhance_idx]).sum() / len(
            pred_enhance_idx)
        print(
            f'\t\tpred confident nodes: {len(pred_enhance_idx)} / {total_num_nodes - valpred_point}, pred confident_level: {pred_confident_level}')

        return predict_prob, enhance_idx

    def train_for_firststage(self, model,
                             loss_fcn, loss_sim, optimizer,
                             feats, train_idx, labels, inter_net,
                             simi_gama, ):
        model.train()
        optimizer.zero_grad()
        out, x = model(feats)

        L1 = loss_fcn(out[train_idx], labels[train_idx])
        L2 = loss_sim(x[inter_net[0]], x[inter_net[1]])
        loss = L1 + simi_gama * L2

        loss.backward()
        optimizer.step()

        return loss, out, x

    def train_for_mutistage(self, model,
                            loss_fcn, loss_sim, optimizer,
                            feats, train_idx, enhance_idx, labels, inter_net,
                            enhance_gama,
                            simi_gama,
                            device,
                            predict_prob, ):

        L1_ratio = len(train_idx) * 1.0 / (len(train_idx) + len(enhance_idx))
        L2_ratio = len(enhance_idx) * 1.0 / (len(train_idx) + len(enhance_idx))
        extra_weight, extra_y = predict_prob[enhance_idx].max(dim=1)
        # extra_weight = extra_weight.to(device)
        extra_y = extra_y.to(device)

        model.train()
        optimizer.zero_grad()

        out, x = model(feats)

        L1 = loss_fcn(out[train_idx], labels[train_idx])
        L2 = loss_fcn(out[enhance_idx], extra_y)
        L3 = loss_sim(x[inter_net[0]], x[inter_net[1]])

        loss = L1_ratio * L1 + enhance_gama * L2_ratio * L2 + simi_gama * L3

        loss.backward()
        optimizer.step()

        return loss, out, x

    def get_eval(self, preds, labels, val_idx, pred_idx):
        val_acc = get_acc(preds[val_idx], labels[val_idx])
        pred_acc = get_acc(preds[pred_idx], labels[pred_idx])

        # cluster labels?
        ami = get_AMI(preds[pred_idx], self.adatas[1].obs['leiden'].tolist())
        microF1 = get_F1_score(preds[pred_idx], labels[pred_idx], average='micro')
        macroF1 = get_F1_score(preds[pred_idx], labels[pred_idx], average='macro')
        weightedF1 = get_F1_score(preds[pred_idx], labels[pred_idx], average='weighted')
        return val_acc, pred_acc, ami, microF1, macroF1, weightedF1

    def train(self,
              curdir,
              checkpt_file,
              stages=[200, 200, 200],
              best_point_epoch=10,
              use_class_weights=True,
              use_label_smoothing=False,
              patience=100,
              nfeats=64,
              hidden=64,
              dropout=0.5,
              input_drop=0.2,
              att_drop=0.2,
              residual=True, 
              enhance_gama=10,
              simi_gama=0.1,
              dsnames=['a','b']
              ):
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        labels = torch.LongTensor(self.labels).to(device)
        train_idx, val_idx, pred_idx, trainval_point, valpred_point, total_num_nodes = self.get_idx(self.list_idx)

        for stage in range(len(stages)):

            # if stage == 0:
            #     counts_info = get_type_counts_info(self.adatas, self.key_class, dsnames)
            #     counts_info.to_csv(f'{curdir}/counts_info_.csv')
                

            self.init_best()
            self.init_log_list()
            epochs = stages[stage]

            # =======
            # Expand training set
            # =======
            if stage > 0:
                predict_prob, enhance_idx = self.get_enhance_idx(stage, self.output, labels, train_idx, val_idx, pred_idx,
                                                                 valpred_point, total_num_nodes,
                                                                 threshold=self.threshold)
            model = MYHGNN(self.data_size_cell,
                    nfeats, hidden, self.n_classes,
                    len(self.feats_cell),
                    dropout=dropout,
                    input_drop=input_drop,
                    att_drop=att_drop,
                    act='relu',
                    residual=True)
            model = model.to(device)

            if stage == 0:
                print(model)
                print("# Params:", get_n_params(model))

            if use_class_weights:
                if stage == 0:
                    weight_class = make_class_weights(labels[train_idx])
                else:
                    weight_class = make_class_weights(
                        torch.cat((labels[train_idx], predict_prob[enhance_idx].max(dim=1)[1])))
            else:
                weight_class = None
            if use_label_smoothing:
                loss_fcn = nn.CrossEntropyLoss(weight=weight_class,label_smoothing=0.1)
            else:    
                loss_fcn = nn.CrossEntropyLoss(weight=weight_class)  # type: ignore
            loss_sim = nn.MSELoss()

            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            count = 0
            for epoch in range(epochs):

                start = time.time()
                if stage == 0:
                    loss_train, out, x = self.train_for_firststage(model, loss_fcn, loss_sim, optimizer,
                                                                   self.feats_cell,
                                                                   train_idx, labels, self.inter_net,
                                                                   simi_gama)
                else:
                    loss_train, out, x = self.train_for_mutistage(model, loss_fcn, loss_sim, optimizer,
                                                                  self.feats_cell, train_idx, enhance_idx, labels,
                                                                  self.inter_net, enhance_gama, simi_gama,
                                                                  device=device, predict_prob=predict_prob)
                # all data
                model.eval()
                out,x = model(self.feats_cell)
                preds = out.argmax(dim=-1)
                train_acc = get_acc(preds[train_idx], labels[train_idx])
                log = ""
                log = "Epoch {}, train loss {:.4f}, train acc {:.2f}\n".format(epoch, loss_train, train_acc * 100)

                loss_val = loss_fcn(out[val_idx], labels[val_idx]).item()
                # loss_pred = loss_fcn(out[pred_idx], labels[pred_idx]).item()
                loss_pred = 0
                val_acc, pred_acc, ami, microF1, macroF1, weightedF1 = self.get_eval(preds, labels, val_idx, pred_idx)

                end = time.time()
                log += 'Time: {:.4f},Val loss: {:.4f}, Pred loss: {:.4f}, Val acc: {:.2f}, Pred acc: {:.2f}, ami: {:.4f}, f1: {:.2f}\n'.format(
                    end - start, loss_val, loss_pred, val_acc * 100, pred_acc * 100, ami, weightedF1)

                if epoch % 5 == 0:
                    print(log, flush=True)

                if epoch > best_point_epoch:
                    if ami > self.best_ami:
                        self.best_epoch = epoch
                        self.best_ami = ami
                        self.best_val_acc = val_acc
                        self.best_pred_acc = pred_acc
                        self.best_f1 = weightedF1
                        torch.save(model.state_dict(), checkpt_file + f'_{stage}_epoch{self.best_epoch}.pkl')
                        count = 0
                        log += "[Best Epoch {}],Val {:.4f}, Pred {:.4f}".format(self.best_epoch, self.best_val_acc * 100,
                                                                              self.best_pred_acc * 100)
                        print(log, flush=True)
                    else:
                        count += 1
                        if count >= patience:
                            break


                self.train_loss_list.append(loss_train.detach().cpu().numpy())
                self.val_loss_list.append(loss_val)
                self.train_acc_list.append(train_acc.numpy())
                self.val_acc_list.append(val_acc.numpy())
                self.pre_acc_list.append(pred_acc.numpy())
                self.ami_list.append(ami)
                self.f1_list.append(weightedF1)

            # torch.save(model.state_dict(), checkpt_file+f'_{stage}.pkl')
            print("Best Epoch {}, Val {:.4f}, Pred {:.4f}, F1 {:.4f}, AMI {:.4f}".format(self.best_epoch,
                                                                                         self.best_val_acc * 100,
                                                                                         self.best_pred_acc * 100,
                                                                                         self.best_f1,
                                                                                         self.best_ami))
            model.load_state_dict(torch.load(checkpt_file + f'_{stage}_epoch{self.best_epoch}.pkl'))
            self.output, self.embedding_hidden = model(self.feats_cell)
            torch.save(self.output, checkpt_file + f'_best_out_{stage}.pt')
            torch.save(self.embedding_hidden, checkpt_file + f'_best_embedding_{stage}.pt')
            self.save_train_log(curdir,stage)


    def make_adt_res(self):
        self.adt = sc.AnnData(self.embedding_hidden.detach().numpy())
        self.adt.obs = pd.concat(self.adatas[0].obs+ self.adatas[1].obs)


    # def plot_umap(self, figdir):
    #     plot_umap(self.adt, figdir=figdir)

    
    def get_confuse_matrix(self, is_probability=False):
        df = pd.DataFrame(data=None)
        df['true_label'] = np.concatenate(get_labels_from_adatas(self.adatas, self.key_class))
        df['pre_label'] = self.output.argmax(dim=-1).detach().numpy()
        df['pre_label'].replace(self.encode_and_types[0], self.encode_and_types[1], inplace=True)
        df = df.iloc[self.adatas[0].shape[0]:,:]
        df['count'] = 1
        confuse_matrix = df.pivot_table(index='true_label', columns='pre_label', values='count', aggfunc='sum').fillna(0).astype(int)
        confuse_matrix_pro = confuse_matrix.div(confuse_matrix.sum(axis=1),axis=0)
        return confuse_matrix_pro if is_probability else confuse_matrix
    
    def get_train_log(self):
        train_dict = { 'train_loss': self.train_loss_list,
                        'val_loss': self.val_loss_list,
                        'train_acc': self.train_acc_list,
                        'val_acc': self.val_acc_list,
                        'pre_acc': self.pre_acc_list,
                        'ami':self.ami_list,
                        'weightedF1':self.f1_list}
        train_log = pd.DataFrame(train_dict)

        return train_log
    
    def get_cell_out(self):
        reference_out = pd.DataFrame(data=self.output[:self.ref_nums, :].detach().numpy(), columns=self.encode_and_types[1])
        query_out = pd.DataFrame(data=self.output[self.ref_nums:, :].detach().numpy(), columns=self.encode_and_types[1])
        return reference_out, query_out
    
    def get_out_info(self):
        cell_name = self.adatas[0].obs.index.tolist() + self.adatas[1].obs.index.tolist()

        pre_out = pd.DataFrame(data=None, index=cell_name)
        pre_out['true_label'] = self.reference_label+self.query_label
        pre_out['pre_label'] = self.output.argmax(dim=-1).detach().numpy()
        pre_out['pre_label'].replace(self.encode_and_types[0], self.encode_and_types[1], inplace=True)
        pre_out['pre_label_NUM'] = self.output.argmax(dim=-1).detach().numpy()
        pre_prob = self.output.softmax(dim=1).detach().numpy()
        pre_out['max_prob'] = pre_prob.max(axis=1)
        pre_out[self.encode_and_types[1]] = pre_prob
        return pre_out

    def save_train_log( self,
                        curdir,
                        stage):
        reference_out, query_out = self.get_cell_out()
        pre_out = self.get_out_info()
        train_log = self.get_train_log()
        confuse_matrix = self.get_confuse_matrix()

        confuse_matrix.to_csv(f'{curdir}/res_{stage}/confuse_maxtrix_{stage}.csv')
        pre_out.to_csv(f'{curdir}/res_{stage}/pre_out_{stage}.csv')
        reference_out.to_csv(f'{curdir}/res_{stage}/reference_out_{stage}.csv')
        query_out.to_csv(f'{curdir}/res_{stage}/query_out_{stage}.csv')
        train_log.to_csv(f'{curdir}/res_{stage}/train_log_{stage}.csv')
        pre_out.to_csv(f'{curdir}/res_{stage}/pre_out_{stage}.csv')

    
    def load_embedding(self,path):
        self.embedding_hidden = torch.load(path)

    def load_model_weight(self,path):
        self.model.load_state_dict(torch.load(path))
                                                