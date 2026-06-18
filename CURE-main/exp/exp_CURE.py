import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import copy, time
import numpy as np
from exp.exp import Exp
from lib_gnn_model.node_classifier import NodeClassifier
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from lib_utils.utils import get_dataset_train, get_influence_nodes, get_subgraph, get_dataset_unlearn
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix

torch.cuda.empty_cache()


class ExpCURE(Exp):
    def __init__(self, args):
        super(ExpCURE, self).__init__(args)
        self.logger = logging.getLogger('ExpCURE')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_data()
        self.num_feats = self.data.num_features
        self.train_test_split()
        self.unlearning_request()

        self.target_model_name = self.args['target_model']

        self.determine_target_model()

        self.num_layers = 2

        run_training_time, _ = self._train_model()

        self.original_model_for_mia = copy.deepcopy(self.target_model.model)

        unlearning_method = self.args['method']

        if unlearning_method == 'CURE':
            t, f1, self.unlearned_model = self.CURE()
        else:
            raise ValueError(f"erro: {unlearning_method}")

    def load_data(self):
        self.data = self.data_store.load_raw_data()

    def train_test_split(self):

        if self.args['dataset'] in ['ogbn-arxiv', 'ogbn-products']:
            self.train_indices, self.test_indices = self.data.train_indices.numpy(), self.data.test_indices.numpy()
        else:
            self.train_indices, self.test_indices = train_test_split(np.arange(self.data.num_nodes),
                                                                     test_size=self.args['test_ratio'],
                                                                     random_state=100)

        self.data_store.save_train_test_split(self.train_indices, self.test_indices)

        self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.train_indices))
        self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.test_indices))
        self.data.train_indices = self.train_indices
        self.data.test_indices = self.test_indices

    def unlearning_request(self):
        self.dataset_train = get_dataset_train(self.data)

        if self.args['unlearn_task'] == 'edge':
            self.unique_indices = np.where(self.data.edge_index[0] < self.data.edge_index[1])[0]
            self.remove_indices = np.random.choice(self.unique_indices,
                                                   int(self.unique_indices.shape[0] * self.args['unlearn_ratio']),
                                                   replace=False)
            self.remove_edges = self.data.edge_index[:, self.remove_indices]
            self.unique_nodes = np.unique(self.remove_edges)
            self.influenced_nodes = get_influence_nodes(self.args, self.unique_nodes, self.dataset_train.edge_index,
                                                        1)
            self.dataset_unlearned = get_dataset_unlearn(self.args, self.data, self.unique_nodes, self.remove_indices)
            self.dataset_train_unlearned = get_dataset_train(self.dataset_unlearned)

            self.unlearn_subgraph = get_subgraph(self.influenced_nodes, self.dataset_train)

        else:
            if self.args['dataset'] in ['ogbn-mag', 'ogbn-products']:
                self.unlearning_id = np.random.choice(self.data.train_indices, 100, replace=False)
            else:
                self.unlearning_id = np.random.choice(self.data.train_indices,
                                                      int(len(self.data.train_indices) * self.args['unlearn_ratio']),
                                                      replace=False
                                                      )
            self.influenced_nodes = get_influence_nodes(self.args, self.unlearning_id, self.dataset_train.edge_index)
            self.dataset_train_unlearned = get_dataset_unlearn(self.args, self.dataset_train,
                                                               self.unlearning_id)
            self.dataset_unlearned = get_dataset_unlearn(self.args, self.data, self.unlearning_id)
            if self.args['unlearn_task'] in ['node', 'feature']:
                self.dataset_train_unlearned.train_mask[self.unlearning_id] = False
                self.unlearn_subgraph = get_subgraph(self.unlearning_id, self.dataset_train)

        self.influenced_subgraph = get_subgraph(self.influenced_nodes, self.dataset_train)
        self.influenced_subgraph_unlearned = get_subgraph(self.influenced_nodes, self.dataset_train_unlearned)

        self.dataset_train_loader = NeighborLoader(
            self.dataset_train,
            num_neighbors=[5] * self.args['hops'],
            input_nodes=self.dataset_train.train_mask,
            batch_size=self.args['batch_size'],
            shuffle=True
        )

        self.data_test_loader = NeighborLoader(
            self.data,
            num_neighbors=[5] * self.args['hops'],
            input_nodes=self.data.test_mask,
            batch_size=self.args['test_batch_size'],
            shuffle=False
        )

        self.dataset_train_unlearned_loader = NeighborLoader(
            self.dataset_train_unlearned.contiguous(),
            num_neighbors=[5] * self.args['hops'],
            input_nodes=self.dataset_train_unlearned.train_mask,
            batch_size=self.args['batch_size'],
            shuffle=True
        )

        if self.args['dataset'] in ['ogbn-mag', 'ogbn-products']:
            self.dataset_unlearned_test_loader = NeighborLoader(
                self.dataset_unlearned.contiguous(),
                num_neighbors=[5] * self.args['hops'],
                input_nodes=self.dataset_unlearned.test_mask,
                batch_size=self.args['test_batch_size'],
                shuffle=False
            )
        else:
            self.dataset_unlearned_test_loader = NeighborLoader(
                self.dataset_unlearned.contiguous(),
                num_neighbors=[5] * self.args['hops'],
                input_nodes=self.dataset_unlearned.test_mask,
                batch_size=self.args['test_batch_size'],
                shuffle=False
            )


    def determine_target_model(self):
        num_classes = self.data.num_classes
        self.target_model = NodeClassifier(self.num_feats, num_classes, self.args)

    def _train_model(self):
        start_time = time.time()
        self.target_model.data = self.data
        res = self.train_model()
        train_time = time.time() - start_time
        self.logger.info(f"Model training time: {train_time:.4f}")
        return train_time, res

    def train_model(self):
        train_start = time.time()
        self.target_model.model.train()
        self.retrain_model = copy.deepcopy(self.target_model.model)
        self.target_model.model, self.data = self.target_model.model.to(self.device), self.data.to(self.device)
        self.data.y = self.data.y.squeeze().to(self.device)

        optimizer = torch.optim.Adam(self.target_model.model.parameters(), lr=self.args['lr'],
                                     weight_decay=self.args['wd'])

        criterion = nn.CrossEntropyLoss(label_smoothing=self.args['label_smoothing']).to(self.device)
        for epoch in range(1, self.args['epochs'] + 1):
            self.target_model.model.train()
            loss_all = 0
            train_true_num = 0
            num = 0
            for batch in self.dataset_train_loader:
                optimizer.zero_grad()
                output, _ = self.target_model.model(batch.x.to(self.device), batch.edge_index.to(self.device))
                preds = torch.argmax(output[:batch.batch_size], dim=1)
                train_true_num += torch.sum(preds.cpu() == batch.y[:batch.batch_size])
                loss = criterion(output[:batch.batch_size], batch.y[:batch.batch_size].to(self.device))
                loss.backward()
                optimizer.step()
            train_acc = train_true_num / self.dataset_train.train_mask.sum()
        train_end = time.time()

        self.target_model.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in self.data_test_loader:
                batch = batch
                output, _ = self.target_model.model(batch.x.to(self.device), batch.edge_index.to(self.device))
                preds = torch.argmax(output[:batch.batch_size], dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch.y[:batch.batch_size].cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        f1 = f1_score(all_labels, all_preds, average='micro')

        print("F1 : {:4f},time : {:4f}".format(f1, train_end - train_start))
        training_test_f1 = f1
        return training_test_f1

    def CURE(self):

        device = self.device
        epochs = int(self.args.get('epochs', 20))
        lr = float(self.args.get('lr', self.args['lr']))
        a = float(self.args.get('gamma', 0.5))
        T = 1.5

        if self.args['unlearn_task'] in ['node', 'feature']:
            forget_np = self.unlearning_id
            forget_nodes = torch.as_tensor(list(forget_np), dtype=torch.long, device=device)
        elif self.args['unlearn_task'] == 'edge':
            forget_nodes = torch.as_tensor(list(self.unique_nodes), dtype=torch.long, device=device)
        else:
            raise ValueError(
                f"erro: {self.args['unlearn_task']}"
            )

        student = copy.deepcopy(self.target_model.model).to(device)
        for p in student.parameters():
            p.requires_grad = True

        optimizer = torch.optim.SGD(
            student.parameters(),
            lr=lr,
            weight_decay=float(self.args.get('wd', self.args['wd']))
        )

        full_x = self.dataset_train_unlearned.x.to(device)
        full_edge = self.dataset_train_unlearned.edge_index.to(device)

        y = self.dataset_train.y.to(device).squeeze().long()
        train_mask = self.dataset_train.train_mask.to(device).bool()

        rel_map, _ = self._select_most_related(forget_nodes, top_m=5)

        rel_all = self._select_nodes(forget_nodes=forget_nodes,L=3,ratio=0.8,teleport=0.15,alpha=0.85,)

        diff_label_map = {}
        if forget_nodes.numel() > 0:
            candidate_mask_global = train_mask.clone()
            candidate_mask_global[forget_nodes] = False

            for u in forget_nodes.tolist():
                y_u = y[u].item()
                mask_u = candidate_mask_global & (y != y_u)
                cand_idx = torch.where(mask_u)[0]

                if cand_idx.numel() == 0:
                    all_train_non_forget = torch.where(candidate_mask_global)[0]
                    if all_train_non_forget.numel() == 0:
                        diff_label_map[u] = u
                    else:
                        rid = torch.randint(all_train_non_forget.numel(), (1,), device=device)
                        diff_label_map[u] = all_train_non_forget[rid].item()
                else:
                    rid = torch.randint(cand_idx.numel(), (1,), device=device)
                    diff_label_map[u] = cand_idx[rid].item()


        teacher = copy.deepcopy(self.target_model.model).to(device).eval()

        teacher_x = self.dataset_train.x.to(device)
        teacher_edge = self.dataset_train.edge_index.to(device)
        with torch.no_grad():
            t_logits, _ = teacher(teacher_x, teacher_edge)

        loss_push = torch.tensor(0.0, device=device)
        loss_pull_rand = torch.tensor(0.0, device=device)
        loss_c = torch.tensor(0.0, device=device)
        loss_K = torch.tensor(0.0, device=device)

        t0 = time.time()
        for _ in range(epochs):
            student.train()
            optimizer.zero_grad()
            s_logits, s_feats = student(full_x, full_edge)
            if forget_nodes.numel() > 0:
                anchor_emb = s_feats[forget_nodes]
                lens = [int(rel_map[u].numel()) for u in forget_nodes.tolist()]
                K = max(lens) if len(lens) > 0 else 0
                if K > 0:
                    rows = []
                    masks = []
                    for u in forget_nodes.tolist():
                        r = rel_map[u]
                        len_u = int(r.numel())
                        if len_u == 0:
                            rows.append(torch.zeros((K,), dtype=torch.long, device=device))
                            masks.append(torch.zeros((K,), dtype=torch.float32, device=device))
                            continue
                        if len_u < K:
                            pad = torch.zeros((K - len_u,), dtype=torch.long, device=device)
                            r_pad = torch.cat([r, pad], dim=0)
                            m = torch.cat([
                                torch.ones((len_u,), dtype=torch.float32, device=device),
                                torch.zeros((K - len_u,), dtype=torch.float32, device=device)
                            ], dim=0)
                        else:
                            r_pad = r
                            m = torch.ones((K,), dtype=torch.float32, device=device)
                        rows.append(r_pad)
                        masks.append(m)
                    neg_idx = torch.stack(rows, dim=0)
                    neg_mask = torch.stack(masks, dim=0)
                    neg_emb = s_feats[neg_idx]

                    loss_push = self._sage_push(anchor_emb, neg_emb, neg_mask=neg_mask)
                else:
                    loss_push = torch.tensor(0.0, device=device)
            else:
                loss_push = torch.tensor(0.0, device=device)

            if rel_all.numel() > 0:
                with torch.no_grad():
                    p_t = F.softmax(t_logits[rel_all] / T, dim=1)
                log_p_s = F.log_softmax(s_logits[rel_all] / T, dim=1)
                loss_K = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
            else:
                loss_K = torch.tensor(0.0, device=device)

            if forget_nodes.numel() > 0 and len(diff_label_map) > 0:
                rand_idx = torch.as_tensor(
                    [diff_label_map[u.item()] for u in forget_nodes],
                    dtype=torch.long,
                    device=device
                )
                rand_emb = s_feats[rand_idx]
                loss_pull_rand = self._sage_pull(anchor_emb, rand_emb)
            else:
                rand_idx = torch.empty(0, dtype=torch.long, device=device)
                loss_pull_rand = torch.tensor(0.0, device=device)

            loss_c = loss_push + loss_pull_rand
            loss = (1.0 - a) * loss_c + a * loss_K
            loss.backward()
            optimizer.step()

        elapsed = time.time() - t0
        self.unlearned_model = student
        final_f1 = self._calculate_f1_score(student, self.dataset_unlearned_test_loader)['F1']

        print(
            f"time={elapsed:.2f}s, "
            f"Test_F1={final_f1:.4f}, "
        )

        return elapsed, final_f1, student


    def _select_most_related(self,forget_nodes: torch.Tensor,top_m: int = 5,min_k: int = 1,max_k: int = None,):

        device = self.device

        x = self.dataset_train.x.to(device).float()
        edge_index = self.dataset_train.edge_index.to(device)
        train_mask = self.dataset_train.train_mask.to(device).bool()

        forget_nodes = forget_nodes.to(device).long()

        self.target_model.model.eval()
        with torch.no_grad():
            logits, feats = self.target_model.model(x, edge_index)
            soft = F.softmax(logits, dim=1)

        feats_n = F.normalize(feats, p=2, dim=1)
        soft_n = F.normalize(soft, p=2, dim=1)

        mapping = {}
        chosen = []

        for u in forget_nodes.tolist():
            center = torch.tensor([u], device=device)
            subset_u, sub_edge_index, _, _ = k_hop_subgraph(
                center, 2, edge_index, relabel_nodes=True
            )

            global_ids = subset_u
            is_train = train_mask[global_ids]
            is_forget = torch.isin(global_ids, forget_nodes)
            is_self = (global_ids == u)

            cand_mask = is_train & (~is_forget) & (~is_self)
            cands_sub = torch.where(cand_mask)[0]
            cands_global = global_ids[cands_sub]

            if cands_sub.numel() == 0:
                mapping[u] = torch.empty(0, dtype=torch.long, device=device)
                continue

            emb_cos = (feats_n[cands_global] @ feats_n[u])
            emb01 = (emb_cos - emb_cos.min()) / (emb_cos.max() - emb_cos.min() + 1e-12)

            lbl_cos = (soft_n[cands_global] @ soft_n[u])
            lbl01 = (lbl_cos - lbl_cos.min()) / (lbl_cos.max() - lbl_cos.min() + 1e-12)

            final_score = 0.5 * emb01 + 0.5 * lbl01

            num_cand = cands_global.numel()
            k = int(top_m)

            k = max(k, min_k)
            if max_k is not None:
                k = min(k, max_k)
            k = min(k, num_cand)

            topk = cands_global[torch.topk(final_score, k=k).indices]
            mapping[u] = topk
            chosen.append(topk)

        related_all = (
            torch.unique(torch.cat(chosen))
            if len(chosen) > 0
            else torch.empty(0, dtype=torch.long, device=device)
        )
        return mapping, related_all

    def _select_nodes(self,forget_nodes: torch.Tensor,L: int,ratio: float,teleport: float = 0.15,alpha: float = 0.85,):
        device = self.device

        edge_index = self.dataset_train.edge_index.detach().cpu()
        train_mask = self.dataset_train.train_mask.detach().cpu().bool()
        forget_nodes_cpu = forget_nodes.detach().cpu().long()

        if forget_nodes_cpu.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        L = int(L)
        if L <= 0:
            return torch.empty(0, dtype=torch.long, device=device)

        teleport = float(teleport)
        teleport = max(0.0, min(1.0, teleport))
        alpha = float(alpha)

        alpha = max(0.0, alpha)

        subset2, sub_edge_index, mapping, _ = k_hop_subgraph(
            forget_nodes_cpu, L, edge_index, relabel_nodes=True
        )
        m = int(subset2.numel())
        if m == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        A = to_scipy_sparse_matrix(sub_edge_index, num_nodes=m).tocsr()
        A = A + sp.eye(m, format="csr")

        row_sum = np.asarray(A.sum(axis=1)).reshape(-1)
        row_sum[row_sum == 0.0] = 1.0
        P = sp.diags(1.0 / row_sum).dot(A)

        mapping = np.asarray(mapping, dtype=np.int64)
        if mapping.size == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        p0 = np.zeros((m,), dtype=np.float32)
        p0[mapping] = 1.0 / float(mapping.size)

        p = p0.copy()
        score = np.zeros((m,), dtype=np.float32)

        for t in range(L):

            p = (1.0 - teleport) * (p @ P) + teleport * p0

            score += (alpha ** t) * p

        subset2_global = subset2.numpy()
        subset2_global_t = torch.from_numpy(subset2_global).long()
        score_t = torch.from_numpy(score)

        forget_set = set(forget_nodes_cpu.tolist())
        is_forget2 = torch.tensor(
            [int(int(n) in forget_set) for n in subset2_global_t.tolist()],
            dtype=torch.bool
        )
        cand_mask = train_mask[subset2_global_t] & (~is_forget2)
        cand_idx = torch.where(cand_mask)[0]
        if cand_idx.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        cand_global = subset2_global_t[cand_idx]
        cand_score = score_t[cand_idx]

        ratio = float(ratio)
        ratio = max(0.0, min(1.0, ratio))
        k = int(np.ceil(ratio * cand_global.numel()))
        k = max(1, k)
        k = min(k, cand_global.numel())

        topk_idx = torch.topk(cand_score, k=k, largest=True).indices
        kl_nodes_cpu = torch.unique(cand_global[topk_idx])

        return kl_nodes_cpu.to(device)

    def _sage_push(self,anchor_emb: torch.Tensor,neg_emb: torch.Tensor,neg_mask: torch.Tensor = None) -> torch.Tensor:
        import torch.nn.functional as F
        if neg_emb.dim() == 2:
            dot = (anchor_emb * neg_emb).sum(dim=-1)
            return -(F.logsigmoid(-dot).mean())

        B, K, D = neg_emb.shape
        dot = (anchor_emb.unsqueeze(1) * neg_emb).sum(dim=-1)
        loss_each = F.logsigmoid(-dot)

        if neg_mask is None:

            return -(loss_each.mean())

        neg_mask = neg_mask.to(loss_each.device).float()
        denom = neg_mask.sum(dim=1).clamp_min(1.0)
        loss_per_anchor = (loss_each * neg_mask).sum(dim=1) / denom
        return -(loss_per_anchor.mean())

    def _sage_pull(self,anchor_emb: torch.Tensor,pos_emb: torch.Tensor,pos_mask: torch.Tensor = None) -> torch.Tensor:
        import torch.nn.functional as F
        if pos_emb.dim() == 2:
            dot = (anchor_emb * pos_emb).sum(dim=-1)
            return -(F.logsigmoid(dot).mean())

        dot = (anchor_emb.unsqueeze(1) * pos_emb).sum(dim=-1)
        loss_each = F.logsigmoid(dot)

        if pos_mask is None:
            return -(loss_each.mean())

        pos_mask = pos_mask.to(loss_each.device).float()
        denom = pos_mask.sum(dim=1).clamp_min(1.0)
        loss_per_anchor = (loss_each * pos_mask).sum(dim=1) / denom
        return -(loss_per_anchor.mean())

    def _calculate_f1_score(self, model, loader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                output, _ = model(batch.x, batch.edge_index)
                preds = torch.argmax(output[:batch.batch_size], dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch.y[:batch.batch_size].cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        f1 = f1_score(all_labels, all_preds, average='micro')
        model.train()
        return {'F1': f1}











