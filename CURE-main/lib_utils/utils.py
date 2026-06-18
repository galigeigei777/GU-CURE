import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph


def filter_edge_index(edge_index, node_indices, reindex=True):

    assert np.all(np.diff(node_indices) >= 0), 'node_indices must be sorted'
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu()

    node_index = np.isin(edge_index, node_indices)
    col_index = np.nonzero(np.logical_and(node_index[0], node_index[1]))[0]
    edge_index = edge_index[:, col_index]

    if reindex:
        return np.searchsorted(node_indices, edge_index)
    else:
        return edge_index

def get_dataset_train(data):
    train_indices = np.nonzero(data.train_mask.cpu().numpy())[0]
    edge_index = filter_edge_index(data.edge_index, train_indices, reindex=False)

    if isinstance(edge_index, np.ndarray):
        edge_index = torch.from_numpy(edge_index)

    if edge_index.shape[1] == 0:
        edge_index = torch.tensor([[1, 2], [2, 1]])

    # 克隆，避免共享同一底层存储
    dataset_train = Data(
        x=data.x.clone(),
        edge_index=edge_index.clone(),
        y=data.y.clone(),
        train_mask=data.train_mask.clone(),
        test_mask=data.test_mask.clone(),
        train_indices=data.train_indices,
        test_indices=data.test_indices
    )
    return dataset_train


def get_influence_nodes(args, unlearn_nodes, edge_index, hops=2):

    influenced_nodes = unlearn_nodes


    for _ in range(hops):
        target_nodes_location = np.isin(edge_index[0], influenced_nodes)
        neighbor_nodes = edge_index[1, target_nodes_location]
        influenced_nodes = np.append(influenced_nodes, neighbor_nodes)
        influenced_nodes = np.unique(influenced_nodes)

    if args['unlearn_task'] == 'node':
        neighbor_nodes = np.setdiff1d(influenced_nodes, unlearn_nodes)
    else:
        neighbor_nodes = influenced_nodes

    return neighbor_nodes


def get_subgraph(node_id, data, hops=2):
    node_id = node_id[np.isin(node_id, data.edge_index.cpu().numpy())]

    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        torch.tensor(node_id),
        hops,
        data.edge_index,
        relabel_nodes=True
    )

    subgraph = Data(
        x=data.x[subset],
        edge_index=edge_index,
        y=data.y[subset],
        batch_size=len(node_id),
        mapping=mapping
    )
    return subgraph

def get_dataset_unlearn(args, data, unlearning_id, delete_edge_index=None):
    if args['unlearn_task'] == 'feature':
        x = data.x.clone()
        x[unlearning_id] = 0
        train_mask = data.train_mask.clone()
        test_mask = data.test_mask.clone()
        dataset_unlearn = Data(
            x=x,
            edge_index=data.edge_index,
            y=data.y,
            train_mask=train_mask,
            test_mask=test_mask,
            train_indices=data.train_indices,
            test_indices=data.test_indices
        )
    else:
        if args['unlearn_task'] == 'node':
            edge_index_unlearn = update_edge_index_unlearn(args, data.edge_index.cpu().numpy(), unlearning_id)
        elif args['unlearn_task'] == 'edge':
            edge_index_unlearn = update_edge_index_unlearn(args, data.edge_index.cpu().numpy(), unlearning_id, delete_edge_index)
        train_mask = data.train_mask.clone()
        test_mask = data.test_mask.clone()
        dataset_unlearn = Data(
            x=data.x,
            edge_index=edge_index_unlearn,
            y=data.y,
            train_mask=train_mask,
            test_mask=test_mask,
            train_indices=data.train_indices,
            test_indices=data.test_indices
        )
    return dataset_unlearn


def update_edge_index_unlearn(args, edge_index, delete_nodes, delete_edge_index=None):
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    self_loop_idx = np.where(edge_index[0] == edge_index[1])[0]

    unique_idx = np.where(edge_index[0] < edge_index[1])[0]
    unique_idx_not = np.where(edge_index[0] > edge_index[1])[0]

    if args['unlearn_task'] == 'edge':
        if delete_edge_index is None:
            delete_edge_index = np.array([], dtype=unique_idx.dtype)
        remain_idx = np.setdiff1d(unique_idx, delete_edge_index)
    else:
        unique_edges = edge_index[:, unique_idx]
        delete_mask = np.logical_or(np.isin(unique_edges[0], delete_nodes),
                                    np.isin(unique_edges[1], delete_nodes))
        remain_idx = unique_idx[~delete_mask]

    base = int(edge_index.max()) + 1
    remain_code = edge_index[0, remain_idx] * base + edge_index[1, remain_idx]
    not_code = edge_index[1, unique_idx_not] * base + edge_index[0, unique_idx_not]

    sort_idx = np.argsort(not_code)
    sorted_not_code = not_code[sort_idx]

    pos = np.searchsorted(sorted_not_code, remain_code)
    valid = (pos < sorted_not_code.size) & (sorted_not_code[pos] == remain_code)
    remain_idx_not = unique_idx_not[sort_idx[pos[valid]]]

    all_remain = np.union1d(remain_idx, remain_idx_not)

    return torch.from_numpy(edge_index[:, all_remain]).long()



