import logging
import os
import pickle
import shutil

import torch
import numpy as np
import torch_geometric
import torch_geometric.transforms as T
from torch.serialization import add_safe_globals
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, Flickr
from torch_geometric.utils import add_remaining_self_loops, to_undirected
from ogb.nodeproppred import PygNodePropPredDataset

import config


class DataStore:
    def __init__(self, args):
        self.logger = logging.getLogger('data_store')
        self.args = args

        self.dataset_name = self.args['dataset']
        self.num_features = {
            "cora": 1433,
            "pubmed": 500,
            "citeseer": 3703,
            "Physics": 8415,
        }
        self.target_model = self.args['target_model']

        self.determine_data_path()

    def determine_data_path(self):
        embedding_name = '_'.join(('embedding', self.args['unlearn_task'], str(self.args['unlearn_ratio'])))

        processed_data_prefix = config.PROCESSED_DATA_PATH + self.dataset_name + "/"
        self.train_test_split_file = processed_data_prefix + "train_test_split" + str(self.args['test_ratio'])
        self.train_data_file = processed_data_prefix + "train_data"
        self.train_graph_file = processed_data_prefix + "train_graph"
        self.embedding_file = processed_data_prefix + embedding_name

        self.unlearned_file = processed_data_prefix + '_'.join(
            ('unlearned', self.args['unlearn_task'], str(self.args['unlearn_ratio'])))

        dir_lists = [s + self.dataset_name for s in [config.PROCESSED_DATA_PATH, config.MODEL_PATH]]
        for dir in dir_lists:
            self._check_and_create_dirs(dir)

    def _check_and_create_dirs(self, folder):
        if not os.path.exists(folder):
            try:
                os.makedirs(folder, exist_ok=True)
            except OSError:
                shutil.rmtree(folder)
                os.mkdir(folder)
        else:
            pass

    def load_raw_data(self):
        try:
            safe_classes = [
                torch_geometric.data.Data,
                torch_geometric.data.data.DataEdgeAttr,
                torch_geometric.data.data.DataTensorAttr,
                torch_geometric.data.storage.GlobalStorage,
            ]
            registered = []
            for cls in safe_classes:
                if cls not in registered:
                    add_safe_globals([cls])
                    registered.append(cls)

        except Exception as e:
            print(f"[Warning] Safe globals registration skipped or partially applied: {e}")
            pass

        # ====== datasets ======
        if self.dataset_name in ["cora", "pubmed", "citeseer"]:
            dataset = Planetoid(config.RAW_DATA_PATH, self.dataset_name, transform=T.NormalizeFeatures())
            data = dataset[0]

        elif self.dataset_name in ["CS", "Physics"]:
            dataset = Coauthor(config.RAW_DATA_PATH, name=self.dataset_name, pre_transform=T.NormalizeFeatures())
            data = dataset[0]
        else:
            raise Exception('unsupported dataset')

        data.name = self.dataset_name
        data.num_classes = dataset.num_classes

        return data

    def save_train_data(self, train_data):
        pickle.dump(train_data, open(self.train_data_file, 'wb'))

    def load_train_data(self):
        return pickle.load(open(self.train_data_file, 'rb'))

    def save_train_graph(self, train_data):
        pickle.dump(train_data, open(self.train_graph_file, 'wb'))

    def load_train_graph(self):
        return pickle.load(open(self.train_graph_file, 'rb'))

    def save_train_test_split(self, train_indices, test_indices):
        pickle.dump((train_indices, test_indices), open(self.train_test_split_file, 'wb'))

    def load_train_test_split(self):
        return pickle.load(open(self.train_test_split_file, 'rb'))

    def save_embeddings(self, embeddings):
        pickle.dump(embeddings, open(self.embedding_file, 'wb'))

    def load_embeddings(self):
        return pickle.load(open(self.embedding_file, 'rb'))

    def load_unlearned_data(self, suffix):
        file_path = '_'.join((self.unlearned_file, suffix))
        return pickle.load(open(file_path, 'rb'))

    def save_unlearned_data(self, data, suffix):
        pickle.dump(data, open('_'.join((self.unlearned_file, suffix)), 'wb'))
