import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=["cora", "citeseer", "pubmed", "Physics"])
    parser.add_argument('--method', type=str, default="CURE")
    parser.add_argument('--target_model', type=str, default='GCN', choices=["GAT", "GCN", "GIN", "SAGE","APPNP","SIGN"])
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--unlearn_task', type=str, default='node',choices=["edge", "node", 'feature'])
    parser.add_argument('--unlearn_ratio', type=float, default=0.05)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.005, help='learning_rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight_decay')
    parser.add_argument('--optimizer', type=str, default="Adam", help='Adam or SGD')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--hops', type=int, default=2, help='gnn layers')
    parser.add_argument('--erase_ratio', type=float, default=0.01)
    parser.add_argument('--l', type=float, default=0.3)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=64)


    args = vars(parser.parse_args())

    return args
