import networkx as nx
import numpy as np
import torch
import dgl
from dgl import DGLGraph
import os
import yaml


def load_config(args):
    with open('src/config.yaml', 'r') as yamlfile:
        config = yaml.safe_load(yamlfile)
    args.hidden_dim = config[args.data]['hidden_dim']
    args.epochs = config[args.data]['epochs']
    args.lr = float(config[args.data]['lr'])
    args.n_hidden = config[args.data]['n_hidden']
    args.n_graphs = config[args.data]['n_graphs']
    aug_alpha = config[args.data]['aug_alpha']
    if aug_alpha == 'None':
        args.aug_alpha = None
    else:
        args.aug_alpha = float(aug_alpha)
    return args

def degree_bucketing(graph, max_degree=64):
    features = torch.zeros([graph.number_of_nodes(), max_degree])
    for i in range(graph.number_of_nodes()):
        try:
            features[i][min(graph.in_degrees(i), max_degree-1)] = 1
        except:
            features[i][0] = 1
    return features

def get_data(dataset, domain, d="s", offset = 0.1):
    current_work_dir = './'
    if dataset == 'airport':
        edge_path = current_work_dir + '/data/airport/{}-airports.edgelist'.format(domain)
        attribute_path = None
        label_path = current_work_dir + '/data/airport/{}-airports-labels.txt'.format(domain)
    elif dataset == 'citation':
        edge_path = current_work_dir + '/data/citation/{}-citation.edgelist'.format(domain)
        attribute_path = current_work_dir + '/data/citation/{}-citation-attribute.txt'.format(domain)
        label_path = current_work_dir + '/data/citation/{}-citation-labels.txt'.format(domain)
    elif dataset == 'social':
        edge_path = current_work_dir + '/data/social/{}-social.edgelist'.format(domain)
        attribute_path = current_work_dir + '/data/social/{}-social-attribute.txt'.format(domain)
        label_path = current_work_dir + '/data/social/{}-social-labels.txt'.format(domain)

    edge_list = torch.from_numpy(np.loadtxt(edge_path, dtype=np.int64))
    labels = torch.from_numpy(np.loadtxt(label_path, dtype=np.int64))
    if attribute_path is not None:
        features = torch.from_numpy(np.loadtxt(attribute_path, dtype=np.float32))
    else:
        features = None
    g = DGLGraph()
    g.add_nodes(len(labels))
    for edge in edge_list:
        g.add_edges(edge[0], edge[1])
        g.add_edges(edge[1], edge[0])
    g = dgl.remove_self_loop(g)  # remove self-loops if any
    g = dgl.add_self_loop(g)  # ensure self-loops are added back
    
    if features is None:
        features = degree_bucketing(g, 64)
    if d == "t":    # manually add attribute shift
        features += offset
    return g, labels, features
