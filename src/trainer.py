from __future__ import division
from __future__ import print_function
from models import *
from gadget.geomix import geomix
from utils import load_config
import gc
from tqdm import tqdm
import torch.nn.functional as F
import time


'''
    Trainer for BaseGNN
'''
def BaseGNN_train(args, data_s, data_t, model_name, PCA_dim=64, eval=True, weight=None, ent=None):
    """
    Trainer for BaseGNN model.

    Args:
        args (argparse.Namespace): Training arguments.
        data_s (tuple): Source data (graph, labels, features).
        data_t (tuple): Target data (graph, labels, features).
        model_name (str): Name of the GNN model (e.g., 'GCN', 'APPNP').
        PCA_dim (int, optional): Dimension for PCA reduction. Defaults to 64.
        eval (bool, optional): Whether to evaluate on target domain. Defaults to True.
        weight (torch.tensor, optional): Sample weights for loss calculation. Defaults to None.
        ent (float, optional): Entropy regularization weight. Defaults to None.

    Returns:
        tuple: (accuracy, predictions, model)
    """
    t1=time.time()
    args = load_config(args)
    g_s, labels_s, features_s = data_s  # source graph
    g_t, labels_t, features_t = data_t  # target graph

    if features_s.shape[1] > PCA_dim:
        features = torch.cat((features_s, features_t), dim=0)
        n_s = features_s.shape[0]
        _, _, V = torch.pca_lowrank(features, q=int(1.2*PCA_dim))
        pca_features = features @ V[:, :PCA_dim]
        features_s, features_t = pca_features[:n_s], pca_features[n_s:]
    in_feats = features_s.shape[1]
    n_classes = labels_s.max().item() + 1
    g_s = g_s.to(device=args.device)
    g_t = g_t.to(device=args.device)
    features_s = features_s.to(device=args.device)
    labels_s = labels_s.to(device=args.device)
    features_t = features_t.to(device=args.device)
    
    model = BaseGNN(in_feats, n_classes, model_name, args).to(device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = model(g_s, features_s, labels_s, weight=weight, ent=ent)
        loss.backward()
        optimizer.step()

    if eval:
        model.eval()
        with torch.no_grad():
            pred, _ = model.inference(g_t, features_t)
            _, indices = torch.max(pred, dim=1)
            labels_t = labels_t.to(device=args.device)
            correct = torch.sum(indices == labels_t)
            acc = correct.item() * 1.0 / len(labels_t)
    else:
        with torch.no_grad():
            pred, _ = model.inference(g_t, features_t)
        acc = None
    print('BaseGNN train time: {}'.format(time.time()-t1))
    return acc, pred, model


'''
    Trainer for GRADE
'''
def GRADE_train(args, data_s, data_t, model_name, PCA_dim=64, eval=True, weight=None):
    """
    Trainer for GRADE model.

    Args:
        args (argparse.Namespace): Training arguments.
        data_s (tuple): Source data (graph, labels, features).
        data_t (tuple): Target data (graph, labels, features).
        model_name (str): Name of the GNN model.
        PCA_dim (int, optional): Dimension for PCA reduction. Defaults to 64.
        eval (bool, optional): Whether to evaluate on target domain. Defaults to True.
        weight (torch.tensor, optional): Sample weights. Defaults to None.

    Returns:
        tuple: (accuracy, predictions, model)
    """
    args = load_config(args)
    g_s, labels_s, features_s = data_s
    g_t, labels_t, features_t = data_t
    ## use PCA to reduce feature dimension for computational efficiency
    if features_s.shape[1] > PCA_dim:
        features = torch.cat((features_s, features_t), dim=0)
        n_s = features_s.shape[0]
        _, _, V = torch.pca_lowrank(features, q=PCA_dim)
        pca_features = features @ V
        features_s, features_t = pca_features[:n_s], pca_features[n_s:]
    in_feats = features_s.shape[1]
    n_classes = labels_s.max().item() + 1
    g_s = g_s.to(device=args.device)
    g_t = g_t.to(device=args.device)
    features_s = features_s.to(device=args.device)
    labels_s = labels_s.to(device=args.device)
    features_t = features_t.to(device=args.device)

    model = GRADE(g_s, g_t, in_feats, n_classes, model_name, args).to(device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        alpha = 2 / (1 + np.exp(- 10 * epoch / args.epochs)) - 1
        loss = model(features_s, labels_s, features_t, alpha, weight=weight)
        loss.backward()
        optimizer.step()

    if eval:
        model.eval()
        with torch.no_grad():
            pred, _ = model.inference(features_t)
            _, indices = torch.max(pred, dim=1)
            labels_t = labels_t.to(device=args.device)
            correct = torch.sum(indices == labels_t)
            acc = correct.item() * 1.0 / len(labels_t)
    else:
        with torch.no_grad():
            pred, _ = model.inference(features_t)
        acc = None
    return acc, pred, model



'''
    Trainer for Base domain adaptation methods (CORAL, MMD)
'''
def BaseDA_train(args, data_s, data_t, model_name, PCA_dim=64, eval=True, weight=None):
    """
    Trainer for Base domain adaptation methods (CORAL, MMD).

    Args:
        args (argparse.Namespace): Training arguments.
        data_s (tuple): Source data.
        data_t (tuple): Target data.
        model_name (str): Name of the GNN model.
        PCA_dim (int, optional): Dimension for PCA reduction. Defaults to 64.
        eval (bool, optional): Whether to evaluate. Defaults to True.
        weight (torch.tensor, optional): Sample weights. Defaults to None.

    Returns:
        tuple: (accuracy, predictions, model)
    """
    args = load_config(args)
    g_s, labels_s, features_s = data_s
    g_t, labels_t, features_t = data_t
    ## use PCA to reduce feature dimension for computational efficiency
    if features_s.shape[1] > PCA_dim:
        features = torch.cat((features_s, features_t), dim=0)
        n_s = features_s.shape[0]
        _, _, V = torch.pca_lowrank(features, q=PCA_dim)
        pca_features = features @ V
        features_s, features_t = pca_features[:n_s], pca_features[n_s:]
    in_feats = features_s.shape[1]
    n_classes = labels_s.max().item() + 1
    g_s = g_s.to(device=args.device)
    g_t = g_t.to(device=args.device)
    features_s = features_s.to(device=args.device)
    labels_s = labels_s.to(device=args.device)
    features_t = features_t.to(device=args.device)
    
    model = BaseDA(g_s, g_t, in_feats, n_classes, model_name, args).to(device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        loss = model(features_s, labels_s, features_t, weight=weight)
        loss.backward()
        optimizer.step()

    if eval:
        model.eval()
        with torch.no_grad():
            pred, _ = model.inference(features_t)
            _, indices = torch.max(pred, dim=1)
            labels_t = labels_t.to(device=args.device)
            correct = torch.sum(indices == labels_t)
            acc = correct.item() * 1.0 / len(labels_t)
    else:
        with torch.no_grad():
            pred, _ = model.inference(features_t)
        acc = None
    return acc, pred, model


'''
    Trainer for AdaGCN
'''
def AdaGCN_train(args, data_s, data_t, model_name, PCA_dim=64, eval=True, weight=None):
    """
    Trainer for AdaGCN model.

    Args:
        args (argparse.Namespace): Training arguments.
        data_s (tuple): Source data.
        data_t (tuple): Target data.
        model_name (str): Name of the GNN model.
        PCA_dim (int, optional): Dimension for PCA reduction. Defaults to 64.
        eval (bool, optional): Whether to evaluate. Defaults to True.
        weight (torch.tensor, optional): Sample weights. Defaults to None.

    Returns:
        tuple: (accuracy, predictions, model)
    """
    args = load_config(args)
    g_s, labels_s, features_s = data_s
    g_t, labels_t, features_t = data_t
    ## use PCA to reduce feature dimension for computational efficiency
    if features_s.shape[1] > PCA_dim:
        features = torch.cat((features_s, features_t), dim=0)
        n_s = features_s.shape[0]
        _, _, V = torch.pca_lowrank(features, q=PCA_dim)
        pca_features = features @ V
        features_s, features_t = pca_features[:n_s], pca_features[n_s:]
    in_feats = features_s.shape[1]
    n_classes = labels_s.max().item() + 1
    g_s = g_s.to(device=args.device)
    g_t = g_t.to(device=args.device)
    features_s = features_s.to(device=args.device)
    labels_s = labels_s.to(device=args.device)
    features_t = features_t.to(device=args.device)

    model = AdaGCN(g_s, g_t, in_feats, n_classes, model_name, args).to(device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        gp_loss, da_loss = model(features_s, labels_s, features_t, weight=weight)
        gp_loss.backward(retain_graph=True)
        da_loss.backward()
        optimizer.step()
    
    if eval:
        model.eval()
        with torch.no_grad():
            pred, _ = model.inference(features_t)
            _, indices = torch.max(pred, dim=1)
            labels_t = labels_t.to(device=args.device)
            correct = torch.sum(indices == labels_t)
            acc = correct.item() * 1.0 / len(labels_t)
    else:
        with torch.no_grad():
            pred, _ = model.inference(features_t)
        acc = None
    return acc, pred, model



def Gadget_train(args, data_s, data_t, PCA_dim=64, n_graphs = None, rank=None):
    """grdual domain adaptation

    Args:
        args (parser): training parameters
        data_s (dgl.graph): source graph
        data_t (dgl.graph): target graph
        PCA_dim (int, optional): reduce input feature dimension by PCA. Defaults to 64.
        n_graphs (int, optional): number of intermediate domains

    Returns:
        acc (float): prediction accuracy 
        pred (torch.tensor): model output pred
    """
    args = load_config(args)
    if n_graphs is not None:
        args.n_graphs = n_graphs
    g_s, labels_s, features_s = data_s
    g_t, labels_t, features_t = data_t
    features = torch.cat((features_s, features_t), dim=0)
    n_s = features_s.shape[0]
    log_ = {}
    
    ## use PCA to reduce feature dimension for computational efficiency
    if features_s.shape[1] > PCA_dim:
        features = torch.cat((features_s, features_t), dim=0)
        _, _, V = torch.pca_lowrank(features, q=int(1.2*PCA_dim))
        features = features @ V[:, :PCA_dim]
    features = features / (features.norm(dim=1, keepdim=True) + 1e-16)
    features_s, features_t = features[:n_s], features[n_s:]
    ## augment data
    g_s, labels_s, features_s = g_s.to(args.device), labels_s.to(args.device), features_s.to(args.device)
    g_t = g_t.to(args.device)
    labels_t = labels_t.to(args.device)
    features_t = features_t.to(args.device)
    t1 = time.time()
    g_list = geomix(g_s, g_t, features_s, features_t, args, rank=rank)
    t2 = time.time()
    print('Generation time: {}'.format(t2-t1))
    log_['gt'] = t2-t1
    for i in range(len(g_list)):
        g_list[i] = (g_list[i][0].to(device=args.device), g_list[i][1].to(device=args.device))
    n_classes = labels_s.max().item() + 1
    
    labels_prev, weight_prev = {}, {}
    for model_name in args.model_list:
        labels_prev[model_name] = labels_s
        weight_prev[model_name] = torch.ones(len(labels_s)).to(device=args.device) / len(labels_s)
    print('Model training...')
    for i in tqdm(range(len(g_list)-1)):
        g_prev, features_prev = g_list[i]
        g_post, features_post = g_list[i+1]
        
        for model_name in args.model_list:
            _, pred, model = direct_train(args, (g_prev, labels_prev[model_name], features_prev), (g_post, None, features_post), model_name, eval=False, weight=weight_prev[model_name], ent=0.1)
            _, labels_prev[model_name] = torch.max(pred, dim=1)
            ent = -torch.sum(pred*torch.exp(pred), dim=1).detach()   # pred are output of logsoftmax
            weight_prev[model_name] = 1 - (ent-ent.min()) / (ent.max()-ent.min())
            weight_prev[model_name] = weight_prev[model_name] / weight_prev[model_name].sum()
    
    acc = []
    t3 = time.time()
    print('Training time: {}'.format(t3-t2))
    log_['tt'] = t3-t2
    for model_name in args.model_list:
        correct = torch.sum(labels_prev[model_name] == labels_t)
        tmp_acc = correct.item() * 1.0 / len(labels_t)
        acc.append(tmp_acc)
    return acc, pred, log_


def direct_train(args, data_s, data_t, model_name, eval=True, weight=None, ent=None):
    """dirct domain adaptation

    Args:
        args (parser): training parameters
        data_s (dgl.graph): source graph
        data_t (dgl.graph): target graph
        model_name (string): gnn model
        eval (bool, optional): evaluate model performance or not. Defaults to True.
        weight (torch.tensor, optional): pseudo label weights. Defaults to None.

    Returns:
        acc (float): prediction accuracy 
        pred (torch.tensor): model output pred
    """
    if args.method == 'VANILLA':
        acc, pred, model = BaseGNN_train(args, data_s, data_t, model_name, eval=eval, weight=weight, ent=ent)
    elif args.method == 'GRADE':
        acc, pred, model = GRADE_train(args, data_s, data_t, model_name, eval=eval, weight=weight)
    elif args.method in ['CORAL', 'MMD']:
        acc, pred, model = BaseDA_train(args, data_s, data_t, model_name, eval=eval, weight=weight)
    elif args.method == 'AdaGCN':
        acc, pred, model = AdaGCN_train(args, data_s, data_t, model_name, eval=eval, weight=weight)
    return acc, pred, model
