import torch
from gadget.geomix_utils import lgw, proj_graph, Square_Euclidean_Distance
import dgl
import ot
import numpy as np

def geomix(g_s, g_t, x_s, x_t, args, clip_eps=0.1, discretize=False, rank=None):
    """
    Generates intermediate graphs between source and target graphs using geometric mixing.

    Args:
        g_s (dgl.DGLGraph): Source graph.
        g_t (dgl.DGLGraph): Target graph.
        x_s (torch.tensor): Source node features.
        x_t (torch.tensor): Target node features.
        args (argparse.Namespace): Arguments containing configuration parameters.
        clip_eps (float, optional): Threshold for clipping edge weights. Defaults to 0.1.
        discretize (bool, optional): Whether to discretize the intermediate adjacency matrices. Defaults to False.
        rank (int, optional): Rank for OT factorization. Defaults to None.
    """
    print('generating intermediate graphs...')
    
    n_s, n_t = g_s.number_of_nodes(), g_t.number_of_nodes()
    adj_s, adj_t = g_s.adj().to_dense(), g_t.adj().to_dense()
    l = [i/(args.n_graphs+1) for i in range(1,args.n_graphs+1)]
    if rank is None:
        rank = min(n_s, n_t)
    Q, R, g = lgw(adj_s, adj_t, x_s, x_t, rank, alpha = args.aug_alpha)
    coarsen_adj_s, coarsen_adj_t, coarsen_x_s, coarsen_x_t = proj_graph(Q, R, adj_s, adj_t, x_s, x_t)
    source_sparsity, target_sparsity = adj_s.sum()/len(adj_s)**2, adj_t.sum()/len(adj_t)**2
    
    ## to record the fgw distance between subsequent sample pairs (Gt, Gt+1)
    dist = []
    adj_prev, x_prev = adj_s, x_s

    g_list = [(g_s, x_s)]
    for i in range(args.n_graphs):
        mixed_adj = (1-l[i]) * coarsen_adj_s + l[i] * coarsen_adj_t
        if discretize:
            clip_eps = (1-l[i]) * source_sparsity + l[i] * target_sparsity
            mixed_num_edges = int(clip_eps * len(mixed_adj)**2)
            _, indices = torch.topk(mixed_adj.flatten(), k=mixed_num_edges)
            indices = np.array(np.unravel_index(indices.cpu().numpy(), mixed_adj.shape)).T
            mixed_adj = torch.zeros_like(mixed_adj)
            mixed_adj[indices[:,0], indices[:,1]] = 1
        mixed_x = (1-l[i]) * coarsen_x_s + l[i] * coarsen_x_t
        mixed_adj.masked_fill_(mixed_adj.le(args.clip_eps), 0) # mask out edges with small weights
        ind = torch.nonzero(mixed_adj)
        src, dst = ind[:,0], ind[:,1]
        weight = mixed_adj[src, dst]
        g = dgl.graph((src, dst), num_nodes = mixed_adj.shape[0])
        g.edata['w'] = weight
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        g_list.append((g.to(args.device), mixed_x.to(args.device)))
        if args.debug:
            M = Square_Euclidean_Distance(x_prev, mixed_x)
            dist.append(ot.fused_gromov_wasserstein2(M, adj_prev, mixed_adj, alpha = args.aug_alpha))
        adj_prev, x_prev = mixed_adj, mixed_x
        
    g_list.append((g_t, x_t))
    return g_list