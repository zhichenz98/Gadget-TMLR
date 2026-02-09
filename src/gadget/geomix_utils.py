import ot
from sklearn.cluster import KMeans
import torch



eps = 1e-10 # avoid division over 0

def log_space_product(A,B):
    """
    Computes the log-space product of two matrices.

    Args:
        A (torch.tensor): First matrix.
        B (torch.tensor): Second matrix.
    """
    Astack = torch.permute(A.unsqueeze(0).repeat(B.shape[1],1,1), (1,0,2))
    Bstack = torch.permute(B.unsqueeze(0).repeat(A.shape[0],1,1), (0,2,1))
    return torch.logsumexp(Astack+Bstack, dim=2)


def Square_Euclidean_Distance(X, Y):
    """
    Computes the squared Euclidean distance matrix between rows of X and Y.

    Args:
        X (torch.tensor): First matrix, shape = (n1, d).
        Y (torch.tensor): Second matrix, shape = (n2, d).
    """
    device = X.device
    d = X.shape[1]
    return torch.sum(X**2, dim=1, keepdim=True) + torch.sum(Y.T**2, dim=0, keepdim=True) - 2*X@Y.T


def KL(A, B, eps = 1e-10):
    """
    Computes the Kullback-Leibler divergence between A and B.

    Args:
        A (torch.tensor): First distribution/matrix.
        B (torch.tensor): Second distribution/matrix.
        eps (float, optional): Epsilon for numerical stability. Defaults to 1e-10.
    """
    Ratio_trans = torch.log((A + eps) / (B + eps))
    return torch.sum(A * Ratio_trans)


def log_LR_Dykstra(log_K1, log_K2, log_K3, a, b, alpha=1e-10, max_iter=500, delta=1e-9):
    """
    Solves the Low-Rank Sinkhorn problem in log domain using Dykstra's algorithm.

    Args:
        log_K1 (torch.tensor): Log of first kernel matrix.
        log_K2 (torch.tensor): Log of second kernel matrix.
        log_K3 (torch.tensor): Log of third kernel matrix.
        a (torch.tensor): Source marginal distribution.
        b (torch.tensor): Target marginal distribution.
        alpha (float, optional): Parameter for Dykstra. Defaults to 1e-10.
        max_iter (int, optional): Maximum number of iterations. Defaults to 500.
        delta (float, optional): Error threshold for convergence. Defaults to 1e-9.
    """
    device = log_K2.device
    Q = log_K1
    R = log_K2
    log_a = torch.log(a.reshape(-1,1))
    log_b = torch.log(b.reshape(-1,1))
    g_old = log_K3.reshape(-1,1)
    alpha = torch.log(torch.tensor(alpha, device = device))

    r = len(g_old)
    n1 = len(log_a)
    n2 = len(log_b)
    
    v1_old, v2_old = torch.zeros(r,1,device=device), torch.zeros(r,1,device=device)
    u1, u2 = torch.zeros(n1,1,device=device), torch.zeros(n2,1,device=device)
    q_gi, q_gp = torch.zeros(r,1,device=device), torch.zeros(r,1,device=device)
    q_Q, q_R = torch.zeros(r,1,device=device), torch.zeros(r,1,device=device)

    err = 1
    n_iter = 0
    while n_iter < max_iter:
        u1_prev, v1_prev = u1, v1_old
        u2_prev, v2_prev = u2, v2_old
        g_prev = g_old
        if err > delta:
            n_iter = n_iter + 1
            
            # Line 3
            u1 = log_a - log_space_product(log_K1, v1_old)
            u2 = log_b - log_space_product(log_K2, v2_old)

            ## Line 4
            g = torch.maximum(alpha, g_old + q_gi)
            q_gi = g_old + q_gi - g
            g_old = g.clone()
            
            # Line 5
            v1_trans = log_space_product(log_K1.T, u1)
            v2_trans = log_space_product(log_K2.T, u2)
            g = 1/3 * (g_old + q_gp + v1_old + q_Q + v1_trans + v2_old + q_R + v2_trans)
            
            # Line 6
            v1 = g - v1_trans
            v2 = g - v2_trans
            
            # Line 7
            q_gp = g_old + q_gp - g
            q_Q = q_Q + v1_old - v1
            q_R = q_R + v2_old - v2
            v1_old = v1.clone()
            v2_old = v2.clone()
            g_old = g.clone()

            # Update the error (this is time consuming)
            if n_iter % 10 == 0:    # update error per 10 iterations
                u1_trans = log_space_product(log_K1, v1)
                err_1 = torch.sum(torch.abs(torch.exp(u1 + u1_trans).squeeze() - a))
                u2_trans = log_space_product(log_K2, v2)
                err_2 = torch.sum(torch.abs(torch.exp(u2 + u2_trans).squeeze() - b))
                err = err_1 + err_2

            if (
                torch.any(torch.isnan(u1))
                or torch.any(torch.isnan(v1))
                or torch.any(torch.isnan(u2))
                or torch.any(torch.isnan(v2))
                or torch.any(torch.isinf(u1))
                or torch.any(torch.isinf(v1))
                or torch.any(torch.isinf(u2))
                or torch.any(torch.isinf(v2))
            ):
                print("Error Dykstra: ", n_iter)
                u1, v1 = u1_prev, v1_prev
                u2, v2 = u2_prev, v2_prev
                g = g_prev
                break
        else:
            break
    
    Q = u1.reshape((-1, 1)) + log_K1 + v1.reshape((1, -1))
    R = u2.reshape((-1, 1)) + log_K2 + v2.reshape((1, -1))
    return torch.exp(Q), torch.exp(R), torch.exp(g).reshape(1,-1)


def lgw(adj1, adj2, x1, x2, rank, p1 = None, p2 = None, gamma = 0.1, alpha = None, max_iter = 1000, max_err = 1e-5, ent=0.1):
    """low rank GW in the log domain

    Args:
        adj1 (torch.tensor): adjacency matrix, shape = n1*n1
        adj2 (torch.tensor): adjacency matrix, shape = n2*n2
        x1 (torch.tensor): attribute matrix, shape = n1*d
        x2 (torch.tensor): attribute matrix, shape = n2*d
        rank (int): rank for OT factorization
        p1 (torch.tensor, optional): marginal distribution, shape = (n1,1). Defaults to None.
        p2 (torch.tensor, optional): marginal distribution, shape = (n2,1). Defaults to None.
        gamma (float, optional): step size. Defaults to 0.1
        alpha (float, optional): trade off between Wasserstein and GW distances. Defaults to None.
        max_iter (int, optional): maximum number of iteartions. Defaults to 1000.
        max_err (float, optional): error threshold to break the loop. Defaults to 1e-3.
    """
    n1, n2 = len(adj1), len(adj2)
    
    device = adj1.device
    
    if p1 is None:
        p1 = ot.utils.unif(n1, type_as=adj1)
    if p2 is None:
        p2 = ot.utils.unif(n2, type_as=adj1)

    if alpha == 1.0:
        kmeans_x1 = adj1
        kmeans_x2 = adj2
    else:
        kmeans_x1 = x1
        kmeans_x2 = x2

    ## initialize Q and R via K-means
    g = torch.ones(1, rank, device = device) / rank
    
    if rank < n1 and rank < n2:
    # if False:
        kmeans_X = KMeans(n_clusters=rank, random_state=0).fit(kmeans_x1.cpu())
        Z_X = torch.from_numpy(kmeans_X.cluster_centers_).float().to(device)
        C_trans_X = Square_Euclidean_Distance(kmeans_x1, Z_X)
        C_trans_X = (C_trans_X+eps) / (C_trans_X.max()+eps)
        Q = ot.bregman.sinkhorn_stabilized(p1, ot.utils.unif(rank, type_as=adj1), C_trans_X, reg = 1e-1)
        
        kmeans_Y = KMeans(n_clusters=rank, random_state=0).fit(kmeans_x2.cpu())
        Z_Y= torch.from_numpy(kmeans_Y.cluster_centers_).float().to(device)
        C_trans_Y = Square_Euclidean_Distance(kmeans_x2, Z_Y)
        C_trans_Y = (C_trans_Y+eps) / (C_trans_Y.max()+eps)
        R = ot.bregman.sinkhorn_stabilized(p2, ot.utils.unif(rank, type_as=adj1), C_trans_Y, reg = 1e-1)
    else:
        ## let's try to allow higher-dimension here
        Q = torch.rand(n1, rank, device = device)
        R = torch.rand(n2, rank, device = device)
        Q = ot.bregman.sinkhorn_stabilized(p1, ot.utils.unif(rank, type_as=adj1), Q, reg = 1e-1)
        R = ot.bregman.sinkhorn_stabilized(p2, ot.utils.unif(rank, type_as=adj1), R, reg = 1e-1)
    
    if alpha is not None:   # use attribute distances
        d = x1.shape[1]
        w_c1 = torch.sum(x1**2, dim=1, keepdim=True)
        w_c2 = torch.sum(x2**2, dim=1, keepdim=True)
        w_c3 = x1 @ x2.T
        

    iter = 0
    err = torch.inf

    while iter < max_iter and err > max_err:
        Q_prev = Q
        R_prev = R
        g_prev = g
        stable_g = g+ent
        
        assert 0.0 <= alpha <= 1.0, 'weight for FGW distance should be in [0.0, 1.0]!'
        if alpha != 0.0:    # need gw distance
            gw_c1 = -adj1 @ Q / stable_g
            gw_c2 = R.T @ adj2
            gw_K1 = - 4 * gw_c1 @ (gw_c2 @ R) / stable_g
            gw_K2 = - 4 * gw_c2.T @ (gw_c1.T @ Q) / stable_g
            gw_K3 = - torch.diag(Q.T @ gw_K1) / stable_g
        if alpha != 1.0: # need w distance
            w_c4 = w_c3 @ R / stable_g
            w_c5 = w_c3.T @ Q / stable_g
            w_K1 = - (w_c1 + w_c2.T @ R / stable_g - 2*w_c4)
            w_K2 = - (w_c1.T @ Q /stable_g + w_c2 - 2*w_c5)
            w_K3 = - torch.diag(Q.T @ w_K1) / stable_g

        if alpha == 0.0:    # wasserstein distance
            log_K1, log_K2, log_K3 = w_K1, w_K2, w_K3
        elif alpha == 1.0:
            log_K1, log_K2, log_K3 = gw_K1, gw_K2, gw_K3
        else:
            log_K1 = alpha * gw_K1 + (1-alpha) * w_K1
            log_K2 = alpha * gw_K2 + (1-alpha) * w_K2
            log_K3 = alpha * gw_K3 + (1-alpha) * w_K3

        Q, R, g = log_LR_Dykstra(log_K1 + torch.log(Q+eps)/gamma, log_K2 + torch.log(R+eps)/gamma, log_K3 + torch.log(stable_g)/gamma, p1, p2, alpha=1e-10)

        err_1 = ((1 / gamma) ** 2) * (KL(Q, Q_prev) + KL(Q_prev, Q))
        err_2 = ((1 / gamma) ** 2) * (KL(R, R_prev) + KL(R_prev, R))
        err_3 = ((1 / gamma) ** 2) * (KL(g, g_prev) + KL(g_prev, g))
        err = err_1 + err_2 + err_3
        iter += 1
    return Q, R, g


def compute_lgw_dist(A1, A2, Q, g, R):
    """
    Computes the Low-Rank Gromov-Wasserstein distance.

    Args:
        A1 (torch.tensor): Adjacency matrix of first graph.
        A2 (torch.tensor): Adjacency matrix of second graph.
        Q (torch.tensor): First coupling factor.
        g (torch.tensor): Weight vector/diagonal matrix.
        R (torch.tensor): Second coupling factor.
    """
    T = Q @ torch.diag(1/g.squeeze()) @ R.T
    mu1 = torch.sum(Q, dim = 1).reshape(-1,1)
    mu2 = torch.sum(R, dim = 1).reshape(-1,1)
    c1 = torch.trace(A1**2 @ mu1 @ mu1.T).item()
    c2 = torch.trace(mu2 @  mu2.T @ A2**2).item()
    c3 = torch.trace(A1 @ T @ A2.T @ T.T).item()
    return torch.sqrt(c1 + c2 - 2 * c3)


def proj_graph(Q, R, adj1, adj2, x1 = None, x2 = None, eps = 1e-3):
    """
    Projects the graphs into the latent space defined by the low-rank coupling.

    Args:
        Q (torch.tensor): First coupling factor.
        R (torch.tensor): Second coupling factor.
        adj1 (torch.tensor): Adjacency matrix of first graph.
        adj2 (torch.tensor): Adjacency matrix of second graph.
        x1 (torch.tensor, optional): Feature matrix of first graph. Defaults to None.
        x2 (torch.tensor, optional): Feature matrix of second graph. Defaults to None.
        eps (float, optional): Threshold for filtering small values. Defaults to 1e-3.
    """
    assert Q.shape[0] == len(adj1) and R.shape[0] == len(adj2), 'projection matrix does not match adjacency matrix in dimension 0'
    assert Q.shape[1] == R.shape[1], 'project matrices do not match in dimmension 1'
    r = Q.shape[1]
    
    ## filter out zero columns, only keep nonzero columns
    ind = torch.nonzero(torch.logical_and(torch.sum(Q, dim=0) > eps / r, torch.sum(R, dim=0) > eps / r)).squeeze()
    Q, R = Q.index_select(1, ind), R.index_select(1, ind)
    
    norm_Q = Q / (torch.sum(Q, dim=0, keepdim=True) + 1e-16)
    norm_R = R / (torch.sum(R, dim=0, keepdim=True) + 1e-16)
    
    mix_adj1 = norm_Q.T @ adj1 @ norm_Q
    mix_adj2 = norm_R.T @ adj2 @ norm_R
    mix_x1 = norm_Q.T @ x1
    mix_x2 = norm_R.T @ x2

    edge1 = torch.sum(adj1).item()/2
    edge2 = torch.sum(adj2).item()/2
    mix_edge1 = torch.sum(mix_adj1).item()/2
    mix_edge2 = torch.sum(mix_adj2).item()/2
    
    print(r'Nodes/Edges: Source graph {:.0f}/{:.0f} -> {:.0f}/{:.0f}, Target graph {:.0f}/{:.0f} -> {:.0f}/{:.0f}'.format(norm_Q.shape[0], edge1, norm_Q.shape[1], mix_edge1, norm_R.shape[0], edge2, norm_Q.shape[1], mix_edge2))
    return mix_adj1, mix_adj2, mix_x1, mix_x2