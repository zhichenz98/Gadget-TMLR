import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function
import numpy as np
from gnn_layers import *



class BaseGNN(nn.Module):
    def __init__(self, in_feats, n_classes, model_name, args, activation=F.leaky_relu):
        """
        Initializes the BaseGNN model.
        
        Args:
            in_feats (int): Number of input features.
            n_classes (int): Number of output classes.
            model_name (str): Name of the GNN model to use ('GCN' or 'APPNP').
            args: Arguments containing model hyperparameters.
            activation: Activation function to use (default: F.leaky_relu).
        """
        super(BaseGNN, self).__init__()
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if model_name == 'GCN':
            self.layers.append(GCNLayer(in_feats, args.hidden_dim, activation=activation, dropout=args.dropout))
            for i in range(args.n_hidden):
                self.layers.append(GCNLayer(args.hidden_dim, args.hidden_dim, activation=activation, dropout=args.dropout))
        elif model_name == 'APPNP':
            self.layers.append(APPNPLayer(in_feats, args.hidden_dim, args.appnp_k, args.appnp_alpha, activation=activation, dropout=args.dropout))
            for i in range(args.n_hidden):
                self.layers.append(APPNPLayer(args.hidden_dim, args.hidden_dim, args.appnp_k, args.appnp_alpha, activation=activation, dropout=args.dropout))
        else:
            raise ValueError('Model name not recognized!')
        self.fc = nn.Linear(args.hidden_dim, self.n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for ele in self.layers:
            ele.reset_parameters()
        torch.nn.init.kaiming_uniform_(self.fc.weight)
    
    def forward(self, g_s, features_s, labels_s, weight=None, ent=None):
        for i, layer in enumerate(self.layers):
            features_s = layer(g_s, features_s)
        preds_s = self.fc(features_s)
        preds_s = torch.log_softmax(preds_s, dim=-1)
        if weight is not None:
            assert weight.shape[0] == preds_s.shape[0], 'weight shape does not match input size'
            loss = F.nll_loss(preds_s, labels_s, reduction='none')
            loss = (weight * loss).sum()
        else:
            loss = F.nll_loss(preds_s, labels_s)
        if ent is not None: # avoid over-confidence
            loss = loss - ent * self.entropy_loss(preds_s)
        return loss

    def inference(self, g_t, features_t):
        for i, layer in enumerate(self.layers):
            features_t = layer(g_t, features_t)
        preds_t = self.fc(features_t)
        return torch.log_softmax(preds_t, dim=-1), features_t
    
    def entropy_loss(self, prob):
        # probabilities: tensor of shape [batch_size, num_classes]
        # Compute entropy for each sample and then take the mean over the batch
        ent = -torch.sum(torch.exp(prob) * prob, dim=1)
        return ent.mean()
    
    def mlp_classifier(self, features_t):
        preds_t = self.fc(features_t)
        return preds_t



'''
    GRADE model
'''
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Computes Gaussian kernel matrix.

    Args:
        source (torch.tensor): Source features.
        target (torch.tensor): Target features.
        kernel_mul (float, optional): Multiplier for bandwidth. Defaults to 2.0.
        kernel_num (int, optional): Number of kernels. Defaults to 5.
        fix_sigma (float, optional): Fixed sigma value. Defaults to None.

    Returns:
        torch.tensor: Kernel matrix.
    """
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    # L2_distance = torch.norm(total0-total1, dim = 2) ** 2
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Computes MMD loss using RBF kernel.

    Args:
        source (torch.tensor): Source features.
        target (torch.tensor): Target features.
        kernel_mul (float, optional): Multiplier for bandwidth. Defaults to 2.0.
        kernel_num (int, optional): Number of kernels. Defaults to 5.
        fix_sigma (float, optional): Fixed sigma value. Defaults to None.

    Returns:
        torch.tensor: MMD loss.
    """
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GRADE(nn.Module):
    def __init__(self, g_s, g_t, in_feats, n_classes, model_name, args, activation=F.leaky_relu):
        """
        Initializes the GRADE model.

        Args:
            g_s (torch.tensor): Source graph.
            g_t (torch.tensor): Target graph.
            in_feats (int): Number of input features.
            n_classes (int): Number of output classes.
            model_name (str): Name of the GNN model to use ('GCN' or 'APPNP').
            args: Arguments containing model hyperparameters.
            activation: Activation function to use (default: F.leaky_relu).
        """
        super(GRADE, self).__init__()
        self.disc = args.disc
        self.g_s = g_s
        self.g_t = g_t
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if model_name == 'GCN':
            self.layers.append(GCNLayer(in_feats, args.hidden_dim, activation=activation, dropout=args.dropout))
            for i in range(args.n_hidden):
                self.layers.append(GCNLayer(args.hidden_dim, args.hidden_dim, activation=activation, dropout=args.dropout))
        elif model_name == 'APPNP':
            self.layers.append(APPNPLayer(in_feats, args.hidden_dim, args.appnp_k, args.appnp_alpha, activation=activation, dropout=args.dropout))
            for i in range(args.n_hidden):
                self.layers.append(APPNPLayer(args.hidden_dim, args.hidden_dim, args.appnp_k, args.appnp_alpha, activation=activation, dropout=args.dropout))
        else:
            raise ValueError('Model name not recognized!')
        self.fc = nn.Linear(args.hidden_dim, self.n_classes)

        if self.disc == "JS":
            self.discriminator = nn.Sequential(
                nn.Linear(args.hidden_dim*(args.n_hidden+1) + self.n_classes, 2)
            )
        else:
            self.discriminator = nn.Sequential(
                nn.Linear(args.hidden_dim*(args.n_hidden+1) + self.n_classes * 2, 2)
            )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features_s, labels_s, features_t, alpha=1.0, weight=None):
        s_f = []
        t_f = []
        for i, layer in enumerate(self.layers):
            features_s = layer(self.g_s, features_s)
            features_t = layer(self.g_t, features_t)
            s_f.append(features_s)
            t_f.append(features_t)
        features_s = self.fc(features_s)
        features_t = self.fc(features_t)
        s_f.append(features_s)
        t_f.append(features_t)
        preds_s = torch.log_softmax(features_s, dim=-1)
        
        if weight is not None:
            assert weight.shape[0] == preds_s.shape[0], 'weight shape does not match input size'
            class_loss = F.nll_loss(preds_s, labels_s, reduction='none')
            class_loss = (weight * class_loss).sum()
        else:
            class_loss = F.nll_loss(preds_s, labels_s)

        s_f = torch.cat(s_f, dim=1)
        t_f = torch.cat(t_f, dim=1)
        domain_loss = 0.
        if self.disc == "JS":
            domain_preds = self.discriminator(ReverseLayerF.apply(torch.cat([s_f, t_f], dim=0), alpha))
            domain_labels = np.array([0] * features_s.shape[0] + [1] * features_t.shape[0])
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=features_s.device)
            domain_loss = self.criterion(domain_preds, domain_labels)
        elif self.disc == "MMD":
            mind = min(s_f.shape[0], t_f.shape[0])
            domain_loss = mmd_rbf_noaccelerate(s_f[:mind], t_f[:mind])
        elif self.disc == "C":
            ratio = 8
            s_l_f = torch.cat([s_f, ratio * self.one_hot_embedding(labels_s)], dim=1)
            t_l_f = torch.cat([t_f, ratio * F.softmax(features_t, dim=1)], dim=1)
            domain_preds = self.discriminator(ReverseLayerF.apply(torch.cat([s_l_f, t_l_f], dim=0), alpha))
            domain_labels = np.array([0] * features_s.shape[0] + [1] * features_t.shape[0])
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=features_s.device)
            domain_loss = self.criterion(domain_preds, domain_labels)
        loss = class_loss + domain_loss * 0.02
        return loss

    def inference(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(self.g_t, x)
        pred = self.fc(x)
        return torch.log_softmax(pred, dim=-1), x

    def one_hot_embedding(self, labels):
        y = torch.eye(self.n_classes, device=labels.device)
        return y[labels]
    


'''
    CORAL/DAN model
'''
class BaseDA(nn.Module):
    def __init__(self, g_s, g_t, in_feats, n_classes, model_name, args, activation=F.leaky_relu):
        """
        Initializes the BaseDA model.
        
        Args:
            g_s (torch.tensor): Source graph.
            g_t (torch.tensor): Target graph.
            in_feats (int): Number of input features.
            n_classes (int): Number of output classes.
            model_name (str): Name of the GNN model to use ('GCN' or 'APPNP').
            args: Arguments containing model hyperparameters.
            activation: Activation function to use (default: F.leaky_relu).
        """
        super(BaseDA, self).__init__()
        self.g_s = g_s
        self.g_t = g_t
        self.n_classes = n_classes
        self.method = args.method
        self.layers = nn.ModuleList()
        if model_name == 'GCN':
            self.layers.append(GCNLayer(in_feats, args.hidden_dim, activation=activation, dropout=args.dropout))
            for i in range(args.n_hidden):
                self.layers.append(GCNLayer(args.hidden_dim, args.hidden_dim, activation=activation, dropout=args.dropout))
        elif model_name == 'APPNP':
            self.layers.append(APPNPLayer(in_feats, args.hidden_dim, args.appnp_k, args.appnp_alpha, activation=activation, dropout=args.dropout))
            for i in range(args.n_hidden):
                self.layers.append(APPNPLayer(args.hidden_dim, args.hidden_dim, args.appnp_k, args.appnp_alpha, activation=activation, dropout=args.dropout))
        else:
            raise ValueError('Model name not recognized!')
        self.fc = nn.Linear(args.hidden_dim, self.n_classes)

    
    def compute_covariance(self, input_data):
        """
        Compute Covariance matrix of the input data

        Args:
            input_data (torch.tensor): Input features.

        Returns:
            torch.tensor: Covariance matrix.
        """
        n = input_data.size(0)  # batch_size
        id_row = torch.ones(n).resize(1, n).to(device=input_data.device)
        sum_column = torch.mm(id_row, input_data)
        mean_column = torch.div(sum_column, n)
        term_mul_2 = torch.mm(mean_column.t(), mean_column)
        d_t_d = torch.mm(input_data.t(), input_data)
        c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)
        return c
    
    def forward(self, features_s, labels_s, features_t, weight=None):
        for i, layer in enumerate(self.layers):
            features_s = layer(self.g_s, features_s)
            features_t = layer(self.g_t, features_t)
        output_s = self.fc(features_s)
        output_t = self.fc(features_t)
        preds_s = torch.log_softmax(output_s, dim=-1)
        if weight is not None:
            assert weight.shape[0] == preds_s.shape[0], 'weight shape does not match input size'
            class_loss = F.nll_loss(preds_s, labels_s, reduction='none')
            class_loss = (weight * class_loss).sum()
        else:
            class_loss = F.nll_loss(preds_s, labels_s)

        if self.method == 'CORAL':
            d = output_s.shape[1]  # dim vector
            source_c = self.compute_covariance(output_s)
            target_c = self.compute_covariance(output_t)
            domain_loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))
            domain_loss = domain_loss / (4 * d * d)
        elif self.method == 'MMD':
            mind = min(features_s.shape[0], features_t.shape[0])
            domain_loss = mmd_rbf_noaccelerate(features_s[:mind], features_t[:mind])
        loss = class_loss + domain_loss * 0.5
        return loss

    def inference(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(self.g_t, x)
        pred = self.fc(x)
        return torch.log_softmax(pred, dim=-1), x



'''
    AdaGCN model
'''
class AdaGCN(nn.Module):
    def __init__(self, g_s, g_t, in_feats, n_classes, model_name, args, activation=F.leaky_relu):
        """
        Initializes the AdaGCN model.
        
        Args:
            g_s (torch.tensor): Source graph.
            g_t (torch.tensor): Target graph.
            in_feats (int): Number of input features.
            n_classes (int): Number of output classes.
            model_name (str): Name of the GNN model to use ('GCN' or 'APPNP').
            args: Arguments containing model hyperparameters.
            activation: Activation function to use (default: F.leaky_relu).
        """
        super(AdaGCN, self).__init__()
        self.g_s = g_s
        self.g_t = g_t
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if model_name == 'GCN':
            self.layers.append(GCNLayer(in_feats, args.hidden_dim, activation=activation, dropout=args.dropout))
            for i in range(args.n_hidden):
                self.layers.append(GCNLayer(args.hidden_dim, args.hidden_dim, activation=activation, dropout=args.dropout))
        elif model_name == 'APPNP':
            self.layers.append(APPNPLayer(in_feats, args.hidden_dim, args.appnp_k, args.appnp_alpha, activation=activation, dropout=args.dropout))
            for i in range(args.n_hidden):
                self.layers.append(APPNPLayer(args.hidden_dim, args.hidden_dim, args.appnp_k, args.appnp_alpha, activation=activation, dropout=args.dropout))
        else:
            raise ValueError('Model name not recognized!')
        self.fc = nn.Linear(args.hidden_dim, self.n_classes)
        self.discriminator = nn.Linear(self.n_classes, 1)
        self.gp_para = args.gp_para
        self.da_para = args.da_para


    def forward(self, features_s, labels_s, features_t, weight=None):
        for i, layer in enumerate(self.layers):
            features_s = layer(self.g_s, features_s)
            features_t = layer(self.g_t, features_t)
        features_s = self.fc(features_s)
        features_t = self.fc(features_t)
        preds_s = torch.log_softmax(features_s, dim=-1)
        preds_t = torch.log_softmax(features_t, dim=-1)
        features_cat = torch.cat((features_s, features_t), dim=0)
        domain = self.discriminator(features_cat)
        grad_outputs = torch.ones(domain.size(), device=features_s.device, requires_grad=False)
        gradients = torch.autograd.grad(
            outputs=domain,
            inputs=features_cat,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        grad_loss = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        wd_loss = torch.mean(domain[:features_s.shape[0]]) - torch.mean(domain[features_s.shape[0]:])
        gp_loss = -wd_loss + self.gp_para * grad_loss
        if weight is not None:
            assert weight.shape[0] == preds_s.shape[0], 'weight shape does not match input size'
            da_loss = F.nll_loss(preds_s, labels_s, reduction='none')
            da_loss = (weight * da_loss).sum()
        else:
            da_loss = F.nll_loss(preds_s, labels_s)
        
        da_loss += self.da_para * wd_loss
        return gp_loss, da_loss

    def inference(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(self.g_t, x)
        pred = self.fc(x)
        return torch.log_softmax(pred, dim=-1), x
