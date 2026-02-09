import argparse
from utils import *
from trainer import *
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


## Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='social', choices='airport|social|citation|csbm')
parser.add_argument("--attr_offset", type=float, default=0.1)
parser.add_argument("--csbm_set", type=str, default='homophily', choices='homophily|degree|attribute')

## model parameters
parser.add_argument("--method", type=str, default='VANILLA', choices='VANILLA|GRADE|CORAL|MMD|AdaGCN')
parser.add_argument("--model_list", type=list, default=['GCN','APPNP'])
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--n_hidden', type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument('--appnp_k', type=int, default=3, help='number of iterations for APPNP')
parser.add_argument('--appnp_alpha', type=int, default=0.5, help='teleport probability for APPNP')

## training parameters
parser.add_argument('--num_repeat', type=int, default=1)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--cuda', type=int, default=3)
parser.add_argument('--disc', type=str, default='JS', help='Distance metric for GRADE')
parser.add_argument('--gp_para', type=float, default=10, help='gradient penalty weight for AdaGCN')
parser.add_argument('--da_para', type=float, default=1, help='domain adaptation weight for AdaGCN')

## augmentation parameters
parser.add_argument('--aug', type=str, default='gadget', help='augmentation type', choices='base|gadget')
parser.add_argument('--n_graphs', type=int, default=5, help='number of intermediate graphs')
parser.add_argument('--aug_alpha', type=float, default=None, help='weight for Gromov-Wasserstein part')
parser.add_argument('--clip_eps', type=float, default=5e-2, help='threshold to filter out low value edges')

parser.add_argument('--debug', type=bool, default=False, help='debug mode')
args = parser.parse_args()


def set_random_seed(seed=0):
    # seed setting
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    print("Experiment time: {}".format(datetime.now()))
    print("Augmentation type: {}".format(args.aug))
    print("Results on {} dataset".format(args.data))
    if args.data == "airport":
        domains = ["usa", "brazil", "europe"]
    elif args.data == "citation":
        domains = ["acm", "dblp"]
    elif args.data == 'social':
        domains = ["Blog1", "Blog2"]
    else:
        raise KeyError('Dataset not recognized!')

    print(r'Dataset:{}, Model:{}, Method:{}, Aug:{}'.format(args.data, args.model_list, args.method, args.aug))
    all_acc = []
    set_random_seed(seed=args.seed)
    setting_name = []
    acc = []
    std = []
    for s_domain in domains:
        for t_domain in domains:
            if s_domain == t_domain:
                continue
            setting_name.append('{}-{}'.format(s_domain, t_domain))
            data_s = get_data(args.data, s_domain)
            data_t = get_data(args.data, t_domain, d="t", offset=args.attr_offset)
            record = []
            for i in range(args.num_repeat):
                if args.aug == 'base':
                    tmp_acc = []
                    for model_name in args.model_list:
                        tmp_acc_single, _, _ = direct_train(args, data_s, data_t, model_name)
                        tmp_acc.append(tmp_acc_single)
                elif args.aug == 'gadget':
                    tmp_acc, _, log_ = Gadget_train(args, data_s, data_t)
                else:
                    raise ValueError('Invalid augmentation type')
                record.append(tmp_acc)
                
            record = np.array(record)
            acc.append([])
            std.append([])
            print(r'From {} to {}'.format(s_domain, t_domain))
            for i in range(record.shape[1]):
                print(r'{} acc: {:.1f}; std: {:.1f}'.format(args.model_list[i], np.mean(record[:,i]) * 100, np.std(record[:,i]) * 100))
                acc[-1].append(np.mean(record[:,i]) * 100)
                std[-1].append(np.std(record[:,i]) * 100)
            
            torch.cuda.empty_cache()

    ## construct result dataframe
    result = np.array([['' for _ in range(len(setting_name)+1)] for _ in range(len(args.model_list)+1)], dtype = '<U20')
    result[0,1:] = setting_name
    result[1:,0] = np.array(args.model_list)
    for i in range(len(args.model_list)):
        for j in range(len(setting_name)):
            result[i+1][j+1] = '{:.2f}/{:.2f}'.format(acc[j][i], std[j][i])
    if args.data == 'csbm':
        path = './result/{}/{}'.format(args.data, args.csbm_set)
    else:
        path = './result/{}'.format(args.data)
    if not os.path.exists(path):
        os.makedirs(path)
    if args.aug == 'base':
        path = path + '/{}-base.csv'.format(args.method)
    elif args.aug == 'gadget':
        path = path + '/{}-gadget.csv'.format(args.method)
    else:
        raise ValueError('Invalid augmentation type')
    np.savetxt(path, result, delimiter=',', fmt='%s')
