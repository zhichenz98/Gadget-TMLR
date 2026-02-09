import numpy as np

def save_benchmark():
    backbone = ['VANILLA', 'CORAL', 'AdaGCN', 'GRADE']
    gadget = ['base', 'gadget']
    model = ['GCN', 'APPNP']
    model_flag = {'GCN': True, 'APPNP': True}
    data = ['airport', 'citation']
    path = 'result'
    
    result = {}
    for ele_d in data:
        result[ele_d] = {}
        for ele_bb in backbone:
            for ele_g in gadget:
                me = ele_bb + '-' + ele_g
                result[ele_d][me] = {}
                tmp = np.loadtxt(path + '/{}/{}.csv'.format(ele_d, me), delimiter=',', dtype='object')
                for i in range(len(model)):
                    tmp[i+1][0] = me
                    result[ele_d][me][model[i]] = tmp[i+1].reshape(1,-1)
                
                
    for ele_d in data:
        tmp_res = np.loadtxt(path + '/{}/VANILLA-base.csv'.format(ele_d), delimiter=',', dtype='object')[0].reshape(1,-1)    # settings
        for ele_mo in model:
            if not model_flag[ele_mo]:
                continue
            for ele_bb in backbone:
                for ele_g in gadget:
                    me = ele_bb + '-' + ele_g
                    tmp_res = np.concatenate((tmp_res, result[ele_d][me][ele_mo]), axis=0)
        np.savetxt(path + '/{}/result.csv'.format(ele_d), tmp_res, delimiter=',', fmt='%s')
        
    
if __name__ == "__main__":
    save_benchmark()