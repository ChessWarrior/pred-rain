from predrain import *

P = None
#sz, nt, bs, num_gpus, gpu_start, mt_idx, pred_mode, data_idx, load_mt_idx,  load_sz,
#epochs, max_lr = None, None, None, None, None, None, None, None, None, None, None, None

def start(sz, nt, bs, num_gpus, gpu_start, mt_idx, pred_mode, data_idx, epochs, max_lr,
          load_mt_idx=None, load_sz=None, load_idx=None, train=True, comment=''):
    global P
    #global sz, nt, bs, num_gpus, gpu_start, mt_idx, pred_mode, data_idx, load_mt_idx,\
    #load_sz, epochs, max_lr 
        
    P = Predrain()
    
    mt = ModelType(mt_idx)
    P.set_config(sz, nt, bs, mt, num_gpus, gpu_start, pred_mode, comment=comment,
                 allow_growth=False)
    P.get_data(pred_mode=pred_mode, idx=data_idx)
    
    P.get_model(mt, output_mode='error')
    
    if load_mt_idx:
        load_mt = ModelType(load_mt_idx)
        P.load(load_mt, load_sz, load_idx)
    
    if train:
        P.train(epochs, max_lr)
    
    
if __name__ == '__main__':
    fire.Fire(start)
    