from .imports import *

class PredMode(IntEnum):
    Skip1 = 1
    Contiguous1 = 2
    Ensumble = 3
    

def fn_idx_to_dir_names(idx, is_trn):
    """ 
    Returns:
        a list of names of data folders e.g. ['SRAD2018_TRAIN_001']
    """
    is_test = not is_trn
    
    train_mode = 'TEST' if is_test else 'TRAIN'
    if isinstance(idx, int):
        idx = [idx]
    
    # a hack
    # I want to complain about the naming
    if is_test:
        fns = ['SRAD2018_Test_' + str(o) for o in idx]
    else:
        fns = ['SRAD2018_' + train_mode + '_' + str(o).zfill(3) for o in idx]
    
    return fns
    
def fn_idx_to_stats(PATH, is_test, idx):
    """
    Arguments:
        PATH: working path
        is_test
        idx  
    """
    dir_stats = Path(PATH)/'stats'
    dir_names = fn_idx_to_dir_names(is_test, idx)
    fns = [str(dir_stats/(o + '.npy')) for o in dir_names]
    return fns
    
def fn_to_record(base_path, pred_mode, is_test, idx, nt):
    """
    Arguments:
        base_path: the folder that contains the tfrecord files
        pred_mode: either contiguous or skip
        is_test
        idx: 
        nt: number of time steps
    """
    assert pred_mode in ['contiguous', 'skip'], 'pred_mode must be either contiguous or skip'
    if isinstance(idx, int):
        idx = [idx]
    
    train_mode = 'test' if is_test else 'train'
    names = [train_mode + '_' + str(o) + '_' + pred_mode + '_' + str(nt) + '.tfrecord' for o in idx]
    fns = [str(Path(base_path)/o) for o in names]
    return fns