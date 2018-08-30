import tensorflow as tf
from utils.imports import *


def open_greyscale(fn):
    """ Opens an image using OpenCV given the file path.

    Arguments:
        fn: the file path of the image

    Returns:
        The image in greyscale format as numpy array of floats normalized to range between 0.0 - 1.0
    """
    flags = cv2.IMREAD_GRAYSCALE
    if not os.path.exists(fn) and not str(fn).startswith("http"):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn) and not str(fn).startswith("http"):
        raise OSError('Is a directory: {}'.format(fn))
    else:
        #res = np.array(Image.open(fn), dtype=np.float32)/255
        #if len(res.shape)==2: res = np.repeat(res[...,None],3,2)
        #return res
        try:
            if str(fn).startswith("http"):
                req = urllib.urlopen(str(fn))
                image = np.asarray(bytearray(req.read()), dtype="uint8")
                im = cv2.imdecode(image, flags).astype(np.float32)/255
            else:
                im = cv2.imread(str(fn), flags).astype(np.float32)/255
            if im is None: raise OSError(f'File not recognized by opencv: {fn}')
            return im
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e


def calc_mean_std_subdir(subdir):
    """ Returns the mean and std of png grayscale images contained in subdir.
    Caution: All images are read into memery at once so be aware that big folders might cause OOM.

    Argument:
        subdir: the path to a directory that contains grayscale png files
            e.g. data/SRAD2018_TRAIN_001/RAD_206482464212530/
    """
    ims = np.asarray([open_greyscale(o) for o in Path(subdir).glob('*.png')])
    mean, std = np.mean(ims), np.std(ims)
    return mean, std


def calc_mean_std_par_dir(par_dir):
    """ Return the mean and std of png grayscale images in each subdirs.
    
    Argument:
        par_dir: The parent folder whose subfolders each contain equal number of grayscale png images
            e.g. data/SRAD2018_TRAIN_001
    
    Returns:
        The mean and std of all png images in subdirectories of the parent directory.
    """

    subdirs = [o for o in Path(par_dir).iterdir() if o.is_dir()]
    means, stds = zip(*[calc_mean_std_subdir(o) for o in tqdm(subdirs)])
    mean, std = np.mean(stds), np.mean(means)
    return mean, std


def write_mean_std_count(fn, mean, std, count):
    with open(fn, 'w') as f:
        np.array([mean, std, count]).tofile(f)
        

def avg_stats(stats):
    """ Return the average of stats 
    
    Arguments:
        stats: a list or np array of stats
        
    Return:
        The average of stats
    """
    raise NotImplementedError
    

def fn_idx_to_dir_names(is_test, idx):
    """ 
    Returns:
        a list of names of data folders e.g. ['SRAD2018_TRAIN_001']
    """
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
    dir_stats = PATH/'stats'
    dir_names = fn_idx_to_dir_names(is_test, idx)
    fns = [dir_stats/(o + '.npy') for o in dir_names]
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