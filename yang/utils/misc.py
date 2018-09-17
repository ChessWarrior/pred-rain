import tensorflow as tf
from utils.imports import *

class PredMode(IntEnum):
    Skip1 = 1
    Contiguous1 = 2
    Ensumble = 3

def open_greyscale(fn, squeeze=True):
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
                if not squeeze:
                    im = np.expand_dims(im, -1)
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
    
    with ProcessPoolExecutor() as e:
        stats = e.map(calc_mean_std_subdir, subdirs)
    means, stds = zip(*tqdm(stats))
    
    #means, stds = zip(*[calc_mean_std_subdir(o) for o in tqdm_notebook(subdirs)])
    mean, std = np.mean(stds), np.mean(means)
    return mean, std


def write_mean_std_count(fn, mean, std, count, sep=','):
    with open(fn, 'w') as f:
        np.array([mean, std, count]).tofile(f, sep=sep)
        

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

def get_stats(PATH, idx, is_test, sep=','):
    fn_stats = fn_idx_to_stats(PATH, is_test, idx)
    stats = [np.fromfile(o, sep=sep) for o in fn_stats]
    return stats

# hard coding stats
def get_num_samples(num_records, pred_mode, is_test):
    if pred_mode == 'skip':
        num_samples = 5000 * 5
    elif pred_mode == 'contiguous':
        if is_test:
            num_samples = 5000 * 3
        else:
            num_samples = 5000 * 6
    else:
        raise ValueError
        
    return num_samples * num_records
            
    
def str_version(mt, pred_mode, sz, comment=''):
    if comment:
        comment = '_' + comment
    MODEL_VERSION = mt.name + '_' + str(sz) + '_' + pred_mode + comment
    return MODEL_VERSION
