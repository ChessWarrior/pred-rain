import tensorflow as tf
from utils.imports import *
from tensorflow.keras.preprocessing.image import Iterator


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
    means, stds = zip(*[calc_mean_std_subdir(o) for o in tqdm_notebook(subdirs)])
    mean, std = np.mean(stds), np.mean(means)
    return mean, std

def write_mean_std(fn, mean, std):
    with open(fn, 'w') as f:
        np.array([mean, std]).tofile(f, ',')
