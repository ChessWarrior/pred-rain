from utils.imports import *
import torchvision.transforms as transforms

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img.load()
        return img

def open_grayscale(fn, squeeze=True, norm=True):
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
                im = cv2.imdecode(image, flags).astype(np.float32)
                if norm:
                    im /= 255
            else:
                im = cv2.imread(str(fn), flags).astype(np.float32)
                if norm:
                    im /= 255
                if not squeeze:
                    im = np.expand_dims(im, -1)
            if im is None: raise OSError(f'File not recognized by opencv: {fn}')
            return im
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e
            
            
# Transformations
class Resize4d(transforms.Resize):
    
    def __call__(self, img):
        #set_trace()
        return [super(Resize4d, self).__call__(o) for o in img]
    
    
class ToTensor4d(transforms.ToTensor):
    
    def __call__(self, pic):
        pic = [super(ToTensor4d, self).__call__(o) for o in pic]
        return torch.stack(pic)
        
    
def normalize4d(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.
    .. note::
        This transform acts in-place, i.e., it mutates the input tensor.
    See :class:`~torchvision.transforms.Normalize` for more details.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.
    Returns:
        Tensor: Normalized Tensor image.
    """
#    if not _is_tensor_image(tensor):
    if not torch.is_tensor(tensor):
        raise TypeError('tensor is not a torch image.')

    # This is faster than using broadcasting, don't change without benchmarking
    tensor.sub_(mean).div_(std)
    return tensor


class Normalize4d(transforms.Normalize):
    
    def __call__(self, tensor):
        return normalize4d(tensor, self.mean, self.std)

class ZeroMissing():
    """ Replace missing values with 0
    
    """
    def __call__(self, tensor):
        return tensor * (tensor != 1).float()