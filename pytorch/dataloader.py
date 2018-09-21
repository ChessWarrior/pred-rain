from utils.imports import *
from preprocessing import *

# dataset specific imports
import torch.utils.data as data
from fastai.transforms import RandomLighting
from torch.utils.data.distributed import DistributedSampler
from fastai.dataset import ModelData
from functools import reduce

def sequence_index(pred_mode, k, max_len=61, skip_fac=5):
    """ Generate the sequence for truncated BPTT
    
    Args:
        k: the number of steps to unfold the network
    """
    if pred_mode == PredMode.Skip1:
        idx = [range(o, max_len, skip_fac) for o in range(skip_fac)]
        idx = [o[:k] for o in idx if len(o) >= k] # truncate
    
    elif pred_mode == PredMode.Contiguous1:
        num_sequences = max_len // k
        idx = [range(i * k, (i + 1) * k) for i in range(num_sequences)]
        
    else:
        raise NotImplementedError
    return [list(o) for o in idx]


class SRAD2018(data.Dataset):
    """ A SRAD 2018 data loader for a single dataset where the samples are arranged in this way:
    
        root (SRAD2018_TRAIN_003 or test)/RAD_xxxxxx/RAD_xxxxxx123.png
        
    Args:
        root (string): Root directory path.
        is_trn: whether is training folder
        pred_mode
        k: the number of steps to unfold the network
        transform (callable, optional)
    """
    
    def __init__(self, root, is_trn, pred_mode, k=12, transform=None,
                trn_val_split=0.8):
        super().__init__()
        dirs = [o for o in Path(root).iterdir() if o.is_dir()]
        dirs.sort()
        num_trn_dirs = int(len(dirs) * trn_val_split)
        self.transform = transform
        self.dirs = dirs[:num_trn_dirs] if is_trn else dirs[num_trn_dirs:]
        
        self.is_trn = is_trn
        self.pred_mode = pred_mode
        self.k = k
        
        max_len = 61 if is_trn else 31
        self.sequence_index = sequence_index(self.pred_mode, self.k, max_len)

    def __getitem__(self, index):
        """
        Precondition: each subfolder has the same number of images
        Args:
            index (int)
            
        Returns:
            list: of size [nt] + im_shape containing the PIL images in the indexed folder
        """
        dir_idx, seq_idx = divmod(index, len(self.sequence_index))
        subdir = self.dirs[dir_idx]
        
        fns = np.array(sorted(subdir.iterdir()))
        fns = fns[self.sequence_index[seq_idx]]
        
        #samples = [open_grayscale(o, True, False) for o in fns]
        samples = [pil_loader(o) for o in fns]
        if self.transform is not None:
            samples = self.transform(samples)
            
        errors_weights = torch.zeros(self.k)
        errors_weights[1:] = 1 / (self.k - 1)
        return samples, errors_weights  
    
    def __len__(self):
        return len(self.dirs) * len(self.sequence_index)
    
class TorchModelData(ModelData):
    def __init__(self, path, sz, trn_dl, val_dl, aug_dl=None):
        super().__init__(path, trn_dl, val_dl)
        self.aug_dl = aug_dl
        self.sz = sz
        
# Seems to speed up training by ~2%
class DataPrefetcher():
    def __init__(self, loader, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(async=True)
            self.next_target = self.next_target.cuda(async=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self.preload()
            count += 1
            yield input, target
            if type(self.stop_after) is int and (count > self.stop_after):
                break
                
def torch_loader(data_path, dataset_idx, pred_mode, size, trn_val_split=0.8, debug=False, zero_missing=False):
    # TODO: average more stats
    # TODO: return test set
    data_path = Path(data_path)
    pred_mode = PredMode(pred_mode)
    
    if debug:
        class args:
            pass
        args.bs = 5
        args.sz = 256
        args.workers = 1
        args.world_size = 1
        args.distributed = args.world_size > 1
        args.trn_val_split = 0.8
        args.pred_mode = 1
        pred_mode = PredMode(args.pred_mode)
        args.k = 12
        args.prof = True
        args.dist_backend = 'nccl'
        args.dist_url = 'file://sync.file'
        
    # re-calculate stats with zero missing
    mean, std = 0.22, 0.90
    resize = Resize4d(args.sz)
    normalize = Normalize4d(mean, std)
    totensor = ToTensor4d()
    
    tfms = [resize, totensor]
    if zero_missing:
        tfms.append(ZeroMissing())
    tfms.append(normalize)

    fn_trn_datasets = [data_path/o for o in fn_idx_to_dir_names(dataset_idx, True)]
    #     fn_val_datasets = fn_idx_to_dir_names(dataset_idx, False)

    # TODO: check if the transforms applys to data of four dimension data
    #trn_tfms = transforms.Compose([random_lighting] + tfms)
    trn_tfms = transforms.Compose(tfms)
    # TODO: use lighting
    # append random lighting here
    trn_dataset = reduce(sum, [SRAD2018(o, True, pred_mode, args.k, trn_tfms, args.trn_val_split)
                   for o in fn_trn_datasets])
    trn_sampler = (DistributedSampler(trn_dataset, args.world_size)
                    if args.distributed else None)
    trn_loader = data.DataLoader(
        trn_dataset, batch_size=args.bs, shuffle=(trn_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=trn_sampler)

    val_tfms = transforms.Compose(tfms)
    val_dataset = reduce(sum, [SRAD2018(o, False, pred_mode, args.k, val_tfms, args.trn_val_split)
                   for o in fn_trn_datasets])
    val_loader = data.DataLoader(
        val_dataset, args.bs, False, num_workers=args.workers, pin_memory=True)

    trn_loader = DataPrefetcher(trn_loader)
    val_loader = DataPrefetcher(val_loader)

    if args.prof:
        trn_loader.stop_after = 200
        val_loader.stop_after = 0

    md = TorchModelData(data_path, args.sz, trn_loader, val_loader)
    return md, trn_sampler
   