from predrain import *

P, steps, batch_gen = None, None, None

def predict(sz, nt, bs, num_gpus, gpu_start, mt_idx, pred_mode,
      load_mt_idx, load_sz, load_idx, PATH='../data', test_idx=1, comment='', start=False):
    """ A complete prediction method from commandline.
    Arguments:
        sz: the size of model input (input images are resized to sz automatically)
        nt: number of timesteps. (possibly not used)
        bs: prediction batch size
        num_gpus: number of gpus to use
        gpu_start: the index of the first gpu (sorted by PCI_BUS_ID)
        mt_idx: ModelType index. See model factory for detail
        pred_mode: a enum value (PredMode). See utils/misc.py
        load_mt_idx: ModelType index of the weights file. See model factory for detail
        load_sz: the size of model input the weights are trained on
        load_idx: weights version index
        test_idx: index of the test dataset(s) to predict
        comment: weights file comment
    """
    global P, steps, batch_gen
    if isinstance(test_idx, int):
        test_idx = [test_idx]
    if len(test_idx) > 1:
        raise NotImplementedError("not prepared for multiple datasets")

    PATH = Path(PATH)
    pred_mode = PredMode(pred_mode)

    P = Predrain()

    mt = ModelType(mt_idx)
    P.set_config(sz, nt, bs, mt, num_gpus, gpu_start, pred_mode, allow_growth=False)
    # use manual prediction
    P.get_tfms()
    P.get_model(mt, output_mode='prediction', stateful=True)
    P.model.load_weights('../data/models/checkpoints/weights.PredNetLeakyRelu_256_skip_nt5_from_scratch.02.h5', by_name=True)

    #######TODO: zyc
    data_path = PATH/'SRAD2018'/(fn_idx_to_dir_names(True, test_idx)[0])
    assert data_path.exists()

    save_path = PATH/'submission'/str_version(mt, pred_mode.name, sz, comment)
    save_path.mkdir(parents=True, exist_ok=True)
    
    ### Load data
    dirs = [o for o in data_path.iterdir() if o.is_dir()] # len = 10000
    
    test_size = 10000
    assert(len(dirs) == test_size)
    steps = int(test_size // bs)
    assert(test_size == steps * bs)
    batch_gen = get_batch(dirs, bs, pred_mode, steps, P.tfms)
    
    ### Do predict
    # TODO: rewrite using keras generator
    if start:
        def norm_ims(ims):
            ret = np.empty_like(ims)
            for i in range(ims.shape[0]):
                im = ims[i]
                norm = colors.Normalize(im.min(), im.max())
                ret[i] = norm(im)
            return ret
        for x, names in tqdm_notebook(batch_gen, total=steps):
            y = do_predict(x, pred_mode, 6)
            # TODO
            #y = norm_ims(y) * 255
            y = P.denorm(y) * 255
            save_batch(y, names, save_path)
            
def save_batch(y, batch_names, save_path):
    assert(len(y) == len(batch_names))
    for ims, name in zip(y, batch_names):
        for i, im in enumerate(ims):
            fn = save_path/f'{name}_f{str(i + 1).zfill(3)}.png'
            cv2.imwrite(str(fn), im)

def do_predict(x, pred_mode, future_nt):
    """
    Arguments: 
        x: [b, t, h, w, 1]
        pred_mode: PredMode
        future_nt: number of future elements to predict
    """
    global P 
    def resize_batch_preds(preds):
        preds_resized = [cv2.resize(o[0], (501, 501)) for o in preds]
        return np.expand_dims(preds_resized, 1)

    if pred_mode == PredMode.Skip1:
        bs, x_nt, h, w, _ = x.shape
        x = np.moveaxis(x, 0, 1)  # [t, b, h, w, 1]
        
        # setup state
        for t in x:  # for each timestep
            preds = P.model.predict(t[:, None])  # expand time dimension to 1
        
        # actual predictions
        batch_preds = [resize_batch_preds(preds)]  # initialize results list
        for _ in range(future_nt - 1):   # predict future_nt frames into future
            preds = P.model.predict(preds)  # preds comes from the last timestep
            batch_preds.append(resize_batch_preds(preds))
        
        P.model.reset_states() 
        # restore to  [b, t, h, w]
        batch_preds = np.moveaxis(batch_preds, 0, 1).squeeze()
    else:
        raise NotImplementedError("not finished Contiguous")

    return batch_preds

def get_batch(dirs, bs, pred_mode, steps, tfms):
    """ A generator that generates a batch of images per iteration from a list of directories
    Arguemnts:
        dirs: a list of Path objects of test subdirs of length 10000
        bs
        pred_mode
        steps: len(dirs) // bs
    """
    dirs = np.asarray(dirs)
    for step in range(steps):
        batch_dirs = dirs[range(step * bs, (step + 1) * bs)]
        ims = []
        for subdir in batch_dirs:
            fn_ims = sorted(subdir.glob('*.png'))
            if pred_mode == PredMode.Skip1:
                ims.append([tfms(open_greyscale(fn_ims[o], squeeze=False)) 
                            for o in range(0, 31, 5)])
            else:
                raise NotImplementedError
                
        yield np.expand_dims(ims, -1), [o.name for o in batch_dirs]
        

if __name__ == '__main__':
    fire.Fire(predict)