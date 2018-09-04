import os

import cv2 
import numpy as np 

from predrain import *

P = None 

class PredictMode(IntEnum):
    Skip1 = 1
    Contiguous1 = 2
    Ensumble = 3

class Submission:
    
    def predict(self, sz, nt, bs, num_gpus, gpu_start, mt_idx, pred_mode,
          load_mt_idx, load_sz, load_idx, PATH='../data', test_idx=1, comment=''):
        """ A complete prediction method from commandline.
        Arguments:
            sz: the size of model input (input images are resized to sz automatically)
            nt: number of timesteps. (possibly not used)
            bs: prediction batch size
            num_gpus: number of gpus to use
            gpu_start: the index of the first gpu (sorted by PCI_BUS_ID)
            mt_idx: ModelType index. See model factory for detail
            pred_mode: either 'contiguous' or 'skip'
            load_mt_idx: ModelType index of the weights file. See model factory for detail
            load_sz: the size of model input the weights are trained on
            load_idx: weights version index
            test_idx: index of the test dataset(s) to predict
            comment: weights file comment
        """
        global P
        if isinstance(test_idx, int):
            test_idx = [test_idx]
        if len(test_idx) > 1:
            raise NotImplementedError("not prepared for multiple datasets")
            
        PATH = Path(PATH)
            
        P = Predrain()

        mt = ModelType(mt_idx)
        P.set_config(sz, nt, bs, mt, num_gpus, gpu_start, pred_mode, allow_growth=False)
        # use manual prediction
        #P.get_data(pred_mode=pred_mode, idx=data_idx)

        P.get_model(mt, output_mode='prediciton')

        load_mt = ModelType(load_mt_idx)
        P.load(load_mt, load_sz, load_idx)

        predict_mode = PredictMode(predict_mode)
        
        #######TODO: zyc
        data_path = PATH/'SRAD2018'/(fn_idx_to_dir_names(True, test_idx)[0])
        save_path = PATH/str_version(mt, pred_mode, sz, comment)
        ### Load data
        dirs = data_path.iterdir()
        ### Do predict
        for sub_dir in dirs:
            if sub_dir.is_dir:
                pred_imgs = do_predict(sub_dir, predict_mode)

            # Save
            for i, img in enumerate(pred_imgs):
                cv2.imwrite(sub_dir + "pred_" + i*5 + ".png", img)
        
    def do_predict(self, sub_dir, predict_mode):
        global P 
        imgs = os.listdir(sub_dir)
        
        assert(len(imgs) == 31)

        input_imgs = []
        pred_imgs = []

        if predict_mode == PredictMode.Skip1:
            # Inputs: 0 5 10 15 20 25 30
            # Outputs: 5 10 15 20 25 30
            for i in range(0, len(imgs), 5):
                img = cv2.imread(imgs[i])
                img = P.tfms(img)
                input_imgs.append(img)
            
            input_imgs = np.array(input_imgs)
            pred_imgs = P.predict(input_imgs)
        else:
            raise NotImplementedError("not finished Contiguous")

        

        return pred_imgs
        

            
        
    
if __name__ == '__main__':
    fire.Fire(Submission)