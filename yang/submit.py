from predrain import *


class PredictMode(IntEnum):
    Skip1 = 1
    Contiguous1 = 2
    Ensumble = 3

class Submission:
    self.P = None
    
    
    def predict(sz, nt, bs, num_gpus, gpu_start, mt_idx, pred_mode, data_idx, epochs, max_lr, predict_mode
          load_mt_idx, load_sz, load_idx, PATH='../data', test_idx=1, comment=''):

        if isinstance(test_idx, int):
            test_idx = [test_idx]
        if len(test_idx) > 1:
            raise NotImplementedError("not prepared for multiple datasets")
            
        PATH = Path(PATH)
            
        self.P = Predrain()

        mt = ModelType(mt_idx)
        self.P.set_config(sz, nt, bs, mt, num_gpus, gpu_start, pred_mode, allow_growth=False)
        # use manual prediction
        #P.get_data(pred_mode=pred_mode, idx=data_idx)

        self.P.get_model(mt, output_mode='error')

        load_mt = ModelType(load_mt_idx)
        self.P.load(load_mt, load_sz, load_idx)

        predict_mode = PredictMode(predict_mode)
        
        #######TODO: zyc
        data_path = PATH/'SRAD2018'/(fn_idx_to_dir_names(True, data_idx)[0])
        save_path = PATH/str_version(mt, pred_mode, sz, comment)
        ### Load data
        dirs = data_path.iterdir()
        ### Do predict
        for sub_dir in dirs:
            if sub_dir.is_dir:
                y = do_predict(sub_dir, predict_mode)
        ### Write to disk
        
        
    def do_predict(self, predict_mode):
        pass
    
    
if __name__ == '__main__':
    fire.Fire(Submission)