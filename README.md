# IEEE ICDM 2018 Global A.I. Challenge on Meteorology

> Catch Rain If You Can

## The A.I. Challenge

A.I. marvels will be given big data of heavy rain suspects in form of historical weather radar pictures (collectively referred to as SRAD2018, jointly developed by SZMB and HKO). Different suspects have different rainfall severity and tracking patterns, which have to be learned thoroughly and classified precisely during the data mining process using cutting-edge machine learning / A.I. technologies. In the end, the A.I. “shields” so built shall be smart enough to catch new heavy rain suspects given unseen clues with foresights on their future crime pattern up to three hours ahead (i.e. where? when? and how severe to rain?).

## References

- [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214)
- [Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning](https://arxiv.org/abs/1605.08104)
- [Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model](https://arxiv.org/abs/1706.03458)

## TODO

### Next Actions

- [ ] Plot images as gif animation (reference: [Berkeley's vedio prediction repo](https://alexlee-gk.github.io/video_prediction/))
- [x] Calculate tfrecords length
- [x] Create modified model versions and label them appropriately
- [x] Calculate batch_size (bs) according to the number of model parameters and variable size. (rejected)
- [x] Refactor and encapsulate with Google Fire

### 数据

- [x] 封装 playground.ipynb 中的 tfrecords 读取和数据增强代码
- [ ] 数据增强
    - [x] 把 Fastai 中的数据增强库翻译至tf（部分完成）
    - [ ] Gaussian Blur （tf.images 中缺少高斯模糊函数，自己实现）
    - [ ] Random Lighting
    - [ ] Random Crop?
- 实现提议的序列拆分方式
    - [x] ![fig1](https://github.com/ChessWarrior/pred-rain/raw/master/docs/pics/sequence.jpg)
- [x] Automate stats calculation

### 模型

- [x] Refactor PredNet into atomic RNN cells
- [x] Inherit Keras RNN module
- Integrate sequence data training technics
    - [ ] Go bidirectional
    - [ ] [QANet](https://arxiv.org/abs/1804.09541)
    - [ ] ["Google's Neural Machine Translation System"](https://arxiv.org/abs/1609.08144)
    - [ ] ["HAR Stacked residual bidir LSTMs"](https://arxiv.org/abs/1708.08989)
-  Experiments
    - Needs a calibrated dataset to verify proposals.
    - [ ] Dilation: Deeplab
    - [ ] [Multiscale encoders](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w4/Zhou_D-LinkNet_LinkNet_With_CVPR_2018_paper.pdf)
    - [x] Use strides for Conv2d to replace Maxpool (rejected)
    - [x] Use Leaky Relu
    - [ ] Compare the activation functions used in PredNet vs. GAN
    - [x] PredNet 3 layers vs. 5 layers (rejected)
    - [ ] Skip connections
        - [ ] Lower layers use the concatenated representations from upper layers
        - [ ] Skip layers: ["HAR Stacked redisual bidir LSTMs"](https://arxiv.org/abs/1708.08989)
        - [ ] Skip connections between lower layers and upper layers that can accelerate training (preferably providing spatial information to higher layers, e.g. peek holes)
        - [ ] Propagate convoluted results in supplement of errors

### Loss

> "Training objective and evaluation metric should be as close as possible."
- [ ] What is the yes/no (raining) algorithm?

### 训练

- Learning Rate Scheduler ([One Cycle Learning Rate Policy for Keras](https://github.com/titu1994/keras-one-cycle))
    - [x] Super Convergence
    - [x] Cosine Annealing with Warm Restarts
- [x] Progressive resizing

### Docs
Check grammar
