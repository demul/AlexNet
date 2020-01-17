import data_preprocessor
import train
import os
import gif_maker

###########################################################
# differences with reference paper(ImageNet Classification with Deep Convolutional Neural Networks) :
# our dataset is 'cat vs dog'
# our initial learning rate(hyper parameter) is 1/10 of reference paper's
###########################################################

if not(os.path.exists('/kaggle/input/data_preprocessed')):
    DP = data_preprocessor.DataPreprocessor()
    DP.run()

input_size = 500
lr = 0.001    # 1/10 of reference paper's initial learning rate
momentum = 0.9
decaying_factor = 0.0005
LRN_depth = 5
LRN_bias = 2
LRN_alpha = 0.0001
LRN_beta = 0.75
keep_prob = 0.5

max_epoch = 100
loss_sampling_step = 20
acc_sampling_step = 1

alexnet = train.AlexNet(input_size, lr, momentum, decaying_factor, LRN_depth, LRN_bias, LRN_alpha, LRN_beta, keep_prob)
alexnet.run(max_epoch, loss_sampling_step, acc_sampling_step)
alexnet.save_acc()
alexnet.save_loss()
gif_maker.run(max_epoch)
