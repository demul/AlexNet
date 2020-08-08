import os
import numpy as np
import torch
import data_loader
import model
import time
import matplotlib.pyplot as plt

class AlexNet:
    def __init__(self, input_size, lr=0.01, momentum=0.9, decaying_factor=0.0005,
                 LRN_depth=5, LRN_bias=2, LRN_alpha=0.0001, LRN_beta=0.75, keep_prob=0.8):
        self.input_size = input_size
        self.lr = lr
        self.momentum = momentum
        self.decaying_factor = decaying_factor

        self.LRN_depth = LRN_depth
        self.LRN_bias = LRN_bias
        self.LRN_alpha = LRN_alpha
        self.LRN_beta = LRN_beta
        self.drop_rate = 1 - keep_prob

        self.loss_sampling_step = None
        self.acc_sampling_step = None

        ###for plotting
        self.metric_list = dict()
        self.metric_list['losses'] = []
        self.metric_list['train_acc'] = []
        self.metric_list['val_acc'] = []

        self.graph = None
        self.model = model.AlexNetModel(LRN_depth=self.LRN_depth, LRN_bias=self.LRN_bias,
                                        LRN_alpha=self.LRN_alpha, LRN_beta=self.LRN_beta, drop_rate=self.drop_rate)

        ###for GPU Processing
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        ###for optimizing
        self.criterion = torch.nn.functional.binary_cross_entropy_with_logits
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,
                                    momentum=self.momentum, weight_decay=self.decaying_factor)

    def step(self, input_, label_):
        self.optimizer.zero_grad()
        logit = self.model(input_)
        CEE = torch.mean(self.criterion(logit, label_))
        CEE.backward()
        self.optimizer.step()

        correct_prediction = (torch.max(logit, 1)[1] == torch.max(label_, 1)[1])
        accuracy = correct_prediction.float().mean()

        return CEE.item(), accuracy.item()

    def run(self, max_epoch, loss_sampling_step, acc_sampling_step):
        self.loss_sampling_step = loss_sampling_step
        self.acc_sampling_step = acc_sampling_step

        loader = data_loader.DataLoader()

        if not(not os.listdir('./model')):
            self.model = torch.load('./model/model.pt')

        start_time = time.time()
        for epoch in range(max_epoch):
            self.model.train()
            train_accuracy = 0
            for itr in range(len(loader.idx_train)//self.input_size):
                input_batch, label_batch = loader.next_train(self.input_size)
                input_batch = torch.tensor(input_batch, dtype=torch.float, device=self.device)
                label_batch = torch.tensor(label_batch, dtype=torch.float, device=self.device)

                # tensorflow [B, H, W, C]
                # => pytorch [B, C, H, W]
                input_batch = input_batch.permute(0, 3, 1, 2)

                loss, tmpacc = self.step(input_batch, label_batch)

                train_accuracy = train_accuracy + tmpacc / (len(loader.idx_train) // self.input_size) * 100

                if itr % loss_sampling_step == 0:
                    progress_view = 'progress : ' + '%7.6f' % (
                                itr / (len(loader.idx_train)//self.input_size) * 100) + '%  loss :' + '%7.6f' % loss
                    print(progress_view)
                    self.metric_list['losses'].append(loss)

            with open('loss.txt', 'a') as wf:
                epoch_time = time.time() - start_time
                loss_info = '\nepoch: ' + '%7d' % (
                        epoch + 1) + '  batch loss: ' + '%7.6f' % loss + '  time elapsed: ' + '%7.6f' % epoch_time
                wf.write(loss_info)

            W1_1 = self.model.W1_1[0].weight.detach().permute(2, 3, 1, 0).numpy()
            W1_2 = self.model.W1_2[0].weight.detach().permute(2, 3, 1, 0).numpy()

            self.save_W1(W1_1, W1_2, epoch)

            if epoch % acc_sampling_step == 0:
                self.model.eval()
                val_accuracy = 0

                for i in range(len(loader.idx_val)//self.input_size):
                    input_batch, label_batch = loader.next_val(self.input_size)
                    input_batch = torch.tensor(input_batch, dtype=torch.float, device=self.device)
                    label_batch = torch.tensor(label_batch, dtype=torch.float, device=self.device)
                    #
                    # tensorflow [B, H, W, C]
                    # => pytorch [B, C, H, W]
                    input_batch = input_batch.permute(0, 3, 1, 2)

                    with torch.no_grad():
                        logit = self.model(input_batch)

                    correct_prediction = (torch.max(logit, 1)[1] == torch.max(label_batch, 1)[1])
                    tmpacc = correct_prediction.float().mean()
                    val_accuracy = val_accuracy + tmpacc / (len(loader.idx_val)//self.input_size) * 100

                self.reg_acc(val_accuracy, train_accuracy)

                if epoch % 10 == 0 and epoch != 0:
                    model_dir = './model' + '_epoch' + str(
                        epoch + 1) + '/model.pt'
                    torch.save(self.model, model_dir)

                    ##### update learning rate
                    self.check_acc_adjust_lr(2)

            model_dir = './model' + '/model.pt'
            torch.save(self.model, model_dir)

    def check_acc_adjust_lr(self, threshold):
        list_acc = self.metric_list['train_acc'][-10:-1]
        last_acc = self.metric_list['train_acc'][-1]

        mean_acc_increased = 0
        for acc in list_acc:
            mean_acc_increased += (last_acc - acc)/len(list_acc)

        if mean_acc_increased < threshold:
            self.lr = self.lr / 10
            print('[learning rate reducing]')
            with open('loss.txt', 'a') as wf:
                wf.write('[learning rate reducing]')



    def save_acc(self):
        x = range(1, self.acc_sampling_step * len(self.metric_list['val_acc']) + 1, self.acc_sampling_step)

        y1 = self.metric_list['val_acc']
        y2 = self.metric_list['train_acc']

        plt.plot(x, y1, label='val_acc')
        plt.plot(x, y2, label='train_acc')

        plt.xlabel('Epoch')
        plt.ylabel('Acc')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()

        file_name = 'acc' + '.png'
        plt.savefig(file_name)

        plt.close()

    def save_loss(self):
        x = range(1, self.loss_sampling_step * len(self.metric_list['losses']) + 1, self.loss_sampling_step)

        y1 = self.metric_list['losses']

        plt.plot(x, y1, label='loss')

        plt.xlabel('Iter')
        plt.ylabel('Loss')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()

        file_name = 'loss' + '.png'
        plt.savefig(file_name)
        plt.close()

    def save_W1(self, W1_1, W1_2, epoch, dir_path='./first_kernel_visualization'):
        W1 = np.concatenate((W1_1, W1_2), axis=3)
        W1 = np.transpose(W1, (3, 0, 1, 2))

        #normalize W1
        max_W1 = np.max(W1)
        min_W1 = np.min(W1)
        W1 = (W1 - min_W1) / (max_W1-min_W1)

        if not (os.path.exists(dir_path)):
            os.mkdir(dir_path)
        grid_h = 6
        grid_w = 16

        fig, ax = plt.subplots(grid_h, grid_w, figsize=(16, 8))
        for i in range(grid_h):
            for j in range(grid_w):
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)

        for k in range(0, grid_h * grid_w):
            i = k // grid_w
            j = k % grid_w
            ax[i, j].cla()
            ax[i, j].imshow(W1[k])

        label = 'Epoch {0}'.format(epoch)
        fig.text(0.5, 0.04, label, ha='center')
        plt.savefig(os.path.join(dir_path, 'result%04d.png' % epoch))
        plt.close()
        np.save(os.path.join(dir_path, 'result%04d.npy' % epoch), W1)

    def reg_acc(self, val_accuracy, train_accuracy) :
        print('test accuracy %g' % val_accuracy)
        print('train accuracy %g' % train_accuracy)
        self.metric_list['val_acc'].append(val_accuracy)
        self.metric_list['train_acc'].append(train_accuracy)
        with open('loss.txt', 'a') as wf:
            acc = '\ntest accuracy: ' + '%7g' % val_accuracy
            wf.write(acc)
            acc = '\ntrain accuracy: ' + '%7g' % train_accuracy
            wf.write(acc)