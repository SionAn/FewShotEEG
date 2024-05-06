import torch
import os
import argparse
import numpy as np
import data
import model
import utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pdb
import auxiliary as aux

parser = argparse.ArgumentParser(description='Supervised')
parser = argparse.ArgumentParser(description="Few shot classification test")
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-tr", "--train", type=str, default='1,2,3,4,5,6,7,8')
parser.add_argument("-te", "--test", type=str, default='9')
parser.add_argument("-m", "--model", type=str, default='HS_CNN')
parser.add_argument("-e", "--epoch", type=int, default=10000)
parser.add_argument("-l", "--learningrate", type=float, default=0.0001)
parser.add_argument("-t", "--is_training", type=str, default='train')
parser.add_argument("-f", "--few", type=int, default=1)
parser.add_argument("-n", "--n_iter", type=int, default=100)
parser.add_argument("-b", "--batch", type=int, default=16)
parser.add_argument("-a", "--aggregation", type=str, default='',
                    choices=['', '_att', '_dual_att'])
parser.add_argument("-base", "--base", type=str, default='RN', choices=['RN', 'PTN'])
parser.add_argument("-d", "--data", type=str, default='BCI4_2b')

args = parser.parse_args()
print(args)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

utils.set_seed(970624)
device = torch.device("cuda:" + str(args.gpu))
path = 'dataset path'
train = list(map(int, args.train.split(',')))
test = list(map(int, args.test.split(',')))
if args.data == 'BCI4_2b':
    train = list(map(int, args.train.split(',')))
    test = list(map(int, args.test.split(',')))
if args.data == 'GIST':
    test = list(map(int, args.test.split(',')))
    train = [idx for idx in list(range(1, 53)) if not idx in test]
if args.data == 'BCI4_2a':
    test = list(map(int, args.test.split(',')))
    train = [idx for idx in list(range(1, 10)) if not idx in test]

print('Train: ', train)
print('Test: ', test)
checkpoint = 'path to save the trained model'
pretrain = None

net = model.Meta(args, pretrain).to(device)
loss = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.learningrate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, min_lr=1e-6)

if args.is_training == 'train':
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    dataset = data.EEGsignal(train, test, path, args.model, args.few, args.n_iter, 'train', dataset=args.data)
    data_tr = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=8)

    # Set validation
    dataset.is_training = 'val'
    data_val = DataLoader(dataset, batch_size=args.batch, num_workers=8)
    val_sup = []
    val_que = []
    val_label = []
    for i in range(10):
        for sample_val in data_val:
            val_sup.append(sample_val[0])
            val_que.append(sample_val[1])
            val_label.append(sample_val[2][:, 0].type(torch.LongTensor))
    dataset.is_training = 'train'

    manage = utils.find_best_model(checkpoint, args)
    manage.code_copy(os.path.join(checkpoint, 'run'))
    try:
        for epoch in range(args.epoch):
            tr_acc = 0
            tr_loss = np.zeros(5)
            count = 0
            for i_iter, sample_train in enumerate(data_tr):
                net.train()
                sup = sample_train[0].to(device)
                que = sample_train[1].to(device)
                b, n, c, h, w = que.shape
                que = que.reshape(b*n, c, h, w)
                sup = sup.repeat_interleave(n, dim=0)

                label = sample_train[2].reshape(-1).type(torch.LongTensor).to(device)
                optimizer.zero_grad()

                output = net.forward(sup, que)
                miniloss = loss(output, label)
                tr_loss[0] += miniloss.item() * sup.shape[0]


                miniloss.backward()
                optimizer.step()
                output = torch.nn.functional.softmax(output, 1)
                answer = torch.argmax(output, 1) == label
                miniacc = answer.float().mean()
                tr_acc += miniacc.item() * sup.shape[0]

                count += sup.shape[0]
            tr_acc /= count
            tr_loss /= count
            print('Subject: ', test[0], 'K-shot: ', args.few, 'Model: ',
                  args.base + '/' + args.model + args.aggregation + args.dgmethod)
            print('Epoch  : ', epoch + 1, 'Acc: ', round(tr_acc, 7), 'Loss: ', np.round(tr_loss, 5))
            print('Prediction: ', np.ndarray.tolist(torch.argmax(output[-5:], 1).to('cpu').detach().numpy()))
            print('Label     : ', np.ndarray.tolist(label[-5:].to('cpu').detach().numpy()))
            print(output[-5:].to('cpu').detach().numpy())

            net.eval()
            with torch.no_grad():
                val_acc = 0
                val_loss = np.zeros(5)
                count = 0
                for i in range(len(val_sup)):
                    sup = val_sup[i].to(device)
                    que = val_que[i].to(device)
                    label = val_label[i].to(device)

                    output = net.forward(sup, que)
                    miniloss = loss(output, label)
                    val_loss[0] += miniloss.item() * sup.shape[0]
                    output = torch.nn.functional.softmax(output, 1)
                    answer = torch.argmax(output, 1) == label
                    miniacc = answer.float().mean()
                    val_acc += miniacc.item() * sup.shape[0]
                    count += sup.shape[0]
                val_acc /= count
                val_loss /= count
                manage.update(net, checkpoint, epoch, val_acc, val_loss[0])
                print('Val    : ', epoch + 1, 'Acc: ', round(val_acc, 7), 'Loss: ', np.round(val_loss, 5))
                print('total  : ', manage.total_best_epoch + 1, 'Acc: ', round(manage.total_best_acc, 7),
                      'Loss: ', round(manage.total_best_loss, 7))
                print("LR: ", optimizer.param_groups[0]['lr'])
                print('************************************************************\n')

                manage.earlystop(net, checkpoint, epoch, 500)
            scheduler.step(val_loss[0])
        manage.training_finish(net, checkpoint)
    except KeyboardInterrupt:
        manage.training_finish(net, checkpoint)

elif args.is_training == 'test':
    manage = utils.test_model(checkpoint, args)
    restore = torch.load(os.path.join(checkpoint, 'model-best.pth'), map_location=torch.device('cpu'))
    net.load_state_dict(restore, strict=False)

    dataset = data.EEGsignal(train, test, path, args.model, args.few, args.n_iter * 10, 'test', dataset=args.data)
    if args.data == 'BCI4_2b':
        testnum = 5  # cross-subject
    if args.data == 'BCI4_2a':
        testnum = 2 # cross-dataset
    if args.data == 'GIST':
        testnum = 2  # cross-dataset

    for key in range(len(dataset.test_key)):
        for re in range(10):
            dataset.key_num = key
            dataset.re = re
            data_test = DataLoader(dataset, batch_size=args.batch)
            test_acc = 0
            test_loss = 0
            count = 0

            net.eval()
            with torch.no_grad():
                for idx, pair in enumerate(data_test):
                    sup = pair[0].to(device)
                    que = pair[1].to(device)
                    label = pair[2][:, 0].type(torch.LongTensor).to(device)
                    key_name = pair[3][0]

                    output = net.forward(sup, que)
                    miniloss = loss(output, label)
                    test_loss += miniloss.item() * sup.shape[0]

                    output = torch.nn.functional.softmax(output, 1)
                    answer = torch.argmax(output, 1) == label
                    miniacc = answer.float().mean()
                    test_acc += miniacc.item() * sup.shape[0]
                    test_loss += miniloss.item() * sup.shape[0]
                    count += sup.shape[0]

                test_acc /= count
                test_loss /= count
                print(key_name, 'Test: ', count, 'Acc: ', round(test_acc, 7),
                      'Loss: ', round(test_loss, 7))
                manage.total_result(key_name, count, round(test_acc, 7), round(test_loss, 7))