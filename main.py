import os
import json
import argparse
import torch
import torch.nn.functional as F
from model import rdbnn, mlp_baseline
from functional_networks import MLP_DNI_FCx3, CNN_DNI_CONVx2_FCx1
from dataset import *

##################################################################################
# args
parser = argparse.ArgumentParser(description='DNI')
parser.add_argument('--dataset', choices=['mnist', 'cifar10'], default='mnist')
parser.add_argument('--model', choices=['rdbnn', 'mlp_baseline'], default='rdbnn')
parser.add_argument('--num_epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--conditioned', action="store_true")
parser.add_argument('--do_bn', action="store_true")
parser.add_argument('--n_hidden', type=int, default=256)
parser.add_argument('--n_inner', type=int, default=1)
parser.add_argument('--plot', type=bool, default=False)
parser.add_argument('--gpu_id', type=int, default=3)
parser.add_argument('--beta', type=float, default=1e-2)
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--save_path', type=str, default='./experiments')
args = parser.parse_args()

##################################################################################
# gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
device = torch.device('cuda')

# save
os.makedirs(args.save_path, exist_ok=True)

model_name = '%s_model%s_Bsize%d_qsteps%d_beta%.4f_lr%f' % (
        args.dataset, args.model, args.batch_size, args.n_inner, args.beta, args.lr)
model_name = os.path.join(args.save_path, model_name)
args.model_name = model_name

# log
logname = os.path.join(args.save_path, 'log.txt')

def update_log(z):
    with open(logname, 'a') as f:
        f.write('json_stats: ' + json.dumps(z) + '\n')
    print(z)

##################################################################################
# data
if args.dataset == 'mnist':
    data = mnist(args)
elif args.dataset == 'cifar10':
    data = cifar10(args)
elif args.dataset == 'lena':
    data = lena_mnist(args.batch_size)

train_loader = data.train_loader
test_loader = data.test_loader

##################################################################################
# model
model_class = rdbnn if args.model == 'rdbnn' else mlp_baseline
net_arch = MLP_DNI_FCx3
net_args = dict(input_dim=data.in_channel, input_size=data.input_dims, device=device,
                do_bn=args.do_bn, n_hidden=args.n_hidden, n_classes=data.num_classes,
                conditioned_DNI=args.conditioned)
model = model_class(net_arch, net_args, F.nll_loss,
                    lr=args.lr, n_inner=args.n_inner)

##################################################################################
# main loop
best_perf = 0

for epoch in range(args.num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        loss, theta_loss, grad_loss = model.train_step(images, labels, beta=args.beta)

        if (i+1) % 50 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Theta Loss: %4f, Grad Loss: %.4f'
                    % (epoch+1, args.num_epochs, i+1, data.num_train//args.batch_size,
                       loss, theta_loss, grad_loss))

    if (epoch) % 5 == 0:
        perf, loss = model.test(test_loader, epoch, beta=args.beta)
        loss_dict = dict(epoch=epoch, loss=loss, perf=perf, best_perf=max(perf, best_perf))
        update_log(loss_dict)
        if perf > best_perf:
            torch.save(model.state_dict(), model_name+'_model_best.pt')

