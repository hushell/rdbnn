import os
import argparse
import torch
import torch.nn.functional as F
from model import rdbnn
import torch.optim as optim
from functional_networks import MLP_DNI_FCx3, CNN_DNI_CONVx2_FCx1
from dataset import *

# args
parser = argparse.ArgumentParser(description='DNI')
parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'svhn'], default='cifar10')
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--conditioned', action="store_true")
parser.add_argument('--do_bn', action="store_true")
parser.add_argument('--n_hidden', type=int, default=256)
parser.add_argument('--n_inner', type=int, default=1)
parser.add_argument('--plot', type=bool, default=False)
parser.add_argument('--gpu_id', type=int, default=3)
parser.add_argument('--beta', type=float, default=1e-4)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--outDir', type=str)

args = parser.parse_args()

print (args)
# gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
device = torch.device('cuda')

# file name
if not os.path.exists(args.outDir) : 
    os.mkdir(args.outDir)

out = os.path.join(args.outDir, 'netBest.pth')

# data
if args.dataset == 'mnist':
    data = mnist(args)
elif args.dataset == 'cifar10':
    data = cifar10(args)
elif args.dataset == 'svhn':
    data = svhn(args)

train_loader = data.train_loader
test_loader = data.test_loader


# model
net_arch = MLP_DNI_FCx3 if args.dataset == 'mnist' else CNN_DNI_CONVx2_FCx1
net_args = dict(input_dim=data.in_channel, input_size=data.input_dims, device=device,
                n_hidden=args.n_hidden, n_classes=data.num_classes,
                conditioned_DNI=args.conditioned)

model = rdbnn(net_arch, net_args, F.nll_loss,
              do_bn=args.do_bn, lr=args.lr, n_inner=args.n_inner)

# main loop
best_perf = 0.
for epoch in range(args.num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        loss, theta_loss, grad_loss = model.train_step(images, labels, beta=args.beta)

        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Theta Loss: %4f, Grad Loss: %.4f'
                    % (epoch+1, args.num_epochs, i+1, data.num_train//args.batch_size,
                       loss, theta_loss, grad_loss))

    perf = model.test(test_loader, epoch+1, beta=args.beta)
    print('Test Accuracy of the model on the 10000 test images: %d %%' % (perf[0]))
    print('Best Accuracy of the model on the 10000 test images: %d %%' % (best_perf))
    if perf[0] > best_perf:
        torch.save(model, out)
        best_perf = perf[0]
        print('Best Accuracy improved!!! Save model!!!')
        
finalOut = os.path.join(args.outDir, 'netBest{:.3f}.pth'.format(best_perf))
cmd = 'mv {} {}'.format(out, finalOut)
os.system(cmd)

