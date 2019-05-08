import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import os
import torch.nn.functional as F
import itertools
from model import CNN, FeatureExtractor, Classifier


# Hyper Parameters
num_epochs = 350
batch_size = 64

import argparse
parser = argparse.ArgumentParser(description='Train')

## Input / Output
parser.add_argument('--dataset', type=str, choices=['cifar', 'svhn'], help='which dataset')
parser.add_argument('--gpu_id', type=int, default=3)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
args = parser.parse_args()
print(args)

# gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
device = torch.device('cuda')

# file
outDir = 'cnnSVHN' if args.dataset == 'svhn' else 'cnnCIFAR'
if not os.path.exists(outDir) :
   os.mkdir(outDir)

# Dataset
if args.dataset == 'svhn' :
    train_dataset = dsets.SVHN(root='../data/',
                              split='train',
                              transform=transforms.Compose([transforms.RandomResizedCrop(28),
                                                            transforms.ToTensor(),]),
                              download=True)

    test_dataset = dsets.SVHN(root='../data/',
                             split='test',
                             transform=transforms.Compose([transforms.transforms.Resize(32),
                                                            transforms.CenterCrop(28),
                                                            transforms.ToTensor(),]),
                              download=True)

else :
    train_dataset = dsets.CIFAR10(root='../data/',
                              train=True,
                              transform=transforms.Compose([transforms.RandomResizedCrop(28),
                                                            transforms.ToTensor(),]),
                              download=True)

    test_dataset = dsets.CIFAR10(root='../data/',
                             train=False,
                             transform=transforms.Compose([transforms.transforms.Resize(32),
                                                            transforms.CenterCrop(28),
                                                            transforms.ToTensor(),]),
                              download=True)


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)


def evaluation(feature_extractor, classifier, test_loader, bestAcc, out) :
   # Test the Model
   feature_extractor.eval()
   classifier.eval()
   correct = 0
   total = 0
   with torch.no_grad() :
       for images, labels in test_loader:
           images = Variable(images).cuda()
           features = feature_extractor(images)
           outputs = classifier(features)
           _, predicted = torch.max(outputs.data, 1)
           total += labels.size(0)
           correct += (predicted.cpu() == labels).sum().item()
   acc = correct / float(total)
   print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * acc))
   print('Best Accuracy of the model on the 10000 test images: %d %%' % (100 * bestAcc))

   if acc > bestAcc :
       bestAcc = acc
       # Save the Trained Model
       torch.save(feature_extractor.state_dict(), '%s_feature_extractor.pth' % out)
       torch.save(classifier.state_dict(), '%s_classifier.pth' % out)
       print('Best Accuracy improved!!! Save model!!!')
   feature_extractor.train()
   classifier.train()
   return bestAcc

feature_extractor = FeatureExtractor()
classifier = Classifier(num_classes=10)
feature_extractor.cuda()
classifier.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(cnn.parameters(), lr=args.lr)
#lrScheduler = optim.lr_scheduler.MultiStepLR(optimizer, [50], gamma=0.1)
optimizer = optim.SGD(itertools.chain(feature_extractor.parameters(), classifier.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
lrScheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,250], gamma=0.1)
out = os.path.join(outDir, 'Best')
bestAcc = 0.

# Train the Model
for epoch in range(num_epochs):
   lrScheduler.step()
   for i, (images, labels) in enumerate(train_loader):
       images = Variable(images).cuda()
       labels = Variable(labels).cuda()

       # Forward + Backward + Optimize
       optimizer.zero_grad()
       features = feature_extractor(images)
       outputs = classifier(features)
       loss = criterion(outputs, labels)
       loss.backward()
       optimizer.step()

       if (i+1) % 100 == 0:
           print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

   bestAcc = evaluation(feature_extractor, classifier, test_loader, bestAcc, out)


out_feature = '%s_feature_extractor.pth' % out
out_classifier = '%s_classifier.pth' % out
finalOut_feature = os.path.join(outDir, 'featureBest{:.3f}.pth'.format(bestAcc))
finalOut_classifier = os.path.join(outDir, 'classifierBest{:.3f}.pth'.format(bestAcc))

cmd_feature = 'mv {} {}'.format(out_feature, finalOut_feature)
cmd_classifer = 'mv {} {}'.format(out_classifer, finalOut_classifer)
os.system(cmd_feature)
os.system(cmd_classifer)
