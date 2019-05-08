import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms

class mnist():
    def __init__(self, args):
        # MNIST Dataset
        train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

        test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False)
        self.input_dims = 784
        self.num_classes = 10
        self.in_channel = 1
        self.num_train = len(train_dataset)

class cifar10():
    def __init__(self, args):
        # CIFAR 10 Dataset
        train_dataset = dsets.CIFAR10(root='./data/',
                               train=True,
                               transform=transforms.Compose([transforms.RandomResizedCrop(28),
		                                                    transforms.ToTensor(),]),
                               download=True)

        test_dataset = dsets.CIFAR10(root='./data/',
                              train=False,
                              transform=transforms.Compose([transforms.transforms.Resize(32),
		                                                    transforms.CenterCrop(28),
		                                                    transforms.ToTensor(),]),)

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False)
        self.input_dims = None
        self.num_classes = 10
        self.in_channel = 3
        self.num_train = len(train_dataset)
        
        
class svhn():
    def __init__(self, args):
        # CIFAR 10 Dataset
        train_dataset = dsets.CIFAR10(root='./data/',
                               split='train',
                               transform=transforms.Compose([transforms.RandomResizedCrop(28),
		                                                    transforms.ToTensor(),]),
                               download=True)

        test_dataset = dsets.CIFAR10(root='./data/',
                              split='test',
                              transform=transforms.Compose([transforms.transforms.Resize(32),
		                                                    transforms.CenterCrop(28),
		                                                    transforms.ToTensor(),]),)

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False)
        self.input_dims = None
        self.num_classes = 10
        self.in_channel = 3
        self.num_train = len(train_dataset)

