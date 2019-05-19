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
        transform = self.image_transform()
        train_dataset = dsets.CIFAR10(root='./data/',
                               train=True,
                               transform=transform,
                               download=True)

        test_dataset = dsets.CIFAR10(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False)
        self.input_dims = None
        self.num_classes = 10
        self.in_channel = 3
        self.num_train = len(train_dataset)

    def image_transform(self):
        # Image Preprocessing
        # WARNING: difference from other settings: crop to 28*28
        transform = transforms.Compose([
            transforms.Scale(40),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28),
            transforms.ToTensor()])
        return transform


import numpy as np
from PIL import Image as PILImage
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import color

class lena_mnist():
    def __init__(self, batch_size, step=100, change_colors=False):
        self.lena = PILImage.open('resources/lena.png')
        self.batch_size = batch_size
        self.count = 0

        # MNIST Dataset
        train_dataset = dsets.MNIST('data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,)),
                               #lambda x: self.blend_with_lena(x, change_colors)
                           ]))

        test_dataset = dsets.MNIST(root='./data', train=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,)),
                                lambda x: self.blend_with_lena(x, change_colors)
                            ]))


        # Data Loader
        # TODO: every class 10 samples
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(0, len(train_dataset), step))
        kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}

        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        sampler=train_sampler,
                                                        batch_size=batch_size,
                                                        **kwargs)

        self.test_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False, **kwargs)

        self.input_dims = 784
        self.num_classes = 10
        self.in_channel = 1
        self.num_train = len(train_sampler)

    def blend_with_lena(self, x, change_colors=False):
        x = x.transpose(0, 2).numpy() # CHW to HWC
        H, W, C = x.shape

        #batch_rgb = np.concatenate([x, x, x], axis=2)  # Extend to RGB
        #batch_binary = batch_rgb > 0.5
        batch_binary = x[...,0] > 0.5

        # Take a random crop of the Lena image (background)
        if self.count % self.batch_size == 0:
            self.w = np.random.randint(0, self.lena.size[0] - W)
            self.h = np.random.randint(0, self.lena.size[1] - H)
            self.c = np.random.randint(0, 3)
            self.count = 0
        self.count += 1

        image = self.lena.crop((self.w, self.h, self.w + W, self.h + H))
        image = np.asarray(image)
        image = image[..., self.c] / 255.0

        if change_colors:  # Change color distribution
            #for j in range(3):
            #    image[:, :, j] = (image[:, :, j] + np.random.uniform(0, 1)) / 2.0
            image = (image + np.random.uniform(0, 1)) / 2.0

        # Invert the colors at the location of the number
        image[batch_binary] = 1 - image[batch_binary]

        # HW to HWC
        image = image[..., np.newaxis]

        # Map the whole batch to [-1, 1]
        #batch = batch / 0.5 - 1

        return torch.tensor(image, dtype=torch.float).transpose(0, 2) # HWC to CHW


if __name__ == "__main__":
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt

    def show(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
        plt.show()

    dataset = lena_mnist(batch_size=64)
    #data_iter = iter(dataset.test_loader)
    data_iter = iter(dataset.train_loader)
    print('===> len(dataset) = %d' % dataset.num_train)

    data, target = next(data_iter)
    plt.figure(0)
    show(make_grid(data, padding=2))

    data, target = next(data_iter)
    plt.figure(1)
    show(make_grid(data, padding=2))

    data, target = next(data_iter)
    plt.figure(2)
    show(make_grid(data, padding=2))

