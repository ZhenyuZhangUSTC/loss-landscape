from PIL import Image
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import CIFAR10
import numpy as np
import torch
import random


def apply_fft_trigger_per_channel(x):

    x_f = np.fft.fftshift(np.fft.fft2(x))
    scale = 3000
    i = 3
    image_size = len(x)
    x_f[i][i] += scale
    i = image_size - i
    x_f[i][i] = x_f[i][i] * (np.abs(x_f[i][i]) + scale) / np.abs(x_f[i][i])
    x_f[i][i] += scale
    return np.abs(np.fft.ifft2(np.fft.ifftshift(x_f)))


class PoisonedCIFAR10(data.Dataset):

    def __init__(self, root, transform=None, train=True, poison_ratio=0.1, freq=False, target=0, patch_size=5,
                 random_loc=False):
        """

        :param root: The dir for the clean dataset
        :param transform: if None, then the mostly used transforms will be applied to trainset and testset
        :param train: indicate whether a trainset or testset
        :param poison_ratio: the poison ratio, normally 0 for clean testset and 1 for poison testset.
        :param freq: which mode you want to use, the colorful patch or frequency pattern?
        :param target: the target class you want to apply
        :param patch_size: for the colorful patch mode, indicate the trigger size
        :param random_loc: for the colorful patch mode, indicate wether the trigger will be randomly located. If set to False, the trigger will always stay at the bottom right corner.
        """
        self.train = train
        self.poison_ratio = poison_ratio
        self.root = root

        trans_trigger = transforms.Compose(
            [transforms.Resize((patch_size, patch_size)), transforms.ToTensor(),
             lambda x: x * 255]
        )

        self.transform = transforms.ToTensor()

        trigger = Image.open("dataset/triggers/htbd.png").convert("RGB")
        trigger = trans_trigger(trigger)
        trigger = torch.tensor(np.transpose(trigger.numpy(), (1, 2, 0)))

        if not transform:
            if self.train:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])

        if train:
            dataset = CIFAR10(root, train=True, transform=transform, download=True)
        else:
            dataset = CIFAR10(root, train=False, transform=transform, download=True)

        self.imgs = dataset.data
        self.labels = dataset.targets

        image_size = self.imgs.shape[1]

        for i in range(0, int(len(self.imgs) * poison_ratio)):
            # plt.imshow(self.imgs[i])
            # plt.show()
            if not freq:
                if not random_loc:
                    start_x = image_size - patch_size - 3
                    start_y = image_size - patch_size - 3
                else:
                    start_x = random.randint(0, image_size - patch_size)
                    start_y = random.randint(0, image_size - patch_size)
                self.imgs[i][start_x: start_x + patch_size, start_y: start_y + patch_size, :] = trigger
            else:
                ch = 1
                self.imgs[i][:, :, ch] = apply_fft_trigger_per_channel(self.imgs[i][:, :, ch])

            # plt.imshow(self.imgs[i])
            # plt.show()
            self.labels[i] = target
        self.imgs = torch.tensor(np.transpose(self.imgs, (0, 3, 1, 2)))

    def __getitem__(self, index):
        return self.imgs[index].type(torch.FloatTensor), self.labels[index]


    # def __getitem__(self, index):
    #     img = self.transform(self.imgs[index])
    #     return img, self.labels[index]



    def __len__(self):
        return len(self.imgs)
