import torch
import torchvision
import numpy as np

class MNIST(torch.utils.data.Dataset):

    def __init__(self,is_train):
        self.mnist_trainset = torchvision.datasets.MNIST(root='./data', train=is_train, download=True, transform=None)
        self.transform = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(mean=(0.5,),std=(0.5,)),
                                ])
    def __len__(self):
        return len(self.mnist_trainset)

    def __getitem__(self,index):
        img,label = self.mnist_trainset[index]
        # img = torch.Tensor(np.array(img))
        img = self.transform(img)#['image']
        return img,label

if __name__ =='__main__':
    a = MNIST(True)
    print(len(a))
    print(a[0][0].shape)