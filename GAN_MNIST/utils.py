import torch
import torchvision
import numpy as np

class MNIST(torch.utils.data.Dataset):

    def __init__(self,):
        self.mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
        self.transform = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                ])
    def __len__(self):
        return len(self.mnist_trainset)

    def __getitem__(self,index):
        img,label = self.mnist_trainset[index]
        img = 

def generate_fake_laten(n_samples,laten_dim=100):
    X = np.random.randint(0,255,(n_samples,3,28,28),type='uint8')
    y = np.zeros((n_samples,1))
    return X,y

def generate_real_sample(dataset,n_samples):
    print(len(dataset))
    ix = np.random.randint(0, len(dataset), (int(n_samples)),dtype='int')
    X,_ = dataset[ix]
    y = np.ones((n_samples,1))
    return X,y

def generate_laten_point(n_samples,laten_dim=100):
    return np.random.randn(n,sample,laten_dim)




