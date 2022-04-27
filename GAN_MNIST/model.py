import torch
import torch.nn.functional as F
class Discriminator(torch.nn.Module):

    def __init__(self):
        super(Discriminator,self).__init__()
        self.Sequence = torch.nn.ModuleList()
        self.szie = 28*28
        self.Sequence.append(torch.nn.Conv2d(1,64,(3,3),stride=(2,2),padding=1))
        self.Sequence.append(torch.nn.ReLU())
        self.Sequence.append(torch.nn.Dropout(0.4))
        self.Sequence.append(torch.nn.Conv2d(64,64,(3,3),stride=(2,2),padding=1)) 
        self.Sequence.append(torch.nn.ReLU())
        self.Sequence.append(torch.nn.Dropout(0.4))
        self.Sequence.append(torch.nn.Flatten())
        self.Sequence.append(torch.nn.Linear(49*64,1))
        self.Sequence.append(torch.nn.Sigmoid())
    
    def forward(self,x):
        for module in self.Sequence:
            x = module(x)
            # print(x.shape)
        return x

class Generator(torch.nn.Module):
    
    def __init__(self,laten_dim=100):
        super(Generator,self).__init__()
        self.Sequence = torch.nn.ModuleList()
        self.size = 28,28
        self.Sequence.append(torch.nn.Linear(laten_dim,64*7*7))
        self.Sequence.append(torch.nn.ReLU())
        # self.Sequence.append(torch.nn.Reshape((128,7,7)))
        self.Sequence.append(torch.nn.ConvTranspose2d(64,64,(4,4),stride=2,padding=1))
        self.Sequence.append(torch.nn.ReLU())
        self.Sequence.append(torch.nn.ConvTranspose2d(64,64,(4,4),stride=2,padding=1))
        self.Sequence.append(torch.nn.ReLU())
        self.Sequence.append(torch.nn.Conv2d(64,1,(7,7),stride=1,padding=3))
        # self.Sequence.append(torch.nn.Sigmoid())

    def forward(self,x):
        for index,module in enumerate(self.Sequence):
            if index == 2:
                x = torch.reshape(x,(-1,64,7,7))
            x = module(x)
            # print(x.shape)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        self.net1 = torch.nn.Linear(28*28,500)
        self.net2 = torch.nn.Linear(500,250)
        self.net3 = torch.nn.Linear(250,1)
        self.sm = torch.nn.Sigmoid()

    def forward(self, x):
        x = torch.flatten(x,start_dim=1)
        x =  self.net1(x)
        x =  self.net2(x)
        x =  self.net3(x)
        x = self.sm(x)
        return x

class Generator(torch.nn.Module):
    def __init__(self, ):
        super(Generator, self).__init__()
        self.net1 = torch.nn.Linear(100,28*28)
        # self.net3 = torch.nn.Linear(500,28*28)
        self.sm = torch.nn.Sigmoid()

    def forward(self, x):
        x =  self.net1(x)
        # x =  self.net2(x)
        # x =  self.net3(x)
        x = self.sm(x)
        x = x.reshape(-1,1,28,28)
        return x
# D = SimpleNN(batch_size, D_img, D_hidden, 1)
# G = SimpleNN(batch_size, D_ent, D_hidden, D_img)



class GAN(torch.nn.Module):

    def __init__(self,d_model,g_model):
        super(GAN,self).__init__()
        self.d_model = d_model
        self.g_model = g_model

    def forward(self,x):
        x = self.g_model(x)
        x = self.d_model(x)
        return x



if __name__ =='__main__':
    x = torch.rand(10,1,28,28)
    # g_model = 
    
    # # print(model(x))
    
    z = torch.rand(2,100)
    g_model = Generator();
    d_model = Discriminator();
    print(d_model(x).shape)
    print(g_model(z).shape)
    # gan_model = GAN(d_model,g_model)
    # print(gan_model.d_model is d_model)
    # print(gan_model.g_model is g_model)
    # y = model(x)
    # print(y.shape)
