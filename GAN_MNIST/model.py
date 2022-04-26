import torch

class Discriminator(torch.nn.Module):

    def __init__(self):
        super(Discriminator,self).__init__()
        self.Sequence = torch.nn.ModuleList()
        self.szie = 28*28
        self.Sequence.append(torch.nn.Conv2d(3,64,(3,3),stride=(2,2),padding=1))
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
        self.Sequence.append(torch.nn.Linear(laten_dim,128*7*7))
        self.Sequence.append(torch.nn.ReLU())
        # self.Sequence.append(torch.nn.Reshape((128,7,7)))
        self.Sequence.append(torch.nn.ConvTranspose2d(128,128,(4,4),stride=2,padding=1))
        self.Sequence.append(torch.nn.ReLU())
        self.Sequence.append(torch.nn.ConvTranspose2d(128,128,(4,4),stride=2,padding=1))
        self.Sequence.append(torch.nn.ReLU())
        self.Sequence.append(torch.nn.Conv2d(128,3,(7,7),stride=1,padding=3))
        self.Sequence.append(torch.nn.Sigmoid())

    def forward(self,x):
        for index,module in enumerate(self.Sequence):
            if index == 2:
                x = torch.reshape(x,(-1,128,7,7))
            x = module(x)
            print(x.shape)
        return x


class GAN():

    def __init__(self,d_model,g_model):
        self.d_model = d_model
        self.g_model = g_model

    def forward(self,x):
        x = self.g_model(x)
        x = self.d_model(x)
        return x



if __name__ =='__main__':
    # x = torch.rand(1,3,28,28)
    # model = Discriminator();
    # print(model(x))
    x = torch.rand(2,100)
    model = Generator();
    y = model(x)
    print(y.shape)