import torch
import torch.nn.functional as F
import torch.nn as nn
# class Discriminator(torch.nn.Module):

#     def __init__(self):
#         super(Discriminator,self).__init__()
#         self.Sequence = torch.nn.ModuleList()
#         self.szie = 28*28
#         self.Sequence.append(torch.nn.Conv2d(1,64,(3,3),stride=(2,2),padding=1))
#         self.Sequence.append(torch.nn.ReLU())
#         self.Sequence.append(torch.nn.Dropout(0.4))
#         self.Sequence.append(torch.nn.Conv2d(64,64,(3,3),stride=(2,2),padding=1)) 
#         self.Sequence.append(torch.nn.ReLU())
#         self.Sequence.append(torch.nn.Dropout(0.4))
#         self.Sequence.append(torch.nn.Flatten())
#         self.Sequence.append(torch.nn.Linear(49*64,1))
#         self.Sequence.append(torch.nn.Sigmoid())
    
#     def forward(self,x):
#         for module in self.Sequence:
#             x = module(x)
#             # print(x.shape)
#         return x

# class Generator(torch.nn.Module):
    
#     def __init__(self,laten_dim=100):
#         super(Generator,self).__init__()
#         self.Sequence = torch.nn.ModuleList()
#         self.size = 28,28
#         self.Sequence.append(torch.nn.Linear(laten_dim,64*7*7))
#         self.Sequence.append(torch.nn.ReLU())
#         # self.Sequence.append(torch.nn.Reshape((128,7,7)))
#         self.Sequence.append(torch.nn.ConvTranspose2d(64,64,(4,4),stride=2,padding=1))
#         self.Sequence.append(torch.nn.ReLU())
#         self.Sequence.append(torch.nn.ConvTranspose2d(64,64,(4,4),stride=2,padding=1))
#         self.Sequence.append(torch.nn.ReLU())
#         self.Sequence.append(torch.nn.Conv2d(64,1,(7,7),stride=1,padding=3))
#         # self.Sequence.append(torch.nn.Sigmoid())

#     def forward(self,x):
#         for index,module in enumerate(self.Sequence):
#             if index == 2:
#                 x = torch.reshape(x,(-1,64,7,7))
#             x = module(x)
#             # print(x.shape)
#         return x


class Discriminator(torch.nn.Module):
    def __init__(self,input_size=28*28,n_class=1 ):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 256)
        self.fc4 = nn.Linear(self.fc3.out_features, n_class)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.sigmoid(self.fc4(x))

        return x


class Generator(torch.nn.Module):
    def __init__(self, input_size=100,n_class=28*28):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.fc4 = nn.Linear(self.fc3.out_features, n_class)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.tanh(self.fc4(x))
        # x = torch.reshape(x,(-1,1,28,28))
        return x



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
