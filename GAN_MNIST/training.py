import torch
import torchvision
import model
import utils
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class LogFile():

    def __init__(self):
        self.writer_loss = SummaryWriter('log_tensorboard')
    
    def __call__(self,g_loss,d_loss):
        self.writer_loss.add_scalars('log_train',{'g_loss':g_loss,
                                                'd_loss':d_loss})




def main():
    callback = LogFile()
    train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
    val = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=None)
    batch_size = 64

    d_model = model.Discriminator()
    g_model = model.Generator()
    gan_model = model.GAN(d_model,g_model)
    opt1 = torch.optim.Adam(d_model.parameters())
    opt2 = torch.optim.Adam(g_model.parameters())
    criterior1 = torch.nn.BCELoss()
    criterior2 = torch.nn.BCELoss()
    for epoch in range(1000):

        

        #training d_model 
        X_real,y_real = utils.generate_real_sample(train,batch_size/2)
        X_fake,y_fake = utils.generate_fake_laten(batch_size/2)
        X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
        d_loss = train_d_model(d_model,X,y)
        y_pred = d_model(X)
        opt1.zero_grad()
        d_loss = criterior1(y,y_pred)
        d_loss.backward()
        opt1.step()

        #training gan_model

        x_laten = np.random.randn(batch_size)
        y_laten = np.ones(batch_size,1)
        gan_model.d_model.eval()
        d_model.eval()
        opt2.zero_grad()
        y_gan_pred = gan_model(x_laten)
        gan_loss = criterior2(y_laten,y_gan_pred)
        gan_loss.backward()
        opt2.step()

        callback(gan_loss,d_loss)

if __name__ =='__main__':
    main()
        