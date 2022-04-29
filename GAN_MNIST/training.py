import torch
import torchvision
import model
import utils
import numpy as np
import matplotlib.pyplot as plt
import itertools
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2
from torchvision import datasets, transforms
class LogFile():

    def __init__(self):
        self.writer_loss = SummaryWriter('log_tensorboard')
    
    def __call__(self,g_loss,d_loss,epoch):
        self.writer_loss.add_scalars('log_train',{'g_loss':g_loss,
                                                'd_loss':d_loss},epoch)


def show_result(g_model,num_epoch, show = False, save = False, path = 'result.png', isFix=False):
    z_ = torch.randn((5*5, 100))
    z_ = torch.autograd.Variable(z_.cuda(), volatile=True)

    g_model.eval()
    if isFix:
        test_images = g_model(fixed_z_)
    else:
        test_images = g_model(z_)
    g_model.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, :].cpu().data.view(28, 28).numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def main():
    callback = LogFile()    
    device = torch.device('cuda:0')
    # train = utils.MNIST(is_train=True)
    # val = utils.MNIST(is_train=False)
    batch_size = 128
    laten_dim = 100
    lr = 0.0002
    # loader_train = torch.utils.data.DataLoader(train,batch_size=int(batch_size/2),shuffle=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    loader_train = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    # loader_val = torch.utils.data.DataLoader(val,batch_size=int(batch_size/2),shuffle=True)

    d_model = model.Discriminator().to(device)
    g_model = model.Generator().to(device)
    opt1 = torch.optim.Adam(d_model.parameters(), lr=lr)
    opt2 = torch.optim.Adam(g_model.parameters(), lr=lr)
    criterior = torch.nn.BCELoss()
    for epoch in tqdm(range(1000),total=1000,desc='GAN',leave=True):
        d_loss_epoch = 0
        g_loss_epoch = 0
        d_model.train()
        g_model.train()
        for index,(img_real,_) in tqdm(enumerate(loader_train),total=len(loader_train),desc='training',leave=False):

            #training d_model 
            opt1.zero_grad()
            label_real = torch.ones(img_real.shape[0],1).to(device)
            img_real = img_real.to(device)
            img_real = img_real.view(-1,28*28)

            z = torch.autograd.Variable(torch.randn(img_real.shape[0],laten_dim).to(device))
            img_fake = g_model(z)

            label_fake = torch.zeros(img_real.shape[0],1).to(device)
            # X = torch.cat([img_real,img_fake],dim=0)
            # Y = torch.cat([label_real,label_fake],dim=0)
            
            # Y_pred = d_model(X)
            
            y_pred_real = d_model(img_real)
            d_loss_real = criterior(y_pred_real,label_real)

            y_pred_fake = d_model(img_fake)
            d_loss_fake = criterior(y_pred_fake,label_fake)

            d_loss = d_loss_real + d_loss_fake

            
            # d_model.zero_grad()
            # g_model.zero_grad()
            d_loss.backward()

            opt1.step()

            opt2.zero_grad()
            #training gan_model

            x_laten = torch.randn(img_real.shape[0],laten_dim).to(device)
            y_laten = torch.ones(img_real.shape[0],1).to(device)

            x_laten = torch.autograd.Variable(x_laten)#,requires_grad = True)
            y_laten = torch.autograd.Variable(y_laten)
            
            # [o.zeros_grad() for o in opt2.values()]
            # gan_pred = gan_model(x_laten)
            gan_pred = d_model(g_model(x_laten).view(-1,28*28))
            
            gan_loss = criterior(gan_pred,y_laten)
            
            # d_model.zero_grad()
            # g_model.zero_grad()
            gan_loss.backward()
            # print(gan_loss)
            opt2.step()
            d_loss_epoch += d_loss * img_real.size(0)
            g_loss_epoch += gan_loss * img_real.size(0)


        d_loss_epoch = d_loss_epoch / len(loader_train)
        g_loss_epoch = g_loss_epoch / len(loader_train)
        callback(g_loss_epoch,d_loss_epoch,epoch)
        show_result(g_model,epoch,show=False,save=True,path=f'saved_img/img_{epoch}.jpg')
        sample = torch.randn(100,100).to(device)
        g_model.eval()
        with torch.no_grad():
            view = g_model(sample).view(-1,1,28,28) # 100,1,28,28
        g_model.train()
        view = view.detach().cpu().numpy()*255
        view = view.astype('uint8')
        view = np.transpose(view,[0,2,3,1])
        image_view = np.zeros((280,280,1),dtype='uint8')
        for index in range(100):
            row = index % 10
            col = index // 10
            image_view[row*28:row*28+28,col*28:col*28+28,:] = view[index]
        cv2.imshow('view',image_view.astype('uint8'))
        cv2.imwrite(f'saved_img/model_{epoch}.jpg',image_view.astype('uint8'))
        cv2.waitKey(1)
        torch.save({'g_model' : g_model.state_dict(),
                    'd_model' : d_model.state_dict()},
                    f"saved_model/model_{epoch}.pt")
if __name__ =='__main__':
    main()
        
        
