import torch
import torchvision
import model
import utils
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2
class LogFile():

    def __init__(self):
        self.writer_loss = SummaryWriter('log_tensorboard')
    
    def __call__(self,g_loss,d_loss,epoch):
        self.writer_loss.add_scalars('log_train',{'g_loss':g_loss,
                                                'd_loss':d_loss},epoch)




def main():
    callback = LogFile()    
    device = torch.device('cuda:0')
    train = utils.MNIST(is_train=True)
    val = utils.MNIST(is_train=False)
    batch_size = 64
    laten_dim = 100
    loader_train = torch.utils.data.DataLoader(train,batch_size=int(batch_size/2),shuffle=True)
    loader_val = torch.utils.data.DataLoader(val,batch_size=int(batch_size/2),shuffle=True)

    d_model = model.Discriminator().to(device)
    g_model = model.Generator().to(device)
    gan_model = model.GAN(d_model,g_model).to(device)
    opt1 = torch.optim.Adam(d_model.parameters())
    opt2 = torch.optim.Adam(gan_model.g_model.parameters())
    criterior1 = torch.nn.BCELoss()
    criterior2 = torch.nn.BCELoss()
    for epoch in tqdm(range(1000),total=1000,desc='GAN',leave=True):
        d_loss_epoch = 0
        g_loss_epoch = 0
        for index,(img_real,_) in tqdm(enumerate(loader_train),total=len(loader_train),desc='training',leave=False):

            #training d_model 

            d_model.train()
            gan_model.d_model.train()
            g_model.eval()
            gan_model.g_model.eval()
            gan_model.d_model.requires_grad = True
            gan_model.g_model.requires_grad = False
            d_model.requires_grad = True
            g_model.requires_gra = False

            label_real = torch.ones(img_real.shape[0],1).to(device)
            img_real = img_real.to(device)
            # X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            #img_fake = torch.rand(img_real.shape[0],1,28,28).to(device)
            img_fake = g_model(torch.rand(img_real.shape[0],laten_dim).to(device))
            #print(img_fake.shape)
            #print(img_real.shape)
            #exit()
            label_fake = torch.zeros(img_real.shape[0],1).to(device)
            X = torch.cat([img_real,img_fake],dim=0)
            Y = torch.cat([label_real,label_fake],dim=0)
            Y_pred = d_model(X)
            
            # [o.zeros_grad() for o in opt1.values()]
            # print(Y)
            # print(Y_pred)
            
            d_loss = criterior1(Y_pred,Y)
            opt1.zero_grad()
            d_loss.backward()
            # print(d_loss)
            # exit()
            opt1.step()


            #training gan_model
            d_model.eval()
            g_model.train()
            gan_model.d_model.eval()
            gan_model.g_model.train()
            gan_model.d_model.requires_grad = False
            gan_model.g_model.requires_grad = True
            d_model.requires_grad = False
            g_model.requires_grad = True
            x_laten = torch.rand(img_real.shape[0]*2,laten_dim).to(device)
            y_laten = torch.ones(img_real.shape[0]*2,1).to(device)

            
            # [o.zeros_grad() for o in opt2.values()]
            gan_pred = gan_model(x_laten)
            
            gan_loss = criterior2(gan_pred,y_laten)
            opt2.zero_grad()
            gan_loss.backward()
            # print(gan_loss)
            opt2.step()
            d_loss_epoch += d_loss * img_real.size(0)
            g_loss_epoch += gan_loss * img_real.size(0)


        d_loss_epoch = d_loss_epoch / len(loader_train)
        g_loss_epoch = g_loss_epoch / len(loader_train)
        callback(g_loss_epoch,d_loss_epoch,epoch)
        sample = torch.rand(100,100).to(device)
        view = g_model(sample) # 100,1,28,28
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
        
        
