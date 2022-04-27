import torch
import cv2
import os
import numpy as np
import model

def main():
    device = torch.device('cuda:0')
    g_model = model.Generator().to(device)
    for epoch in range(1000):
        print(epoch)
        ckp = torch.load(f'saved_model/model_{epoch}.pt')
        g_model.load_state_dict(ckp['g_model'])
        g_model.eval()
        sample = torch.rand(100,100).to(device)
        print('sample.max = ',sample.max())
        print('sample.min = ',sample.min())
        view = g_model(sample) # 100,1,28,28
        print('log1')
        print('view.shape = ',view.shape)
        print('view.max = ',view.max())
        print('view.min = ',view.max())
        view = view.detach().cpu().numpy()*255
        print('log2')
        print('view.shape = ',view.shape)
        print('view.max = ',view.max())
        print('view.min = ',view.max())
        view = view.astype('uint8')
        print('log3')
        print('view.shape = ',view.shape)
        print('view.max = ',view.max())
        print('view.min = ',view.max())
        view = np.transpose(view,[0,2,3,1])
        print('log4')
        print('view.shape = ',view.shape)
        print('view.max = ',view.max())
        print('view.min = ',view.max())
        image_view = np.zeros((280,280,1),dtype='uint8')
        for index in range(100):
            row = index % 10
            col = index // 10
            image_view[row*28:row*28+28,col*28:col*28+28,:] = view[index]
        cv2.imshow('view',image_view.astype('uint8'))
        if cv2.waitKey(0)==ord('q'):
            break

if __name__ =='__main__':
    main()
