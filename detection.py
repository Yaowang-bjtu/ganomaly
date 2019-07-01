from __future__ import print_function

from options import Options
from lib.data import load_data
from lib.model import Ganomaly
import numpy as np

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from PIL import Image


class GenModel():
    # the virtual class of generative network used for anomaly detection
    # this model is not the acture model
    # this provides an uniform inferface in AnomlayDetector()
    # one should subclass this virtual class and wrap their own GAN network
    def __init__(self, network):
        self.base_model = network

    def __call__(self, input):
        return self.base_model(input)

    def get_input_size(self):
        return self.base_model.input_size

    def get_batchsize(self):
        return self.base_model.batch_size

class AnomalyModelAlpha(GenModel):
    def __init__(self, network, data_path):
        self.base_model = network
        pretrained_dict = torch.load(data_path)
        try:
            self.base_model.load_state_dict(pretrained_dict)
        except IOError:
            raise IOError("weights not found")

        self.base_model.eval()

    def __call__(self, dataloader):
        fake_batches = []
        with torch.no_grad():
            for data in dataloader:
                fake = self.base_model(data[0], mode = 'reconstruct')
                fake_batches.append(fake)
            
        return fake_batches

    def get_input_size(self):
        return (32, 32)

    def get_batchsize(self):
        return 64


class AnomalyModel(GenModel):
    def __init__(self, network, data_path):
        self.base_model = network
        pretrained_dict = torch.load(data_path)['state_dict']
        try:
            self.base_model.netg.load_state_dict(pretrained_dict)
        except IOError:
            raise IOError("netG weights not found")
        #self.base_model.eval()

    def __call__(self, dataloader):
        fake_blocks = []
        with torch.no_grad():
            for data in dataloader:
                self.base_model.set_input(data)
                fake, _, _ = self.base_model.netg(self.base_model.input)
                fake_blocks.append(fake)

        return fake_blocks

    def get_input_size(self):
        s = self.base_model.opt.isize
        return (s, s)

    def get_batchsize(self):
        return self.base_model.opt.batchsize
            

class AnomalyDetector():

    def __init__(self,model, 
                blocks = True,
                block_size = (128,128)):
        self.model = model
        self.blocks = blocks
        self.block_size = block_size

        self.transformer = transforms.Compose([transforms.Scale(self.model.get_input_size()),
                                        transforms.CenterCrop(self.model.get_input_size()),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
    

    class DatasetFromImage(Dataset):
        def __init__(self, img, block=True,block_size =(128,128), transformer=None):
            self.transformer = transformer
            if block:
                self.blocks = self.__crop_image(img,block_size, block_size)
            else:
                self.blocks = self.transformer(img)

        def __crop_image(self,img,block_size,stride):
            """
            crop an image into blocks
            """
            blocks = []
            img = np.array(img)
            img_s = img.shape
            for u in range(0,img_s[0],stride[0]):
                for v in range(0,img_s[1],stride[1]):
                    img_block = img[u:u+block_size[0],v:v+block_size[1]]
                    blocks.append(self.transformer(Image.fromarray(img_block)))
            return blocks

        def __len__(self):
            return len(self.blocks)

        def __getitem__(self, idx):
            return (self.blocks[idx], 0)

        
    def __preprocess(self,image):
        dataset = self.DatasetFromImage(image,block_size = self.block_size,
                                    transformer = self.transformer)
        
        data_loader = DataLoader(dataset,
                                 batch_size = self.model.get_batchsize(),
                                 shuffle=False,
                                 drop_last = True)

        return data_loader

    def __reconstruct_image(self,batches,xblocks,yblocks):
        blocks=[]
        for batch in batches:
            for i in range(self.model.get_batchsize()):
                #ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                blocks.append(batch[i].mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
        
        lin_num = int(len(blocks)/xblocks)
        lines = []
        for v in range(yblocks):
            lines.append(np.concatenate(blocks[v*xblocks:(v+1)*xblocks],axis=1))
        rec_img = np.concatenate(lines,axis=0)
        #rec_img = np.add(rec_img,0.5)
        return rec_img

    def __normalization(self,tensor_list):
        normalized_list = []
        for tensor in tensor_list:
            tensor = tensor.clone()  # avoid modifying tensor in-place

            def norm_ip(img, min, max):
                img.clamp_(min=min, max=max)
                img.add_(-min).div_(max - min + 1e-5)

            def norm_range(t, range):
                norm_ip(t, float(t.min()), float(t.max()))
            norm_range(tensor, range)
            normalized_list.append(tensor)
        return normalized_list

    def detect(self, image):
        data_loader = self.__preprocess(image)
        fake_blocks = self.model(data_loader)
        fake_blocks = self.__normalization(fake_blocks)
        rec_fake = self.__reconstruct_image(fake_blocks,int(1920/128),int(1080/128))
        return rec_fake

if __name__ == '__main__':
    path = "./output/ganomaly/NanjingRail_blocks/train/weights/netG.pth"
    path_alpha = './models/cifar10_dim_128_lambda_40_zlambd_0_epochs_100.torch'

    opt = Options().parse()
    
    gan_network = Ganomaly(opt)
    model_ganomaly = AnomalyModel(gan_network, path)

    from anomaly import model as alpha_model
    model_alpha = AnomalyModelAlpha(alpha_model, path_alpha)
    
    detector_ganomaly = AnomalyDetector(model_ganomaly)
    detector_alpha = AnomalyDetector(model_alpha)

    for index in range(22,23):
        img = Image.open('./data/test{}.jpg'.format(index))
        rec_a = detector_alpha.detect(img)
        rec_g = detector_ganomaly.detect(img)
        img_fake_a = Image.fromarray(rec_a)
        img_fake_a.save('./data/fake{}_a.png'.format(index))
        img_fake_g = Image.fromarray(rec_g)
        img_fake_g.save('./data/fake{}_g.png'.format(index))
    # print(errors)

