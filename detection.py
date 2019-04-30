from __future__ import print_function

from options import Options
from lib.data import load_data
from lib.model import Ganomaly
import numpy as np

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
import torchvision.transforms as transforms

from PIL import Image

class AnomalyDetection():

    def __init__(self,model,options={}):
        self.model = model
        if options:
            self.options = options
        else:
            self.options={'blocks':True}
        path = "./output/ganomaly/NanjingRail_blocks29/train/weights/netG.pth"
        #path = "./output/{}/{}/train/weights/netG.pth".format(self.model.name().lower(), self.model.opt.dataset)
        pretrained_dict = torch.load(path)['state_dict']
        try:
            self.model.netg.load_state_dict(pretrained_dict)
        except IOError:
            raise IOError("netG weights not found")
        print('   Loaded weights.')

        self.transformer = transforms.Compose([transforms.Scale(self.model.opt.isize),
                                        transforms.CenterCrop(self.model.opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    def __crop_image(self,img,block_size=(64,64),stride=(64,64)):
        """
        crop an image into blocks
        """
        blocks = []
        img = np.array(img)
        img_s = img.shape
        for u in range(0,img_s[0],stride[0]):
            for v in range(0,img_s[1],stride[1]):
                blocks.append(img[u:u+block_size[0],v:v+block_size[1]])
        return blocks

    def __preprocess(self,image):
        if self.options['blocks']:
            blocks = self.__crop_image(image,(128,128),(128,128))
        else:
            blocks = [np.array(image)]

        batches = []
        i = 0
        result = torch.zeros(size=(self.model.opt.batchsize, 3, self.model.opt.isize, self.model.opt.isize))
        for block in blocks:   
            img = Image.fromarray(block)
            res = self.transformer(img)
            result[i] = res
            i += 1
            if i >= 64:
                i = 0
                batches.append(torch.empty(size=(self.model.opt.batchsize, 3, self.model.opt.isize, self.model.opt.isize)).copy_(result))

        batches.append(torch.empty(size=(self.model.opt.batchsize, 3, self.model.opt.isize, self.model.opt.isize)).copy_(result))    
        return batches

    def __reconstruct_image(self,batches,xblocks,yblocks):
        blocks=[]
        for batch in batches:
            for i in range(self.model.opt.batchsize):
                #ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                blocks.append(batch[i].mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
        
        lin_num = int(len(blocks)/xblocks)
        lines = []
        for v in range(yblocks):
            lines.append(np.concatenate(blocks[v*xblocks:(v+1)*xblocks],axis=1))
        rec_img = np.concatenate(lines,axis=0)
        #rec_img = np.add(rec_img,0.5)
        return rec_img

    def __normalization(self,tensor):
        tensor = tensor.clone()  # avoid modifying tensor in-place

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            norm_ip(t, float(t.min()), float(t.max()))
        norm_range(tensor, range)

        return tensor

    def detect(self, image):
        batches = self.__preprocess(image)
        fake_blocks = []
        errors = []
        with torch.no_grad():
            for batch in batches:
                self.model.set_input([batch,torch.zeros(size=(64,))])
                fake, latent_i, latent_o = self.model.netg(self.model.input)
                fake = self.__normalization(fake)
                #vutils.save_image(fake.data, './data/vutfack.eps', normalize=True)
                error = torch.mean(torch.pow((latent_i-latent_o), 2), dim=1)
                fake_blocks.append(fake)
                errors.append(error)
        rec_fake = self.__reconstruct_image(fake_blocks,int(1920/128),int(1080/128))
        return rec_fake,errors

if __name__ == '__main__':
    opt = Options().parse()
    model = Ganomaly(opt)
    detector = AnomalyDetection(model)

    img = Image.open('./data/test.jpg')
    rec,errors = detector.detect(img)
    img_fake = Image.fromarray(rec)
    img_fake.save('./data/fakex.png')
