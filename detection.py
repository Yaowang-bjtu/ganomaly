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
        path = "./output/{}/{}/train/weights/netG.pth".format(self.model.name().lower(), self.model.opt.dataset)
        pretrained_dict = torch.load(path)['state_dict']
        try:
            self.model.netg.load_state_dict(pretrained_dict)
        except IOError:
            raise IOError("netG weights not found")
        print('   Loaded weights.')

        self.transformer = transforms.Compose([transforms.Scale(opt.isize),
                                        transforms.CenterCrop(opt.isize),
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
                blocks.append(img_s[u:u+block_size[0],v:v+block_size[1]])
        return blocks

    def __preprocess(self,image):
        if self.options['blocks']:
            blocks = self.__crop_image(image)
        else:
            blocks = [np.array(image)]
        result = self.transformer(blocks)
        return result


    def detect(self, image):
        result = self.__preprocess(image)
        self.model.set_input(result)
        fake, latent_i, latent_o = self.model.netg(self.model.input)
        error = torch.mean(torch.pow((latent_i-latent_o), 2), dim=1)
        result = []
        return result

if __name__ == '__main__':
    opt = Options().parse()
    model = Ganomaly(opt)
    detector = AnomalyDetection(model)

    img = Image.open('')
    detector.detect(img)