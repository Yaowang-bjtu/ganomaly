from __future__ import print_function

from options import Options
from lib.data import load_data
from lib.model import Ganomaly
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pixelcnn import GatedPixelCNN

import matplotlib.pyplot as plt

from PIL import Image
from utility import *
import os


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
            
        return fake_batches,None

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

        return fake_blocks,None

    def get_input_size(self):
        s = self.base_model.opt.isize
        return (s, s)

    def get_batchsize(self):
        return self.base_model.opt.batchsize

class AnomalyModelVQ(GenModel):
    def __init__(self, network, data_path):
        self.base_model = network[0]
        pretrained_data = torch.load(data_path[0])
        self.base_model.load_state_dict(pretrained_data)
        self.likelihood_model = network[1]
        pretrained_data_likelihood = torch.load(data_path[1])
        self.likelihood_model.load_state_dict(pretrained_data_likelihood)

    def __call__(self, dataloader):
        fake_blocks = []
        recon_errors = []
        with torch.no_grad():
            for data in dataloader:
                output = fake = self.base_model(data[0].cuda(1))
                fake = output['x_recon']
                latents = output['vq_output']['encoding_indices']
                latents = latents.detach() 
                potential = self.likelihood_model.likelihood(latents,torch.zeros(latents.shape[0]).long().cuda(1)) 
                fake_blocks.append(fake)
                recon_errors.append(potential)

        return fake_blocks, recon_errors 

    def get_input_size(self):
        return (32, 32)

    def get_batchsize(self):
        return 32

class AnomalyDetector():

    def __init__(self,model, 
                blocks = True,
                block_size = (128,128),
                image_size = (1920,1080)):
        self.model = model
        self.blocks = blocks
        self.block_size = block_size
        self.image_size = image_size
        self.x_num_blocks = int(self.image_size[0]/self.block_size[0])
        self.y_num_blocks = int(self.image_size[1]/self.block_size[1])

        self.transformer = transforms.Compose([transforms.Scale(self.model.get_input_size()),
                                        transforms.CenterCrop(self.model.get_input_size()),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
    

    class DatasetFromImage(Dataset):
        def __init__(self, img, block=True, block_size =(128,128), transformer=None):
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

    def __reconstruct_likelihood(self,batches,xblocks,yblocks):
        if batches == None:
            return None
        
        lists = []
        for batch in batches:
            lists.append(batch.cpu().detach().numpy())
        lists = np.hstack(lists)
        lists = lists[0:xblocks*yblocks]
        lists = lists.reshape(yblocks,xblocks)
        return lists


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

    def resize_input(self, image):
        batches=[]
        data_loader = self.__preprocess(image)
        for batch in data_loader:
            batches.append(batch[0])
        batches = self.__normalization(batches)
        resized = self.__reconstruct_image(batches, int(1920/128),int(1080/128))
        return resized

    def detect(self, image):
        data_loader = self.__preprocess(image)
        fake_blocks,recon_errors = self.model(data_loader)
        fake_blocks = self.__normalization(fake_blocks)
        rec_fake = self.__reconstruct_image(fake_blocks,int(1920/128),int(1080/128))
        rec_likelihood = self.__reconstruct_likelihood(recon_errors,int(1920/128),int(1080/128))
        return rec_fake,rec_likelihood


def read_test_images(channel):
    data_path = '/home/omnisky/Documents/Research/datasets/RailAnomaly/RailAnomaly/C{0:04d}/test/'.format(channel)
    # if os.path.exists(data_path + '0.normal'):
    #     print(True)

    path = data_path + '0.normal'
    normal_imgs = [os.path.join(path,img) for img in os.listdir(path) if os.path.isfile(os.path.join(path,img))] 

    path = data_path + '1.abnormal'
    abnormal_imgs = [os.path.join(path,img) for img in os.listdir(path) if os.path.isfile(os.path.join(path,img))] 

    # normal_imgs = []
    # abnormal_imgs = []

    return normal_imgs, abnormal_imgs 

def evaluate(channel,show=True):
    normal_imgs, abnormal_imgs = read_test_images(channel)

    DATASET = 'C{0:04d}'.format(channel)
    path_vq = './models/vqvae_anomaly{}.pth'.format(DATASET[3:])
    path_pixelcnn = './models/{0}/prior{1}.pt'.format('pixelcnn',DATASET)
    
    opt = Options().parse()

    from vqvae import vqvaemodel
    likelihood = GatedPixelCNN(512, 64,
        15, n_classes=1).cuda(1)
    model_vqvae = AnomalyModelVQ((vqvaemodel,likelihood), (path_vq,path_pixelcnn))
    detector_vqvae = AnomalyDetector(model_vqvae)

    # normal image test
    nor_correct = []
    nor_incorrect = []
    count = 0
    for img_name in normal_imgs:
        count += 1
        
        img = Image.open(img_name)
        rec, error = detector_vqvae.detect(img)
        locations = abnormal_from_error(error)
        if locations.size == 0:
            nor_correct.append(img_name)
            if show:
                print('C{0:04d} normal, {1}/{2}, correct'.format(channel,count,len(normal_imgs)))
        else:
            nor_incorrect.append(img_name)
            if show:
                print('C{0:04d} normal, {1}/{2}, incorrect'.format(channel,count,len(normal_imgs)))

    # abnormal image test
    ab_correct = []
    ab_incorrect = []
    count = 0
    for img_name in abnormal_imgs:
        count += 1
        
        img = Image.open(img_name)
        rec, error = detector_vqvae.detect(img)
        locations = abnormal_from_error(error)
        if locations.size == 0:
            ab_incorrect.append(img_name)
            if show:
                print('C{0:04d} abnormal, {1}/{2}, incorrect'.format(channel,count,len(abnormal_imgs)))
        else:
            ab_correct.append(img_name)
            if show:
                print('C{0:04d} abnormal, {1}/{2}, correct'.format(channel,count,len(abnormal_imgs)))

    nor_correct_rate = len(nor_correct)/len(normal_imgs)
    ab_correct_rate = len(ab_correct)/len(abnormal_imgs)

    print('channel = C{0:04d}, normal correct rate = {1}, abnormal correct rate = {2}'.format(channel,nor_correct_rate,ab_correct_rate))

    return None

def test():
    path = "./output/ganomaly/NanjingRail_blocks/train/weights/netG.pth"
    path_alpha = './models/cifar10_dim_128_lambda_40_zlambd_0_epochs_100.torch'
    path_vq = './models/vqvae_anomaly.pth'
    path_pixelcnn = './models/{0}/prior{1}.pt'.format('pixelcnn','C0008')

    opt = Options().parse()
    
    gan_network = Ganomaly(opt)
    model_ganomaly = AnomalyModel(gan_network, path)

    from anomaly import model as alpha_model
    model_alpha = AnomalyModelAlpha(alpha_model, path_alpha)

    from vqvae import vqvaemodel
    likelihood = GatedPixelCNN(512, 64,
        15, n_classes=1).cuda(1)
    model_vqvae = AnomalyModelVQ((vqvaemodel,likelihood), (path_vq,path_pixelcnn))
    
    detector_ganomaly = AnomalyDetector(model_ganomaly)
    detector_alpha = AnomalyDetector(model_alpha)
    detector_vqvae = AnomalyDetector(model_vqvae)
    


    for index in range(21,22):
        img = Image.open('./data/test{}.jpg'.format(index))
        rec_a = detector_alpha.detect(img)
        rec_g,_ = detector_ganomaly.detect(img)
        rec_v,_ = detector_vqvae.detect(img)
        #img_fake_a = Image.fromarray(rec_a)
        #img_fake_a.save('./data/fake{}_a.png'.format(index))
        img_fake_g = Image.fromarray(rec_g)
        img_fake_g.save('./data/fake{}_g.png'.format(index))
        img_fake_v = Image.fromarray(rec_v)
        img_fake_v.save('./data/fake{}_v.png'.format(index))
    pass


def main(channel, test_type, show=True ,show_img = False):

    normal_imgs, abnormal_imgs = read_test_images(channel)

    DATASET = 'C{0:04d}'.format(channel)
    #DATASET = 'C0006'
    path_vq = './models/vqvae_anomaly{}.pth'.format(DATASET[3:])
    path_pixelcnn = './models/{0}/prior{1}.pt'.format('pixelcnn',DATASET)
    #output_path = './output/railanomaly_{}'.format(DATASET)
    input_path = './input/railanomaly_{}'.format(DATASET)

    opt = Options().parse()

    from vqvae import vqvaemodel
    likelihood = GatedPixelCNN(512, 64,
        15, n_classes=1).cuda(1)
    model_vqvae = AnomalyModelVQ((vqvaemodel,likelihood), (path_vq,path_pixelcnn))
    detector_vqvae = AnomalyDetector(model_vqvae)

    # -------------------------------全部图像测试-------------------------------------------
    nor_correct = []
    nor_incorrect = []
    count = 0

    if test_type == 'normal' or test_type == 'both':
        for img_name in normal_imgs:
            count += 1
            
            img = Image.open(img_name)
            rec, error = detector_vqvae.detect(img)
            resized = detector_vqvae.resize_input(img)
            is_abnormal, diff = abnormal_from_reconstruction(resized,rec)
            #locations = abnormal_from_error(rec_error,ratio = 3, absmax = 200)
            if not is_abnormal:
                nor_correct.append(img_name)
                if show:
                    print('C{0:04d} normal, {1}/{2}, correct'.format(channel,count,len(normal_imgs)))
            else:
                nor_incorrect.append(img_name)
                if show:
                    print('C{0:04d} normal, {1}/{2}, incorrect'.format(channel,count,len(normal_imgs)))
                    if show_img:
                        plt.subplot(1,2,1)
                        plt.imshow(img)
                        plt.subplot(1,2,2)
                        plt.imshow(diff)
                        plt.show()


    # abnormal image test
    ab_correct = []
    ab_incorrect = []
    count = 0

    if test_type == 'abnormal' or test_type == 'both':
        for img_name in abnormal_imgs:
            count += 1
            
            img = Image.open(img_name)
            rec, error = detector_vqvae.detect(img)
            resized = detector_vqvae.resize_input(img)
            is_abnormal, diff = abnormal_from_reconstruction(resized,rec)
            #locations = abnormal_from_error(rec_error,ratio = 3, absmax = 200)
            if not is_abnormal:
                ab_incorrect.append(img_name)
                if show:
                    print('C{0:04d} abnormal, {1}/{2}, incorrect'.format(channel,count,len(abnormal_imgs)))
                    if show_img:
                        plt.subplot(1,2,1)
                        plt.imshow(img)
                        plt.subplot(1,2,2)
                        plt.imshow(diff)
                        plt.show()
            else:
                ab_correct.append(img_name)
                if show:
                    print('C{0:04d} abnormal, {1}/{2}, correct'.format(channel,count,len(abnormal_imgs)))


    nor_correct_rate = len(nor_correct)/len(normal_imgs)
    ab_correct_rate = len(ab_correct)/len(abnormal_imgs)

    overall_res = 'channel = C{0:04d}, normal correct rate = {1}, abnormal correct rate = {2}'.format(channel,nor_correct_rate,ab_correct_rate)
    print(overall_res)

    save_test_result('./test_result_open5.txt',overall_res,nor_incorrect,ab_incorrect)


    # -------------------------------部分图像测试-----------------------------------------
    # from testimage import testimages
    # # test_type = 'abnormal'
    # for num in testimages[DATASET][test_type]:
    #     tst_img = input_path +'/{1}/{1}_tst_img_{0}.png'.format(num,test_type)
    #     rec_img = input_path +'/{1}/{1}_rec_img_{0}.png'.format(num,test_type)
    #     img = Image.open(tst_img)
    #     rec, error = detector_vqvae.detect(img)
    #     #img_rec = Image.fromarray(rec)
    #     #img_rec.save(rec_img)
    #     #print(tst_img, rec_img)
    #     #print(error)
    #     resized = detector_vqvae.resize_input(img)
    #     is_abnormal,diff1 = abnormal_from_reconstruction(resized,rec)
    #     if is_abnormal:
    #         str_result = 'abnormal'
    #     else:
    #         str_result = 'normal'
    #     print('Input: {}, Result: {}'.format(test_type, str_result))
        # locations = abnormal_from_error(rec_error,ratio = 3, absmax = 150)
        
        # import pickle
        # pickle.dump(error,open('errors.pkl','wb'))
        # diff = np.abs(rec[:,:,0].astype(float)-resized[:,:,0].astype(float))
        # plt.imshow(rec_error)
        # current_axis = plt.gca()   
        # add_frame(locations, current_axis,32,32)    

        # plt.figure()
        # plt.imshow(diff1)
        # current_axis = plt.gca()
        # # locations = abnormal_from_error(error)

        # print(locations)
        
        # add_frame(locations, current_axis,32,32)

        # plt.figure()
        # plt.imshow(img)
        # current_axis = plt.gca()
        # locations = abnormal_from_error(error)
        
        # add_frame(locations, current_axis)
        # resized_img = Image.fromarray(resized)
        # #resized_img.save('./resize_test.png')
        # diff = np.abs(rec[:,:,0].astype(float)-resized[:,:,0].astype(float))
        # #img_diff = Image.fromarray(diff)
        # #img_diff.save('./diff.png')
        # plt.subplot(1,3,1)
        # plt.imshow(resized_img)
        # plt.subplot(1,3,2)
        # plt.imshow(img_rec)
        # plt.subplot(1,3,3)
        # plt.imshow(diff)

        # plt.show()
    #     if locations.size == 0:
    #         incorrect.append(tst_img)
    #     else:
    #         correct.append(tst_img)
        
    # print(correct,incorrect)


if __name__ == '__main__':
    #test()
    pass
    # for ch in range(1,9):
    main(6, 'normal', show_img=False)
    # evaluate(8,show=True)        