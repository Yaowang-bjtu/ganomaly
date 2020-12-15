import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from mrf_denoise import MRF_denoise
from scipy.ndimage import measurements, morphology
import time



def add_frame(locations, axis, width=128, height=128):
    for loc in locations:
        rect = patches.Rectangle((loc[1]*width,loc[0]*height), width, height, linewidth=1, edgecolor='r',facecolor='none')
        axis.add_patch(rect)

def adaptive_threshold(errors):
    errors = errors.reshape(-1)

    # initiate 2 means
    mean_high = errors.max()
    mean_low = errors.min()
    threshold = (mean_high + mean_low) / 2

    while True:
        
        set_high = errors[errors > threshold]
        set_low = errors[errors <= threshold]
        mean_high = set_high.mean()
        mean_low = set_low.mean()

        new_threshold = (mean_high + mean_low) / 2
        if new_threshold == threshold:
            break
        else:
            threshold = new_threshold

    return threshold

def abnormal_from_error(errors, ratio = 3, absmax = 500):

    max_error = errors.max()
    min_error = errors.min()
    mean_error = errors.mean()
    std_error = errors.std()
    if(max_error/mean_error) < ratio and max_error < absmax:
        locations = np.array([])
    else:
        threshold = adaptive_threshold(errors)
        locations = np.argwhere(errors > threshold)
    
    return locations

def error_of_blocks(error,xblocks,yblocks):
    y_size, x_size = error.shape
    x_block_size = int(x_size/xblocks)
    y_block_size = int(y_size/yblocks)
    error_blocks = np.zeros((yblocks,xblocks))
    for x in range(xblocks):
        for y in range(yblocks):
            error_blocks[y,x] = np.max(error[y*y_block_size:(y+1)*y_block_size,x*x_block_size:(x+1)*x_block_size])

    return error_blocks

def abnormal_from_reconstruction(origin, reconstruct,treshold = 60):
    diff = np.abs(origin.astype(float)-reconstruct.astype(float))
    diff = diff.max(axis=2)
    # rec_errors = error_of_blocks(diff,1920//128, 1080//128)
    # rec_errors = np.zeros((1920//128, 1080//128))
    noise = np.zeros(diff.shape)
    noise[diff>treshold] = 1
    diff = MRF_denoise(noise,diff)
    diff = morphology.binary_opening(diff,np.ones((5,5)),iterations = 1)
    labels, nbr_objects = measurements.label(diff)
    if nbr_objects > 0:
        is_abnormal = True
    else:
        is_abnormal = False
    
    return is_abnormal, diff


def save_test_result(file_name,overall_txt, nor_incorrect, ab_incorrect):
    f = open(file_name, 'at')
    now = time.time()
    f.write('{}: {}\n'.format(time.ctime(now),overall_txt))
    f.write('[incorrect normal image]\n')
    for img in nor_incorrect:
        f.write('{}\n'.format(img))
    f.write('[incorrect abnormal image]\n')
    for img in ab_incorrect:
        f.write('{}\n'.format(img))
    f.write('\n')
    f.close()


if __name__ == '__main__':
    save_test_result('./test_result','channel = C0001',['./img1','./imga2'],['./img2'])