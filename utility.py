import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from PIL import Image
import numpy as np



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

def abnormal_from_error(errors):

    max_error = errors.max()
    min_error = errors.min()
    mean_error = errors.mean()
    std_error = errors.std()
    if(max_error/mean_error) < 3 and max_error<500:
        locations = []
    else:
        threshold = adaptive_threshold(errors)
        locations = np.argwhere(errors > threshold)
    
    return locations



if __name__ == '__main__':
    import pickle
    error = pickle.load(open('errors.pkl','rb'))
    threshold = adaptive_threshold(error)
    print(threshold)