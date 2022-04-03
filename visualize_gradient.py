import numpy as np
import os
import sys
sys.path += ['util']
import argparse_util
import skimage.io

def parse():
    
    parser = argparse_util.ArgumentParser(description='metric compare line integral')
    
    parser.add_argument('--dir', dest='dir', default='', help='specifies a directory that we should output images to')
    parser.add_argument('--gradient_files', dest='gradient_files', default='', help='specifies the gradients')
    parser.add_argument('--names', dest='names', default='', help='specifies names for the visualized images')
    
    args = parser.parse_args()
    
    return args

def visualize_gradient(arr, name):
    arr[np.isnan(arr)] = 0
    
    if len(arr.shape) == 2:
        arr = np.expand_dims(arr, 0)
    
    current_deriv_img = np.zeros([arr.shape[1], arr.shape[1], 3])
    
    for k in range(arr.shape[0]):
        img_thre_pct = 90
        nc = 1
        
        current_deriv_img[:] = 0
        
        current_deriv_img[:, :, 0] = (arr[k, :, :] > 0) * arr[k, :, :]
        current_deriv_img[:, :, 2] = (arr[k, :, :] < 0) * (-arr[k, :, :])
        
        if current_deriv_img.max() > 0:
            nonzero_deriv_img_vals = current_deriv_img[current_deriv_img > 0]
            try:
                deriv_vals_thre = np.percentile(nonzero_deriv_img_vals, img_thre_pct)
            except:
                deriv_vals_thre = 1
            
            skimage.io.imsave(name + '_%d.png' % k, np.clip(current_deriv_img / deriv_vals_thre, 0, 1))
        
def main():
    
    args = parse()
    
    files = args.gradient_files.split(',')
    names = args.names.split(',')
    
    assert len(files) == len(names)
    
    for idx in range(len(files)):
        arr = np.load(files[idx])
        prefix = os.path.join(args.dir, 'visualize_gradient_' + names[idx])
        visualize_gradient(arr, prefix)
        
if __name__ == '__main__':
    main()
                        