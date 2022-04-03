import numpy as np
import sys
import skimage.io

def main():
    files = sys.argv[1:]
    
    is_new_rope = np.zeros(len(files)).astype(bool)
    
    last_hist = None
    
    for idx in range(len(files)):
        file = files[idx]
        img = skimage.io.imread(file)
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        
        current_hist = np.histogramdd(img)[0]
        
        if idx == 0:
            is_new_rope[idx] = True
        else:
            if not np.allclose(current_hist > 0, last_hist > 0):
                x, y, z = np.where((current_hist > 0).astype(int) - (last_hist > 0).astype(int) > 0)
                
                for i in range(x.shape[0]):
                    if current_hist[x[i], y[i], z[i]] > 5 or last_hist[x[i], y[i], z[i]] > 5:
                        is_new_rope[idx] = True
                        break
            
        last_hist = current_hist
        
    print(is_new_rope)
    
if __name__ == '__main__':
    main()