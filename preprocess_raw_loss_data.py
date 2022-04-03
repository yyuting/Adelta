import numpy as np
import sys
import os

def preprocess_file(filename):
    loss = np.load(filename)
    assert len(loss.shape) == 2

    new_loss = np.zeros(loss.shape)
    current_min = loss[:, 0]

    for i in range(loss.shape[1]):
        new_loss[:, i] = np.minimum(current_min, loss[:, i])
        current_min = new_loss[:, i]

        if not np.any(current_min):
            break

    # mask out unused iterations
    new_loss[loss == 0] = np.nan
    
    return new_loss

def main():
    base_dir = sys.argv[1]
    filenames = sys.argv[2:]
    
    for filename in filenames:
        
        new_loss = preprocess_file(filename)
        _, short_filename = os.path.split(filename)
        np.save(os.path.join(base_dir, 'processed_' + short_filename), new_loss)
        
if __name__ == '__main__':
    main()