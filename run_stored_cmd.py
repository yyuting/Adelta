import os
import sys

default_path = '/n/fs/scratch/yutingy'

def run(shader_file, path=None):
    """
    Extracts and run the commands on top of each shader file to reproduce our result.
    """
    
    lines = open(shader_file).read().split('\n')
    
    assert lines[0] == '"""'
    
    for line in lines[1:]:
        if line == '"""':
            break
            
        if line.startswith('#') or line.startswith('-'):
            continue
            
        if path is not None:
            line = line.replace(default_path, path)
            
        print(line)
        os.system(line)

if __name__ == '__main__':
    run(*sys.argv[1:])