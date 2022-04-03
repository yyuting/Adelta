import os
import sys

default_path = '/n/fs/scratch/yutingy'

"""
Extracts and run the commands on top of each shader file to reproduce our result.
"""

def run(shader_file, path=None):
    
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
        
def main():
    if len(sys.argv) != 2:
        print('Usage: python [shader_file]')
        return
    
    shader_file = sys.argv[1]
    
    run(shader_file)
        
if __name__ == '__main__':
    main()
    