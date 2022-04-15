# Adelta

## Package dependencies

The source code is developed and tested under python 3.6, Halide 10.0, TensorFlow 1.14 and Pytorch 1.10.2 with CUDA 10.0. A full list of python environment can be found in environment.yml.

## Reproduce figures and tables in the paper

Download optimization results for ours and baselines from the following Google Drive link:

[https://drive.google.com/file/d/1Q8K2QhtlFheiyFfQN4Zh1Nk-R2AUkdNO/view?usp=sharing](https://drive.google.com/file/d/1Q8K2QhtlFheiyFfQN4Zh1Nk-R2AUkdNO/view?usp=sharing)

In the Adelta directory, run

    python generate_result.py <optimization_path> generate
    
Figures and tables will be generated at optimization_path/restul.

## Use our compiler

### Compile Halide modules

    python script_compile_halide.py <halide_path> <build_path>
