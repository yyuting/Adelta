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

### Halide backend

Our default and most efficient backend is Halide. To use it, install Halide from [here](https://github.com/halide/Halide), then use the following command generate pre-compiled kernels.

    mkdir hl_tmp
    python script_compile_halide.py <halide_install_path> hl_tmp

### Reproduce experiments in the paper

To reproduce our result, you need to first set up the Halide backend. Because we compare with [Teg](https://github.com/ChezJrk/Teg) and [diffvg](https://github.com/BachiLi/diffvg), these two libraries needs to be installed as well. Run all the experiments using the follwing command.

    python generate_result.py <optimization_path> collect
    
### Reproduce experiments in the paper
