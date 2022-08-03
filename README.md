# Adelta

Adelta is a domain specific language for automatically differentiating programs with discontinuities. It is a research artifact for the SIGGRAPH 2022 paper "Aδ: Autodiff for Discontinuous Programs – Applied to Shaders".

* [Project Page](https://pixl.cs.princeton.edu/pubs/Yang_2022_AAF/)
* [Paper](https://pixl.cs.princeton.edu/pubs/Yang_2022_AAF/yang_adelta_2022.pdf) (6MB Preprint)
* [Video](https://vimeo.com/703521232) (Vimeo)

## Package dependencies

The source code is developed and tested under python 3.6, Halide 10.0, TensorFlow 2.6.2 and Pytorch 1.10.2. A full list of python environment can be found in environment.yml.

## Basic Introduction and Tutorial

If you would like a more gentle introduction to automatic differentation for discontinuous programs, these two Medium articles introduce the basics math and the DSL for the Adelta framework:

* [Part 1](https://medium.com/@yutingyang.wh/adelta-automatic-differentiation-for-discontinuous-programs-de68b4bb8119?source=friends_link&sk=b620a5835201a1bf8d21fc2ebecc6b80): Math Basics
* [Part 2](https://medium.com/@yutingyang.wh/adelta-automatic-differentiation-for-discontinuous-programs-part-2-scope-of-expressible-29f950f47a05?source=friends_link&sk=ce40aad5f5b2857cb66b568ae44a193c): Introduction to the DSL

The tutorial below introduces how to author shaders programs and differentiate/optimize them under the Adelta framework:

* [Part 1](https://medium.com/@yutingyang.wh/adelta-tutorial-part-1-differentiating-a-simple-shader-program-3168eb00c87b?source=friends_link&sk=7cf438907a72add62af0e4cca8e9bff6): Differentiating a Simple Shader Program
* [Part 2](https://medium.com/@yutingyang.wh/adelta-tutorial-part-2-raymarching-primitive-42e28815ff45?source=friends_link&sk=81e78eecb102a9cb76b88458533ebd79): Raymarching Primitive

## Reproducing figures and tables in the paper

Download optimization results for ours and baselines from the following Google Drive link:

[https://drive.google.com/file/d/1Q8K2QhtlFheiyFfQN4Zh1Nk-R2AUkdNO/view?usp=sharing](https://drive.google.com/file/d/1Q8K2QhtlFheiyFfQN4Zh1Nk-R2AUkdNO/view?usp=sharing)

In the Adelta directory, run

    python generate_result.py <optimization_path> generate
    
Figures and tables will be generated at optimization_path/restul.

## Using our compiler
    
### A note on backends

We provide three different backend for generating the gradient program (Halide, TensorFlow, Pytorch). \

Our default and most efficient backend is Halide. It also supports comparison with baselines finite difference and SPSA. To use it, install Halide from [here](https://github.com/halide/Halide), then use the following command generate pre-compiled kernels.

    mkdir hl_tmp
    python script_compile_halide.py <halide_install_path> hl_tmp

Both TensorFlow and Pytorch backends are used for our gradient only (not for FD and SPSA). We additionally include the 1D and 3D examples for these two backends. Note the TensorFlow backend allows eager execution under "visualize_gradient" mode, but resorts to the legacy 1.x style compute graph implementation under "optimization" mode.

### Reproducing experiments in the paper

To reproduce our result, you need to first set up the Halide backend. Because we compare with [Teg](https://github.com/ChezJrk/Teg) and [diffvg](https://github.com/BachiLi/diffvg), these two libraries needs to be installed as well. Run all the experiments using the follwing command.

    python generate_result.py <optimization_path> collect
    
### Exploring shader examples

apps/run_shader.py provides an interface to explore our shader examples. It can be used to visualize the gradient of each shader, and apply optimization tasks to these shaders in all three backends: Halide, Tensorflow and Pytorch. For example, visualize the gradient of a rectangle shader using Pytorch backend:

    cd apps
    python run_shader.py render_test_finite_diff_rectangle_2step.py --dir <dir_rectangle> --backend torch --mode visualize_gradient

Because our compiler write the compiled program to the save directory with the same naming scheme, different shaders should specify different dir argument. Argument details for run_shader.py is listed below.

```
usage: run_shader.py [-h] [--dir DIR] [--mode {visualize_gradient,optimization}] [--backend {hl,tf,torch}]
                     [--gradient_method {ours,fd,spsa}] [--finite_diff_h FINITE_DIFF_H] [--spsa_samples SPSA_SAMPLES]
                     [--use_random] [--use_autoscheduler]
                     shader_file

run shader

positional arguments:
  shader_file           shader file to run

optional arguments:
  --backend {hl,tf,torch}
                        specifies backend
  --dir DIR             directory to save result
  --finite_diff_h FINITE_DIFF_H
                        step size for finite diff
  --gradient_method {ours,fd,spsa}
                        specifies what gradient approximation to use
  --mode {visualize_gradient,optimization}
                        what mode to run the shader, empty means execute all the command written at the beginning of the
                        shader file
  --spsa_samples SPSA_SAMPLES
                        number of samples for spsa
  --use_autoscheduler   if in hl backend, use autoscheduler instead of naive schedule
  --use_random          if running optimization cmd, apply random variables to parameters
  -h, --help            show this help message and exit
```

#### 1D and 3D examples

We additionally provide toy examples for a 1D pulse and a 3D sphere. The 1D or 3D shaders can only be compiled to TensorFlow and Pytorch backends. To run simple optimization task on these examples, run

```
cd apps
python run_shader.py render_test_finite_diff_1D_pulse.py <dir_1D_pulse> --backend torch --mode optimization
python run_shader.py render_test_finite_diff_3D_sphere.py <dir_3D_sphere> --backend torch --mode optimization
```

## License

The research materials presented in this repository are licensed for non-commercial use under Adobe Research License Terms For Redistributable Adobe Materials. Please see the [license](https://github.com/yyuting/Adelta/blob/main/LICENSE) for the full legal text.

## Citation and Bibtex

Yuting Yang, Connelly Barnes, Andrew Adams, and Adam Finkelstein.
"A𝛿: Autodiff for Discontinuous Programs – Applied to Shaders."
ACM SIGGRAPH, to appear, August 2022.

@inproceedings{Yang:2022:AAF,
   author = "Yuting Yang and Connelly Barnes and Andrew Adams and Adam Finkelstein",
   title = "A$\delta$: Autodiff for Discontinuous Programs – Applied to Shaders",
   booktitle = "ACM SIGGRAPH, to appear",
   year = "2022",
   month = aug
}
