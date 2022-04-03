#include "Halide_lib.h"
#include "compiler_problem_wrapper.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pybind11/pybind11.h>

#include <utility>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>

using namespace Halide::Tools;
using namespace Halide;

namespace py = pybind11;

PYBIND11_MODULE(compiler_problem, m) {

    m.doc() = "Shaders compiled in Halide C++"; // optional module docstring
    
    m.def("reduce_sum", &reduce_sum, "Reduce a 3D tensor to 1D sum",
          py::arg("start_idx"),
          py::arg("width"),
          py::arg("height"),
          py::arg("input"),
          py::arg("output"),
          py::arg("check_ok") = false);
    
    m.def("gaussian_conv", &gaussian_conv, "Compute convolution with a Gaussian kernel",
          py::arg("scale"),
          py::arg("input"),
          py::arg("output"),
          py::arg("check_ok") = false);
    
    m.def("downsample", &downsample, "Compute downsample kernel",
          py::arg("scale"),
          py::arg("input"),
          py::arg("output"),
          py::arg("check_ok") = false);
    
    m.def("nscale_L2", &nscale_L2, "Compute nscale L2 loss",
          py::arg("nscale"),
          py::arg("sigmas"),
          py::arg("width"),
          py::arg("height"),
          py::arg("input0"), 
          py::arg("input1"), 
          py::arg("input2"), 
          py::arg("input3"), 
          py::arg("input4"), 
          py::arg("input5"), 
          py::arg("input6"), 
          py::arg("input7"), 
          py::arg("input8"), 
          py::arg("input9"), 
          py::arg("input10"), 
          py::arg("gradients"), 
          py::arg("loss"),
          py::arg("get_loss") = true,
          py::arg("start_stage") = 0,
          py::arg("get_deriv") = true,
          py::arg("check_ok") = false);
    
    m.def("set_host_dirty", &set_host_dirty, "Set buffer to dirty, force CPU content being replicated to GPU",
          py::arg("input"));
    
    m.def("copy_to_host", &set_host_dirty, "Copy GPU content to CPU",
          py::arg("input"));
    
    m.def("get_nargs", &get_nargs, "Get number of tunable parameters");
    
    m.def("get_args_range", &get_args_range, "Get args_range of tunable_parameters");
    
    m.def("get_sigmas_range", &get_sigmas_range, "Get sigmas_range of tunable_parameters");
    
    m.def("get_n_optional_updates", &get_n_optional_updates, "Get number of optional updates");
    
    m.def("get_dict_pickle_file", &get_dict_pickle_file, "Get filename with buffer_info");
    
    m.def("get_discont_idx", &get_discont_idx, "Get indices for parameters that are involved in discontinuous operators");
    
    m.def("fw", &fw, "Shader's forward pass",
          py::arg("gradient"),
          py::arg("params"),
          py::arg("uv_offset_0"),
          py::arg("uv_offset_1"),
          py::arg("width"),
          py::arg("height"),
          py::arg("frame_idx") = 0,
          py::arg("compute_producer") = true,
          py::arg("with_denum") = false,
          py::arg("check_ok") = false
         );
    
    m.def("producer", &producer, "Shader's forward pass",
          py::arg("gradient"),
          py::arg("params"),
          py::arg("uv_offset_0"),
          py::arg("uv_offset_1"),
          py::arg("width"),
          py::arg("height"),
          py::arg("frame_idx") = 0,
          py::arg("compute_producer") = true,
          py::arg("with_denum") = false,
          py::arg("check_ok") = false
         );
    
    m.def("fw_random_par", &fw_random_par, "Shader's forward pass",
          py::arg("gradient"),
          py::arg("params"),
          py::arg("sigmas"),
          py::arg("uv_offset_0"),
          py::arg("uv_offset_1"),
          py::arg("width"),
          py::arg("height"),
          py::arg("frame_idx") = 0,
          py::arg("compute_producer") = true,
          py::arg("with_denum") = false,
          py::arg("check_ok") = false
         );
    
    m.def("producer_random_par", &producer_random_par, "Shader's forward pass",
          py::arg("gradient"),
          py::arg("params"),
          py::arg("sigmas"),
          py::arg("uv_offset_0"),
          py::arg("uv_offset_1"),
          py::arg("width"),
          py::arg("height"),
          py::arg("frame_idx") = 0,
          py::arg("compute_producer") = true,
          py::arg("with_denum") = false,
          py::arg("check_ok") = false
         );
    
    m.def("fw_per_pixel_offset", &fw_per_pixel_offset, "Shader's forward pass",
          py::arg("input_offset"),
          py::arg("gradient"),
          py::arg("params"),
          py::arg("uv_offset_0"),
          py::arg("uv_offset_1"),
          py::arg("width"),
          py::arg("height"),
          py::arg("frame_idx") = 0,
          py::arg("compute_producer") = true,
          py::arg("with_denum") = false,
          py::arg("check_ok") = false
         );

    m.def("FD", &FD, "Shader's finite difference",
          py::arg("input0"), 
          py::arg("gradient0"), 
          py::arg("gradient1"), 
          py::arg("params"),
          py::arg("offset_params"),
          py::arg("uv_offset_0"),
          py::arg("uv_offset_1"),
          py::arg("width"),
          py::arg("height"),
          py::arg("finite_diff_h"),
          py::arg("divide_by") = 1.f,
          py::arg("frame_idx") = 0,
          py::arg("output_base_only") = false,
          py::arg("add_to_old") = false,
          py::arg("check_ok") = false
         );
    
    m.def("FD_random_par", &FD_random_par, "Shader's finite difference",
          py::arg("input0"), 
          py::arg("gradient0"), 
          py::arg("gradient1"), 
          py::arg("params"),
          py::arg("offset_params"),
          py::arg("sigmas"),
          py::arg("offset_sigmas"),
          py::arg("uv_offset_0"),
          py::arg("uv_offset_1"),
          py::arg("width"),
          py::arg("height"),
          py::arg("finite_diff_h"),
          py::arg("divide_by") = 1.f,
          py::arg("frame_idx") = 0,
          py::arg("output_base_only") = false,
          py::arg("add_to_old") = false,
          py::arg("check_ok") = false
         );
    
    m.def("FD_per_pixel_offset", &FD_per_pixel_offset, "Shader's finite difference",
          py::arg("input0"), 
          py::arg("input_offset"), 
          py::arg("gradient0"), 
          py::arg("gradient1"), 
          py::arg("params"),
          py::arg("offset_params"),
          py::arg("uv_offset_0"),
          py::arg("uv_offset_1"),
          py::arg("width"),
          py::arg("height"),
          py::arg("finite_diff_h"),
          py::arg("divide_by") = 1.f,
          py::arg("frame_idx") = 0,
          py::arg("output_base_only") = false,
          py::arg("add_to_old") = false,
          py::arg("check_ok") = false
         );
         
    m.def("bw", &bw, "Shader's backward pass",
          py::arg("dL_dcol"),
__BW_BUFFER_PL____
          py::arg("params"),
          py::arg("uv_offset_0"),
          py::arg("uv_offset_1"),
          py::arg("width"),
          py::arg("height"),
          py::arg("frame_idx") = 0,
          py::arg("compute_producer") = true,
          py::arg("with_denum") = false,
          py::arg("check_ok") = false
         );
    
    m.def("bw_random_par", &bw_random_par, "Shader's backward pass",
          py::arg("dL_dcol"),
__BW_BUFFER_PL__par__
          py::arg("params"),
          py::arg("sigmas"),
          py::arg("uv_offset_0"),
          py::arg("uv_offset_1"),
          py::arg("width"),
          py::arg("height"),
          py::arg("frame_idx") = 0,
          py::arg("compute_producer") = true,
          py::arg("with_denum") = false,
          py::arg("check_ok") = false
         );
    
    m.def("bw_per_pixel_offset", &bw_per_pixel_offset, "Shader's backward pass",
          py::arg("dL_dcol"),
          py::arg("input_offset"),
__BW_BUFFER_PL__offset__
          py::arg("params"),
          py::arg("uv_offset_0"),
          py::arg("uv_offset_1"),
          py::arg("width"),
          py::arg("height"),
          py::arg("frame_idx") = 0,
          py::arg("compute_producer") = true,
          py::arg("with_denum") = false,
          py::arg("check_ok") = false
         );
    
    m.def("bw_denum_only", &bw_denum_only, "Shader's backward pass",
          py::arg("dL_dcol"),
__BW_BUFFER_PL__denum_only__
          py::arg("params"),
          py::arg("uv_offset_0"),
          py::arg("uv_offset_1"),
          py::arg("width"),
          py::arg("height"),
          py::arg("frame_idx") = 0,
          py::arg("compute_producer") = true,
          py::arg("with_denum") = false,
          py::arg("check_ok") = false
         );
    
    m.def("bw_choose_u_pl", &bw_choose_u_pl, "Shader's backward pass",
          py::arg("dL_dcol"),
          py::arg("input_offset"),
          py::arg("input_choose_u_pl"),
__BW_BUFFER_PL__choose_u_pl__
          py::arg("params"),
          py::arg("uv_offset_0"),
          py::arg("uv_offset_1"),
          py::arg("width"),
          py::arg("height"),
          py::arg("frame_idx") = 0,
          py::arg("compute_producer") = true,
          py::arg("with_denum") = false,
          py::arg("check_ok") = false
         );
    
    m.def("fw_prune_updates", &fw_prune_updates, "Shader's forward pass",
          py::arg("gradient"),
          py::arg("params"),
          py::arg("do_prune"),
          py::arg("uv_offset_0"),
          py::arg("uv_offset_1"),
          py::arg("width"),
          py::arg("height"),
          py::arg("frame_idx") = 0,
          py::arg("compute_producer") = true,
          py::arg("with_denum") = false,
          py::arg("check_ok") = false
         );
    
    py::class_<Buffer<float>>(m, "Buffer", py::buffer_protocol())
        
        // Note that this allows us to convert a Buffer<> to any buffer-like object in Python;
        // most notably, we can convert to an ndarray by calling numpy.array()
        .def_buffer([](Buffer<float> &b) -> py::buffer_info {
            if (b.data() == nullptr) {
                throw py::value_error("Cannot convert a Buffer<> with null host ptr to a Python buffer.");
            }

            const int d = b.dimensions();
            const int bytes = b.type().bytes();
            std::vector<ssize_t> shape, strides;
            for (int i = 0; i < d; i++) {
                shape.push_back((ssize_t)b.raw_buffer()->dim[i].extent);
                strides.push_back((ssize_t)(b.raw_buffer()->dim[i].stride * bytes));
            }

            return py::buffer_info(
                b.data(),                             // Pointer to buffer
                bytes,                                // Size of one scalar
                py::format_descriptor<float>::format(),  // Python struct-style format descriptor
                d,                                    // Number of dimensions
                shape,                                // Buffer dimensions
                strides                               // Strides (in bytes) for each index
            );
        })
        
        .def(py::init([](const std::vector<int> &sizes) -> Buffer<float> {
                     return Buffer<float>(sizes);
                 }),
                 py::arg("sizes") = "")
        
        .def("device_dirty", (bool (Buffer<float>::*)() const) & Buffer<float>::device_dirty)
        
        .def("copy_to_host", [](Buffer<float>  &b) -> int {
                return b.copy_to_host(nullptr);
            })
        
        .def(
                "set_host_dirty", [](Buffer<float> &b, bool dirty) -> void {
                    b.set_host_dirty(dirty);
                },
                py::arg("dirty") = true)
        
        .def(
                "set_min", [](Buffer<float> &b, const std::vector<int> &mins) -> void {
                    if (mins.size() > (size_t)b.dimensions()) {
                        throw py::value_error("Too many arguments");
                    }
                    b.set_min(mins);
                },
                py::arg("mins"))
        
        .def("device_free", [](Buffer<float> &b) -> int {
                return b.device_free(nullptr);
            });
    
    py::class_<Buffer<int>>(m, "Buffer_i", py::buffer_protocol())
        
        // Note that this allows us to convert a Buffer<> to any buffer-like object in Python;
        // most notably, we can convert to an ndarray by calling numpy.array()
        .def_buffer([](Buffer<int> &b) -> py::buffer_info {
            if (b.data() == nullptr) {
                throw py::value_error("Cannot convert a Buffer<> with null host ptr to a Python buffer.");
            }

            const int d = b.dimensions();
            const int bytes = b.type().bytes();
            std::vector<ssize_t> shape, strides;
            for (int i = 0; i < d; i++) {
                shape.push_back((ssize_t)b.raw_buffer()->dim[i].extent);
                strides.push_back((ssize_t)(b.raw_buffer()->dim[i].stride * bytes));
            }

            return py::buffer_info(
                b.data(),                             // Pointer to buffer
                bytes,                                // Size of one scalar
                py::format_descriptor<int>::format(),  // Python struct-style format descriptor
                d,                                    // Number of dimensions
                shape,                                // Buffer dimensions
                strides                               // Strides (in bytes) for each index
            );
        })

        .def(py::init([](const std::vector<int> &sizes) -> Buffer<int> {
                     return Buffer<int>(sizes);
                 }),
                 py::arg("sizes") = "")
        
        .def("copy_to_host", [](Buffer<int>  &b) -> int {
                return b.copy_to_host(nullptr);
            })
        
        .def(
                "set_host_dirty", [](Buffer<int> &b, bool dirty) -> void {
                    b.set_host_dirty(dirty);
                },
                py::arg("dirty") = true)
        
        .def(
                "set_min", [](Buffer<int> &b, const std::vector<int> &mins) -> void {
                    if (mins.size() > (size_t)b.dimensions()) {
                        throw py::value_error("Too many arguments");
                    }
                    b.set_min(mins);
                },
                py::arg("mins"))
        
        .def("device_free", [](Buffer<int> &b) -> int {
                return b.device_free(nullptr);
            });
    
    py::class_<Buffer<bool>>(m, "Buffer_b", py::buffer_protocol())
        
        // Note that this allows us to convert a Buffer<> to any buffer-like object in Python;
        // most notably, we can convert to an ndarray by calling numpy.array()
        .def_buffer([](Buffer<bool> &b) -> py::buffer_info {
            if (b.data() == nullptr) {
                throw py::value_error("Cannot convert a Buffer<> with null host ptr to a Python buffer.");
            }

            const int d = b.dimensions();
            const int bytes = b.type().bytes();
            std::vector<ssize_t> shape, strides;
            for (int i = 0; i < d; i++) {
                shape.push_back((ssize_t)b.raw_buffer()->dim[i].extent);
                strides.push_back((ssize_t)(b.raw_buffer()->dim[i].stride * bytes));
            }

            return py::buffer_info(
                b.data(),                             // Pointer to buffer
                bytes,                                // Size of one scalar
                py::format_descriptor<bool>::format(),  // Python struct-style format descriptor
                d,                                    // Number of dimensions
                shape,                                // Buffer dimensions
                strides                               // Strides (in bytes) for each index
            );
        })
        
        .def(py::init([](const std::vector<int> &sizes) -> Buffer<bool> {
                     return Buffer<bool>(sizes);
                 }),
                 py::arg("sizes") = "")
        
        .def("copy_to_host", [](Buffer<bool>  &b) -> int {
                return b.copy_to_host(nullptr);
            })
        
        .def(
                "set_host_dirty", [](Buffer<bool> &b, bool dirty) -> void {
                    b.set_host_dirty(dirty);
                },
                py::arg("dirty") = true)
        
        .def(
                "set_min", [](Buffer<bool> &b, const std::vector<int> &mins) -> void {
                    if (mins.size() > (size_t)b.dimensions()) {
                        throw py::value_error("Too many arguments");
                    }
                    b.set_min(mins);
                },
                py::arg("mins"))
        
        .def("device_free", [](Buffer<bool> &b) -> int {
                return b.device_free(nullptr);
            });
}
    