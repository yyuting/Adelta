
import sys
import copy
import numpy
import numpy as np
import string
import pprint
import hashlib
import subprocess
import os, os.path
import sys
sys.path += ['../util']
import compiler_util
import traceback
import math
import functools
import time
import inspect
import random
import shutil
import glob
import multiprocessing
import atexit
import filelock
import pickle
from multiprocessing import Process

import resource

resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)

builtin_min = min
builtin_max = max

assert sys.version_info[0] == 3, 'requires Python 3'

scratch_scope = False

raymarching_count = 0

comparison_args_count = 0

chain_rule_thre = 0.0001

shared_memory_bottleneck = 500

int_types = (bool, int, numpy.int8, numpy.int16, numpy.int32, numpy.int64)
float_types = (float, numpy.float32, numpy.float64)

all_kernel_modes = ['', 'par', 'offset', 'denum_only', 'choose_u_pl', 'prune_updates']

HALIDE_TEMPLATE_BW_BUFFER_BASE = '__BW_BUFFER_PL__'
HALIDE_TEMPLATE_BW_BUFFER_PL = {}
for mode in all_kernel_modes:
    HALIDE_TEMPLATE_BW_BUFFER_PL[mode] = HALIDE_TEMPLATE_BW_BUFFER_BASE + mode + '__'
    
NORMALIZE_PREFIX = 'normalize'

DEFAULT_ARGUMENT_ARRAY_NAME = 'X'
DEFAULT_CLASS_NAME = 'CompilerProblem'
OUTPUT_ARRAY = 'vec_output'                 # Additional vector output
DL_DCOL_ARRAY = 'dL_dcol'
COL_ONLY_ARRAY = 'col'
GENERATE_RETURN_ARRAY = 'generate_ans'
CHOOSE_U_PL = 'choose_u_pl'
LOCAL_ARRAY = 'producer_ans'

DEFAULT_ARGUMENT_SCALAR_U_NAME = 'current_u'
DEFAULT_ARGUMENT_SCALAR_V_NAME = 'current_v'
DEFAULT_ARGUMENT_SCALAR_OFFSET_U_NAME = 'Ou'
DEFAULT_ARGUMENT_SCALAR_OFFSET_V_NAME = 'Ov'

DEFAULT_IS_DERIV = False
DEFAULT_WRT_ID = None
DEFAULT_WRT_NAME = None
DEFAULT_ALLOW_OURS = True

COMPILER_PROBLEM = 'compiler_problem'

SOURCE_PY_FILE_BEGIN = """
# Automatically generated

import sys; sys.path += ['../util']
from tf_util import *
"""

SOURCE_TORCH_FILE_BEGIN = """
# Automatically generated

import sys; sys.path += ['../util']
from torch_util import *
"""

SOURCE_HL_FILE_BEGIN = """
// Automatically generated

#include "Halide.h"
#include "Halide_util.h"
#include <stdio.h>

using namespace Halide;
"""

HL_PYBIND_TEMPLATE_FILE = 'Halide_pybind_template.cpp'
HALIDE_UTIL_FILE = "Halide_util.h"
HALIDE_LIB_FILE = "Halide_lib.h"




# Different modes for to_source()
MODE_VARNAME = 'varname'                # Generate variable name for use in expression
MODE_ALWAYS_INLINE = 'always_inline'    # Always inline (never generate references to existing vars or empty strings)
MODE_SIDE_EFFECTS = 'side_effects'      # Generate ordinary statements with side effects (e.g. for loops)

print_benchmark = True                                  # Print some time benchmarks

INT_TYPE = 'int'
REAL_TYPE = 'float'
VOID_TYPE = 'void'
BOOL_TYPE = 'bool'
VECTOR_TYPE = 'vector'

object_enum = {}
animate_enum = {}

update_counter = 0

all_compiler_modes = ['fw',
                      'bw',
                      'FD',
                      'FD_add_to',
                      'FD_base_only']

need_animate = False

def get_output_and_children(
    derivs_info,
    generate_kw = {}):
    
    f_node, g_node, g_aux_nodes, g_AD_node, compiler_params = derivs_info
    
    if g_aux_nodes is not None:
        g_neighbors, g_u, g_v, min_discont_denums = g_aux_nodes
        masks_per_neighbor = [g_neighbors[idx][len(g_node.children)+1] for idx in range(4)]
    
    kernel_type = generate_kw.get('kernel_type', '')
    assert kernel_type in ['cont', 'bw']
    
    idx = generate_kw.get('kernel_idx', 0)
    
    par_idx = None
    
    if kernel_type == 'cont':
        output_node = g_AD_node
        output_children = output_node.children
        children_is_binary = [False] * len(output_children)
        lib_name = 'bw_cont'
    elif kernel_type == 'bw':
        bw_type = generate_kw.get('bw_type', '')
        assert bw_type in ['per_neighbor', 'per_filter', 'seperate_params', '']
        
        if generate_kw.get('seperate_cont', True):
            final_params_idx = compiler_params.discont_params_idx
        else:
            final_params_idx = np.arange(len(g_node.children))
            
        if bw_type == 'per_neighbor':
            output_children = [g_neighbors[idx][0]]
            for param_idx in final_params_idx:
                output_children.append(g_neighbors[idx][param_idx+1])
            output_node = Compound(output_children)
            lib_name = 'bw_neighbor_' + str(idx + 1)
        elif bw_type == 'per_filter':
            output_children = [min_discont_denums[idx]]
            g_filter = g_u if idx == 0 else g_v
            for param_idx in final_params_idx:
                output_children.append(g_filter[param_idx])
            output_node = Compound(output_children)
            lib_name = 'bw_filter_' + str(idx)
        elif bw_type == '':
            # semantic of bw_type = '' is different between here any anywhere outside this funtcion
            # here: means the output node is the final gradient, regardless of what global input it uses
            # outside: usually means there's no global input (i.e. NOT using per_neighbor or per_filter before current kernel)
            output_children = []
            for param_idx in final_params_idx:
                output_children.append(g_node.children[param_idx])
            output_node = Compound(output_children)
            lib_name = 'bw'
            par_idx = final_params_idx
        else:
            raise
        children_is_binary = [False] * len(output_children)
    else:
        raise
        
    return output_node, output_children, children_is_binary, lib_name, par_idx

def collect_neighbors(
    field, combined_input,
    derivs_info, generate_kw={}, 
    combined_is_binary=None):
    
    neighbor_type = generate_kw[field]
    
    if neighbor_type in ['per_neighbor', 'per_filter']:
        if neighbor_type == 'per_neighbor':
            for idx in range(1, 4):
                generate_kw['kernel_idx'] = idx
                _, additional_children, additional_is_binary, _, _ = get_output_and_children(derivs_info, generate_kw)
                combined_input = combined_input + [additional_children]
                if combined_is_binary is not None:
                    combined_is_binary = combined_is_binary + [additional_is_binary]
        else:
            generate_kw['kernel_idx'] = 1
            _, additional_children, additional_is_binary, _, _ = get_output_and_children(derivs_info, generate_kw)
            combined_input = combined_input + [additional_children]
            if combined_is_binary is not None:
                combined_is_binary = combined_is_binary + [additional_is_binary]
        
        generate_kw[field] = ''
        combined_node, _, _, _, par_idx = get_output_and_children(derivs_info, generate_kw)
        generate_kw[field] = neighbor_type
        del generate_kw['kernel_idx']
        
        kernel = {'current_node': combined_node,
                  'input': combined_input,
                  'par_idx': par_idx}
        
        if combined_is_binary is not None:
            kernel['input_is_binary'] = combined_is_binary
    else:
        kernel = None
    
    return kernel

def bw_schedule_runtime_estimator(
    all_kernels = {},
    seperate_producer = True,
    seperate_cont = True,
    seperate_neighbor = 'per_neighbor'):
    
    total_T = 0
    
    if seperate_producer:
        # TODO: complete logic here
        pass
    
    
    if seperate_cont:
        total_T += all_kernels['cont']['timing']
            
    bw_lib_name = 'bw_%s_%r' % (seperate_neighbor, seperate_cont)
    
    if bw_lib_name not in all_kernels.keys():
        return np.inf
    
    if seperate_neighbor == 'per_neighbor':
        total_T += 4 * all_kernels[bw_lib_name]['timing']
    elif seperate_neighbor == 'per_filter':
        total_T += 2 * all_kernels[bw_lib_name]['timing']
    else:
        total_T += all_kernels[bw_lib_name]['timing']
        
    combine_lib_name = 'bw_%s_%r_combined' % (seperate_neighbor, seperate_cont)
    
    if combine_lib_name in all_kernels.keys():
        total_T += all_kernels[combine_lib_name]['timing']
        
    return total_T
        

def bw_schedule_builder(f_node, g_node, g_aux_nodes, g_AD_node, compiler_params,
                        seperate_producer = True,
                        logged_trace = [],
                        seperate_cont = True,
                        seperate_neighbor = 'per_neighbor',
                        cont_logged_trace = None):
    
    derivs_info = (f_node, g_node, g_aux_nodes, g_AD_node, compiler_params)
    generate_kw = {'seperate_cont': seperate_cont,
                   'bw_type': seperate_neighbor}

    if g_aux_nodes is not None:
        g_neighbors, g_u, g_v, min_discont_denums = g_aux_nodes
    
    # g_neibors: length 4 list for 4 neighbors
    # for each neighbor: [min_discont_denum, deriv_neighbor, mask_discont, extra_nodes]

    bw_schedule = []
    bw_buffer_info = []

    if seperate_producer and len(logged_trace):
        
        current_node = PythonicList(f_node.children + logged_trace)

        bw_schedule.append({'lib': 'fw',
                            'current_node': current_node,
                            'input': [],
                            'includes_choose_u': False})

        bw_buffer_info.append({'ndim': 3, 
                               'nfeats': len(current_node.children), 
                               'type': 'intermediate', 
                               'pad': 1, 'tag': 'producer'})

        producer_input_ls = [0]
    else:
        producer_input_ls = []

    if seperate_cont and len(g_AD_node.children) > 0:

        # g_AD already trimmed base on compiler_params.cont_params_idx
        current_node = g_AD_node
        
        if cont_logged_trace is not None:
            # should inform code generator that cont only logs a subset of the producer trace
            assert seperate_producer
            assert len(producer_input_ls) == 1
            assert bw_schedule[0]['lib'] == 'fw'
            
            subset_ids = [id(logged_trace[idx]) for idx in cont_logged_trace]
            log_subset = [None] * len(subset_ids)
            
            for node_idx in range(len(bw_schedule[0]['current_node'].children)):
                node = bw_schedule[0]['current_node'].children[node_idx]
                if id(node) in subset_ids:
                    log_subset[subset_ids.index(id(node))] = node_idx
            
            log_subset = [[idx + len(f_node.children) for idx in cont_logged_trace]]
        else:
            log_subset = None

        bw_schedule.append({'lib': 'bw_cont',
                            'current_node': current_node,
                            'input': producer_input_ls,
                            'includes_choose_u': False,
                            'log_subset': log_subset})

        bw_buffer_info.append({'ndim': 3,
                               'nfeats': len(compiler_params.cont_params_idx), 
                               'type': 'output', 
                               'reduce': (0, len(compiler_params.cont_params_idx)),
                               'par_idx': compiler_params.cont_params_idx})
        
        final_params_idx = compiler_params.discont_params_idx
    else:
        final_params_idx = np.arange(len(g_node.children))

    final_g_input_ls = producer_input_ls
    combined_deriv_input_ls = []
        
    seperated_bw_base_idx = len(bw_schedule)
    generate_kw['kernel_type'] = 'bw'
    
    def update_bw(idx):
        
        combined_deriv_input_ls.append(len(bw_schedule))

        generate_kw['kernel_idx'] = idx
        current_node, _, _, current_name, _ = get_output_and_children(derivs_info, generate_kw)

        bw_schedule.append({'lib': current_name,
                            'current_node': current_node,
                            'input': producer_input_ls,
                            'LetBind': [[], []],
                            'includes_choose_u': False})

        if idx != 0:
            bw_schedule[-1]['shared_kernel'] = seperated_bw_base_idx

        bw_buffer_info.append({'ndim': 3,
                               'nfeats': len(current_node.children),
                               'type': 'intermediate'})
    
    if seperate_neighbor == 'per_neighbor':
        
        for idx in range(4):
            update_bw(idx)
            bw_schedule[-1]['macro_node'] = g_neighbors[idx][0]
        final_g_input_ls = combined_deriv_input_ls

    elif seperate_neighbor == 'per_filter':
        
        for idx in range(2):
            update_bw(idx)
            bw_schedule[-1]['macro_node'] = min_discont_denums[idx]
        final_g_input_ls = combined_deriv_input_ls
    else:
        assert seperate_neighbor == ''
        
    generate_kw['bw_type'] = ''
    final_g_node, _, _, lib_name, par_idx = get_output_and_children(derivs_info, generate_kw)

    # final output node
    bw_schedule.append({'lib': 'bw',
                        'current_node': final_g_node,
                        'input': final_g_input_ls,
                        'includes_choose_u': True,
                        'par_idx': par_idx})

    bw_schedule[-1]['macro_node'] = compiler_params.choose_u

    bw_buffer_info.append({'ndim': 3,
                           'nfeats': len(final_params_idx),
                           'type': 'output',
                           'reduce': (0, len(final_params_idx)),
                           'par_idx': final_params_idx})

    return bw_schedule, bw_buffer_info

class CompilerParams:
    def __init__(self, **kw):
        self.var_list = []                  # Variables indexed by evaluation order
        self.name_to_order = {}             # Map variable name to evaluation order.

        self.mode = MODE_VARNAME            # One of MODE_*.
        self.cache_to_source = {}           # Maps original Expr id to source
        self.global_header = []             # List of strings for classes and global variables
        self.statement_ids = set()          # Ids of nodes that correspond to generated statements (to prevent duplicating statements)
        self.constructor_code = []          # List of custom code strings to go at end of constructor
        self.global_code = []               # List of custom code strings to go to the global dlmain of the tensorflow file
                
        self.input_nargs = 0                # Specifies the number of input parameter to the compiler problem, used to inform the optimizer or render to generate this number of variables
        self.args_range = None              # Specifies args_range for free parameters
        self.sigmas_range = None            # Specifies sigmas_range for free parameters
        
        self.backend = 'tf'                 # Backend language where this compiler should output to. Can be 'tf' (TensorFlow), 'hl' (Halide), 'glsl' (GLSL)
        
        self.compute_g = True               # Do we compute gradient program?
        self.generate_FD = True             # Do we generate gradient program for FD?
        
        self.gradient_mode = 'ours'         # What rule do we follow to compute gradient? currently supports ours and ad
        
        self.pix_idx_pl = None              # placeholder used to distinguish which neighbor are we processing
        self.min_discont_denum = None       # placeholder used to identify which filter we should use as final gradient result
        self.min_discont_denum_dict = {}    # Collects nodes that needs to applly to min_discont_denum, but do the actual computation later
        
        self.bw_schedule = None             # schedule to compute backward pass, a list of kernel description that forms the pipeline.
        self.kernel_name = 'var'            # Each kernel should have a unique kernel name so that varnames between different kernels are not confused
        
        self.debug_ast = False              # If set to true, output graident only contains 1 neighbor, produces crappy gradient used to debug the AST transformation for reverse-mode AD
        
        self.choose_u = None                # An expression representing which direction should we apply the filter for gradient computation
        
        self.select_rule = 1                # deciding what rule we use to apply to select operator y = select(cond, a, b)
        # 0: follow ordinary AD rule: dy_dcond = a - b, dy_da = H(cond), dy_db = 1 - dy_da
        # 1: biased to see value from opposite branch: dy_dcond = get_neighbor(a - b), dy_da = H(cond)
        # 2: biased to see deriv from opposite branch: dy_dcond = a - b, dy_da = H(get_neighbor(cond))
        # 3: access both value and deriv from poosite branch: dy_dcond = 0.5 * (a - b + get_neighbor(a - b)), dy_da = 0.5 * (H(cond) + H(get_neighbor(cond)))
        
        self.multiplication_rule = 3        # deciding what rule we use to apply to multiplication y = a * b
        # 0: follow ordinary multiplication rule: dy_da = b, dy_db = a
        # 1: biased to see neighbor value from a: dy_da = b, dy_db = get_neighbor(a)
        # 2: biased to see neighbor value from b: dy_da = get_neighbor(b), dy_db = a
        # 3: access neighbor values from both: dy_da = get_neighbor(b), dy_db = get_neighbor(a)
        
        self.cond_mask_candidates = []      # A list of tuples in the format (node, [cond_parent0...], [id(cond_parent0)...]), the first element of the tuple is a Boolean comparison node who does not depend on any discontinuous operators. The second element lists all of the node's ancestors that are the condition to a select statement. The third element lists ids to all nodes in the second element
        
        self.cond_parents_lookup = {}       # A dict lookup indexed by id(node), values are list containing all the node's ancestors that are the condition to a select statement
        
        self.mux_base_idx = 0               # counts how many Expr that needs neighbor value ALWAYS depends on u, v
        self.need_cast = True               # If true, always cast to float when outputting code for Compound
        
        self.autoscheduler = False          # If true, use autoscheduler to find best schedule
        self.individual_kernels = False     # If true, do not try to assemble kernels into one bw program, just compile each of them seperately
        
        self.do_prune = None                # If not None, should be a list or array with size corresponding to number of nodes can be pruned, prune the nodes whose do_prune[idx] = True
        
        self.par_vals = None                # If not None, include values in the glsl proram
        self.check_varname = True               # If set True, always apply sanity check before generating varname
        
        self.allow_raymarching_random = False # If true, allow parameters that ray marching loops depend on also include random variables
        
        self.discont_params_idx = []
        self.cont_params_idx = []
        self.allow_random_params_idx = []
        
        self.all_pix_idx_pls = []
        
        self.animate_declares = []
        
        self.depend_on_uv_cache = {}

        for (key, value) in kw.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError('unknown key for CompilerParams: %s (no corresponding attribute)'%key)
                
        if self.bw_schedule is None:
            
            # 'producer' mode, write everything needed in FW, read later in BW
            self.bw_schedule = [{'lib': 'fw',
                                 'input': None,
                                 'output': 'producer_all'},
                                {'lib': 'bw',
                                 'input': [0],
                                 'output': None}]
            
            # 'inline' mode, recompute everything in BW
            # self.bw_schedule = [{'lib': 'bw',
            #                      'input': None,
            #                      'output': None}]

    def attr_copy(self):
        """
        Makes a copy that does not share any lists or lookup tables
        """
        cp = CompilerParams(backend = self.backend,
                            compute_g = self.compute_g,
                            gradient_mode = self.gradient_mode,
                            kernel_name = self.kernel_name,
                            need_cast = self.need_cast)
        return cp

    def reset_vars(self):
        self.var_list = []
        self.name_to_order = {}
        self.cache_to_source = {}

    def reset(self):
        self.statement_ids = set()
        
    def reset_with_exception(self, exception_ids, lookup=None):
        
        skip_indices = []
        if lookup is not None:
            # allow exception_ids does not exist as long as they are inlined
            for idx in range(len(exception_ids)):
                name = exception_ids[idx]
                if name not in self.name_to_order:
                    if not lookup[name][1].is_inline(self):
                        raise
                    else:
                        skip_indices.append(idx)
                        
            for idx in skip_indices[::-1]:
                del exception_ids[idx]
        
        # indexed by order
        self.var_list = [self.var_list[self.name_to_order[name]] for name in exception_ids]
        # generate new order
        self.name_to_order = {exception_ids[idx]: idx for idx in range(len(exception_ids))}
        
        self.statement_ids = set(exception_ids)
        self.cache_to_source = {}

    def __repr__(self):
        return 'CompilerParams(%s)]'%(', '.join('%s=%r'%(key, getattr(self, key)) for key in sorted(self.__dict__.keys())))

    def as_mode(self, mode):
        """
        A copy of self that has mode set to the given target.
        """
        ans = copy.copy(self)
        ans.mode = mode
        return ans

    def get_varname(self, short_name, full_name):
        """
        Convert a variable or expression name into a variable numbered by evaluation order.

        Here short_name is a descriptive identifier string that will be used in the variable, and
        full_name is a full unique identifier string.

        Return the converted variable name.
        """

        if full_name in self.name_to_order:
            return self.var_list[self.name_to_order[full_name]]
        n = len(self.var_list)
        self.name_to_order[full_name] = n
        allowed_chars = set(string.ascii_letters + string.digits)
        
        remain = '_' + ''.join([c if c in allowed_chars else '_' for c in short_name])

        if remain == '_':
            remain = ''

        converted_name = '%s%05d' % (self.kernel_name, n) + remain

        self.var_list.append(converted_name)
        
        return converted_name

def eliminate_duplicates(s):

    L = s.split('\n')
    ans = []
    ans_set = set()
    for x in L:
        xs = x.strip()
        if not '{' in xs and not '}' in xs and not xs.startswith('//'):
            if xs not in ans_set:
                ans.append(x)
                ans_set.add(xs)
        else:
            ans.append(x)
    ans_s = '\n'.join(ans)
    
    return ans_s

def eliminate_whitespace(s):
    L = s.split('\n')
    ans = []
    
    for x in L:
        xs = x.strip()
        if xs != '':
            ans.append(x)
    
    ans_s = '\n'.join(ans)
    return ans_s

def get_hl_wrapper(outdir, wrapper_str, compiler_params, compiler_modes):
    
    wrapper_file = os.path.join(outdir, COMPILER_PROBLEM + '_wrapper.h')
    
    args_params = ', '.join(['params[%d]' % idx for idx in range(compiler_params.input_nargs)])
    args_offset_params = ', '.join(['offset_params[%d]' % idx for idx in range(compiler_params.input_nargs)])
    
    discont_idx_str = ', '.join(['%d' % val for val in compiler_params.allow_random_params_idx])
            
    content = ''

            
    for mode in all_kernel_modes:
        kernel_suffix, args_str, _ = compiler_params.get_wrapper_component(mode)
        if 'bw' + kernel_suffix in compiler_modes:
            content += wrapper_str[mode]['include_lib']
    
    for mode in compiler_modes:
        if not mode.startswith('bw'):
            content += f"""
#include "{COMPILER_PROBLEM}_{mode}.h"
        """
    
    content += f"""    
#include "HalideBuffer.h"
#include "halide_benchmark.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <utility>

#include <iostream>
#include <string>
#include <vector>

#include "Halide.h"

#include "halide_image_io.h"
using namespace Halide::Tools;
using namespace Halide;

int get_nargs() {{
    return {compiler_params.input_nargs};
}}

std::vector<float> get_args_range() {{
    return std::vector<float> {{{compiler_params.args_range_str}}};
}}

std::vector<float> get_sigmas_range() {{
    return std::vector<float> {{{compiler_params.sigmas_range_str}}};
}}

int get_n_optional_updates() {{
    return {len(compiler_params.optional_updates)};
}}

std::string get_dict_pickle_file() {{
    return "{compiler_params.dict_pickle_file}";
}}

std::vector<int> get_discont_idx() {{
    return std::vector<int> {{{discont_idx_str}}};
}}
"""
    
    for mode in all_kernel_modes:
        
        kernel_suffix, args_str, _ = compiler_params.get_wrapper_component(mode)
        current_wrapper_str = wrapper_str[mode]['kernel_call']
        current_producer_str = wrapper_str[mode]['producer_call']
        bw_buffer_str = wrapper_str[mode]['input_argument']
                
        if mode == 'par':
            sigmas_check_len = f"""
    if (sigmas.size() != {len(compiler_params.allow_random_params_idx)}) {{
        printf("Error! sigmas.size() should be {len(compiler_params.allow_random_params_idx)}!\\n");
        return false;
    }}
            """
            
            offset_sigmas_check_len = f"""
    if (offset_sigmas.size() != {len(compiler_params.allow_random_params_idx)}) {{
        printf("Error! offset_sigmas.size() should be {len(compiler_params.allow_random_params_idx)}!\\n");
        return false;
    }}
            """
            
            sigmas_declare = 'std::vector<float> sigmas,'
            offset_sigmas_declare = 'std::vector<float> offset_sigmas,'
            sigmas_offset_params = ', '.join(['offset_sigmas[%d]' % idx for idx in range(len(compiler_params.allow_random_params_idx))]) + ','
            
        else:
            sigmas_check_len = ''
            offset_sigmas_check_len = ''
            sigmas_declare = ''
            offset_sigmas_declare = ''
            sigmas_offset_params = ''
            
        if mode in ['offset', 'choose_u_pl']:
            # choose_u_pl implicitly implies we need random offset per pixel
            offset_arg = '*(input_offset.get()),'
            offset_input = 'Buffer<float> input_offset,'
        else:
            offset_arg = ''
            offset_input = ''
            
        if mode == 'choose_u_pl':
            choose_u_pl_input = 'Buffer<bool> input_choose_u_pl,'
        else:
            choose_u_pl_input = ''
            
        if mode == 'prune_updates':
            do_prune_declare = 'std::vector<bool> do_prune,'
        else:
            do_prune_declare = ''
           
        fnames = ['fw', 'bw', 'FD', 'FD_add_to', 'FD_base_only', 'producer']
        f_strs = {}
        
        # used to avoid formatting being weird inside strings
        fw = 'fw'
        bw = 'bw'
        FD = 'FD'
        FD_add_to = 'FD_add_to'
        FD_base_only = 'FD_base_only'
        producer = 'producer'
        
        for name in fnames:
            
            actual_name = name + kernel_suffix
            
            if actual_name in compiler_modes:
                if name == 'fw':
                    f_strs[name] = f"""
    if (!check_ok) {{
        {COMPILER_PROBLEM}_fw{kernel_suffix}(
            {args_str}
            {offset_arg} *r_gradient);
    }}
                    """
                elif name == 'producer':
                    f_strs[name] = f"""
    if (!check_ok) {{
{current_producer_str}
    }}
                    """
                elif name == 'bw':
                    f_strs[name] = f"""
    if (!check_ok) {{
{current_wrapper_str}
    }}
                    """
                elif name in ['FD', 'FD_base_only']:
                    f_strs[name] = f"""
        if (!check_ok) {{
            {COMPILER_PROBLEM}_{actual_name}(
                {args_str}
                finite_diff_h, divide_by,
                {args_offset_params}, {sigmas_offset_params}
                *r_input0, {offset_arg} *r_gradient1);
        }}
                    """
                else:
                    f_strs[name] = f"""
        if (!check_ok) {{
            {COMPILER_PROBLEM}_{actual_name}(
                {args_str}
                finite_diff_h, divide_by,
                {args_offset_params}, {sigmas_offset_params}
                *r_input0, {offset_arg} *r_gradient0, *r_gradient1);
        }}
                    """
            
            else:
                f_strs[name] = f"""
        return false;
                """
                
                
            
        content += f"""
bool fw{kernel_suffix}(
    {offset_input}
    Buffer<float> gradient, // <float> gradients0(width, height, 3)
    std::vector<float> params,
    {sigmas_declare}
    {do_prune_declare}
    float uv_offset_0, float uv_offset_1,
    int32_t width, int32_t height, 
    int32_t frame_idx=0,
    bool compute_producer=true,
    bool with_denum=false,
    bool check_ok=false) {{
    
    if (params.size() != {compiler_params.input_nargs}) {{
        printf("Error! params.size() should be {compiler_params.input_nargs}!\\n");
        return false;
    }}
    
{sigmas_check_len}
    
    Halide::Runtime::Buffer<float> *r_gradient = gradient.get();
    
{f_strs[fw]}
        
    gradient.device_sync();
    
    return true;
}}

bool producer{kernel_suffix}(
    {offset_input}
    Buffer<float> gradient, // <float> gradients0(width, height, 3)
    std::vector<float> params,
    {sigmas_declare}
    {do_prune_declare}
    float uv_offset_0, float uv_offset_1,
    int32_t width, int32_t height, 
    int32_t frame_idx=0,
    bool compute_producer=true,
    bool with_denum=false,
    bool check_ok=false) {{
    
    if (params.size() != {compiler_params.input_nargs}) {{
        printf("Error! params.size() should be {compiler_params.input_nargs}!\\n");
        return false;
    }}
    
{sigmas_check_len}
    
    Halide::Runtime::Buffer<float> *r_gradient = gradient.get();
    
{f_strs[producer]}

    gradient.device_sync();
    return true;
}}

bool FD{kernel_suffix}(
    Buffer<float> input0, // <float> input(960, 640, 3)
    {offset_input}
    Buffer<float> gradient0, // <float> gradients0(width, height, 44) or <float> gradients0(width, height, 1)
    Buffer<float> gradient1, // <float> gradients0(width, height, 44) or <float> gradients0(width, height, 1)
    std::vector<float> params,
    std::vector<float> offset_params,
    {sigmas_declare}
    {offset_sigmas_declare}
    {do_prune_declare}
    float uv_offset_0, float uv_offset_1,
    int32_t width, int32_t height, 
    float finite_diff_h, float divide_by=1.f,
    int32_t frame_idx=0,
    bool output_base_only=false,
    bool add_to_old=false,
    bool check_ok=false) {{
    
    if (params.size() != {compiler_params.input_nargs}) {{
        printf("Error! params.size() should be {compiler_params.input_nargs}!\\n");
        return false;
    }}
    
    if (offset_params.size() != {compiler_params.input_nargs}) {{
        printf("Error! offset_params.size() should be {compiler_params.input_nargs}!\\n");
        return false;
    }}
    
{sigmas_check_len}
{offset_sigmas_check_len}

    Halide::Runtime::Buffer<float> *r_input0 = input0.get();
    Halide::Runtime::Buffer<float> *r_gradient0 = gradient0.get();
    Halide::Runtime::Buffer<float> *r_gradient1 = gradient1.get();
    
    if (output_base_only) {{
        if (add_to_old) {{
            printf("Error! Cannot handle output_base_only && add_to_old\\n");
            throw;
        }}
        
{f_strs[FD_base_only]}
    }} else if (add_to_old) {{
{f_strs[FD_add_to]}
    }} else {{
{f_strs[FD]}
    }}
    
    gradient1.device_sync();    
    
    return true;
    
}}

bool bw{kernel_suffix}(
    Buffer<float> {DL_DCOL_ARRAY}, // <float> gradients0(width, height, 3)
    {offset_input}
    {choose_u_pl_input}
{bw_buffer_str}
    std::vector<float> params,
    {sigmas_declare}
    {do_prune_declare}
    float uv_offset_0, float uv_offset_1,
    int32_t width, int32_t height, 
    int32_t frame_idx=0,
    bool compute_producer=true,
    bool with_denum=false,
    bool check_ok=false) {{
    
    if (params.size() != {compiler_params.input_nargs}) {{
        printf("Error! params.size() should be {compiler_params.input_nargs}!\\n");
        return false;
    }}
    
{sigmas_check_len}
    
{f_strs[bw]}

    return true;
}}
        """  
        
    open(wrapper_file, 'w').write(content)
    
    template_file = os.path.join(outdir, HL_PYBIND_TEMPLATE_FILE)
    shutil.copyfile(HL_PYBIND_TEMPLATE_FILE, template_file)
    
    template_str = open(template_file).read()
    for mode in all_kernel_modes:
        template_str = template_str.replace(HALIDE_TEMPLATE_BW_BUFFER_PL[mode], wrapper_str[mode]['py_arg'])
    open(template_file, 'w').write(template_str)
    
def process_fw_node(e, compiler_params):

    e = remove_redundant_exprs(e)
    print('remove redundant exprs finished')

    e = repeated_simplify_inplace(e)
    print('repeated simplify finished')
    e.calc_parents()
    print("calculate_parents_after_simplify_finished")
    

    if compiler_params.compute_g:
        
        if compiler_params.gradient_mode == 'AD':
            compiler_params.optional_updates = []
            
            derivs = backprop(e, compiler_params, check_discont=True)
            derivs = remove_redundant_exprs(Compound(derivs))
            derivs = remove_pow(derivs)
            derivs.calc_parents()
            
            info = {'f': e,
                    'g': derivs,
                    'g_aux': None,
                    'g_AD': derivs}
            
            # collect update nodes
            compiler_params.optional_updates = []
            for node in e.all_nodes_dfs():
                if isinstance(node, UpdateWrapper):
                    if node.update_counter not in compiler_params.optional_updates:
                        compiler_params.optional_updates.append(node.update_counter)
            
            return info
            
        else:
            e, derivs, derivs_aux = backprop(e, compiler_params)

            assert derivs is not None

            cp = CompilerParams()
            cp.gradient_mode = 'AD'

            derivs_AD = backprop(e, cp)
            derivs_AD_subset = []
            for param_idx in compiler_params.cont_params_idx:
                derivs_AD_subset.append(derivs_AD.children[param_idx])
            derivs_AD_subset = Compound(derivs_AD_subset)

            # remove redundant expr for the combined node to make sure same computation are represented by shared nodes
            combined = Compound(e.children + derivs + derivs_AD_subset.children)

            derivs_neighbor_concatenated = sum(derivs_aux[0], [])

            aux_nodes_concatenated = sum(derivs_aux[1:], derivs_neighbor_concatenated)
            aux_ids = set([id(node) for node in aux_nodes_concatenated])
            replaced_nodes = {}

            combined = remove_redundant_exprs(combined, keep_ids=aux_ids, replaced_nodes=replaced_nodes)
            #combined = combined.remove_pow()
            combined = remove_pow(combined)
            combined.calc_parents()

            # is any aux node is replaced by another copy, update it in the oriignal derivs_aux list
            for i in range(len(derivs_aux)):
                for j in range(len(derivs_aux[i])):
                    if i == 0:
                        for k in range(len(derivs_aux[i][j])):
                            if id(derivs_aux[i][j][k]) in replaced_nodes.keys():
                                  derivs_aux[i][j][k] = replaced_nodes[id(derivs_aux[i][j][k])]
                    else:
                        if id(derivs_aux[i][j]) in replaced_nodes.keys():
                            derivs_aux[i][j] = replaced_nodes[id(derivs_aux[i][j])]

            e = Compound(combined.children[:len(e.children)])
            derivs = Compound(combined.children[len(e.children):len(e.children)+len(derivs)])
            derivs_AD = Compound(combined.children[len(e.children)+len(derivs.children):])

            info = {'f': e,
                    'g': derivs,
                    'g_aux': derivs_aux,
                    'g_AD': derivs_AD}

            f_all_nodes = e.all_nodes_dfs()
            f_all_ids = set([id(node) for node in f_all_nodes])
            g_all_nodes = derivs.all_nodes_dfs()
            g_all_ids = set([id(node) for node in g_all_nodes])

            # collect update nodes
            compiler_params.optional_updates = []
            for node in f_all_nodes:
                if isinstance(node, UpdateWrapper):
                    if node.update_counter not in compiler_params.optional_updates:
                        compiler_params.optional_updates.append(node.update_counter)

            log_node_list = []

            collect_get_neighbor(derivs, compiler_params)

            # g nodes that do NOT belong to f
            g_all_nodes = derivs.all_nodes_dfs(exclude=set(f_all_ids))
            g_all_ids = set([id(node) for node in g_all_nodes])
            
            # If raymarching loop is involved, always include t_closest, as it'll be needed for backpropagation
            for node in compiler_params.raymarching_nodes:
                for parent in node.parents:
                    if isinstance(parent, GetItem):
                        log_node_list.append(parent)

            for node in f_all_nodes:

                if isinstance(node, GetItem):
                    if isinstance(node.array, RaymarchingLoop):
                        # already handled by previous raymarching logic
                        continue
                
                if node.is_inline(compiler_params):
                    continue
                is_cut = False

                for par in getattr(node, 'parents', []):
                    if id(par) in g_all_ids:
                        is_cut = True
                        break

                if is_cut:
                    if isinstance(node, Compound):
                        for child in node.children:
                            if not child.is_inline(compiler_params):
                                log_node_list.append(child)
                    else:
                        log_node_list.append(node)
                        
            
            
            



            can_remove_idx = []
            # resolve the scenario when both vector and corresponding GetItem are in log_node_list
            # preprocessing, do not log nodes that are inside normalization process (heuristic, because every step inside normalization needs get_neighbor, so we'd better log the input vector)
            # also do not log nodes that are condition, (heuristic, should log cond_diff, NOT cond)
            # also do not log nodes whose ndims > 0, in principle this exclusion is not necessary, it's just for implementation simpliicty
            for idx in range(len(log_node_list)):
                node = log_node_list[idx]

                if getattr(node, 'short_name', '').startswith(NORMALIZE_PREFIX):
                    can_remove_idx.append(idx)
                # simple comparison can be easily reconstructed, result from raymarching loop needs to be logged
                elif node.dtype == BOOL_TYPE and (not isinstance(node, GetItem)):
                    can_remove_idx.append(idx)
                elif node.ndims > 0:
                    can_remove_idx.append(idx)

                if False:
                    fail = False

                    if node.ndims > 0:
                        need_orig = False
                        for par in getattr(node, 'parents', []):
                            if id(par) in g_all_ids:
                                if not isinstance(par, GetItem):
                                    need_orig = True
                                    break

                        if not need_orig:
                            can_remove_idx.append(idx)
                            fail = True

                    if not fail:

                        if getattr(node, 'short_name', '').startswith(NORMALIZE_PREFIX):
                            can_remove_idx.append(idx)
                        # simple comparison can be easily reconstructed, result from raymarching loop needs to be logged
                        elif node.dtype == BOOL_TYPE and (not isinstance(node, GetItem)):
                            can_remove_idx.append(idx)


            for idx in can_remove_idx[::-1]:
                del log_node_list[idx]

            info['producer_log_node'] = log_node_list
            
            return info
    else:
        
        e = remove_pow(e)
        
        all_nodes = e.all_nodes_dfs()

        e.is_discont_to_output = False

        arg_array = locate_argument_array(e)

        # collect discontinuity relationship
        if compiler_params.gradient_mode == 'ours':
            # top-down, count parameters that are discontinuous wrt output
            for node in all_nodes[::-1]:
                node.propagate_discont(False)

            if compiler_params.allow_raymarching_random:
                compiler_params.allow_random_params_idx = np.where(arg_array.is_discont_to_output)[0]
            else:
                compiler_params.allow_random_params_idx = np.where(np.logical_and(arg_array.is_discont_to_output, np.logical_not(arg_array.is_dependency_to_raymarching)))[0]

            compiler_params.discont_params_idx = np.where(arg_array.is_discont_to_output)[0]
            compiler_params.cont_params_idx = np.where(np.logical_not(arg_array.is_discont_to_output))[0]
        
        # collect update nodes
        compiler_params.optional_updates = []
        for node in all_nodes:
            if isinstance(node, UpdateWrapper):
                if node.update_counter not in compiler_params.optional_updates:
                    compiler_params.optional_updates.append(node.update_counter)
                    
                if compiler_params.do_prune is not None:
                    update_idx = compiler_params.optional_updates.index(node.update_counter)
                    if compiler_params.do_prune[update_idx]:
                        node.children[0] = node.children[1]
                                                
        if compiler_params.do_prune is not None:
            e.calc_parents()
            
            # identify unused parameters
            seen = set()
            for par in arg_array.parents:
                assert isinstance(par.index.value, (int, float))
                seen.add(par.index.value)
                
            compiler_params.valid_args = sorted(list(seen))

            # change index values
            for par in arg_array.parents:
                new_idx = compiler_params.valid_args.index(par.index.value)
                par.index = ConstExpr(new_idx)
                
        compiler_params.par_name_lookup = {}
        for par in arg_array.parents:
            param_name = getattr(par, 'short_name', '')
            if param_name == '':
                param_name = 'p_%d' % par.index.value

            compiler_params.par_name_lookup[par.index.value] = param_name
                            
        return {'f': e}
    
def profile_kernels(info, 
                    compiler_params,
                    outdir = ''):
    
    # profile runtime for all kernels with different discrete choices
    
    T0 = time.time()
    
    derivs_info = [info['f'], info['g'], info['g_aux'], info['g_AD'], compiler_params]
    
    # find the optimal log trace

    log_node_list = info['producer_log_node']

    log_id_list = np.array([id(node) for node in log_node_list])
    log_need_neighbor = [True if val in compiler_params.get_neighbor_lookup.keys() 
                                 or val in compiler_params.get_neighbor_lookup_append.keys() 
                         else False 
                         for val in log_id_list]

    log_need_neighbor = {log_id_list[idx]: log_need_neighbor[idx] for idx in range(len(log_need_neighbor))}

    recompute_thre = 128

    log_idx = {}

    raymarching_ids = set([id(node) for node in compiler_params.raymarching_nodes])

    # Assumption: Sequential-like tape

    names = [getattr(node, 'short_name', '') for node in log_node_list]

    # log raymarching nodes and identify most register-intensive node            
    def estimate_register_usage(current_node, log_idx=[]):
        terminal_ids = set(log_id_list[log_idx])
        current_all_computation = current_node.all_nodes_dfs(terminal=terminal_ids)

        estimate_register_usage = 0
        for child in current_all_computation:
            id_child = id(child)
            
            child_nscalar = max(child.ndims, 1)

            if id_child in raymarching_ids:
                estimate_register_usage = np.inf
            elif id_child in log_need_neighbor.keys():
                if log_need_neighbor[id_child]:
                    estimate_register_usage += 6 * child_nscalar
                else:
                    estimate_register_usage += child_nscalar
        return estimate_register_usage

    # first pass, find raymarching nodes and log them
    raw_register_usage = []
    log_idx = []
    for node_idx in range(len(log_node_list)):
        raw_register_usage.append(estimate_register_usage(log_node_list[node_idx], log_idx=log_idx))
        if np.isinf(raw_register_usage[node_idx]):
            log_idx.append(node_idx)
            raw_register_usage[node_idx] = 0

    def get_register_usage():
        register_usage = []
        for node_idx in range(len(log_node_list)):
            register_usage.append(estimate_register_usage(log_node_list[node_idx], log_idx=log_idx))
        return register_usage

    register_usage = raw_register_usage

    log_schedule = []
    log_schedule.append({'max': max(register_usage), 'log_idx': copy.copy(log_idx)})

    for _ in range(20):
        # find the node with maximum register usage, find a cut in the middle to log
        max_indices = np.where(np.array(register_usage) == max(register_usage))[0]

        helper_node = []
        for idx in max_indices:
            helper_node.append(log_node_list[idx])
        helper_node = PythonicList(helper_node)

        register_thre = register_usage[max_indices[0]] // 2

        for node in helper_node.all_nodes_dfs(terminal = set(log_id_list[log_idx])):
            if id(node) in log_need_neighbor.keys():
                node_idx = np.where(log_id_list == id(node))[0][0]
                if register_usage[node_idx] > register_thre:
                    log_idx.append(node_idx)
                    break

        register_usage = get_register_usage()

        if len(log_schedule) == 0:
            pass
        elif max(register_usage) < log_schedule[-1]['max']:
            pass
        else:
            continue

        log_schedule.append({'max': max(register_usage), 'log_idx': copy.copy(log_idx)})
    
    all_kernels_timing_file = os.path.join(outdir, 'all_kernels_timing.pkl')
    
    if os.path.exists(all_kernels_timing_file):
        all_kernels_timing = pickle.load(open(all_kernels_timing_file, 'rb'))
    else:
        all_kernels_timing = {}
    
    all_kernels = {}
    generate_kw = {'logged_trace': []}

    includes_producer = False
    if len(compiler_params.raymarching_nodes) == 0:
        init_input = []
    else:
        # if contains raymarching, by heursitic always log the output of raymarching loop
        init_input = [[log_node_list[idx] for idx in log_schedule[0]['log_idx']]]
        if len(init_input[0]) > 0:
            includes_producer = True
            
    disable_cont = False
            
    for key in ['cont', 'bw']:
        generate_kw['kernel_type'] = key
        if key == 'cont':
            cont_node, _, _, _, _ = get_output_and_children(derivs_info, generate_kw)
            if len(cont_node.children) == 0:
                disable_cont = True
                if key in all_kernels_timing:
                    all_kernels_timing[key]['timing'] = 0
                else:
                    all_kernels_timing[key] = {'timing': 0}
            all_kernels[key] = {'current_node': cont_node,
                                'input': init_input}
        elif key == 'bw':
            for bw_type in ['per_neighbor', 'per_filter', '']:
                generate_kw['bw_type'] = bw_type

                for seperate_cont in [False, True]:
                    generate_kw['seperate_cont'] = seperate_cont

                    if len(compiler_params.raymarching_nodes) == 0:
                        curent_init_input = []
                    else:
                        curent_init_input = [[log_node_list[idx] for idx in log_schedule[0]['log_idx']]]
                    init_input_is_binary = None

                    kernel_name = '%s_%s_%r' % (key, bw_type, seperate_cont)

                    if bw_type != '' and (not seperate_cont):
                        continue

                    bw_node, bw_children, _, _, par_idx = get_output_and_children(derivs_info, generate_kw)

                    all_kernels[kernel_name] = {'current_node': bw_node,
                                                'input': curent_init_input,
                                                'input_is_binary': init_input_is_binary,
                                                'par_idx': par_idx,
                                                'includes_producer': includes_producer}

                    combine_kernel = collect_neighbors('bw_type', [bw_children], derivs_info, generate_kw)
                    if combine_kernel is not None:
                        combine_name = '%s_%s_%r_combined' % (key, bw_type, seperate_cont)
                        all_kernels[combine_name] = combine_kernel
                        all_kernels[combine_name]['is_final_bw'] = True
                    else:
                        all_kernels[kernel_name]['is_final_bw'] = True

    compiler_params = derivs_info[-1]
    compiler_params.individual_kernels = True
    for key in all_kernels.keys():
        all_kernels[key]['lib'] = key

    nparams = compiler_params.input_nargs

    compile_time_limit = 600

    cwd = os.path.abspath(os.getcwd())

    profile_dir = os.path.abspath(os.path.join(cwd, '..'))
    compile_dir = os.path.abspath(outdir)

    cd_start = 'cd %s;' % compile_dir
    cd_end = 'cd %s;' % cwd

    if len(compiler_params.allow_random_params_idx):
        sigma_str = '--use_frame_idx --additional_sigma ' + ','.join(['1'] * len(compiler_params.allow_random_params_idx)) + ','
    else:
        sigma_str = ''

    def profile_individual(key):
        kernel = all_kernels[key]

        need_compile = kernel['need_compile']
        compile_cmd = 'exec ' + kernel['cmd'].replace('\n', '')

        kernel_name = kernel['kernel_name']

        if isinstance(kernel['current_node'], Binary2Float):
            p_bound = 1
        else:
            p_bound = len(kernel['current_node'].children)

            if kernel.get('par_idx', None) is not None:
                p_bound += len(set(kernel['par_idx']).intersection(set(compiler_params.allow_random_params_idx)))

        if (not key.startswith('fw')):
            # first element corresponds to dL_dcol
            input_dims = ['3']
        else:
            input_dims = []
        
        for input_ls in kernel['input']:
            input_dims.append(str(len(input_ls)))
            
        if len(input_dims) > 0:
            input_dims_str = ' --input_dims ' + ','.join(input_dims) + ' '
            input_vals = ['1'] * len(input_dims)
            input_dims_str += ' --input_vals ' + ','.join(input_vals) + ' '
        else:
            input_dims_str = ''
        
        if kernel['lib'].startswith('consumer') or kernel.get('includes_producer', False):
            producer_args = ' --producer_pad 1 '
        else:
            producer_args = ''

        profile_cmd = f"""python {profile_dir}/Halide_profile_generator.py --lib_name {kernel_name} --niters 100 --p_bound {p_bound} --nparams {nparams} --target_dir {outdir} {input_dims_str} {sigma_str} {producer_args}"""

        success = True
        if need_compile:
            
            print(compile_cmd)
            
            # compile with a 1min time limit
            
            with subprocess.Popen(cd_start + compile_cmd + cd_end, shell=True) as process:
                try:
                    stdout, stderr = process.communicate(timeout=compile_time_limit)
                except subprocess.TimeoutExpired:
                    print('Process %s was killed by timeout.' % kernel['lib'])
                    success = False
                finally:
                    process.terminate()
                    process.kill()
            
            #try:
            #    subprocess.run(cd_start + compile_cmd + cd_end, timeout=compile_time_limit, shell=True)
            #except:
            #    print("kernel %s time out or fails" % kernel['lib'])
            #    success = False

        if success:
            profile_timing = float(subprocess.check_output(profile_cmd, shell=True))
        else:
            profile_timing = np.inf

        kernel['timing'] = profile_timing
        
    def compile_per_kernel(keys_ls, copy_only=False):
        nonlocal kernel_idx_base
        for key in keys_ls:
            if key not in all_kernels_timing.keys():
                if copy_only:
                    all_kernels[key]['timing'] = np.inf
                else:
                    skip_runtime = os.path.exists(os.path.join(outdir, 'runtime.a'))
                    to_source_and_compile(outdir, [all_kernels[key]], compiler_params, kernel_idx_base=kernel_idx_base, skip_runtime=skip_runtime)
                    profile_individual(key)
            else:
                all_kernels[key]['timing'] = all_kernels_timing[key]['timing']
            kernel_idx_base += 1
            
    def log_timing():
        
        nonlocal all_kernels_timing, T0
        
        T1 = time.time()
        
        old_timing = all_kernels_timing
    
        all_kernels_timing = {key: {'timing': all_kernels[key].get('timing', np.inf), 'cmd': all_kernels[key].get('cmd', '')} for key in all_kernels.keys()}

        if str(old_timing) != str(all_kernels_timing):
            # some kernels are newly profiled:

            pickle.dump(all_kernels_timing, open(all_kernels_timing_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

            # Move all profile kernels to a subfolder, in case we need to debug
            profile_subfolder = os.path.join(outdir, 'profile')
            if not os.path.isdir(profile_subfolder):
                os.makedirs(profile_subfolder)

            for file in os.listdir(outdir):
                if file.startswith('compiler_problem'):
                    os.rename(os.path.join(outdir, file), os.path.join(profile_subfolder, file))
                    
            autoscheduler_timing_file = os.path.join(outdir, 'autoscheduler_timing.txt')
            open(autoscheduler_timing_file, 'a+').write('Took %f s\n' % (T1 - T0))

    # compile kernels in a strategy that allows early termination

    kernel_idx_base = 0
    
    # first pass, check whether it's needed to use cont or not
    keys = ['cont', 'bw__False', 'bw__True']
    compile_per_kernel(keys)

    # allow some noise
    if (all_kernels['cont']['timing'] + all_kernels['bw__True']['timing']) > 1.2 * all_kernels['bw__False']['timing'] and len(raymarching_ids) == 0:
        # always search for producer if raymarching loop is present
        log_timing()
        # reach early termination criterion
        return {'seperate_cont': False,
                'seperate_neighbor': ''}

    # second pass, check which sepeate_neighbor mode we should use to explore producers
    keys = [key for key in all_kernels.keys() if key.startswith('bw') and key not in ['bw__False', 'bw__True'] and 'combined' not in key]
    compile_per_kernel(keys)

    # do not allow noise because this always excludes a nontrivial runtime for combine
    if all_kernels['bw_per_neighbor_True']['timing'] * 4 < all_kernels['bw__True']['timing']:
        copy_neighbor_combined = False
    else:
        copy_neighbor_combined = True
    compile_per_kernel(['bw_per_neighbor_True_combined'], copy_only=copy_neighbor_combined)
    
    if all_kernels['bw_per_filter_True']['timing'] * 2 < all_kernels['bw__True']['timing']:
        copy_filter_combined = False
    else:
        copy_filter_combined = True
    compile_per_kernel(['bw_per_filter_True_combined'], copy_only=copy_filter_combined)
    
    current_best_T = 1e8
    current_best_configure = None
    # assuming using the same producer or None, so there's no need to compute producer runtime at the current point
    for seperate_neighbor in ['per_neighbor', 'per_filter', '']:
        current_configure = {'seperate_cont': not disable_cont,
                             'seperate_neighbor': seperate_neighbor}

        current_T = bw_schedule_runtime_estimator(all_kernels, **current_configure)

        if current_T < current_best_T:
            current_best_T = current_T
            current_best_configure = current_configure

    # find the best schedule without producer
    best_configure = None
    best_T = 1e8

    # assuming using the same producer or None, so there's no need to compute producer runtime at the current point
    for seperate_cont in [False, True]:
        for seperate_neighbor in ['per_neighbor', 'per_filter', '']:
            current_configure = {'seperate_cont': seperate_cont,
                                 'seperate_neighbor': seperate_neighbor}

            current_T = bw_schedule_runtime_estimator(all_kernels, **current_configure)

            if current_T < best_T or best_configure is None:
                best_configure = current_configure
                best_T = current_T

    if disable_cont:
        best_configure['seperate_cont'] = False
        
    if len(init_input) > 0:
        best_configure['logged_trace'] = init_input[0]

    print('current best configure:', best_T)
    print(best_configure)

    producer_target_bw_type = best_configure['seperate_neighbor']
    
    if producer_target_bw_type != '' or len(raymarching_ids) > 0:

        best_consumer_T = best_T

        if producer_target_bw_type == 'per_neighbor':
            scale = 4
        elif producer_target_bw_type == 'per_filter':
            scale = 2
        else:
            scale = 1

        if producer_target_bw_type != '':
            producer_combine_name = 'bw_%s_True_combined' % (producer_target_bw_type)
            if 'timing' not in all_kernels[producer_combine_name].keys():
                compile_per_kernel([producer_combine_name])

        additional_cont = 0
        if best_configure is not None:
            if best_configure.get('seperate_cont', False):
                # This is not accurate for different producer logs, but we'll use it as a proxy
                additional_cont = all_kernels['cont']['timing']


        # assuming producer is roughly similar, exclude it from estimate at the current point
        if producer_target_bw_type == 'per_neighbor':
            def estimate_producer_runtime(raw):
                return 4 * raw + all_kernels[producer_combine_name]['timing'] + additional_cont
        elif producer_target_bw_type == 'per_filter':
            def estimate_producer_runtime(raw):
                return 2 * raw + all_kernels[producer_combine_name]['timing'] + additional_cont
        else:
            def estimate_producer_runtime(raw):
                return raw + additional_cont

        for producer_idx in range(len(log_schedule)):
            generate_kw = {'logged_trace': log_schedule[producer_idx]['log_idx'],
                           'kernel_type': 'bw',
                           'bw_type': producer_target_bw_type,
                           'seperate_cont': not disable_cont}

            current_node, _, _, _, _ = get_output_and_children(derivs_info, generate_kw)

            producer_input = []
            for node_idx in log_schedule[producer_idx]['log_idx']:
                producer_input.append(log_node_list[node_idx])

            lib_name = 'consumer_bw_%s_%d' % (producer_target_bw_type, producer_idx)

            all_kernels[lib_name] = \
            {'lib': lib_name,
             'current_node': current_node,
             'input': [producer_input],
             'includes_producer': True}

            compile_per_kernel([lib_name])

            # early termination, global I/O becomes the bottleneck now, should not try more producer logs
            # allow some noise in evaluation
            estimate_T = estimate_producer_runtime(all_kernels[lib_name]['timing'])
            if best_consumer_T * 1.2 < estimate_T:
                break

            if estimate_T < best_consumer_T:
                best_consumer_T = estimate_T

        max_producer_idx = producer_idx

        if max_producer_idx > 0:
            # use a linear model to estimate producer runtime
            keys = ['fw', 'fw_producer_%d' % max_producer_idx]
            all_kernels['fw'] = {'lib': 'fw',
                                 'current_node': info['f'],
                                 'input': []}
            all_kernels[keys[-1]] = {'lib': keys[-1],
                                     'current_node': Compound(info['f'].children + producer_input),
                                     'input': []}
            compile_per_kernel(keys)

            best_producer_bw_T = 1e8
            best_producer_idx = None

            for producer_idx in range(max_producer_idx):
                lib_name = 'fw_producer_%d' % producer_idx
                # linear estimate
                estimate_timing = all_kernels['fw']['timing'] + \
                                  (len(log_schedule[producer_idx]['log_idx']) / len(log_schedule[max_producer_idx]['log_idx'])) * \
                                  (all_kernels[keys[-1]]['timing'] - all_kernels['fw']['timing'])
                all_kernels[lib_name] = {'timing': estimate_timing}

                consumer_name = 'consumer_bw_%s_%d' % (producer_target_bw_type, producer_idx)
                estimate_producer_bw_T = estimate_timing + estimate_producer_runtime(all_kernels[consumer_name]['timing'])

                if estimate_producer_bw_T < best_producer_bw_T:
                    best_producer_bw_T = estimate_producer_bw_T
                    best_producer_idx = producer_idx

                if producer_idx == 0 and len(init_input) > 0:
                    # if the schedules before searching producer log already uses a minimum producer, should add the producer runtime now
                    best_T += estimate_timing

            if not disable_cont:
                # only compile cont with best possible producer for bw discont
                cont_name = 'cont_producer_%s_%d' % (producer_target_bw_type, best_producer_idx)
                cont_node, _, _, _, _ = get_output_and_children(derivs_info, {'kernel_type': 'cont'})

                producer_input = []
                for node_idx in log_schedule[best_producer_idx]['log_idx']:
                    producer_input.append(log_node_list[node_idx])

                all_kernels[cont_name] = {'lib': cont_name,
                                          'current_node': cont_node,
                                          'input': [producer_input]}
                compile_per_kernel([cont_name])

                if all_kernels['cont']['timing'] < all_kernels[cont_name]['timing']:
                    # Use less log gives faster cont
                    # this uses strictly a subset of the final producer log
                    # so it can share the same producer as other bw kernels
                    cont_min_kernel = 'cont'
                else:
                    cont_min_kernel = cont_name
            else:
                cont_min_kernel = 'cont'

            if best_producer_bw_T + all_kernels[cont_min_kernel]['timing'] - additional_cont < best_T:
                best_configure = {'seperate_cont': not disable_cont,
                                  'seperate_neighbor': producer_target_bw_type,
                                  'logged_trace': producer_input}

                if cont_min_kernel == 'cont' and len(all_kernels['cont']['input']):
                    # should be the first log_schedule
                    best_configure['cont_logged_trace'] = np.arange(len(log_schedule[0]['log_idx'])).tolist()
            
    log_timing()
    
    return best_configure
    
def to_source_and_compile(outdir, schedule, compiler_params, f_node=None, compiler_modes=None, skip_runtime=False, kernel_idx_base=0):
    source, wrapper_str = to_source(schedule, compiler_params, f_node)
    
    if outdir == '':
        outdir = '../apps'
        
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
        
    filename = COMPILER_PROBLEM
    
    if compiler_params.backend == 'hl':
        if compiler_params.gradient_mode == 'AD':
            filename += '_Ad'
        filename += '.cpp'
    elif compiler_params.backend == 'glsl':
        filename += '.frag'
    elif compiler_params.backend == 'torch':
        filename += '_torch.py'
    else:
        filename += '.py'
          
    T1 = time.time()
    filename_full = os.path.join(outdir, filename)
    
    if os.path.exists(filename_full):
        old_source = open(filename_full).read()
    else:
        old_source = ''
        
    if old_source == source:
        update_source = False
        source_time = os.path.getmtime(filename_full)
    else:
        update_source = True
        source_time = None
        open(filename_full, 'w').write(source)
        
    if compiler_params.backend == 'hl':
        if not compiler_params.individual_kernels:
            dict_pickle_file = os.path.join(outdir, COMPILER_PROBLEM + '_buffer.pkl')
            compiler_params.dict_pickle_file = os.path.abspath(dict_pickle_file)
            
            with open(dict_pickle_file, 'wb') as f:
                pickle.dump([compiler_params.fw_info, compiler_params.bw_info, compiler_params.producer_info], f, pickle.HIGHEST_PROTOCOL)
                
        if compiler_modes is None:
            if compiler_params.individual_kernels:
                if len(compiler_params.allow_random_params_idx):
                    compiler_modes = ['bw_random_par']
                else:
                    compiler_modes = ['bw']
            else:
                compiler_modes = all_compiler_modes
                
        Halide_lib_dir = os.path.abspath('../')
        
        shutil.copyfile(os.path.join('../util', HALIDE_UTIL_FILE), os.path.join(outdir, HALIDE_UTIL_FILE))
        shutil.copyfile(os.path.join(Halide_lib_dir, HALIDE_LIB_FILE), os.path.join(outdir, HALIDE_LIB_FILE))
        
        halide_lib_o_file = os.path.join(Halide_lib_dir, HALIDE_LIB_FILE.replace('.h', '.o'))
        
        cwd = os.getcwd()
        
        
        lib_str = ''
                
        cmd = f"""
        cd {outdir}; 
        g++ {filename_full} $HALIDE_TOOL_PATH/GenGen.cpp -g -std=c++11 -fno-rtti -I $HALIDE_INCLUDE_PATH -L $HALIDE_LIB_PATH -lHalide -lpthread -ldl -Wl,-rpath,${{HALIDE_LIB_PATH}} -o {COMPILER_PROBLEM};
        """
        
        def check_need_compile(name):
            if not update_source:
                success = True
                for suffix in ['.a', '.h']:
                    lib_filename = os.path.join(outdir, name + suffix)
                    if not os.path.exists(lib_filename):
                        # lib file does not exist
                        success = False
                        break
                    if os.path.getmtime(lib_filename) < source_time:
                        # lib file is compiled before source file last updated
                        success = False
                        break
            else:
                success = False
            return not success
        
        for mode in all_kernel_modes:
            kernel_suffix, args_str, _ = compiler_params.get_wrapper_component(mode)
            
            allow_replace = False
            if mode == 'choose_u_pl':
                replace_mode = 'offset'
                replace_suffix, _, _ = compiler_params.get_wrapper_component(replace_mode)
                
                if 'bw' + replace_suffix in compiler_modes:
                    allow_replace = True
                    
            if 'bw' + kernel_suffix in compiler_modes:
                extra_args = ''
                
                if kernel_suffix == '_random_par':
                    extra_args += ' use_random_par=true '
                if kernel_suffix in ['_per_pixel_offset', '_choose_u_pl']:
                    extra_args += ' use_per_pixel_offset=true '
                if kernel_suffix == '_denum_only':
                    extra_args += ' denum_only=true '
                    
                for idx in wrapper_str[mode]['kernel_indices'].keys():
                    kernel_name = f"""{COMPILER_PROBLEM}_kernel{idx + kernel_idx_base}{kernel_suffix}"""
                    lib_str += f""" {kernel_name}.a """
                    
                    
                    
                    current_extra_args = extra_args + wrapper_str[mode]['kernel_indices'][idx]
                    
                    need_compile = check_need_compile(kernel_name)
                    
                    if kernel_name in wrapper_str[mode]['replace_kernel']:
                        replace_name = wrapper_str[mode]['replace_kernel'][kernel_name]
                        
                        if allow_replace or (not check_need_compile(replace_name)):
                            need_compile = False
                            wrapper_str[mode]['include_lib'] = wrapper_str[mode]['include_lib'].replace(kernel_name, replace_name)
                            wrapper_str[mode]['kernel_call'] = wrapper_str[mode]['kernel_call'].replace(kernel_name, replace_name)
                    
                    current_cmd = f"""
                ./{COMPILER_PROBLEM} -o . -g shader -f {kernel_name} -e static_library,h target=host-cuda-no_runtime auto_schedule=false {current_extra_args};"""
                    
                    need_compile = check_need_compile(kernel_name)
                    
                    if compiler_params.individual_kernels:
                        schedule[idx]['kernel_name'] = kernel_name
                        schedule[idx]['cmd'] = current_cmd
                        schedule[idx]['need_compile'] = need_compile
                    elif need_compile:
                        cmd += current_cmd
                        
        if not skip_runtime:
            cmd += f"""./{COMPILER_PROBLEM} -o . -r runtime target=host-cuda auto_schedule=false;"""
                    
        if not compiler_params.individual_kernels:
            # no need to compile non-bw kernels if we're only profiling individual kernels
            for mode in compiler_modes:
                if not mode.startswith('bw'):

                    kernel_name = f"""{COMPILER_PROBLEM}_{mode}"""
                    lib_str += f""" {kernel_name}.a """

                    if check_need_compile(kernel_name):
                        extra_args = ''
                        if 'fw' in mode:
                            extra_args += ' is_fw=true '
                        if 'random_par' in mode:
                            extra_args += ' use_random_par=true '
                        if 'per_pixel_offset' in mode or 'choose_u_pl' in mode:
                            extra_args += ' use_per_pixel_offset=true '
                        if 'choose_u_pl' in mode:
                            extra_args += ' use_choose_u_pl=true '
                        if 'FD' in mode:
                            extra_args += ' is_FD=true '
                        if 'add_to' in mode:
                            extra_args += ' add_to=true '
                        if 'base_only' in mode:
                            extra_args += ' output_base_only=true '
                        if 'prune_updates' in mode:
                            extra_args += ' prune_optional_update=true '

                        cmd += f"""
            ./{COMPILER_PROBLEM} -o . -g shader -f {COMPILER_PROBLEM}_{mode} -e static_library,h target=host-cuda-no_runtime auto_schedule=false kernel_idx=0 {extra_args};
                    """
            cmd += f"""
        g++ {HL_PYBIND_TEMPLATE_FILE} -std=c++11 -I $HALIDE_INCLUDE_PATH -I $HALIDE_TOOL_PATH -L $HALIDE_LIB_PATH {lib_str} runtime.a {halide_lib_o_file} `libpng-config --cflags --ldflags` -ljpeg -ldl -lpthread -fPIC $(python3 -m pybind11 --includes) -Wall -shared -o compiler_problem$(python3-config --extension-suffix) -lpthread -lHalide -I ./;
        cd {cwd}
        """
            
            get_hl_wrapper(outdir, wrapper_str, compiler_params, compiler_modes=compiler_modes)
            
        print('Compiling Halide kernel:')
        print(cmd)
        
        os.system(cmd)    

def check(e, compiler_params, ndims=-1, outdir='', compiler_modes=None, scalar_loss=None):
    """
    Convert Expr to finalized source code and (by default) run to check the correctness or performance of the output code.
    """
    ans = {}

    if ndims <= 0:
        try:
            arg_array = locate_argument_array(e)
            arg_array_ndims = arg_array.ndims
        except NoArgumentArray:
            arg_array_ndims = 1

    T0 = time.time()
    
    if scalar_loss is not None and compiler_params.backend in ['hl', 'tf', 'torch']:
        # we don't want processing scalar_loss makes original compute node affected
        # plus, scalar loss should only include simple arithmatic on parameters (but not uv coordinates), so compute graph should not be large
        scalar_loss = copy.deepcopy(scalar_loss)
        scalar_loss = remove_redundant_exprs(scalar_loss)
        scalar_loss = repeated_simplify_inplace(scalar_loss)
        
        cp = CompilerParams()
        cp.gradient_mode = 'AD'
        scalar_derivs = backprop(scalar_loss, cp)
        
        scalar_derivs = Compound([scalar_loss] + scalar_derivs.children)
        
        scalar_derivs.calc_parents()
        scalar_derivs = remove_redundant_exprs(scalar_derivs)
        scalar_derivs = repeated_simplify_inplace(scalar_derivs)
        
        cp.backend = 'np'
        cp.mode = MODE_SIDE_EFFECTS
        scalar_derivs.root = True
        scalar_src = to_source_nonfinal(scalar_derivs, cp)
        
        scalar_declare = f"""
import sys; sys.path += ['util']
from np_util import *
import numpy as np

def f(X, scalar_loss_scale):
        """
        
        scalar_return = f"""
return {scalar_derivs.to_source(cp.as_mode(MODE_VARNAME))}[0], np.array({scalar_derivs.to_source(cp.as_mode(MODE_VARNAME))}[1:])
        """
        
        scalar_src = scalar_declare + indent(scalar_src + scalar_return)
        
        scalar_src_file = os.path.join(outdir, 'compiler_problem_scalar.py')
        open(scalar_src_file, 'w').write(scalar_src)
    
    info = process_fw_node(e, compiler_params)
    

    if compiler_params.compute_g:
        
        if compiler_params.gradient_mode == 'AD':
            scheduler_and_buffer = bw_schedule_builder(info['f'], info['g'], info['g_aux'], info['g_AD'], compiler_params,
                                                  seperate_producer = False,
                                                  logged_trace = [],
                                                  seperate_cont = False,
                                                  seperate_neighbor = '')
            
        elif compiler_params.backend in ['tf', 'torch'] or (not compiler_params.autoscheduler):
            
            log_list = info['producer_log_node']
            
            if compiler_params.backend == 'hl':
                schedule_key = 'inline'
                
                # use heuristic to see if checkpointing is needed
                # only checkpoint if raymarching loop is found
                
                log_list = []
                
                for idx in range(len(info['producer_log_node'])):
                    node = info['producer_log_node'][idx]
                    
                    if isinstance(node, GetItem) and isinstance(node.array, RaymarchingLoop):
                        log_list.append(node)
                        
                if len(log_list) > 0:
                    # heuristic, try to avoid CUDA error when combining the 4 per_neighbor tensor with too large dimension
                    if len(compiler_params.discont_params_idx) > 30:
                        schedule_key = 'seperate_producer_cont_filter'
                    else:
                        schedule_key = 'seperate_producer_cont_neighbor'
                        
            elif compiler_params.backend == 'glsl':
                schedule_key = 'inline'
            elif compiler_params.backend in ['tf', 'torch']:
                schedule_key = 'seperate_producer'
            else:
                schedule_key = 'inline'
                

            # inline everything
            if schedule_key == 'inline':
                scheduler_and_buffer = bw_schedule_builder(info['f'], info['g'], info['g_aux'], info['g_AD'], compiler_params,
                                                  seperate_producer = False,
                                                  logged_trace = [],
                                                  seperate_cont = False,
                                                  seperate_neighbor = '')
            elif schedule_key == 'seperate_producer':
                scheduler_and_buffer = bw_schedule_builder(info['f'], info['g'], info['g_aux'], info['g_AD'], compiler_params,
                                                  seperate_producer = True,
                                                  logged_trace = log_list,
                                                  seperate_cont = False,
                                                  seperate_neighbor = '')
            elif schedule_key == 'seperate_cont':
                scheduler_and_buffer = bw_schedule_builder(info['f'], info['g'], info['g_aux'], info['g_AD'], compiler_params,
                                                  seperate_producer = False,
                                                  logged_trace = [],
                                                  seperate_cont = True,
                                                  seperate_neighbor = '')
            elif schedule_key == 'seperate_producer_cont':
                scheduler_and_buffer = bw_schedule_builder(info['f'], info['g'], info['g_aux'], info['g_AD'], compiler_params,
                                                  seperate_producer = True,
                                                  logged_trace = log_list,
                                                  seperate_cont = True,
                                                  seperate_neighbor = '')
            elif schedule_key == 'seperate_neighbor':
                scheduler_and_buffer = bw_schedule_builder(info['f'], info['g'], info['g_aux'], info['g_AD'], compiler_params,
                                                  seperate_producer = False,
                                                  logged_trace = log_list,
                                                  seperate_cont = False,
                                                  seperate_neighbor = 'per_neighbor')
            elif schedule_key == 'seperate_filter':
                scheduler_and_buffer = bw_schedule_builder(info['f'], info['g'], info['g_aux'], info['g_AD'], compiler_params,
                                                  seperate_producer = False,
                                                  logged_trace = log_list,
                                                  seperate_cont = False,
                                                  seperate_neighbor = 'per_filter')
            elif schedule_key == 'seperate_producer_neighbor':
                scheduler_and_buffer = bw_schedule_builder(info['f'], info['g'], info['g_aux'], info['g_AD'], compiler_params,
                                                  seperate_producer = True,
                                                  logged_trace = log_list,
                                                  seperate_cont = False,
                                                  seperate_neighbor = 'per_neighbor')
            elif schedule_key == 'seperate_producer_filter':
                scheduler_and_buffer = bw_schedule_builder(info['f'], info['g'], info['g_aux'], info['g_AD'], compiler_params,
                                                  seperate_producer = True,
                                                  logged_trace = log_list,
                                                  seperate_cont = False,
                                                  seperate_neighbor = 'per_filter')
            elif schedule_key == 'seperate_cont_neighbor':
                scheduler_and_buffer = bw_schedule_builder(info['f'], info['g'], info['g_aux'], info['g_AD'], compiler_params,
                                                  seperate_producer = False,
                                                  logged_trace = [],
                                                  seperate_cont = True,
                                                  seperate_neighbor = 'per_neighbor')
            elif schedule_key == 'seperate_cont_filter':
                scheduler_and_buffer = bw_schedule_builder(info['f'], info['g'], info['g_aux'], info['g_AD'], compiler_params,
                                                  seperate_producer = False,
                                                  logged_trace = [],
                                                  seperate_cont = True,
                                                  seperate_neighbor = 'per_filter')
            elif schedule_key == 'seperate_producer_cont_neighbor':
                scheduler_and_buffer = bw_schedule_builder(info['f'], info['g'], info['g_aux'], info['g_AD'], compiler_params,
                                                  seperate_producer = True,
                                                  logged_trace = log_list,
                                                  seperate_cont = True,
                                                  seperate_neighbor = 'per_neighbor')
            elif schedule_key == 'seperate_producer_cont_filter':
                scheduler_and_buffer = bw_schedule_builder(info['f'], info['g'], info['g_aux'], info['g_AD'], compiler_params,
                                                  seperate_producer = True,
                                                  logged_trace = log_list,
                                                  seperate_cont = True,
                                                  seperate_neighbor = 'per_filter')
        else:
            best_configure = profile_kernels(info, compiler_params, outdir=outdir)

            scheduler_and_buffer = bw_schedule_builder(info['f'], info['g'], info['g_aux'], info['g_AD'], compiler_params,
                                                       **best_configure)
    else:
        scheduler_and_buffer = [[], []]
        
    compiler_params.individual_kernels = False
    to_source_and_compile(outdir, scheduler_and_buffer, compiler_params, f_node=info['f'], compiler_modes=compiler_modes)

    return ans

class NoArgumentArray(Exception):
    pass

def is_unknown_array(node):
    """
    Returns whether the given Expr is an ArgumentArray for the unknown array argument of the solver.
    """
    return isinstance(node, ArgumentArray) and node.name == DEFAULT_ARGUMENT_ARRAY_NAME

def locate_argument_array(e):
    """
    Return ArgumentArray instance for unknown array given objective function Expr, or raise exception if not found.
    """
    all_nodes = e.all_nodes()
    for node in all_nodes:
        if is_unknown_array(node):
            return node
    raise NoArgumentArray
    
def locate_argument_uv(e):
    """
    Return ArgumentScalar instances for uv coordinates given objective function Expr, or raise exception if not found.
    """
    ans = [None, None]
    for node in e.all_nodes():
        if isinstance(node, ArgumentScalar):
            if node.name == DEFAULT_ARGUMENT_SCALAR_U_NAME:
                ans[0] = node
            if node.name == DEFAULT_ARGUMENT_SCALAR_V_NAME:
                ans[1] = node
                
            if ans[0] is not None and ans[1] is not None:
                return ans
    return ans

def locate_RaymarchingLoop(e):
    """
    return list of nodes that are instances of RaymarchingLoop
    """
    return [node for node in e.all_nodes() if isinstance(node, RaymarchingLoop)]

def to_source_nonfinal(e, compiler_params, log_node_list=[], read_node_dict=None):
    """
    Convert Expr to non-finalized source code.
    Memoize tape if necessary.
    """
    s = e.to_source(compiler_params)
    
    # no need to
    # 1. the nodes are already processed by remove_redundant_expr, there shouldn't be any duplicates
    # 2. both tf and hl builds a compute graph and compiles in a separte pass, duplicate code only rewrite the compute graph, will NOT hurt performance of the finalkernel
    #s = eliminate_duplicates(s)
    
    s = eliminate_whitespace(s)
    
    log_s = ''
    read_s = ''
    var_count = 0
    
    if compiler_params.backend == 'hl':
        func_name = 'generate'
    elif compiler_params.backend in ['tf', 'np', 'torch']:
        func_name = 'f'
    elif compiler_params.backend == 'glsl':
        func_name = 'mainImage'
    else:
        raise "Unknown backend"

    # TODO: figure out hot wo memoize tape for Halide
    if compiler_params.backend in ['tf', 'np', 'torch']:
        
        if read_node_dict is not None and len(read_node_dict):
            # if reading from log_intermediate, assuming NOT writing to it
            assert len(log_node_list) == 0
            
            read_node_list = sum([val[1] for val in read_node_dict.values()], [])
            
            read_list = [compiler_params.var_list[compiler_params.name_to_order[val]] for val in read_node_list]
            
            for i in range(len(read_list)):
                var_name = read_list[i]
                read_s += '%s = f_log_intermediate[%d]\n' % (var_name, var_count)
                var_count += 1
            
        else:
            if log_node_list is None:
                log_list = []
            else:
                log_list = [compiler_params.var_list[compiler_params.name_to_order[val]] for val in log_node_list]

            for i in range(len(log_list)):
                var_name = log_list[i]
                log_s += '%s_log_intermediate[%d] = %s' %(func_name, var_count, var_name)
                log_s += '\n'

                var_count += 1
    elif compiler_params.backend == 'glsl':
        assert len(log_node_list) == 0
        assert read_node_dict is None or len(read_node_dict) == 0
    else:
        
        # any additional output should go to gradient() directly
        assert len(log_node_list) == 0
        
        if read_node_dict is not None and len(read_node_dict):
            
            for buffer_idx in read_node_dict.keys():
                
                base_idx = read_node_dict[buffer_idx][0]
                
                for i in range(len(read_node_dict[buffer_idx][1])):
                    
                    if read_node_dict[buffer_idx][2][i] == BOOL_TYPE:
                        cast_str = 'Halide::cast<bool> '
                    else:
                        cast_str = ''
                    
                    var_name = compiler_params.var_list[compiler_params.name_to_order[read_node_dict[buffer_idx][1][i]]]
                    read_s += f"""
Expr {var_name};

if (use_per_pixel_offset) {{
    {var_name} = {cast_str} select(Ou == 0 && Ov == 0, (*output{buffer_idx})(u, v, {i + base_idx}),
                                   Ou == -1 && Ov == 0, (*output{buffer_idx + compiler_params.nkernels})(u, v, {i}),
                                   Ou == 1 && Ov == 0, (*output{buffer_idx + 2 * compiler_params.nkernels})(u, v, {i}),
                                   Ou == 0 && Ov == -1, (*output{buffer_idx + 3 * compiler_params.nkernels})(u, v, {i}),
                                   (*output{buffer_idx + 4 * compiler_params.nkernels})(u, v, {i}));
}} else {{
    {var_name} = {cast_str} ((*output{buffer_idx})(u + Ou, v + Ov, {i + base_idx}));
}}
"""
            
    s = read_s + '\n' + s + '\n' + log_s
    
    return s

def repeated_simplify_inplace(e, max_simplifications=5, keep_ids=set(), replaced_nodes={}):
    max_simplifications = 5
    n_simplify = 0
    e_current = ''
    while n_simplify < max_simplifications:
        e_before = e_current
        t1 = time.time()
        e = e.simplify(keep_ids=keep_ids, replaced_nodes=replaced_nodes)
        e = remove_redundant_exprs(e, keep_ids=keep_ids, replaced_nodes=replaced_nodes)
        print('simplify, round', n_simplify, time.time() - t1)
        n_simplify += 1

    return e

def backprop(e, compiler_params, check_discont=False):
    
    # e is already simplified
    
    try:
        arg_array = locate_argument_array(e)
    except:
        arg_array = None
        
        print('Error! cannot find tunable parameters to take gradients wrt!')
        
    if arg_array is None:
        return e, None, None
    
    if isinstance(e, Compound) or e.ndims > 0:
        # multi channel output (e.g. RGB color)
        dL = ArgumentArray(DL_DCOL_ARRAY, ndims=e.ndims)
    else:
        dL = 1
        
    compiler_params.pix_idx_pl = ArgumentScalar('pix_idx')
    orig_min_discont_denum = 1e8
    compiler_params.min_discont_denum = orig_min_discont_denum
    
    loss_name = 'L'
    
    if compiler_params.gradient_mode == 'AD':
        loss_name += '_AD'
    
    compiler_params.backprop_source = loss_name
    
    e.dL_dself[loss_name] = dL

    all_nodes = e.all_nodes_dfs()

    e.is_discont_to_output = False
    e.calc_parents()
    
    if compiler_params.gradient_mode == 'ours' or check_discont:
        # top-down, count parameters that are discontinuous wrt output
        for node in all_nodes[::-1]:
            node.propagate_discont(False)
            
        if compiler_params.allow_raymarching_random:
            compiler_params.allow_random_params_idx = np.where(arg_array.is_discont_to_output)[0]
        else:
            compiler_params.allow_random_params_idx = np.where(np.logical_and(arg_array.is_discont_to_output,
                                                                     np.logical_not(arg_array.is_dependency_to_raymarching)))[0]
        
        compiler_params.discont_params_idx = np.where(arg_array.is_discont_to_output)[0]
        compiler_params.cont_params_idx = np.where(np.logical_not(arg_array.is_discont_to_output))[0]
        
    if compiler_params.gradient_mode == 'ours':

        # bottom-up, collect comparisons that are continuous wrt parameters
        for node in all_nodes:
            node.propagate_discont(True)
            node.propagate_params_dependency(compiler_params)
            
        # seperate parameters into 2 clusters based on their position in f
        if True:
            all_nodes_idx = [id(node) for node in all_nodes]

            info = {}
            clusters = []
            
            def add_cluster(cluster, min_occ, max_occ, indices):
                added = False
                if min_occ >= cluster['min'] and min_occ <= cluster['max']:
                    added = True
                    cluster['max'] = max(cluster['max'], max_occ)
                if max_occ >= cluster['min'] and max_occ <= cluster['max']:
                    added = True
                    cluster['min'] = min(cluster['min'], min_occ)
                if min_occ <= cluster['min'] and max_occ >= cluster['max']:
                    added = True
                    cluster['min'] = min_occ
                    cluster['max'] = max_occ
                if added:
                    cluster['indices'] += indices
                return added
            
            for par in arg_array.parents:
                assert isinstance(par, GetItem)
                idx = par.index.value
                
                # continuous parameters should be in a seperated kernel, can ignore
                if idx in compiler_params.cont_params_idx:
                    continue
                    
                # in case our remove_redundant is not successful
                if idx in info.keys():
                    raise

                occurences = []
                for node in par.parents:
                    occurences.append(all_nodes_idx.index(id(node)))
                info[idx] = [occurences, par]
                
                min_occ = min(occurences)
                max_occ = max(occurences)

                added = False
                
                for cluster in clusters:
                    added = add_cluster(cluster, min_occ, max_occ, [idx])
                    if added:
                        break
                        
                if not added:
                    clusters.append({'indices': [idx], 'min': min_occ, 'max': max_occ})
                    
            # recursively merge clusters if their ranges are overlapping
            converged = False
            while not converged:
                converged = True
                for c_idx in range(len(clusters) - 1, -1, -1):
                    merged = False
                    for alt_idx in range(c_idx):
                        merged = add_cluster(clusters[alt_idx], 
                                             clusters[c_idx]['min'], 
                                             clusters[c_idx]['max'], 
                                             clusters[c_idx]['indices'])
                        if merged:
                            break
                    if merged:
                        del clusters[c_idx]
                        converged = False
                        
            clusters = sorted(clusters, key = lambda i: i['min'])
            
            nargs_to_cut = len(info)
            
            first_half_params = []
            
            for cluster in clusters:
                if len(first_half_params) + len(cluster['indices']) < nargs_to_cut // 2:
                    first_half_params += cluster['indices']
                else:
                    if abs(len(first_half_params) - nargs_to_cut // 2) > abs(len(first_half_params) + len(cluster['indices']) - nargs_to_cut // 2):
                        first_half_params += cluster['indices']
                    break
                    
            second_half_params = list(set(compiler_params.discont_params_idx).difference(set(first_half_params)))
            
            half_program_parents = []
            half_program_parents_ids = []
            for params in [first_half_params, second_half_params]:
                seen = set()
                all_parents = []
                for idx in params:
                    all_parents += info[idx][1].all_parents(seen)
                half_program_parents.append(all_parents)
                half_program_parents_ids.append([id(node) for node in all_parents])

            half_program_cut_nodes = []
            for idx in range(2):
                current_half = idx
                other_half = (idx + 1) % 2

                cut_nodes_candidates = []
                cut_nodes_seen_ids = set()

                def visit_cut_nodes(node):
                    if id(node) in cut_nodes_seen_ids:
                        return
                    cut_nodes_seen_ids.add(id(node))

                    if id(node) not in half_program_parents_ids[current_half] and id(node) in half_program_parents_ids[other_half]:
                        cut_nodes_candidates.append(node)
                        return

                    for child in node.children:
                        if isinstance(child, Expr):
                            visit_cut_nodes(child)

                for node in half_program_parents[current_half]:
                    visit_cut_nodes(node)

                half_program_cut_nodes.append(cut_nodes_candidates)
                
            ncut_first_half = sum([max(node.ndims, 1) for node in half_program_cut_nodes[0]])
            ncut_second_half = sum([max(node.ndims, 1) for node in half_program_cut_nodes[1]])
            
            if ncut_first_half < ncut_second_half:
                # first half as producer
                compiler_params.first_half_params = sorted(first_half_params)
                compiler_params.second_half_params = sorted(second_half_params)
                compiler_params.half_cut_nodes = half_program_cut_nodes[0]
            else:
                # second half as producer
                compiler_params.first_half_params = second_half_params
                compiler_params.second_half_params = first_half_params
                compiler_params.half_cut_nodes = half_program_cut_nodes[1]
            
            
            if False:
                # find parents of first half
                first_half_parents = []
                first_half_ids = set()
                def visit_first_half(node):
                    if id(node) in first_half_ids:
                        return
                    first_half_ids.add(id(node))
                    first_half_parents.append(node)

                    for par in getattr(node, 'parents', []):
                        visit_first_half(par)

                for idx in first_half_params:
                    visit_first_half(info[idx][1])

                # find the children of first_half_parents that doesn't belong to first_half_parents
                cut_nodes_candidates = []
                cut_node_seen_ids = set()

                def visit_cut_nodes(node):
                    if id(node) in cut_node_seen_ids:
                        return
                    cut_node_seen_ids.add(id(node))

                    #if node.is_inline(compiler_params):
                    #    return
                    if id(node) not in first_half_ids:
                        cut_nodes_candidates.append(node)
                        return

                    for child in node.children:
                        if isinstance(child, Expr):
                            visit_cut_nodes(child)

                for node in first_half_parents:
                    visit_cut_nodes(node)

                cut_nodes = []
                # remove nodes that DOES NOT depend on discontinuous parameters
                for node in cut_nodes_candidates:
                    if node is arg_array:
                        continue
                    if node.params_only in [1, 2, 4]:
                        continue
                    cut_nodes.append(node)

                
    
        raymarching_nodes = locate_RaymarchingLoop(e)
        
        compiler_params.raymarching_nodes = raymarching_nodes
        
        assert len(compiler_params.raymarching_nodes) <= 1
    
        compiler_params.raymarching_dependent_comparisons = []
    
        for node in raymarching_nodes:

            if node.wrapper is not None:

                # TODO: in principle, if the surface normal is automatically generated, we could safely assume all related comparisons can be obtained by traversing res0 only
                # however, becasue currently we still use manually implemented surface normal, it is possible that derivs use a different comparison node than res0 and the compiler fails to remove the redundant copy, so we will traverse both res0 and derivs for different purposes

                if node.wrapper.include_derivs:
                    # traverse derivs to collect comparison nodes that should shortcut our regular rule
                    for val in Compound(node.wrapper.derivs).all_nodes_dfs(exclude=set([id(val) for val in node.wrapper.final_pos + node.dependencies])):
                        if val.is_comparison():
                            node.comparison_parents.append(val)

            compiler_params.raymarching_dependent_comparisons += [id(val) for val in node.comparison_parents]

    global DEFAULT_IS_DERIV
    
    DEFAULT_IS_DERIV = True
    
    T0 = time.time()
    # top-down, backpropagation
    
    for node_idx in range(len(all_nodes) - 1, -1, -1):
        
        node = all_nodes[node_idx]
        node.backprop(compiler_params)
            
    T1 = time.time()    
    
    print('Time used to apply backpropagation to existing AST: ', T1 - T0)
    
    if compiler_params.gradient_mode == 'AD':
        if loss_name in arg_array.dL_dself.keys():
            return arg_array.dL_dself[loss_name]
        else:
            return Compound(np.zeros(arg_array.ndims).tolist())
    
    # compute min_discont_denum after collecting nodes needed
    base_denum = None
    if len(compiler_params.raymarching_nodes) > 0:
        # if raymarching loop exists, denum is decided solely based on raymarching at geometry discontinuities
        
        for key in compiler_params.min_discont_denum_dict.keys():
            if key == id(compiler_params.raymarching_nodes[0]):
                base_denum = compiler_params.min_discont_denum_dict[key]
                del compiler_params.min_discont_denum_dict[key]
                break
        assert base_denum is not None
    
    for vals in compiler_params.min_discont_denum_dict.values():
        dL, partial_coord = vals
        do_update = (dL != 0) & (abs(partial_coord) < compiler_params.min_discont_denum)
        compiler_params.min_discont_denum = select(do_update, abs(partial_coord), compiler_params.min_discont_denum)
        
    if base_denum is not None:
        compiler_params.min_discont_denum = select(base_denum[0] != 0, abs(base_denum[1]), compiler_params.min_discont_denum)

    # we include forward node e because simplification will return a copy of the original nodes
    # and we don't want nodes in forward pass become a redundant copy of nodes in backward pass

    nderivs = len(arg_array.dL_dself[loss_name].children) + 1
    
    if isinstance(e, Compound):
        fw_channels = e.children
    else:
        fw_channels = []
        for idx in range(e.ndims):
            fw_channels.append(e[idx])
    
    deriv_pl = Compound([compiler_params.min_discont_denum] + arg_array.dL_dself[loss_name].children + fw_channels)

    # simplify before substiting, hopefully less burden to compiler
    deriv_pl = remove_redundant_exprs(deriv_pl)

    deriv_pl = repeated_simplify_inplace(deriv_pl)

    e = Compound(deriv_pl.children[nderivs:])
    e.calc_parents()

    # collect nodes that should be part of the masking process
    compiler_params.cond_parents_lookup = {}
    for node in e.all_nodes():
        if node.is_comparison() and not node.is_discont_to_params:
            cond_parents = node.get_cond_parents(compiler_params)
            compiler_params.cond_mask_candidates.append((node, cond_parents, [id(par) for par in cond_parents]))

    mask_combine_lookup = {}
    mask_skip_idx = []

    # find conditions that always reside in the same comparison
    # e.g. if we have cond_all = cond_1 && cond_2 && cond_#
    # then cond_2 and cond_3 should be combined with cond_1 (i.e. discontinuous in cond_1, cond_2, and cond_3 should NOT be counted more than once)
    for i in range(len(compiler_params.cond_mask_candidates)):
        for j in range(i + 1, len(compiler_params.cond_mask_candidates)):
            if j not in mask_skip_idx:
                if compiler_params.cond_mask_candidates[i][2] == compiler_params.cond_mask_candidates[j][2]:
                    if i in mask_combine_lookup.keys():
                        mask_combine_lookup[i].append(j)
                    else:
                        mask_combine_lookup[i] = [j]
                    mask_skip_idx.append(j)

    if len(compiler_params.cond_mask_candidates) - len(mask_skip_idx) <= 1:
        multi_discont_mask = ConstExpr(False)
        masked_derivs = deriv_pl.children[1:nderivs]
    else:
        discont_count = 0
        for i in range(len(compiler_params.cond_mask_candidates)):

            if i in mask_skip_idx:
                continue

            current_discont = cast2f(compiler_params.cond_mask_candidates[i][0] != 
                                     get_neighbor(compiler_params.cond_mask_candidates[i][0], compiler_params.pix_idx_pl))

            for j in mask_combine_lookup.get(i, []):
                current_discont += cast2f(compiler_params.cond_mask_candidates[j][0] != 
                                          get_neighbor(compiler_params.cond_mask_candidates[j][0], compiler_params.pix_idx_pl))

            cond_discont = (compiler_params.cond_mask_candidates[i][1][0] !=
                            get_neighbor(compiler_params.cond_mask_candidates[i][1][0], compiler_params.pix_idx_pl))

            for par in compiler_params.cond_mask_candidates[i][1][1:]:
                cond_discont = cond_discont | (par !=
                                                get_neighbor(par, compiler_params.pix_idx_pl))

            discont_count = discont_count + cast2f(cond_discont) * current_discont

        # garbage collect lists once the mask computation is done
        compiler_params.cond_mask_candidates = []
        compiler_params.cond_parents_lookup = {}

        multi_discont_mask = discont_count > 1

        masked_derivs = []
        for deriv in deriv_pl.children[1:nderivs]:
            masked_derivs.append(select(multi_discont_mask, 0, deriv))
        
    cut_dL = []
    for cut_node in compiler_params.half_cut_nodes:
        if loss_name not in cut_node.dL_dself.keys():
            print('cut fails')
            cut_dL = []
            compiler_params.first_half_params = None
            compiler_params.second_half_params = None
            break
        
        dL = cut_node.dL_dself[loss_name]
        if not dL.is_inline(compiler_params):
            cut_dL.append(dL)
        
        if loss_name in cut_node.dL_dself_scalar.keys():
            dL_scalar = cut_node.dL_dself_scalar[loss_name]
            if not isinstance(dL_scalar, (int, float)):
                if not dL_scalar.is_inline(compiler_params):
                    cut_dL.append(dL_scalar)

    # min_discont_denum + masked derivatives + dL_dself from cut_nodes
    deriv_pl = Compound([deriv_pl.children[0]] + masked_derivs + [multi_discont_mask] + cut_dL)
    
    deriv_pl = repeated_simplify_inplace(deriv_pl)
    
    # make parents correct for backward pass
    deriv_pl.calc_parents()
    
    #optimize_get_neighbor(deriv_pl, compiler_params)

    deriv_neighbors = []
    min_discont_denums = []

    if compiler_params.debug_ast:
        nneighbors = 1
    else:
        nneighbors = 4
        
    all_pix_idx_pls = []

    for i in range(1, nneighbors + 1):

        def pix_idx_setter(node):
            node.pix_idx = i

        T2 = time.time()
        
        all_pix_idx_pls.append(ArgumentScalar('pix_idx_%d' % i))

        ans = deriv_pl.subs_scalar([compiler_params.pix_idx_pl],
                                   [all_pix_idx_pls[-1]],
                                   attr_setter=pix_idx_setter, verbose=True)
        T3 = time.time()
        print('Time used to substitute for neighbor %d: ' % i, T3 - T2)

        deriv_neighbors.append(ans.children)

    if compiler_params.debug_ast:
        derivs = deriv_neighbors[0]
    else:
        min_discont_denums = [minimum(deriv_neighbors[0][0], deriv_neighbors[1][0]),
                              minimum(deriv_neighbors[2][0], deriv_neighbors[3][0])]

        choose_u = ChooseU((min_discont_denums[1] > 1e7) | 
                           ((min_discont_denums[0] < 1e7) & (min_discont_denums[0] > min_discont_denums[1])))

        compiler_params.choose_u = choose_u

        derivs = []
        deriv_u = []
        deriv_v = []
        
        for i in range(1, nderivs):
            deriv_u.append(deriv_neighbors[0][i] + deriv_neighbors[1][i])
            deriv_v.append(deriv_neighbors[2][i] + deriv_neighbors[3][i])
            
            derivs.append(0.5 * select(choose_u, 
                                       deriv_u[-1],
                                       deriv_v[-1]))
            
    compiler_params.all_pix_idx_pls = all_pix_idx_pls
        
    DEFAULT_IS_DERIV = False
    
    aux_nodes = [deriv_neighbors, deriv_u, deriv_v, min_discont_denums]

    return e, derivs, aux_nodes

def collect_get_neighbor(e, compiler_params):
    compiler_params.get_neighbor_lookup = {}
    compiler_params.get_neighbor_lookup_append = {}
        
    # collect all nodes that are input arguments to get_neighbor according to node.params_only
    # if node.params_only <= 2, collect in compiler_params.get_neighbor_lookup
    # if node.params_only = 3, collect in compiler_params.get_neighbor_lookup_append
    # if node.params_only = 4, should never reach this branch
    for node in e.all_nodes_dfs():
        if isinstance(node, Call) and node.name in ['get_neighbor', 'get_partial_trace_coord']:
            
            if isinstance(node.children[1], GlobalBuffer):
                continue
            
            if node.children[1].params_only == 4:
                print('Something wrong, should debug!')
                raise
                
            if node.children[1].params_only <= 2:
                if id(node.children[1]) not in compiler_params.get_neighbor_lookup.keys():
                    compiler_params.get_neighbor_lookup[id(node.children[1])] = \
                    (len(compiler_params.get_neighbor_lookup), node.children[1])
            elif node.children[1].params_only == 3:
                if id(node.children[1]) not in compiler_params.get_neighbor_lookup_append.keys():
                    compiler_params.get_neighbor_lookup_append[id(node.children[1])] = \
                    (len(compiler_params.get_neighbor_lookup_append), node.children[1])
            else:
                print('Unknown params_only value')
                raise
                
    compiler_params.mux_base_idx = len(compiler_params.get_neighbor_lookup)

def optimize_get_neighbor(e, compiler_params):
    
    collect_get_neighbor(e, compiler_params)
                
    func_nodes = []
    for lookup in [compiler_params.get_neighbor_lookup, compiler_params.get_neighbor_lookup_append]:
        for val in lookup.values():
            func_nodes.append(HL_Func(val[1]))
    
    # rewrite get_neighbor and get_partial_trace_coord in-place
    for node in e.all_nodes_dfs():
        if isinstance(node, Call) and node.name in ['get_neighbor', 'get_partial_trace_coord']:
            if id(node.children[1]) in compiler_params.get_neighbor_lookup.keys():
                func_idx = compiler_params.get_neighbor_lookup[id(node.children[1])][0]
            else:
                func_idx = compiler_params.get_neighbor_lookup_append[id(node.children[1])][0] + compiler_params.mux_base_idx
                
            node.children[1] = func_nodes[func_idx]
            node.children.append(func_idx)
            
    # garbage collect dicts
    compiler_params.get_neighbor_lookup = {}
    compiler_params.get_neighbor_lookup_append = {}
            
    e.calc_parents()

def to_source(schedule_and_buffer, compiler_params, f_node = None):
    """
    Convert Expr to finalized source code, including global variable and function declarations.
    """
    
    if compiler_params.individual_kernels:
        bw_schedule = schedule_and_buffer
        bw_buffer_info = [{}] * len(bw_schedule)
        fw_info = None
        producer_info = None
        bw_info = None
    else:
        bw_schedule = schedule_and_buffer[0]
        bw_buffer_info = schedule_and_buffer[1]
        
        fw_info = {'regular': {'buffer_info': []},
                   'per_pixel_offset': {'buffer_info': [{'ndim': 3, 'nfeats': 2, 'type': 'input', 'default_val': 0}]}}

        producer_info = {'regular': {'buffer_info': []}}

        bw_info = {'regular': 
                   {'buffer_info': [{'ndim': 3, 'nfeats': 3, 'type': 'input', 'default_val': 1, 'tag': 'dL_dcol'}]},
                   'per_pixel_offset':
                   {'buffer_info': [{'ndim': 3, 'nfeats': 3, 'type': 'input', 'default_val': 1, 'tag': 'dL_dcol'},
                                    {'ndim': 3, 'nfeats': 2, 'type': 'input', 'default_val': 0}]},
                   'denum_only':
                   {'buffer_info': [{'ndim': 3, 'nfeats': 3, 'type': 'input', 'default_val': 1, 'tag': 'dL_dcol'}]},
                   'offset_choose_u_pl':
                   {'buffer_info': [{'ndim': 3, 'nfeats': 3, 'type': 'input', 'default_val': 1, 'tag': 'dL_dcol'},
                                    {'ndim': 3, 'nfeats': 2, 'type': 'input', 'default_val': 0},
                                    {'ndim': 3, 'nfeats': 1, 'type': 'input', 'dtype': bool, 'default_val': False}]}}
        
    compiler_params.mode = MODE_SIDE_EFFECTS
    print("before resetting compiler parameters")
    compiler_params.reset()
    print("successfully reset compiler parameters")
    
    kernel_str = []
    output_buffer = {}
    
    configure_str = ''
                           
    generate_str = ''
    
    wrapper_str = {}
    for mode in all_kernel_modes:
        wrapper_str[mode] = {'kernel_call': '',
                             'include_lib': '',
                             'input_argument': '',
                             'sync_call': '',
                             'py_arg': '',
                             'kernel_indices': {},
                             'producer_call': '',
                             'replace_kernel': {}}
    
    args_params = ', '.join(['params[%d]' % idx for idx in range(compiler_params.input_nargs)])
    sigmas_params = ', '.join(['sigmas[%d]' % idx for idx in range(len(compiler_params.allow_random_params_idx))])
    args_excluding_buffer = f"""
        {args_params},
        uv_offset_0, uv_offset_1,
        width, height,
    """
    args_with_sigma_excluding_buffer = args_excluding_buffer + f"""
        frame_idx,
        {sigmas_params},
    """
    
    
    def get_wrapper_component(mode):
        if mode == 'par':
            return '_random_par', args_with_sigma_excluding_buffer, ''
        elif mode == 'offset':
            return '_per_pixel_offset', args_excluding_buffer, '*(input_offset.get()),'
        elif mode == 'denum_only':
            return '_denum_only', args_excluding_buffer, ''
        elif mode == 'choose_u_pl':
            return '_choose_u_pl', args_excluding_buffer, '*(input_offset.get()), *(input_choose_u_pl.get()),'
        elif mode == 'prune_updates':
            if len(compiler_params.optional_updates) > 0:
                do_prune_str = ', '.join(['do_prune[%d]' % idx for idx in range(len(compiler_params.optional_updates))]) + ', '
            else:
                do_prune_str = ''
            args_str = args_excluding_buffer + do_prune_str
            return '_prune_updates', args_str, ''
        else:
            return '', args_excluding_buffer, ''
        
    compiler_params.get_wrapper_component = get_wrapper_component
    
    fw_kernels = ''
    
    seperate_fw = True
    if len(bw_schedule):
        if bw_schedule[0]['lib'] == 'fw':
            compiler_params.bw_start_idx = 0
            seperate_fw = False
    
    if compiler_params.individual_kernels:
        seperate_fw = False
        

    if seperate_fw:
        compiler_params.reset_with_exception([])
        
        assert f_node is not None
        
        f_node.calc_parents()
        f = to_source_nonfinal(f_node, compiler_params, [], {})
        
        if compiler_params.backend in ['tf', 'np', 'torch']:
            f_return = '\n' + 'return ' + f_node.to_source(compiler_params.as_mode(MODE_VARNAME))
            f_declare = f"""
def f({DEFAULT_ARGUMENT_SCALAR_U_NAME}, {DEFAULT_ARGUMENT_SCALAR_V_NAME}, {DEFAULT_ARGUMENT_ARRAY_NAME}, f_log_intermediate, vec_output, width, height):
    Ou = 0
    Ov = 0
    pix_idx = 0
    """            
            compiler_params.global_code.append('f_log_intermediate_len = 0\n')
        elif compiler_params.backend == 'glsl':
            return_name = f_node.to_source(compiler_params.as_mode(MODE_VARNAME))
            if f_node.ndims == 0:
                f_return = f"""
    fragColor = vec4({return_name}, {return_name}, {return_name}, 1.0);
    return;
}}"""
            else:
                assert f_node.ndims == 3
                f_return = f"""
    fragColor = vec4({return_name}, 1.0);
    return;
}}"""
                                           
            f_declare = f"""

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {{

    fragCoord.y = iResolution.y - fragCoord.y;
    
    float {DEFAULT_ARGUMENT_SCALAR_U_NAME};
    float {DEFAULT_ARGUMENT_SCALAR_V_NAME};
    
    if (preserve_aspect_ratio) {{
        float max_scale = max(width / iResolution.x, height / iResolution.y) * scale_factor;
        
        vec2 padding = (vec2(width, height) - max_scale * iResolution.xy) / 2.;
        
        {DEFAULT_ARGUMENT_SCALAR_U_NAME} = fragCoord.x * max_scale + padding.x;
        {DEFAULT_ARGUMENT_SCALAR_V_NAME} = fragCoord.y * max_scale + padding.y;
    }} else {{
        {DEFAULT_ARGUMENT_SCALAR_U_NAME} = fragCoord.x / iResolution.x * width;
        {DEFAULT_ARGUMENT_SCALAR_V_NAME} = fragCoord.y / iResolution.y * height;
    }}

            """
        else:
            f_return = f"""\nreturn {f_node.to_source(compiler_params.as_mode(MODE_VARNAME))};}}"""
            f_declare = f"""
std::vector<Expr> generate0(std::vector<Expr> params) {{
    Ou = Halide::cast<int> (0);
    Ov = Halide::cast<int> (0);
    
    Expr current_u = Halide::cast<float> (center_pix_u_pos + Ou);
    Expr current_v = Halide::cast<float> (center_pix_v_pos + Ov);
            """
                        
            generate_str += f"""
        if (kernel_idx == 0) {{
            p_bound = 3;
            {GENERATE_RETURN_ARRAY} = generate0(params);
        }}
            """
            
            configure_str += f"""
        if (kernel_idx == 0) {{
            if (use_per_pixel_offset && !is_FD) {{
                per_pixel_offset = add_input<Func>("per_pixel_offset", Float(32), 3);
            }}
        }}
        """
            
        compiler_params.bw_start_idx = 1
        fw_kernels = f_declare + indent(f + f_return)
        
        for key in fw_info.keys():
            fw_info[key]['buffer_info'].append({'ndim': 3, 'nfeats': 3, 'type': 'output', 'tag': 'col'})
        for key in producer_info.keys():
            producer_info[key]['buffer_info'].append({'ndim': 3, 'nfeats': 3, 'type': 'output', 'tag': 'col'})
            
        if compiler_params.backend == 'glsl':
            if getattr(compiler_params, 'valid_args', None) is not None:
                fw_kernels = '\n\n// valid: %s\n' % (', '.join(['%d' % val for val in compiler_params.valid_args])) + fw_kernels
            
            if getattr(compiler_params, 'par_name_lookup', None) is not None:
                named_par_str = ''
                for idx in sorted(compiler_params.par_name_lookup.keys()):
                    named_par_str += f"""
#define {compiler_params.par_name_lookup[idx]}_idx {idx}
float {compiler_params.par_name_lookup[idx]} = X[{compiler_params.par_name_lookup[idx]}_idx];
"""
                    
                if compiler_params.par_vals is not None:
                    if getattr(compiler_params, 'valid_args', None) is not None:
                        valid_idx = compiler_params.valid_args
                    else:
                        valid_idx = np.arange(compiler_params.input_nargs).tolist()
                    
                    orig_X_str = ''
                    for idx in sorted(compiler_params.par_name_lookup.keys()):
                        orig_X_str += f"""{compiler_params.par_vals[valid_idx][idx]}, """
                    orig_X_str = orig_X_str[:-2]
                        
                    animate_str = 'animate_params();\n'
                    
                    fw_kernels = f_declare + indent(animate_str + f + f_return)
                    
                    if raymarching_count > 0:
                        raymarching_nodes = locate_RaymarchingLoop(f_node)
                        
                        raymarching_iter_str = f"""
// Number of iterations for each raymarching loop
                        """
                        
                        for idx in range(raymarching_count):
                            
                            node = raymarching_nodes[idx]
                            
                            raymarching_iter_str += f"""
#define _raymarching_iter_{node.raymarching_count} {node.niters}
                            """
                    else:
                        raymarching_iter_str = ''
                    
                    named_par_str = f"""
// If true, aspect ratio in the original optimization will be preserved, but extra space outside original FOV might reveal artifact.
// If false, FOV is the same as original optimization, but aspect ratio will not be preserved.
#define preserve_aspect_ratio true

// Smaller factor will zoom in the rendering, larger factor will zoom out
float scale_factor = 1.;

{raymarching_iter_str}

#define X float[]({orig_X_str})

""" + named_par_str
                    
                    named_par_str += f"""
void animate_params() {{
}}
                    """
                    
                    for animate_declare in compiler_params.animate_declares:
                        named_par_str += animate_declare + '\n'
                    
                fw_kernels = named_par_str + fw_kernels
            
            
    compiler_params.nkernels = len(bw_schedule)
    
    restricted_shared_memory_idx = []

    if len(bw_schedule) is not None:
        
        for i in range(compiler_params.nkernels):
            
            if compiler_params.individual_kernels:
                def get_kernel_idx(idx):
                    return idx
            else:
                # Leave out the first compiler_params.nkernels * 5 indices 
                # because when use_per_pixel_offset = True, 
                # these indices may be used to represent producer input from neighboring pixels
                def get_kernel_idx(idx):
                    return idx * 5 + compiler_params.bw_start_idx

            kernel_idx = get_kernel_idx(i)

            component = bw_schedule[i]

            current_node = component['current_node']

            if not compiler_params.individual_kernels:
                current_log_node_list = current_node.children
                
                if hasattr(current_node, 'child_indices'):
                    current_log_node_indices = current_node.child_indices
                else:
                    current_log_node_indices = np.arange(len(current_log_node_list))
                
                # if any node in current_log_node_list is Binary2Float, expand it
                is_combined_binary = None
                combined_binary_idx = -1
                for node in current_log_node_list:
                    if isinstance(node, Binary2Float):
                        assert node is current_log_node_list[-1]
                        current_log_node_list = current_log_node_list[:-1]
                        is_combined_binary = [False] * len(current_log_node_list)
                        combined_binary_idx = len(current_log_node_list)
                        current_log_node_list += node.children
                        is_combined_binary += [True] * len(node.children)

                current_log_id_list = [id(node) for node in current_log_node_list]
                if is_combined_binary is None:
                    if isinstance(current_node, Binary2Float):
                        is_combined_binary = [True] * len(current_log_id_list)
                        combined_binary_idx = 0
                    else:
                        is_combined_binary = [False] * len(current_log_id_list)
                output_buffer[kernel_idx] = (current_log_id_list, current_log_node_list, is_combined_binary, combined_binary_idx, current_log_node_indices)
            
                if component['lib'] == 'fw':
                    if 'producer' not in fw_info.keys():
                        fw_info['producer'] = copy.deepcopy(fw_info['regular'])

                    for buffer_key in fw_info.keys():
                        fw_info[buffer_key]['buffer_info'].append({'ndim': 3, 'nfeats': 3, 'type': 'output', 'tag': 'col'})
                    for buffer_key in producer_info.keys():
                        producer_info[buffer_key]['buffer_info'].append({'ndim': 3, 'nfeats': bw_buffer_info[i]['nfeats'], 'type': 'output', 'tag': 'col', 'pad': 1})
                    

                    nfeats_exclude_col = bw_buffer_info[i]['nfeats'] - 3
                    
                    for buffer_key in bw_info.keys():
                        if 'offset' in buffer_key:
                            # per_pixel_offset and offset_choose_u_pl
                            bw_info[buffer_key]['buffer_info'] += 5 * [{'ndim': 3, 'nfeats': nfeats_exclude_col, 'type': 'intermediate'}]
                        elif buffer_key == 'denum_only':
                            bw_info[buffer_key]['buffer_info'].append(copy.copy(bw_buffer_info[i]))
                            bw_info[buffer_key]['buffer_info'][-1]['nfeats'] = bw_info[buffer_key]['buffer_info'][-1]['nfeats'] - 3
                        else:
                            bw_info[buffer_key]['buffer_info'].append(bw_buffer_info[i])
                else:
                    for buffer_key in bw_info.keys():
                        if buffer_key != 'denum_only':
                            bw_info[buffer_key]['buffer_info'].append(bw_buffer_info[i])
                        else:
                            if 'cont' in component['lib']:
                                continue

                            if bw_info[buffer_key]['buffer_info'][-1]['type'] == 'output':
                                exist_previous_bw_kernel = True
                            else:
                                bw_info[buffer_key]['buffer_info'].append({'ndim': 3, 'nfeats': 1, 'type': bw_buffer_info[i]['type']})
                
            if 'shared_kernel' not in component.keys():
                
                buffer_str = ''
                
                compiler_params.kernel_name = 'kernel%d_' % kernel_idx
                
                current_read_id_node_lookup = {}

                if len(component['input']):
                    
                    if compiler_params.individual_kernels:
                        for buffer_idx in range(len(component['input'])):
                            # we don't care converted_idx consistency as long as they're unique per input 
                            converted_idx = buffer_idx
                            
                            read_count = 0
                            for idx in range(len(component['input'][buffer_idx])):
                                key = id(component['input'][buffer_idx][idx])
                                val = component['input'][buffer_idx][idx]
                                
                                if val.ndims == 0:
                                    current_read_id_node_lookup[key] = [converted_idx, read_count, val, idx]
                                    read_count += 1
                                else:
                                    current_read_idx = [read_count + n for n in range(val.ndims)]
                                    current_read_id_node_lookup[key] = [converted_idx, current_read_idx, val, idx]
                                    read_count += val.ndims
                                
                            input_is_binary = None
                            if component.get('input_is_binary', None) is not None:
                                input_is_binary = component['input_is_binary'][buffer_idx]
                                
                            if input_is_binary is None:
                                input_is_binary = [False] * len(component['input'][buffer_idx])
                                combined_binary_idx = 0
                            elif True not in input_is_binary:
                                combined_binary_idx = 0
                            else:
                                combined_binary_idx = input_is_binary.index(True)
                            output_buffer[buffer_idx] = [None, None, input_is_binary, combined_binary_idx, None]
                    else:
                        # TF always uses the simpliest inline everything schedule
                        # because TF can never be highly efficient, and is only used for debugging
                        #assert compiler_params.backend != 'tf'
                        
                        log_subset = component.get('log_subset', None)

                        for raw_idx in range(len(component['input'])):
                            buffer_idx = component['input'][raw_idx]

                            converted_idx = get_kernel_idx(buffer_idx)
                            
                            if log_subset is not None:
                                current_log_subset = log_subset[raw_idx]
                            else:
                                current_log_subset = None
                                
                            current_read_ids = output_buffer[converted_idx][0]
                            current_read_nodes = output_buffer[converted_idx][1]
                            current_read_indices = output_buffer[converted_idx][4]

                            for idx in range(len(output_buffer[converted_idx][0])):
                                key = current_read_ids[idx]
                                val = current_read_nodes[idx]
                                
                                if current_log_subset is not None:
                                    if idx in current_log_subset:
                                        current_read_id_node_lookup[key] = [converted_idx, current_read_indices[idx], val, idx]
                                else:
                                    current_read_id_node_lookup[key] = [converted_idx, current_read_indices[idx], val, idx]
                            
                if len(current_read_id_node_lookup) > shared_memory_bottleneck:
                    restricted_shared_memory_idx.append(kernel_idx)
                
                if isinstance(current_node, Compound):
                    p_bound = len(current_node.children)
                else:
                    assert isinstance(current_node, Binary2Float)
                    p_bound = 1

                exist_previous_bw_kernel = False

                

                local_read_nodes_lookup = {}
                global_read_nodes_lookup = {}
                global_buffer_nodes_lookup = {}
                
                if bw_schedule[0]['lib'] == 'fw':
                    includes_producer = True
                else:
                    includes_producer = False

                

                current_node.calc_parents()

                # substitute global read nodes into GlobalRead Expr, 
                # if found inside get_neighbor/get_partial_trace_coord, substittue with GlobalBuffer Expr
                # keep the list of newly created nodes, we will substitute them back at the end of the iteration

                for key, val in current_read_id_node_lookup.items():

                    buffer_idx = val[0]
                    node_idx = val[1]
                    node = val[2]
                    raw_idx = val[3]
                    is_binary2float = output_buffer[buffer_idx][2]
                    combined_binary_idx = output_buffer[buffer_idx][3]

                    if buffer_idx not in global_buffer_nodes_lookup:
                        global_buffer_nodes_lookup[buffer_idx] = GlobalBuffer(buffer_idx, is_binary2float, combined_binary_idx)

                    if includes_producer and buffer_idx == 0:
                        # GlboalRead from producer
                        extra_producer_arg = True
                    else:
                        extra_producer_arg = False
                        
                    if isinstance(node_idx, list):
                        global_read_node = Compound([global_buffer_nodes_lookup[buffer_idx](idx, node, extra_producer_arg) for idx in node_idx])
                        get_neighbor_args = Compound(node_idx)
                    else:
                        global_read_node = global_buffer_nodes_lookup[buffer_idx](node_idx, node, extra_producer_arg)
                        get_neighbor_args = node_idx

                    global_read_nodes_lookup[key] = []

                    for par in node.parents:
                        if isinstance(par, Call) and getattr(par, 'name', '') in ['get_neighbor', 'get_partial_trace_coord']:
                            assert par.children[1] is node
                            # make the GlobalBuffer substitution

                            par.children[1] = global_buffer_nodes_lookup[buffer_idx]
                            par.children.append(get_neighbor_args)
                            par.children.append(node.dtype == BOOL_TYPE)
                            par.children.append(ConstExpr(extra_producer_arg))
                            par.children.append(ConstExpr(False))
                            
                            assert not is_binary2float[raw_idx]

                            global_read_nodes_lookup[key].append((1, par))
                        else:
                            # make GlobalRead substitution
                            for child_idx in range(len(par.children)):
                                if par.children[child_idx] is node:
                                    par.children[child_idx] = global_read_node
                                    global_read_nodes_lookup[key].append((child_idx, par))
                
                if compiler_params.backend == 'hl':
                    # make some addiitonal mutation if we're outputting Halide code
                    
                    compiler_params.reset_with_exception([])

                    current_node.calc_parents()
                    collect_get_neighbor(current_node, compiler_params)

                    all_nodes = current_node.all_nodes_dfs()
                    # reset is_producer attr
                    for node in all_nodes:
                        node.is_producer = False

                    # set is_producer = True for all nodes that needs neighbor value
                    for lookup in [compiler_params.get_neighbor_lookup, compiler_params.get_neighbor_lookup_append]:
                        for val in lookup.values():
                            val[1].is_producer = True

                    # top-down, set children of producer not to be producer, collect any producer node with non-producer parent
                    cut_nodes = [val[1] for val in compiler_params.get_neighbor_lookup.values()] + \
                                [val[1] for val in compiler_params.get_neighbor_lookup_append.values()]

                    n_needs_neighbor = len(cut_nodes)

                    # top-down
                    for node in all_nodes[::-1]:

                        if node.is_producer:
                            for child in node.children:
                                if isinstance(child, Expr):
                                    child.is_producer = True

                            if id(node) not in compiler_params.get_neighbor_lookup.keys() and \
                               id(node) not in compiler_params.get_neighbor_lookup_append.keys():

                                if is_all_constant(node) or node.is_inline(compiler_params):
                                    continue

                                for par in node.parents:
                                    if not par.is_producer:
                                        cut_nodes.append(node)
                                        break

                    cut_nodes_lookup = {id(cut_nodes[idx]): [idx, cut_nodes[idx]] for idx in range(len(cut_nodes))}

                    if len(cut_nodes):
                        producer_node = Compound(cut_nodes)
                        producer_node.root = True
                        producer_node.calc_parents()

                        compiler_params.need_cast = False
                        producer_f = to_source_nonfinal(producer_node, compiler_params)
                        producer_return_name = producer_node.to_source(compiler_params.as_mode(MODE_VARNAME))

                        compiler_params.need_cast = True


                        producer_f_return = f"""
    if (pix_idx != 0) {{
        if (! use_random_par) {{
            std::vector<Expr> ans({producer_return_name}.begin(), {producer_return_name}.begin() + {n_needs_neighbor});
            return ans;
        }}
    }}
    return {producer_return_name};
}}
                        """
                    
                    else:
                        producer_f = ''
                        producer_f_return = 'return {0.f}; }'

                    producer_f_declare = f"""
std::vector<Expr> generate{kernel_idx}_producer(std::vector<Expr> params, int pix_idx) {{
    Ou = Halide::cast<int> (0);
    Ov = Halide::cast<int> (0);
    
    if (pix_idx == 1) {{
        Ou = Halide::cast<int> (1);
    }} else if (pix_idx == 2) {{
        Ou = Halide::cast<int> (-1);
    }} else if (pix_idx == 3) {{
        Ov = Halide::cast<int> (1);
    }} else if (pix_idx == 4) {{
        Ov = Halide::cast<int> (-1);
    }}
    
    Expr current_u = Halide::cast<float> (center_pix_u_pos + Ou);
    Expr current_v = Halide::cast<float> (center_pix_v_pos + Ov);
                """
                        
                    
                    
                
                    kernel_str.append(producer_f_declare + indent(producer_f + producer_f_return))

                    consumer_nodes = current_node
                    all_nodes = consumer_nodes.all_nodes_dfs()

                    # rewrite get_neighbor/get_partial_trace_coord to GlobalRead nodes int

                    consumer_nodes.calc_parents()
                    local_array = ArgumentArray(LOCAL_ARRAY)

                    # replace node in cut_ondes into local_array reads, similarly we will revert these changes at the end of the iteration

                    for node in cut_nodes:

                        local_read_nodes_lookup[id(node)] = []

                        for par in node.parents:
                            for child_idx in range(len(par.children)):
                                if par.children[child_idx] is node:
                                    # replace with local read

                                    replaced = False

                                    # check if node needs access to neighbor value
                                    if isinstance(par, Call):
                                        if par.name in ['get_neighbor', 'get_partial_trace_coord']:
                                            par.children[child_idx] = local_array
                                            par.children.append(cut_nodes_lookup[id(node)][0])
                                            par.children.append(compiler_params.mux_base_idx)
                                            local_read_nodes_lookup[id(node)].append((child_idx, par))
                                            replaced = True

                                    if not replaced:
                                        # accessing center pixel value
                                        par.children[child_idx] = get_neighbor_f(local_array, 0, cut_nodes_lookup[id(node)][0], 0)
                                        local_read_nodes_lookup[id(node)].append((child_idx, par))

                    compiler_params.reset_with_exception(list(cut_nodes_lookup.keys()), lookup=cut_nodes_lookup)
                else:
                    consumer_nodes = current_node
                    compiler_params.reset_with_exception([])
                    cut_nodes = []

                consumer_nodes.root = True
                
                bind_nodes = [consumer_nodes]
                if component.get('macro_node', None) is not None and compiler_params.backend == 'hl':
                    bind_nodes.append(component['macro_node'])

                for node in bind_nodes:
                    if 'LetBind' in component.keys():
                        LetBind(node,
                                component['LetBind'][0],
                                component['LetBind'][1],
                                compiler_params)
                    else:
                        LetBind(node, 
                                compiler_params.all_pix_idx_pls, 
                                np.arange(len(compiler_params.all_pix_idx_pls)) + 1,
                                compiler_params)

                if component.get('macro_node', None) is not None and compiler_params.backend == 'hl':
                    macro_node = Compound([component['macro_node']])
                    macro_node.root = True
                    macro_node.calc_parents()
                    f_macro = to_source_nonfinal(macro_node, compiler_params)
                    macro_name = macro_node.to_source(compiler_params.as_mode(MODE_VARNAME))
                    macro_node.root = False
                    
                    f_macro += f"""
if (denum_only) {{
    p_bound = 1;
    return {macro_name};
}}
                    """
                else:
                    f_macro = ''

                consumer_nodes.calc_parents()
                f = to_source_nonfinal(consumer_nodes, compiler_params)
                
                names = [node.var_name(compiler_params) for node in consumer_nodes.children]

                return_name = consumer_nodes.to_source(compiler_params.as_mode(MODE_VARNAME))
                f_return, f_declare = get_f_return_declare(compiler_params, component['lib'], return_name, kernel_idx, 3, len(current_node.children))

                if compiler_params.backend == 'hl':
                    # revert back local read nodes
                    for node in cut_nodes:
                        for vals in local_read_nodes_lookup[id(node)]:
                            child_idx = vals[0]
                            par = vals[1]
                            par.children[child_idx] = node

                            if isinstance(par, Call):
                                if par.name in ['get_neighbor', 'get_partial_trace_coord']:
                                    par.children = par.children[:3]

                # revert back global read nodes
                for key, val in current_read_id_node_lookup.items():
                    node = val[2]

                    for vals in global_read_nodes_lookup[key]:
                        child_idx = vals[0]
                        par = vals[1]

                        if isinstance(par.children[child_idx], GlobalBuffer):
                            par.children = par.children[:3]

                        par.children[child_idx] = node


                if compiler_params.backend == 'hl':
                    generate_str += update_generate_str(compiler_params, 
                                                        component['lib'], 
                                                        kernel_idx, 
                                                        p_bound,
                                                        component.get('is_final_bw', False))

                    extra_configure_str, buffer_str = \
                    get_configure_str(compiler_params, component['lib'], kernel_idx, component['input'], get_kernel_idx, includes_producer)
                    configure_str += extra_configure_str

                kernel_str.append(f_declare + indent(f_macro + f + f_return))
            
            if compiler_params.backend == 'hl':
                
                if 'shared_kernel' not in component.keys():
                    shared_idx = None
                else:
                    shared_idx = get_kernel_idx(component['shared_kernel'])
                
                for mode in all_kernel_modes:
                    wrapper_str[mode] = \
                    update_wrapper_str(compiler_params, wrapper_str[mode], mode, component['lib'], kernel_idx, exist_previous_bw_kernel, buffer_str, args_params, shared_kernel_idx=shared_idx, includes_choose_u=component.get('includes_choose_u', False), includes_producer=includes_producer, par_idx=component.get('par_idx', None))
            
    
    bw_kernels = '\n'.join(kernel_str)
        
    for key in wrapper_str.keys():
        wrapper_str[key]['kernel_call'] += wrapper_str[key]['sync_call']
        
    compiler_params.fw_info = fw_info
    compiler_params.producer_info = producer_info
    compiler_params.bw_info = bw_info

    if compiler_params.backend in ['tf', 'np', 'torch']:
        c = fw_kernels + '\n' + bw_kernels
    elif compiler_params.backend == 'glsl':
        assert not compiler_params.compute_g
        c = fw_kernels
    else:
        
        random_idx_str = ', '.join([str(val) for val in compiler_params.allow_random_params_idx])
        
        input_str = f"""Input<Func> *{DL_DCOL_ARRAY}"""
        for idx in range(5 * max(compiler_params.nkernels, 1)):
            input_str += ', *output%d' % idx
            
        if compiler_params.compute_g:
            if compiler_params.gradient_mode == 'ours':
                pix_idx_declare = ', '.join([pl.name for pl in compiler_params.all_pix_idx_pls])
            else:
                pix_idx_declare = '__pl__'
        else:
            pix_idx_declare = '__pl__'
        
        pix_idx_assign_neighbor = ''
        pix_idx_assign_filter = ''
        
        for idx in range(len(compiler_params.all_pix_idx_pls)):
            pix_idx_assign_neighbor += f"""
            {compiler_params.all_pix_idx_pls[idx].name} = neighbor_idx;
            """
                        
            pix_idx_assign_filter += f"""
            {compiler_params.all_pix_idx_pls[idx].name} = filter_idx * 2 + {idx % 2} + 1;
            """
        
        if len(restricted_shared_memory_idx) == 0:
            restricted_cond = 'false'
        else:
            restricted_cond = ' || '.join(['kernel_idx == %d' % val for val in restricted_shared_memory_idx])
        
        
        c = f"""
class Shader : public Halide::Generator<Shader> {{
    
public:

    Input<float[NARGS]> orig_params{{"orig_params"}};
    Input<float[2]> uv_offset{{"uv_offset"}};
    
    Input<int> width{{"width"}};
    Input<int> height{{"height"}};
    
    Output<Buffer<float>> gradients{{"gradients", 3}};
    
    GeneratorParam<int> kernel_idx{{"kernel_idx", 0}};
    
    GeneratorParam<bool> is_fw{{"is_fw", /* default value */ false}};
    GeneratorParam<bool> is_FD{{"is_FD", /* default value */ false}};
    
    GeneratorParam<bool> add_to{{"add_to", /* default value */ false}};
    GeneratorParam<bool> output_base_only{{"output_base_only", /* default value */ false}};
    
    GeneratorParam<bool> use_random_par{{"use_random_par", /* default value */ false}};
    
    GeneratorParam<bool> use_per_pixel_offset{{"use_per_pixel_offset", /* default value */ false}};
    GeneratorParam<bool> use_choose_u_pl{{"use_choose_u_pl", /* default value */ false}};
    
    GeneratorParam<bool> denum_only{{"denum_only", /* default value */ false}};
    
    GeneratorParam<int> neighbor_idx{{"neighbor_idx", /* default value */ -1}};
    GeneratorParam<int> filter_idx{{"filter_idx", /* default value */ -1}};
    
    GeneratorParam<bool> exclude_col{{"exclude_col", /* default value */ false}};
    
    GeneratorParam<bool> prune_optional_update{{"prune_optional_update", /* default value */ false}};
    
    GeneratorParam<std::string> par_idx{{"par_idx", /* defaut value */ ""}};
    
    Expr checkpoint(const Expr &e) {{
        std::vector<Var> args = {{u, v}};
        if (Internal::expr_uses_var(e, p.name())) {{
            args.push_back(p);
        }}
        Func f;
        f(args) = e;
        f.compute_at(innermost);
        return f(args);
    }}
    std::vector<Expr> checkpoint(std::vector<Expr> vec) {{
        for (auto &e : vec) {{
            e = checkpoint(e);
        }}
        return vec;
    }}
    
    Expr get_neighbor(Func f, int pix_idx, int idx, bool bool_cast=false, bool is_producer=false, bool is_binary2float=false, int combined_binary_idx=0) {{
    
        Expr ans;
        
        if (exclude_col && is_producer) {{
            idx -= 3;
        }}
        
        if (is_binary2float) {{
            assert(pix_idx == 0);
            
            int base_idx = idx - combined_binary_idx;
            int scale;
            
            if (base_idx == 0) scale = 1;
            else if (base_idx == 1) scale = 2;
            else if (base_idx == 2) scale = 4;
            else if (base_idx == 3) scale = 8;
            else if (base_idx == 4) scale = 16;
            else throw;
            
            ans = (f(u, v, combined_binary_idx) % (scale * 2)) >= scale;
        }} else if (use_per_pixel_offset && is_producer) {{
            if (pix_idx == 0) {{
                ans = (*output0)(u, v, idx);
            }} else if (pix_idx == 1) {{
                ans = (*output1)(u, v, idx);
            }} else if (pix_idx == 2) {{
                ans = (*output2)(u, v, idx);
            }} else if (pix_idx == 3) {{
                ans = (*output3)(u, v, idx);
            }} else {{
                ans = (*output4)(u, v, idx);
            }}
        }} else {{
            if (pix_idx == 0) {{
                ans = f(u, v, idx);
            }} else if (pix_idx == 1) {{
                ans = f(u + 1, v, idx);
            }} else if (pix_idx == 2) {{
                ans = f(u - 1, v, idx);
            }} else if (pix_idx == 3) {{
                ans = f(u, v + 1, idx);
            }} else {{
                ans = f(u, v - 1, idx);
            }}
        }}
        
        if (bool_cast) {{
            ans = Halide::cast<bool> (ans);
        }}
        
        return ans;
    }}   
    
    float get_scale(int pix_idx) {{
        if (pix_idx == 1 || pix_idx == 3) {{
            return -1.f;
        }} else {{
            return 1.f;
        }}
    }}
    
    Expr get_partial_trace_coord(Func f, int pix_idx, int idx, bool bool_cast=false, bool is_producer=false, bool is_binary2float=false) {{
    
        assert(!is_binary2float);
    
        Expr current_idx = idx;
    
        if (exclude_col && is_producer) {{
            current_idx -= 3;
        }}
    
        Expr ans = Halide::cast<float> (f(u, v, current_idx)) - Halide::cast<float> (get_neighbor(f, pix_idx, idx, false, is_producer));
        if (pix_idx == 1 || pix_idx == 3) {{
            ans = -ans;
        }}
        
        return ans;
    }}
    
    Expr get_neighbor(std::vector<Expr> nodes[5], int pix_idx, int idx, int mux_base_idx) {{
    
        if (idx >= mux_base_idx && ! use_random_par) {{
            return nodes[0][idx];
        }}
        return nodes[pix_idx][idx];
    }}   
    
    Expr get_partial_trace_coord(std::vector<Expr> nodes[5], int pix_idx, int idx, int mux_base_idx) {{
        if (idx >= mux_base_idx) {{
            if (! use_random_par) {{
                return 0.f;
            }}
        }}
        
        Expr ans = Halide::cast<float> (nodes[0][idx]) - Halide::cast<float> ((nodes[pix_idx][idx]));
        if (pix_idx == 1 || pix_idx == 3) {{
            ans = -ans;
        }}
        
        return ans;
    }}
    
    Expr read_par(std::vector<Expr> current_params, int idx) {{
    
        Expr ans = current_params[idx];
        
        bool found = false;
        int random_idx = idx;
        
        if (use_random_par) {{
            auto it = std::find(random_par_idx_int.begin(), random_par_idx_int.end(), idx);
            
            if (it != random_par_idx_int.end()) {{
                found = true;
                random_idx = it - random_par_idx_int.begin();
            }}
        }}
        
        if (found) {{
            ans += random_noise(u + Ou, v + Ov, idx) * current_params[NARGS + random_idx];
        }}
        
        return ans;
        
    }}
    
    void configure() {{
    
        if (use_random_par) {{
            frame_idx = add_input<int>("frame_idx");
            
            random_par_idx_int = {{{random_idx_str}}};
            
            int nsigmas = random_par_idx_int.size();
            
            for (int idx = 0; idx < nsigmas; idx++) {{
                random_sigma.push_back(add_input<float>(" _" + std::to_string(idx)));
            }}
        }}
        
        if (prune_optional_update) {{
            for (int idx = 0; idx < {len(compiler_params.optional_updates)}; idx++) {{
                do_prune.push_back(add_input<bool>("do_prune_" + std::to_string(idx)));
            }}
        }}

{configure_str}

        if (is_FD) {{
            
            finite_diff_h = add_input<float>("finite_diff_h");
            divide_by = add_input<float>("divide_by");
            
            for (int idx = 0; idx < NARGS; idx++) {{
                offset_dir.push_back(add_input<float>("offset_dir_" + std::to_string(idx)));
            }}
            
            if (use_random_par) {{
                int nsigmas = random_par_idx_int.size();
                
                for (int idx = 0; idx < nsigmas; idx++) {{
                    offset_sigma.push_back(add_input<float>("offset_sigma" + std::to_string(idx)));
                }}
            }}

            {DL_DCOL_ARRAY} = add_input<Func>("{DL_DCOL_ARRAY}", Float(32), 3);
            
            if (use_per_pixel_offset) {{
                per_pixel_offset = add_input<Func>("per_pixel_offset", Float(32), 3);
            }}
            
            if (add_to) {{
                old_gradients = add_input<Func>("old_gradients", Float(32), 3);
            }}
        }}
    }}
    
{indent(fw_kernels)}

{indent(bw_kernels)}

    void generate() {{
    
        std::string par_idx_str = par_idx;
        
        if (par_idx_str.compare("") != 0) {{
            std::stringstream ss(par_idx_str);

            while( ss.good() )
            {{
                std::string substr;
                getline( ss, substr, ',' );
                par_idx_int.push_back( std::stoi(substr) );
            }}
        }}
    
        center_pix_u_pos = Halide::cast<float> (u) + uv_offset[0];
        center_pix_v_pos = Halide::cast<float> (v) + uv_offset[1];
        
        if (use_per_pixel_offset) {{
            center_pix_u_pos += (*per_pixel_offset)(u, v, 0);
            center_pix_v_pos += (*per_pixel_offset)(u, v, 1);
        }}
    
        if (use_random_par) {{
            random_noise(u, v, p) = our_random_float({{p, (*frame_idx), u, v}}) - 0.5f;
        }}
    
        std::vector<Expr> params;
        
        for (int idx = 0; idx < NARGS; idx++) {{
            params.push_back(orig_params[idx]);
        }}
        
        if (use_random_par) {{
            for (int idx = 0; idx < random_par_idx_int.size(); idx++) {{
                params.push_back(*(random_sigma[idx]));
            }}
        }}
        
        if (neighbor_idx > 0) {{
{pix_idx_assign_neighbor}
        }} else if (filter_idx >= 0) {{
{pix_idx_assign_filter}
        }}
        
        if (is_FD) {{
            
            std::vector<Expr> params_pos;
            std::vector<Expr> params_neg;
            
            for (int idx = 0; idx < NARGS; idx++) {{
                params_pos.push_back(params[idx] + (*finite_diff_h) * (*offset_dir[idx]));
                params_neg.push_back(params[idx] - (*finite_diff_h) * (*offset_dir[idx]));
            }}
            
            if (use_random_par) {{
                for (int idx = 0; idx < random_par_idx_int.size(); idx++) {{
                    params_pos.push_back((*random_sigma[idx]) + (*finite_diff_h) * (*offset_sigma[idx]));
                    params_neg.push_back((*random_sigma[idx]) - (*finite_diff_h) * (*offset_sigma[idx]));
                }}
            }}
            
            std::vector<Expr> cols[2];
            cols[0] = generate0(params_pos);
            cols[1] = generate0(params_neg);
            
            std::vector<Expr> dL_dcol_val = {{(*{DL_DCOL_ARRAY})(u, v, 0),
                                              (*{DL_DCOL_ARRAY})(u, v, 1),
                                              (*{DL_DCOL_ARRAY})(u, v, 2)}};
            
            Expr base_deriv = dot(dL_dcol_val, sub3(cols[0], cols[1])) / (2.f * (*finite_diff_h));
            
            if (output_base_only) {{
                gradients(u, v, p) = base_deriv;
            }} else {{
                std::vector<Expr> derivs;

                for (int idx = 0; idx < NARGS; idx++) {{
                    Expr current_deriv = base_deriv * (*offset_dir[idx]);

                    if (add_to) current_deriv += (*old_gradients)(u, v, idx);

                    current_deriv /= (*divide_by);

                    derivs.push_back(current_deriv);
                }}
                
                if (use_random_par) {{
                    for (int idx = 0; idx < random_par_idx_int.size(); idx++) {{
                        Expr current_deriv = base_deriv * (*offset_sigma[idx]);
                        
                        if (add_to) current_deriv += (*old_gradients)(u, v, idx + NARGS);
                        
                        current_deriv /= (*divide_by);

                        derivs.push_back(current_deriv);
                    }}
                }}
                
                if (derivs.size() > 1) {{
                    gradients(u, v, p) = mux(p, derivs);
                }} else {{
                    gradients(u, v, p) = derivs[0];
                }}
            }}
            
            p_bound = NARGS;
            if (output_base_only) p_bound = 1;
            else if (use_random_par) p_bound += random_par_idx_int.size();
            
            return;
        }}
    
        std::vector<Expr> {GENERATE_RETURN_ARRAY};
    
{generate_str}

        // Add is_nan check to avoid outputting nan value
        if ({GENERATE_RETURN_ARRAY}.size() > 1) {{
        
            Expr ans = mux(p, {GENERATE_RETURN_ARRAY});
            ans = select(is_finite(ans), ans, 0.f);
        
            gradients(u, v, p) = ans;
        }} else {{
            
            Expr ans = {GENERATE_RETURN_ARRAY}[0];
            ans = select(is_finite(ans), ans, 0.f);
        
            gradients(u, v, p) = ans;
        }}
    
    }}

    void schedule() {{
    
        printf("p_bound: %d\\n", p_bound);

        Var uo, ui, vo, vi;

        gradients.reorder(p, u, v).bound(p, 0, p_bound).unroll(p);
        
        innermost.set({{gradients, ui}});

        if (get_target().has_gpu_feature()) {{
            if ({restricted_cond}) {{
                gradients.compute_root()
                    .gpu_tile(u, v, uo, vo, ui, vi, 1, 1);
            }} else {{
                gradients.compute_root()
                    .gpu_tile(u, v, uo, vo, ui, vi, 32, 8);
            }}
        }} else {{
            gradients.compute_root()
                .split(u, uo, ui, 16).vectorize(ui)
                .split(v, vo, vi, 8).parallel(vo);
        }}
    }}
    
private:
    Var u{{"u"}}, v{{"v"}}, p{{"p"}};
    LoopLevel innermost;
    
    Expr center_pix_u_pos, center_pix_v_pos;
    Expr Ou, Ov;
    
    int p_bound;
    
    Input<float> *finite_diff_h, *divide_by;
    std::vector<Input<float> *> offset_dir, offset_sigma;
    Input<Func> *old_gradients;
    
    Input<int> *frame_idx;
    
    std::vector<int> random_par_idx_int;
    Func random_noise;
    std::vector<Input<float> *> random_sigma;
    std::vector<Input<bool> *> do_prune;
    
    Input<Func> *per_pixel_offset, *{CHOOSE_U_PL};
    
    int {pix_idx_declare};
    
    {input_str};
    
    std::vector<int> par_idx_int;
}};

HALIDE_REGISTER_GENERATOR(Shader, shader)
        """
        
    if compiler_params.args_range is None:
        args_range = np.ones(compiler_params.input_nargs)
    else:
        args_range = compiler_params.args_range
        
    if compiler_params.sigmas_range is None:
        sigmas_range = np.ones(compiler_params.input_nargs)
    else:
        sigmas_range = compiler_params.sigmas_range
        
    compiler_params.args_range_str = ', '.join(['%f' % val for val in args_range])
    compiler_params.sigmas_range_str = ', '.join(['%f' % val for val in sigmas_range])
    
    if compiler_params.backend in ['tf', 'np', 'torch']:
        compiler_params.global_code.append('args_range = np.array([%s])\n' % compiler_params.args_range_str)
        compiler_params.global_code.append('\nnargs = (%d)\n' % (compiler_params.input_nargs))
        compiler_params.global_code.append('\ndiscont_idx = np.array([%s])\n' % ', '.join(['%d' % val for val in compiler_params.allow_random_params_idx]))
        compiler_params.global_code.append('\nsigmas_range = np.array([%s])\n' % compiler_params.sigmas_range_str)
    elif compiler_params.backend == 'hl':
        macro_str = '\n#define NARGS (%d)\n' % (compiler_params.input_nargs)
        compiler_params.global_code.append(macro_str)
        
    if compiler_params.backend == 'torch':
        
        X_str = ', '.join(['X' + str(i) for i in range(compiler_params.input_nargs)])
        
        autograd_str = f"""
class CompilerProblem(torch.autograd.Function):

    @staticmethod
    def forward(ctx, current_u, current_v, {X_str}, width, height):
        X = [{X_str}]
        ans = f(current_u, current_v, X, [], [], width, height)
        vec_output = ans[:3]
        
        ctx.save_for_backward(current_u, current_v, torch.tensor(width).float(), torch.tensor(height).float(), *X, *ans)
        ctx.nargs = {compiler_params.input_nargs}
        
        return torch.stack(vec_output, -1)

    @staticmethod
    def backward(ctx, grad_output):
        trace = ctx.saved_tensors
        current_u, current_v, width, height = trace[:4]
        X = trace[4:4+ctx.nargs]
        trace = trace[4+ctx.nargs:]
        
        dL_dcol = []
        for i in range(3):
            dL_dcol.append(grad_output[..., i])
        
        grad_X = g(current_u, current_v, X, dL_dcol, trace, [], width, height)
        grad = []
        
        for val in grad_X:
            #grad.append(torch.mean(val))
            # zero out boundary for safety
            grad.append(torch.nn.functional.pad(val[:, 1:-1, 1:-1], (1, 1, 1, 1)))
            
        
        assert not (ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[-2] or ctx.needs_input_grad[-1]), "uv coordinate, width and height should not be tunable"
        
        return tuple([None, None] +  grad + [None, None])

        """
        
        compiler_params.global_code.append(autograd_str)
        
    c = '\n'.join(compiler_params.global_code) + '\n' + c

    if compiler_params.backend in ['tf', 'np']:
        c = SOURCE_PY_FILE_BEGIN + c
    elif compiler_params.backend == 'torch':
        c = SOURCE_TORCH_FILE_BEGIN + c
    elif compiler_params.backend == 'hl':
        c = SOURCE_HL_FILE_BEGIN + c

    return c, wrapper_str

def update_wrapper_str(compiler_params, vals, mode, lib_name, kernel_idx, exist_previous_bw_kernel, buffer_str, args_params, shared_kernel_idx=None, includes_choose_u=False, includes_producer=False, par_idx=None):

    kernel_suffix, args_str, input_arg = compiler_params.get_wrapper_component(mode)
    
    extra_args = ''
    if shared_kernel_idx is None:
        extra_args += ' kernel_idx=%d ' % kernel_idx
    else:
        extra_args += ' kernel_idx=%d ' % shared_kernel_idx
        
    if 'neighbor' in lib_name or 'filter' in lib_name:
        try:
            seperator_idx = len(lib_name) - lib_name[::-1].index('_')
            pix_idx = int(lib_name[seperator_idx:])
        except:
            pix_idx = 0
            
        if lib_name.startswith('bw_neighbor'):
            extra_args += ' neighbor_idx=%d ' % pix_idx
        elif lib_name.startswith('bw_filter'):
            extra_args += ' filter_idx=%d ' % pix_idx
        
    if par_idx is not None:
        extra_args += ' par_idx=%s ' % ','.join(['%d' % val for val in par_idx])
    
    kernel_name = f"""{COMPILER_PROBLEM}_kernel{kernel_idx}{kernel_suffix}"""
        
    if 'choose_u_pl' in mode:
        if includes_choose_u:
            extra_args += ' use_choose_u_pl=true '
        else:
            replace_mode = 'offset'
            replace_suffix, _, input_arg = compiler_params.get_wrapper_component(replace_mode)
            vals['replace_kernel'][kernel_name] = f"""{COMPILER_PROBLEM}_kernel{kernel_idx}{replace_suffix}"""
            
    if mode in ['choose_u_pl', 'denum_only', 'offset']:
        # These modes are not used in actual optimziation, so it's not important for the producer to also output color image
        extra_args += ' exclude_col=true '

    if lib_name == 'fw':
        
        vals['kernel_call'] += f"""
            if (compute_producer) {{
                {kernel_name}(
        {indent(args_str)}
                    {input_arg} *(output{kernel_idx}.get()));
            """
        if includes_producer and mode in ['offset', 'choose_u_pl']:
            
            # No need to update producer_call because it will NEVER be used with random offset
            for pix_idx in range(1, 5):
                offset_u = 0
                offset_v = 0
                if pix_idx == 1:
                    offset_u = 1
                elif pix_idx == 2:
                    offset_u = -1
                elif pix_idx == 3:
                    offset_v = 1
                else:
                    offset_v = -1
                
                vals['kernel_call'] += f"""
                {kernel_name}(
        {indent(args_params)},
                    uv_offset_0 + ({offset_u}), uv_offset_1 + ({offset_v}),
                    width, height,
                    {input_arg} *(output{kernel_idx+pix_idx}.get()));
        """
        
        vals['kernel_call'] += f"""
            }}
        """
        vals['producer_call'] += f"""
        {kernel_name}(
    {indent(args_str)}
                {input_arg} *(gradient.get()));
        """
    else:
        if mode == 'denum_only' and (exist_previous_bw_kernel or 'cont' in lib_name):
            pass
        else:
            
            if includes_producer and mode in ['offset', 'choose_u_pl']:
                
                producer_str = '*(output0.get()), '
                
                if producer_str in buffer_str:
                    replace_str = producer_str
                    for idx in range(1, 5):
                        replace_str += '*(output%d.get()), ' % idx
                    buffer_str = buffer_str.replace(producer_str, replace_str)
            
            vals['kernel_call'] += f"""
            {kernel_name}(
        {args_str}
                *({DL_DCOL_ARRAY}.get()), {input_arg} {buffer_str}*(output{kernel_idx}.get()));
            """

    if mode == 'denum_only' and (exist_previous_bw_kernel or 'cont' in lib_name):
        pass
    else:
        vals['include_lib'] += f"""
        #include "{kernel_name}.h"
        """
        vals['input_argument'] += f"""
            Buffer<float> output{kernel_idx},
        """
        vals['sync_call'] += f"""
            output{kernel_idx}.device_sync();
        """
        vals['py_arg'] += f"""
            py::arg("output{kernel_idx}"),
        """
        
        if includes_producer and lib_name == 'fw' and mode in ['offset', 'choose_u_pl']:
            for idx in range(1, 5):
                vals['input_argument'] += f"""
            Buffer<float> output{kernel_idx+idx},
            """
                vals['sync_call'] += f"""
            output{kernel_idx+idx}.device_sync();
            """
                vals['py_arg'] += f"""
            py::arg("output{kernel_idx+idx}"),
            """

        vals['kernel_indices'][kernel_idx] = extra_args
    return vals

def get_f_return_declare(compiler_params, lib_name, return_name, kernel_idx, base_p_bound, n_log):
    if compiler_params.backend in ['tf', 'np', 'torch']:

        f_return = '\n' + 'return ' + return_name

        if lib_name == 'fw':
            f_declare = f"""
def f({DEFAULT_ARGUMENT_SCALAR_U_NAME}, {DEFAULT_ARGUMENT_SCALAR_V_NAME}, {DEFAULT_ARGUMENT_ARRAY_NAME}, f_log_intermediate, vec_output, width, height):
    Ou = 0
    Ov = 0
    pix_idx = 0
        """
            compiler_params.global_code.append('f_log_intermediate_len = %d\n' % n_log)
        else:
            # for tf, we always assume there's only one BW kernel
            f_declare = f"""
def g({DEFAULT_ARGUMENT_SCALAR_U_NAME}, {DEFAULT_ARGUMENT_SCALAR_V_NAME}, {DEFAULT_ARGUMENT_ARRAY_NAME}, {DL_DCOL_ARRAY}, output0, vec_output, width, height):
    Ou = 0
    Ov = 0
    pix_idx = 0
            """
    elif compiler_params.backend == 'hl':
        if lib_name == 'fw':
            # special handling for fw
            f_return = f"""
        if (is_fw || is_FD) {{
            std::vector<Expr> {COL_ONLY_ARRAY}({return_name}.begin(), {return_name}.begin() + {base_p_bound});
            p_bound = {base_p_bound};
            return {COL_ONLY_ARRAY};
        }} else return {return_name};}}"""

        else:
            f_return = f"""return {return_name};}}"""
            
        f_declare = f"""
std::vector<Expr> generate{kernel_idx}(std::vector<Expr> params) {{
    std::vector<Expr> {LOCAL_ARRAY}[5];
    for (int pix_idx = 0; pix_idx < 5; pix_idx++) {{
        {LOCAL_ARRAY}[pix_idx] = generate{kernel_idx}_producer(params, pix_idx);
    }}
    
    int pix_idx = 0;
    
    Ou = Halide::cast<int> (0);
    Ov = Halide::cast<int> (0);
    
    Expr current_u = Halide::cast<float> (center_pix_u_pos + Ou);
    Expr current_v = Halide::cast<float> (center_pix_v_pos + Ov);
        """
            
    return f_return, f_declare

def update_generate_str(compiler_params, lib_name, kernel_idx, p_bound, is_final_bw=False):
    ans = f"""
        if (kernel_idx == {kernel_idx}) {{
            p_bound = {p_bound};
            {GENERATE_RETURN_ARRAY} = generate{kernel_idx}(params);
    """
    
    if lib_name == 'bw' or is_final_bw:
        
        ans += f"""
            if (use_random_par) {{
            
                if (par_idx_int.size() != {GENERATE_RETURN_ARRAY}.size()) {{
                    throw;
                }}
                
                int n_valid_sigma = 0;
                for (int idx = 0; idx < par_idx_int.size(); idx++) {{
                    if (par_idx_int[idx] >= 0) {{
                        auto it = std::find(random_par_idx_int.begin(), random_par_idx_int.end(), par_idx_int[idx]);
                        
                        if (it != random_par_idx_int.end()) {{
                            Expr current_deriv_sigma = random_noise(u, v, par_idx_int[idx]) * {GENERATE_RETURN_ARRAY}[idx];
                            {GENERATE_RETURN_ARRAY}.push_back(current_deriv_sigma);
                            n_valid_sigma += 1;
                        }}
                    }}
                }}
                
                p_bound += n_valid_sigma;
            }}
            """
    elif lib_name == 'fw':
        ans += f"""
            if (exclude_col) {{
                p_bound -= 3;
                std::vector<Expr>({GENERATE_RETURN_ARRAY}.begin()+3, {GENERATE_RETURN_ARRAY}.end()).swap({GENERATE_RETURN_ARRAY});
            }}
        """
        
    ans += f"""
        }}"""
    
    return ans

def get_configure_str(compiler_params, lib_name, kernel_idx, input_ls, lambda_get_kernel, includes_producer=False):
    buffer_str = ''
    
    ans = f"""
        if (kernel_idx == {kernel_idx}) {{"""
    
    if not lib_name.startswith('fw'):
        ans += f"""
            {DL_DCOL_ARRAY} = add_input<Func>("{DL_DCOL_ARRAY}", Float(32), 3);"""
        
    ans += f"""
            if (use_per_pixel_offset && !is_FD) {{
                per_pixel_offset = add_input<Func>("per_pixel_offset", Float(32), 3);
            }}
            
            if (use_choose_u_pl) {{
                {CHOOSE_U_PL} = add_input<Func>("{CHOOSE_U_PL}", Bool(1), 3);
            }}
            """
        
    if input_ls is not None:
        for input_idx in range(len(input_ls)):
            
            if isinstance(input_ls[input_idx], int):
                read_idx = input_ls[input_idx]
            else:
                assert isinstance(input_ls[input_idx], list)
                read_idx = input_idx
            
            converted_idx = lambda_get_kernel(read_idx)
            
            buffer_str += '*(output%d.get()), ' % converted_idx
            
            ans += f"""
            output{converted_idx} = add_input<Func>("output{converted_idx}", Float(32), 3);
            """

            if includes_producer and read_idx == 0:
                ans += f"""
            if (use_per_pixel_offset) {{
                    """
            
                for neighbor_idx in range(1, 5):
                    current_output_idx = converted_idx + neighbor_idx
                    ans += f"""
                output{current_output_idx} = add_input<Func>("output{current_output_idx}", Float(32), 3);
                """
                                    
                ans += f"""
            }}
            """
            
    ans += f"""
        }}"""
    
    return ans, buffer_str

def to_expr(const_or_expr):
    """
    Convert constant or expression typically to Expr type but with special cases for handling None or ConstExpr.
    """
    if isinstance(const_or_expr, int_types + float_types):
        return ConstExpr(const_or_expr)
    elif isinstance(const_or_expr, str):
        return const_or_expr
    elif isinstance(const_or_expr, type(None)):
        return None
    elif isinstance(const_or_expr, Expr):
        return const_or_expr
    elif isinstance(const_or_expr, (np.ndarray, list)):
        return Compound(const_or_expr)

    raise ValueError('unknown case: ', const_or_expr)

def indent(s, count=4):
    lines = s.split('\n')
    return '\n'.join(' '*count + line for line in lines)

#redundant_index = 0

def remove_redundant_exprs(e, keep_ids=set(), replaced_nodes={}):
    """
    Return copy of Expr graph with redundant Exprs consolidated into a single instance.
    """
    
    e.calc_parents()
    all_nodes = e.all_nodes_dfs()
    
    repr_to_expr = {}
    repr_cache = {}
    
    for node in all_nodes:
        r = node.repr(False, repr_cache)
        seen = r in repr_to_expr.keys()
            
        if seen:
            if id(node) in keep_ids:
                replaced_nodes[id(node)] = repr_to_expr[r]
            for par in node.parents:
                for child_idx, child in enumerate(par.children):
                    if node is child:
                        par.children[child_idx] = repr_to_expr[r]
        else:
            repr_to_expr[r] = node
            
    e.calc_parents()
    print('finished calculate_parents')

    return e

def remove_pow(e):
    """
    Remove pow and add safeguard to all divisions
    """
    
    e.remove_pow()
    #e.calc_parents()
    #e.resolve_div()
    
    return e

def identical(a, b):
    """
    Check whether either Expr or list of Exprs a and b are identical (cast elements to Expr using to_expr() if needed).
    """
    if isinstance(a, Expr):
        return a.identical(b)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            av = a[i]
            bv = b[i]
            if not isinstance(av, Expr):
                av = to_expr(av)
            if not isinstance(bv, Expr):
                bv = to_expr(bv)
            if not av.identical(bv):
                return False
        return True

def linenos_from_frame(current_frame, depth=10):
    ans = []
    ans.append(current_frame.f_lineno)
    for i in range(depth):
        current_frame = current_frame.f_back
        if current_frame is None:
            break
        ans.append(current_frame.f_lineno)
    return ans

class Expr:
    """
    Expression type.

    Attributes:
    children --- A list of children for the expression (of any type, but AST nodes are of type Expr).
    dtype --- Type of expression, one of INT_TYPE or REAL_TYPE ('double').
    comment --- Add at the end of the line as comment, for extra information or debugging purpose
    """

    def __init__(self, frame_depth=2):
        self.children = []
        self.recurse_to_source_indices = None
        current_frame = inspect.currentframe()
        self.frame_lineno = linenos_from_frame(current_frame)
        self.comment = ''
        self.to_source_varname = None
        self.root = False
        self.backprop_finished = False
        self.dL_dself = {}
        self.dL_dself_scalar = {}
        self.dL_mask = None
        self.is_deriv = DEFAULT_IS_DERIV
        self.wrt_id = DEFAULT_WRT_ID
        self.wrt_name = DEFAULT_WRT_NAME
        self.pix_idx = None
        self.is_discont_to_output = False
        self.is_discont_to_params = False
        self.is_dependency_to_raymarching = False
        self.params_only = -1
        self.is_scratch = scratch_scope
        self.ndims = 0
        self.allow_ours = DEFAULT_ALLOW_OURS    # Even when set to false, will still go through our gradient for Boolean and Select, but will bypass our rule on blackbox functions and binary floating point operations

    def __copy__(self):
        """
        Create a shallow copy which does not share the children list.
        """
        cls = self.__class__
        ans = cls.__new__(cls)
        ans.__dict__.update(self.__dict__)
        ans.children = list(ans.children)
        return ans
    
    def __getitem__(self, index):
        assert self.ndims > 0
        return GetItem(self, index)
    
    def reset_attr(self, func, seen=set()):
        if id(self) in seen:
            return
        seen.add(id(self))
        
        for child in self.children:
            if isinstance(child, Expr):
                child.reset_attr(func, seen)
                
    def fix_attr(self, old):
        """
        When creating a node to replace an old node
        call this method to make sure attributes in the new node is consistent with old node
        
        For attributes propagated from parent nodes: set it to be the same with old nodes (because the new node replaces old node, therefore they have exactly the same parents)
        For attributes propagated from children nodes: compute them again from children
        """
        self.is_discont_to_output = old.is_discont_to_output
        self.is_dependency_to_raymarching = old.is_dependency_to_raymarching
        
        for child in self.children:
            if isinstance(child, Expr):
                self.is_discont_to_params = self.is_discont_to_params or child.is_discont_to_params
        self.propagate_params_dependency()
    
    def is_comparison(self):
        if isinstance(self, BinaryOp):
            if self.op in ['<', '>', '<=', '>=', '==', '!=']:
                return True
        return False
    
    def get_comparison_primitives(self):
        
        if (not isinstance(self, (BinaryOp, UnaryOp))) or (getattr(self, 'op', '') not in ['||', '&&', '!']):
            return [self]
        
        seen = set()
        ans = []
        
        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))

            for child in node.children:
                if isinstance(child, Expr):
                    if isinstance(child, (BinaryOp, UnaryOp)) and getattr(child, 'op', '') in ['||', '&&', '!']:
                        # if still a Boolean expression, continue search
                        visit(child)
                    else:
                        # otherwise log the primitive
                        ans.append(child)
        visit(self)
        return ans
    
    def depend_on_uv(self, compiler_params):
        
        if id(self) in compiler_params.depend_on_uv_cache:
            return compiler_params.depend_on_uv_cache[id(self)]
        
        if isinstance(self, ArgumentScalar) and getattr(self, 'name', '') in [DEFAULT_ARGUMENT_SCALAR_U_NAME, 
                                                                              DEFAULT_ARGUMENT_SCALAR_V_NAME]:
            ans = True
        elif self.params_only > -1:
            if self.params_only <= 2:
                ans = True
            else:
                ans = False
        else:
            ans = False

            for child in self.children:
                if isinstance(child, Expr):
                    ans = ans or child.depend_on_uv(compiler_params)
                    
        compiler_params.depend_on_uv_cache[id(self)] = ans
        return ans
    
    def get_cond_parents(self, compiler_params):
        if self.root or len(getattr(self, 'parents', [])) == 0:
            return []
        
        if id(self) in compiler_params.cond_parents_lookup.keys():
            return compiler_params.cond_parents_lookup[id(self)]

        ans = []
        self_added = False
        for par in self.parents:
            if isinstance(par, Call) and par.name == 'select' and id(self) == id(par.children[1]):
                if not self_added:
                    ans.append(self)
                    self_added = True
            else:
                ans += par.get_cond_parents(compiler_params)
                
        compiler_params.cond_parents_lookup[id(self)] = ans
        
        return ans
    
    def propagate_params_dependency(self, compiler_params=None):
        """
        Propagate dependency to parameters from child nodes with the following semantic:
        self.params_only = 0: depend on everything
        self.params_only = 1: depend on ArgumentScalar and continuous parameters
        self.params_only = 2: depend on ArgumentScalar only
        self.params_only = 3: depends on parameters only, but with dependency on discontinuous parameters
        self.params_only = 4: depends on continuous parameters only
        """
        if isinstance(self, ArgumentScalar):
            self.params_only = 2
        elif isinstance(self, ArgumentArray):
            if self.name == DEFAULT_ARGUMENT_ARRAY_NAME:
                self.params_only = 3
            else:
                # FW should depend on no other ArgumentArray
                raise
                self.params_only = 0
        elif isinstance(self, GetItem):
            if compiler_params is None:
                # Should never reach this branch
                # compiler_params = None should only be used when calling fix_attr(), which is always on BinaryOp
                raise
            if isinstance(self.array, ArgumentArray) and getattr(self.array, 'name', '') == DEFAULT_ARGUMENT_ARRAY_NAME and is_all_constant(self.index):
                idx_value = eval(self.index.to_source(compiler_params.as_mode(MODE_ALWAYS_INLINE)))

                if idx_value not in compiler_params.allow_random_params_idx:
                    self.params_only = 4
                else:
                    self.params_only = self.array.params_only
            else:
                self.params_only = self.array.params_only
        else:
            children_params_only_set = set([child.params_only for child in self.children if isinstance(child, Expr)])
            if len(children_params_only_set) == 0:
                self.params_only = 4
            elif len(children_params_only_set) == 1:
                self.params_only = children_params_only_set.pop()
            elif 0 in children_params_only_set:
                self.params_only = 0
            else:
                has_scalar = (1 in children_params_only_set) or (2 in children_params_only_set)
                has_cont = (1 in children_params_only_set) or (4 in children_params_only_set)
                has_discont = 3 in children_params_only_set
                
                if has_discont:
                    if has_scalar:
                        self.params_only = 0
                    else:
                        self.params_only = 3
                else:
                    if has_scalar:
                        if has_cont:
                            self.params_only = 1
                        else:
                            self.params_only = 2
                    else:
                        self.params_only = 4
    
    def propagate_discont(self, forward=False):
        """
        Propagate discont from child nodes (forward=False) or from parent nodes (forward=True)
        At certain classes (e.g. Func), propagate_discont is reimplemented such that some node will actually be set to is_discont_to_output=True
        """
        
        if forward:
            node_ls = getattr(self, 'parents', [])
            discont_attr = ['is_discont_to_params']
        else:
            node_ls = self.children
            discont_attr = ['is_discont_to_output', 'is_dependency_to_raymarching']
            
        for node in node_ls:
            if isinstance(node, Expr):
                for attr_name in discont_attr:
                    setattr(node, 
                            attr_name, 
                            getattr(node, attr_name) or getattr(self, attr_name))
    
    def cut_by(self, node_list):
        """
        Returns true if node_list forms a cut of the graph and self is on the leaf side of the cut
        """
        
        if id(self) in node_list:
            return True
        
        if not hasattr(self,'parents') or self.parents is None or len(self.parents) == 0:
            return False

        for parent in self.parents:
            if not parent.cut_by(node_list):
                return False
        return True
    
    def subs_scalar(self, source_ls, dest_ls, attr_setter=None, verbose=False):
        """
        Return a deep copy of self that has every ArgumentScalr in source_ls replaced with corresponding Expr in dest_ls, recursively.
        If attr_setter is not None, run setter for every traversed node
        """
        
        self.clear_parents()
        self = copy.deepcopy(self)
        self.calc_parents()
        
        T0 = time.time()
        for node in self.all_nodes():
            if isinstance(node, ArgumentScalar):
                for source_idx in range(len(source_ls)):
                    source = source_ls[source_idx]
                    if node.name == source.name:
                        node_id = id(node)
                        for parent in node.parents:
                            for i in range(len(parent.children)):
                                if id(parent.children[i]) == node_id:
                                    parent.children[i] = dest_ls[source_idx]
                        break
            if attr_setter is not None and isinstance(node, Expr):
                attr_setter(node)
        T1 = time.time()
        
        if verbose:
            print('time to finishe traversal: ', T1 - T0)
                
        return self
    
    def subs_id(self, source_ls, shallow_dict, dest_ls, stop_ls=set()):
        """
        Return a deep copy of self that has every node whose id are contained in source_ls replaced with corresponding Expr in dest_ls, recursively.
        If stop_ls is nonempty, traversal should stop once the id of current node is contained in stop_ls
        """
        
        self.clear_parents()
        self = copy.deepcopy(self, memo=shallow_dict)
        self.calc_parents(terminals=set(source_ls + list(stop_ls)))
        
        T0 = time.time()
        for node in self.all_nodes(exclude=stop_ls):
            node_id = id(node)
            if node_id in source_ls:
                source_idx = source_ls.index(node_id)
                
                for parent in node.parents:
                    for i in range(len(parent.children)):
                        if id(parent.children[i]) == node_id:
                            parent.children[i] = dest_ls[source_idx]

        T1 = time.time()
                
        return self
    
    def subs(self, source_lambda_ls=None, dest_ls=[], attr_setter=None):
        """
        Return a deep copy of self that has every child node whenever child.source_labmda evaluates to True, being replaced by the corresponding Expr in dest_ls
        If source_lambda_ls is None, defaults to matching repr(child) with repr(dest)
        If attr_setter is not None, run setter for every traversed node
        """
        
        self = copy.deepcopy(self)
        self.calc_parents()
        
        if len(dest_ls) < 1:
            return self
        
        if source_lambda_ls is None:
            def func(source, dest):
                return source.unique_str() == dest.unique_str()
            source_lambda_ls = func
            
        if callable(source_lambda_ls):
            source_lambda_ls = [source_lambda_ls] * len(dest_ls)
        else:
            assert isinstance(source_lambda_ls, list)
            
        assert len(source_lambda_ls) == len(dest_ls)

        for node in self.all_nodes():
            for source_idx in range(len(source_lambda_ls)):
                if source_lambda_ls[source_idx](node, dest_ls[source_idx]):
                    node_id = id(node)
                    for parent in node.parents:
                        for i in range(len(parent.children)):
                            if id(parent.children[i]) == node_id:
                                parent.children[i] = dest_ls[source_idx]
                    break
            if attr_setter is not None and isinstance(node, Expr):
                attr_setter(node)
                
        return self

    def unique_str(self):

        ans = self.repr(False)
        return ans

    def identical(self, b):
        """
        Returns bool for whether self and b are identical expressions without attempting any simplification.
        """
        return self.unique_str() == b.unique_str()
    
    def resolve_div(self):
        """
        Add safeguard to any division
        """
        
        for node in self.all_nodes_dfs():
            if isinstance(node, BinaryOp) and getattr(node, 'op', '') == '/':
                
                safe = True
                safe_val = None
                
                for par in node.parents:
                    if not getattr(par, 'op', '') == 'safe_division':
                        safe = False
                    else:
                        safe_val = par
                        
                if not safe:
                    if safe_val is None:
                        safe_val = safe_division(node.a, node.b, div_val=node)
                        
                    for par in node.parents:
                        if not getattr(par, 'op', '') == 'safe_division':
                            for child_idx in range(len(par.children)):
                                if par.children[child_idx] is node:
                                    par.children[child_idx] = safe_val
    
    def remove_pow(self, seen=None):
        """
        Trying to rewrite power into multiplication / division / sqrt as much as possible
        """
        if seen is None:
            seen = {}
            
        id_self = id(self)
        if id_self in seen.keys():
            return seen[id_self]
        
        for (i, child) in enumerate(self.children):
            if isinstance(child, Expr):
                if id(child) not in seen.keys():
                    cp = child.remove_pow(seen)
                    self.children[i] = cp
                else:
                    self.children[i] = seen[id(child)]
        
        ans = self.remove_pow_impl()
        seen[id_self] = ans
        
        return ans
    
    def remove_pow_impl(self):
        if isinstance(self, BinaryOp):
            if self.op == '**':
                if is_all_constant(self.b):
                    cp = CompilerParams()
                    b_str = self.b_str(cp)
                    
                    b_val = eval(b_str)
                    
                    if b_val < 0:
                        need_div = True
                        b_val *= -1
                    else:
                        need_div = False
                    
                    if b_val == 0:
                        ans = ConstExpr(1)
                    elif b_val == 0.5:
                        ans = sqrt_f(self.a)
                    elif b_val == 1.5:
                        ans = self.a * sqrt_f(self.a)
                    elif b_val == int(b_val):
                        ans = self.a
                        for _ in range(int(b_val) - 1):
                            ans = ans * self.a
                    else:
                        ans = None
                        
                    if ans is not None:
                        if need_div:
                            if hasattr(self, 'short_name'):
                                ans.short_name = self.short_name + '_inv'
                            ans = BinaryOp('/', ConstExpr(1), ans)
                            
                        ans.short_name = getattr(self, 'short_name', '')
                            
                        return ans
                    
            elif self.op == '*':
                cp = CompilerParams()
                arg_list = [(self.a, self.b), (self.b, self.a)]
                for (a_expr, b_expr) in arg_list:
                    if isinstance(a_expr, BinaryOp) and getattr(a_expr, 'op', '') == '/':
                        a_a = a_expr.a_str(cp)
                        if is_constant(a_a, 1):
                            ans = BinaryOp('/', b_expr, a_expr.b)
                            ans.short_name = getattr(self, 'short_name', '')
                            return ans
        return self
                    
    
    

    def simplify(self, seen=None, keep_ids=set(), replaced_nodes={}):
        """
        Simplifies the given Expr in place (recursively), returning the simplified Expr.
        """
        if seen is None:
            seen = {}
        id_self = id(self)
        if id_self in seen:
            return seen[id_self]
        self.simplify_children(seen, keep_ids=keep_ids, replaced_nodes=replaced_nodes)
        ans = self.simplify_impl()
        seen[id_self] = ans
        if id_self in keep_ids and id_self != id(ans):
            replaced_nodes[id_self] = ans
        return ans

    def simplify_impl(self):
        """
        Simplifies the given Expr but not its children in place, returning the simplified Expr.
        """
        return self

    def simplify_children(self, seen, keep_ids=set(), replaced_nodes={}):
        """
        Simplifies the children of the given Expr in place.
        """
        for (i, child) in enumerate(self.children):
            if hasattr(child, 'simplify'):
                if id(child) not in seen:
                    cp = child.simplify(seen, keep_ids=keep_ids, replaced_nodes=replaced_nodes)
                    if not isinstance(cp, Expr) and cp is not None:
                        raise ValueError('Bad type for simplified expression', (cp, type(cp), child))
                    self.children[i] = cp
                else:
                    self.children[i] = seen[id(child)]

    def clear_parents(self):
        for node in self.all_nodes():
            if hasattr(node, 'parents'):
                del node.parents
      
    def clear_dL_dself(self):
        for node in self.all_nodes():
            node.dL_dself = {}
            node.dL_mask = None

    def calc_parents(self, terminals=None):
        all_nodes = self.all_nodes()
        for node in all_nodes:
            node.parents = []
        print("all nodes parents set null")
        for node in self.all_nodes(exclude=terminals):
            
            for child in node.children:
                if isinstance(child, Expr):
                    child.parents.append(node)

    def is_constant(self):
        """
        Is this a constant expression? (ConstExpr or a tree containing operators, calls, and constants)
        """
        return all(isinstance(node, (ConstExpr, BinaryOp, UnaryOp, Call)) for node in self.all_nodes())

    def check_acyclic(self, allow_visit=None):
        """
        Raise an exception if there is a cycle in the Expr graph. Otherwise, return self.
        """
        seen = set()
        ans = []
        def visit(node, parents):
            if id(node) in parents:
                raise ValueError('cycle detected')
            if id(node) in seen:
                return
            seen.add(id(node))
            parents = parents | set([id(node)])
            for child in node.children:
                if isinstance(child, Expr):
                    if (allow_visit is None or allow_visit(node, child)):
                        visit(child, parents)
            ans.append(node)
        visit(self, set())
        return self

    def all_nodes_generator(self):
        """
        A generator yielding all nodes in depth-first order.
        """
        seen = set()
        next = [self]
        while len(next):
            current = next.pop()
            id_current = id(current)
            if not id_current in seen:
                seen.add(id_current)
                if isinstance(current, Expr):
                    yield current
                    next.extend(current.children)
                    
    def all_parents(self, seen=set()):
        ans = []
        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            
            ans.append(node)
            for par in getattr(node, 'parents', []):
                visit(par)
        visit(self)
        return ans

    def all_nodes_dfs(self, seen=set(), allow_visit=None, order='dfs', exclude=set(), terminal=set(), terminal_only=False):
        """
        Returns all children nodes including self (Expr subclasses only), in depth-first order, bottom up.

        This order implies that Exprs are visited in dependency order, so an Expr is visited before any parent that depends on it.
        If exclude is nonempty, recursion will stop at the nodes whose id are in the exclude set, those nodes themselves as well as their children will not be include in the output
        
        if terminal_only = True, only return nodes whose ids are within set terminal
        """
        if order == 'dfs':
            order_normal = True
        elif order == 'dfs_random':
            order_normal = False
        else:
            raise ValueError('unknown order %r' % order)
        seen = seen.union(exclude)
        ans = []
        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            if id(node) in terminal:
                pass
            else:
                for child in (node.children if order_normal else compiler_util.shuffled(node.children)):
                    if isinstance(child, Expr):
                        if (allow_visit is None or allow_visit(node, child)):
                            visit(child)
            
            if id(node) in terminal or (not terminal_only):
                ans.append(node)
        visit(self)
        return ans

    def all_nodes_order(self, order):
        if order in ['dfs', 'dfs_random']:
            return self.all_nodes_dfs(order=order)
        elif order == 'bfs':
            return self.all_nodes()
        else:
            raise ValueError('unknown order')

    def all_nodes(self, allow_visit=None, exclude=None):
        """
        Recursive nodes including self (Expr subclasses only), in breadth-first order, top down.

        If allow_visit is not None then allow_visit(parent, child) should give a bool for whether to visit the given child.
        """
        ans = [self]
        if exclude is not None:
            seen = exclude
            seen.add(id(self))
        else:
            seen = set([id(self)])
        def visit(node):

            visit_next = []
            for child in node.children:
                if isinstance(child, Expr):
                    if id(child) not in seen and (allow_visit is None or allow_visit(node, child)):
                        ans.append(child)
                        seen.add(id(child))
                        visit_next.append(child)
            for child in visit_next:
                visit(child)
        visit(self)
        return ans
    
    def all_grandparents_id(self, all_visit=None):
        """
        Recursive nodes, parents, and grandparents including self (Expr subclass only)
        If allow_visit is not None then allow_visit(parent, child) should give a bool for whether to visit the given child.
        """
        ans = set([id(self)])
        def visit(node):
            visit_next = []
            if hasattr(node, 'parents'):
                for pa in node.parents:
                    if isinstance(pa, Expr):
                        if id(pa) not in ans and (allow_visit is None or allow_visit(pa, node)):
                            ans.add(id(pa))
                            visit_next.append(pa)
            for pa in visit_next:
                visit(pa)
        visit(self)
        return ans
    
    def gradient_wrapper(self, compiler_params):
        if self.ndims == 0:
            child_deriv = self.gradient(compiler_params, None)
        else:
            
            child_deriv = None
            
            for idx in range(self.ndims):
                current_child_deriv = self.gradient(compiler_params, idx)
                
                if child_deriv is None:
                    child_deriv = []
                    for child, _ in current_child_deriv:
                        child_deriv.append((child, [None] * self.ndims))
                
                assert len(child_deriv) == len(current_child_deriv)

                for child_idx in range(len(current_child_deriv)):
                    child_deriv[child_idx][1][idx] = current_child_deriv[child_idx][1]
                        
            for child_idx in range(len(child_deriv)):
                child_deriv[child_idx] = (child_deriv[child_idx][0], Compound(child_deriv[child_idx][1]))
                        
        return child_deriv
    
    def backprop(self, compiler_params):
        """
        Compute propagate dL/dself to children
        """
        
        if compiler_params.backprop_source not in self.dL_dself.keys():
            # skipped
            return
        
        global DEFAULT_WRT_ID
        global DEFAULT_WRT_NAME
        
        DEFAULT_WRT_ID = id(self)
        DEFAULT_WRT_NAME = getattr(self, 'short_name', None)
                        
        child_deriv = self.gradient_wrapper(compiler_params)
        
        for child, deriv in child_deriv:
            
            if isinstance(child, RaymarchingLoop) and self.is_discont_to_output:
                # if self.is_discont_to_output = True, this indicates we are already inside a condition
                # in this case, the backprop method in RaymarchingLoop will find corresponding comparison node directly
                # the current dL_dself already includes undesirable dcond_diff/du, shoudl discard
                continue
            
            DEFAULT_WRT_ID = id(child)
            DEFAULT_WRT_NAME = getattr(child, 'short_name', None)
            
            if isinstance(deriv, list):
                assert isinstance(deriv[0], Expr)
                mask = deriv[1]
                deriv = deriv[0]
            else:
                mask = self.dL_mask
                
            extra_scalar_deriv = self.dL_dself_scalar.get(compiler_params.backprop_source, 1)
                
            if self.ndims > 0 and child.ndims == 0:
                # vector to vector dot product
                dot_elements = deriv * self.dL_dself[compiler_params.backprop_source]
                extra_deriv = sum([dot_elements[idx] for idx in range(self.ndims)], 0)
            else:
                # scalar to scalr
                if self.ndims == 0:
                    extra_deriv = deriv * self.dL_dself[compiler_params.backprop_source]
                elif deriv.ndims == 0:
                    # vector to scalar multiplication
                    extra_deriv = self.dL_dself[compiler_params.backprop_source]
                    extra_scalar_deriv = extra_scalar_deriv * deriv
                else:
                    # pointwise vector to vector multiplication
                    extra_deriv = deriv * self.dL_dself[compiler_params.backprop_source]
                
            if compiler_params.backprop_source not in child.dL_dself.keys():
                child.dL_dself[compiler_params.backprop_source] = extra_deriv
                child.dL_dself_scalar[compiler_params.backprop_source] = extra_scalar_deriv
            else:
                child.dL_dself[compiler_params.backprop_source] = child.dL_dself[compiler_params.backprop_source] + \
                                                                  extra_deriv * extra_scalar_deriv
                
            if compiler_params.gradient_mode == 'implicit_current':
                if isinstance(mask, Expr) and child.dL_mask is None:
                    child.dL_mask = mask
                elif mask is False:
                    child.dL_mask = False
                
        DEFAULT_WRT_ID = None
        DEFAULT_WRT_NAME = None
                
    def gradient(self, compiler_params, get_idx):
        """
        Compute dself/dchildren
        """
        
        raise NotImplementedError(self.__class__)

    def var_name(self, compiler_params, check=True):
        
        if check and not need_animate and compiler_params.check_varname:
            # A sanity check to see if a node is visited at least after one of its parent is visited
            success = False
            if hasattr(self, 'parents') and len(getattr(self, 'parents', [])):
                for parent in self.parents:
                    if id(parent) in compiler_params.name_to_order:
                        success = True
                        break
                if not success:
                    if not isinstance(self, Compound) and not isinstance(self.parents[0], (Compound)):
                        raise 'Check parent varname exist fails'

        if hasattr(self, 'short_name'):
            short_name = self.short_name
        elif hasattr(self, 'name'):
            short_name = self.name
        elif self.is_deriv:
            short_name = '_dans'
            if self.wrt_name is not None:
                short_name += '_' + self.wrt_name
            if self.pix_idx is not None:
                short_name += '_pix' + str(self.pix_idx)
        else:
            short_name = ''
                
        ans = compiler_params.get_varname(short_name, id(self))
        self.to_source_varname = ans

        return ans


    def to_source_expr(self, compiler_params, dummy=None):
        """
        Convert a side-effect free expression to source code.
        """
        raise NotImplementedError(self.__class__)

    def debug_return(self, compiler_params, retval):
        if retval is None:
            
            raise ValueError
        
        return retval

    def is_inline(self, compiler_params):
        
        if compiler_params.mode in [MODE_ALWAYS_INLINE]:
            return True
        return False

    def to_source_inline(self, compiler_params, dummy=None):
        
        if compiler_params.mode in [MODE_VARNAME, MODE_ALWAYS_INLINE]:
            return self.to_source_impl(compiler_params.as_mode(MODE_ALWAYS_INLINE), dummy)
        elif compiler_params.mode == MODE_SIDE_EFFECTS:
            return ''
        else:
            raise ValueError('unknown mode', compiler_params.mode)
        
    def to_source(self, compiler_params, dummy=None):
        
        if self.is_inline(compiler_params):
            ans = self.to_source_inline(compiler_params, dummy)
        else:
            assert dummy is None
            ans = self.to_source_impl(compiler_params, dummy)
            ans = self.debug_return(compiler_params, ans)
        return ans

    def to_source_recurse(self, compiler_params, ans):
        cp = compiler_params.as_mode(MODE_SIDE_EFFECTS)
        
        children = [child for child in self.children if isinstance(child, Expr)]
        if self.recurse_to_source_indices is not None:
            children = children[self.recurse_to_source_indices]
        prepend = ''
        for child in children:
            sub = child.to_source(cp)
            if len(sub):
                prepend = prepend + '\n' + sub
        return prepend + '\n' + ans

    def statement_id(self, compiler_params):
        """
        Get an "id" key for the set CompilerParams.statement_ids.
        """
        return id(self)
    
    def get_declare_str(self, compiler_params):
        if compiler_params.backend in ['tf', 'np', 'torch']:
            declare_str = ''
        elif compiler_params.backend == 'hl':
            if self.ndims == 0:
                declare_str = 'Expr '
            else:
                declare_str = 'std::vector<Expr> '
        elif compiler_params.backend == 'glsl':
            if self.ndims <= 1:
                declare_str = self.dtype + ' '
            elif self.ndims <= 4:
                declare_str = 'vec%d ' % self.ndims
            else:
                declare_str = 'float [%d] ' % self.ndims
        else:
            raise
            
        return declare_str
            

    def to_source_impl(self, compiler_params, dummy=None):
        """
        Convert to source code in the given language.

        In the base class this assumes a side-effect free expression is used, and implemented in to_source_expr().
        However, this behavior can be overridden by implementing a different to_source() in subclasses.
        """

        if compiler_params.mode == MODE_VARNAME:
            assert dummy is None
            return self.var_name(compiler_params)
        elif compiler_params.mode in [MODE_ALWAYS_INLINE]:
            key = id(self)
            if key in compiler_params.cache_to_source:
                assert dummy is None
                return compiler_params.cache_to_source[key]
            ans = self.to_source_expr(compiler_params, dummy)
            compiler_params.cache_to_source[key] = ans
            return ans
        elif compiler_params.mode == MODE_SIDE_EFFECTS:
            assert dummy is None
            
            self_id = self.statement_id(compiler_params)
            if self_id in compiler_params.statement_ids:
                # avoid duplicate
                return ''
            compiler_params.statement_ids.add(self_id)
            
            declare_str = self.get_declare_str(compiler_params)
                    
            lbrace, rbrace = '', ''
            if self.ndims > 0:
                if compiler_params.backend in ['tf', 'np', 'torch']:
                    lbrace, rbrace = '[', ']'
                elif compiler_params.backend == 'hl':
                    lbrace, rbrace = '{', '}'
                
            
            end_str = ';' if compiler_params.backend in ['hl', 'glsl'] else ''
            
            if hasattr(self, 'bind_scalars'):
                #assert self.root
                assign_str = ''
                
                for idx in range(len(self.bind_scalars)):
                    assign_str += f"""
{self.bind_scalars[idx].to_source(compiler_params.as_mode(MODE_VARNAME), dummy)} = {self.bind_values[idx].to_source(compiler_params.as_mode(MODE_VARNAME)), dummy}{end_str}
                    """
            else:
                assign_str = ''
                
            lhs = self.var_name(compiler_params)
            rhs = self.to_source_expr(compiler_params.as_mode(MODE_VARNAME), dummy)

            ans = assign_str
            ans += '\n' + self.to_source_recurse(compiler_params, '')
            
            def ljust(s):
                return (s).ljust(50)

            if self.dtype != VOID_TYPE:

                this_line = declare_str + lhs + ' = ' + lbrace + rhs + rbrace + end_str

                ans += '\n' + ljust(this_line)
            else:
                ans += '\n' + ljust(declare_str + rhs + end_str)
            
            return ans
        else:
            raise ValueError('unhandled mode', compiler_params.mode)

    def repr(self, extra_info=True, cache=None):

        if cache is None:
            cache = dict()
        cache_key = id(self)

        if cache_key in cache:
            extra_s = ''
            if extra_info:
                extra_s = 'id: ' + str(id(self)) + ', '

            return self.__class__.__name__ + '(%srepeated %s)' % (extra_s, cache[cache_key])

        line_length = 80
        sub_repr = [x.repr(extra_info, cache) if hasattr(x, 'repr') else repr(x) for x in self.children]
        sub_repr_len = sum([len(x) for x in sub_repr])


        if extra_info:
            if hasattr(self, 'parents'):        # If calc_parents() called, include extra info in repr().
                sub_repr = ['parents: [' + ', '.join(str(id(p)) for p in self.parents) + ']'] + sub_repr
            sub_repr = ['id: ' + str(id(self))] + sub_repr
        
        if sub_repr_len < line_length and all([not hasattr(node, 'children') or node.children == [] or isinstance(node, ConstExpr) for node in self.children]):
            ans = self.__class__.__name__ + '(' + (', '.join(sub_repr)) + ')'
        else:
            ans = self.__class__.__name__ + '(\n' + indent(',\n'.join(sub_repr)) + '\n)'


        cache[cache_key] = hashlib.md5(ans.encode('utf-8')).hexdigest()


        return ans

    def __repr__(self):
        return self.repr()

    def __str__(self):
        return self.__class__.__name__

    def __iadd__(self, other):
        if self.is_seq(other):
            return numpy.add(self, other)
        return BinaryOp('+', self, other)

    def is_seq(self, other):
        return hasattr(self, '__len__') or hasattr(other, '__len__')

    def __add__(self, other):
        if self.is_seq(other):
            return numpy.add(self, other)
        return BinaryOp('+', self, other)

    def __radd__(self, other):
        if self.is_seq(other):
            return numpy.add(self, other)
        return BinaryOp('+', other, self)

    def __mul__(self, other):
        
        if self.is_seq(other):
            return numpy.multiply(self, other)
        return BinaryOp('*', self, other)

    def __rmul__(self, other):

        if self.is_seq(other):
            return numpy.multiply(self, other)
        return BinaryOp('*', other, self)

    def __mod__(self, other):
        if self.is_seq(other):
            return numpy.mod(self, other)
        return BinaryOp('*', fract(BinaryOp('/', self, other)), other)

    def __rmod__(self, other):
        if self.is_seq(other):
            return numpy.mod(other, self)
        return BinaryOp('*', fract(BinaryOp('/', other, self)), self)

    def __truediv__(self, other):
        if self.is_seq(other):
            return numpy.true_divide(self, other)
        #return BinaryOp('/', self, other)
        # Make gradient simpler by handling one less type of operator
        return BinaryOp('*', self, BinaryOp('**', other, -1.0))

    def __rtruediv__(self, other):
        if self.is_seq(other):
            return numpy.true_divide(other, self)
        #return BinaryOp('/', other, self)
        # Make gradient simpler by handling one less type of operator
        return BinaryOp('*', other, BinaryOp('**', self, -1.0))

    def __floordiv__(self, other):
        if self.is_seq(other):
            return numpy.floor_divide(self, other)
        #return BinaryOp('/', self, other)
        # Make gradient simpler by handling one less type of operator
        return BinaryOp('*', self, BinaryOp('**', other, -1.0))

    def __rfloordiv__(self, other):
        if self.is_seq(other):
            return numpy.floor_divide(other, self)
        #return BinaryOp('/', other, self)
        # Make gradient simpler by handling one less type of operator
        return BinaryOp('*', other, BinaryOp('**', self, -1.0))
    

    def __sub__(self, other):
        if self.is_seq(other):
            return numpy.subtract(self, other)
        return BinaryOp('-', self, other)

    def __rsub__(self, other):
        if self.is_seq(other):
            return numpy.subtract(other, self)
        return BinaryOp('-', other, self)

    def __neg__(self):
        return UnaryOp('-', self)

    def __pos__(self):
        return self

    def __pow__(self, other):
        if self.is_seq(other):
            return numpy.power(self, other)
        if isinstance(other, int_types + float_types):
            if other == 1:
                return self
            elif other == 0:
                return 1.0
        return BinaryOp('**', self, other)

    def __rpow__(self, other):
        if self.is_seq(other):
            return numpy.power(other, self)
        return BinaryOp('**', other, self)

    def __lt__(self, other):
        #return UnaryOp('> 0', BinaryOp('-', other, self))
        return BinaryOp('<', self, other)

    def __le__(self, other):
        #return UnaryOp('>= 0', BinaryOp('-', other, self))
        return BinaryOp('<=', self, other)

    def __eq__(self, other):
        #return UnaryOp('== 0', BinaryOp('-', self, other))
        return BinaryOp('==', self, other)
        #return equal(self, other)

    def __ne__(self, other):
        #return UnaryOp('!= 0', BinaryOp('-', self, other))
        return BinaryOp('!=', self, other)
        #return nequal(self, other)

    def __gt__(self, other):
        #return UnaryOp('> 0', BinaryOp('-', self, other))
        return BinaryOp('>', self, other)

    def __ge__(self, other):
        #return UnaryOp('>= 0', BinaryOp('-', self, other))
        return BinaryOp('>=', self, other)

    def __abs__(self):
        return Func('abs',
                    gradient_lambda=lambda x, *args: [sign(x[0])],
                    tf_name = 'wrapper("abs")')(self)
    
    # Use bitwise operation to represent actual binary operation
    # becasue python does not overwritting the actual "and", "or", "not"
    def __and__(self, other):
        return BinaryOp('&&', self, other)
    
    def __rand__(self, other):
        return BinaryOp('&&', self, other)
    
    def __or__(self, other):
        return BinaryOp('||', self, other)
    
    def __ror__(self, other):
        return BinaryOp('||', self, other)
    
    def __invert__(self):
        return UnaryOp('!', self)

class AlwaysInlineExpr(Expr):
    """
    Expression that is always generated inline. Subclasses should implement to_source_impl() method.
    """
    
    def to_source_impl(self, compiler_params, dummy=None):
        raise NotImplementedError

    def is_inline(self, compiler_params):
        return True

class ConstExpr(AlwaysInlineExpr):

    def __init__(self, value):
        super().__init__()
        self.value = value
        if isinstance(value, int_types):
            self.dtype = INT_TYPE
        elif isinstance(value, float_types):
            self.dtype = REAL_TYPE
        elif isinstance(value, ConstExpr):
            self.value = value.value
            self.dtype = value.dtype
        else:
            raise ValueError('unknown type')
        self.children = [self.value]
        self.ndims = 0

    def repr(self, extra_info=True, cache=None):
        return str(self.value)
    
    def gradient(self, compiler_params, get_idx):
        """
        Compute dself/dchildren
        """
        
        return []

    def __str__(self):
        return super().__str__() + '(' + str(self.value) + ')'

    def to_source_impl(self, compiler_params, dummy=None):
        ans = repr(self)
        if compiler_params.backend in ['hl', 'glsl']:
            if isinstance(self.value, bool):
                if self.value:
                    return '(true)'
                else:
                    return '(false)'
            else:
                if compiler_params.backend == 'hl':
                    # Halide has a problem converting constant of dtype double to float
                    ans = '(float) (' + ans + ')'
                else:
                    ans = 'float(%s)' % ans
        return ans

def gen_attrs(assign_lambdas):
    ans = []
    for (i, assign_lambda) in enumerate(assign_lambdas):
        def append_setter(i, assign_lambda):
            def setter(self, value):
                
                self.children[i] = assign_lambda(value)

            ans.append(property(lambda self: self.children[i], setter))
        append_setter(i, assign_lambda)
    return ans

def Var(name, node):
    if not isinstance(node, Expr):
        node = ConstExpr(node)
    node.short_name = name
    return node

class UpdateWrapper(Expr):
    """
    Wrapper that takes Original value as a children as well
    """
    def __init__(self, val, orig_val, counter):
        super().__init__()
        self.children = [to_expr(val), to_expr(orig_val)]
        self.dtype = val.dtype
        
        self.ndims = val.ndims
        if hasattr(val, 'element_type'):
            self.element_type = val.element_type
        
        self.update_counter = counter
        
    def backprop(self, compiler_params):
        
        if compiler_params.backprop_source in self.dL_dself.keys():
            if compiler_params.backprop_source in self.children[0].dL_dself.keys():
                self.children[0].dL_dself[compiler_params.backprop_source] = \
                self.children[0].dL_dself[compiler_params.backprop_source] + self.dL_dself[compiler_params.backprop_source]
            else:
                self.children[0].dL_dself[compiler_params.backprop_source] = self.dL_dself[compiler_params.backprop_source]
                
            if compiler_params.backprop_source in self.dL_dself_scalar.keys():
                if compiler_params.backprop_source in self.children[0].dL_dself_scalar.keys():
                    self.children[0].dL_dself_scalar[compiler_params.backprop_source] = \
                    self.children[0].dL_dself_scalar[compiler_params.backprop_source] + \
                    self.dL_dself_scalar[compiler_params.backprop_source]
                else:
                    self.children[0].dL_dself_scalar[compiler_params.backprop_source] = \
                    self.dL_dself_scalar[compiler_params.backprop_source]
                
        
        return [(self.children[0], 1)]
                
    def to_source_impl(self, compiler_params, dummy=None):
        if compiler_params.mode in [MODE_VARNAME, MODE_ALWAYS_INLINE]:
            return self.var_name(compiler_params)
        elif compiler_params.mode == MODE_SIDE_EFFECTS:
            self_id = self.statement_id(compiler_params)
            if self_id in compiler_params.statement_ids:
                # avoid duplicate
                return ''
            compiler_params.statement_ids.add(self_id)
            
            lhs = self.to_source(compiler_params.as_mode(MODE_VARNAME))
            
            update_idx = compiler_params.optional_updates.index(self.update_counter)
                    
            rhs_names = []
            return_list = []
            for idx in range(2):
                child = self.children[idx]
                return_list.append(child.to_source(compiler_params))
                rhs_names.append(child.to_source(compiler_params.as_mode(MODE_VARNAME)))
                
            declare_str = self.get_declare_str(compiler_params)
            
            if compiler_params.backend in ['hl', 'glsl']:
                end_str = ';'
            else:
                end_str = ''
                
            if compiler_params.backend == 'hl':
                
                statement = f"""
{declare_str}{lhs};
if (prune_optional_update) {{
    Expr prune_current_{lhs} = Halide::cast<bool> (*(do_prune[{update_idx}]));
    {lhs} = select(prune_current_{lhs}, {rhs_names[1]}, {rhs_names[0]});
}} else {{
    {lhs} = {rhs_names[0]};
}}
                """
            else:
                statement = f"""{declare_str}{lhs} = {rhs_names[0]}{end_str}"""
                
            return '\n'.join(return_list) + statement

class ArgumentArray(AlwaysInlineExpr):
    def __init__(self, name=DEFAULT_ARGUMENT_ARRAY_NAME, dtype=REAL_TYPE, ndims=0):

        super().__init__()
        self.ndims = ndims
        self.children = [str(name)]
        self.dtype = VECTOR_TYPE
        self.element_type = dtype
        self.is_discont_to_output = np.zeros(self.ndims).astype(bool)
        self.is_dependency_to_raymarching = np.zeros(self.ndims).astype(bool)
        self.is_discont_to_params = False

    (name,) = gen_attrs([str])
    
    def backprop(self, compiler_params):
        """
        Compute dL/dself, then propagate dL to children
        """
        return
        print('Error! backprop should never reach ArgumentArray, propagation should always be ended at GetItem!')
        raise

    def __getitem__(self, index):
        return GetItem(self, index)

    def to_source_impl(self, compiler_params, dummy=None):
            
        return self.name
    
class ArgumentScalar(AlwaysInlineExpr):
    def __init__(self, name, dtype=REAL_TYPE):
        
        super().__init__()
        self.children = [str(name)]
        self.dtype = dtype
        
    (name,) = gen_attrs([str])
    
    def gradient(self, compiler_params, get_idx):
        """
        Compute dself/dchildren
        """
        return []
    
    def to_source_impl(self, compiler_params, dummy=None):
        return self.name
        
class Gather(Expr):
    """
    This has identical semantic meaning as GetItem
    But for simplicity in implementation, this is used when the indices are variables, and it is NOT overridden with []
    GetItem is used when the indices are constants
    """
    def __init__(self, array, index):
        super().__init__()
        self.children = [to_expr(array), to_expr(index)]
        
        self.dtype = self.array.dtype
        
        self.ndims = self.array.data.shape[1]
        
    (array, index) = gen_attrs([to_expr, to_expr])
    
    

class GetItem(Expr):
    def __init__(self, array, index):
        super().__init__()
        self.children = [to_expr(array), to_expr(index)]
        
        if not hasattr(self.array, 'element_type'):
            self.dtype = self.array.dtype
        elif isinstance(self.array.element_type, str):
            self.dtype = self.array.element_type
        elif isinstance(self.array.element_type, (list, tuple)):
            index_val = get_const_val(self.index)
            assert index_val is not None
            self.dtype = self.array.element_type[index_val]
        else:
            raise 'Unrecognized array element_type'
        
        self.ndims = 0


    (array, index) = gen_attrs([to_expr, to_expr])
    
    def __getitem__(self, index):
        return GetItem(self, index)
    
    def is_inline(self, compiler_params):
        if isinstance(self.array, AlwaysInlineExpr) and is_all_constant(self.index):
            return True
        else:
            return False
    
    def propagate_discont(self, forward=False):
        
        if forward:
            Expr.propagate_discont(self, forward)
        else:
            if isinstance(self.index, ConstExpr):
                assert isinstance(self.index.value, int) and self.index.value >= 0 and self.index.value < self.array.ndims, 'Error! invalid index %d!' % self.index.value
                
                if isinstance(self.array, RaymarchingLoop):
                    # no need to propagate is_dependency_to_raymarching, it's always True
                    self.array.is_discont_to_output = self.array.is_discont_to_output or self.is_discont_to_output
                else:
                    if isinstance(self.array.is_discont_to_output, (list, tuple, np.ndarray)):
                        self.array.is_discont_to_output[self.index.value] = self.array.is_discont_to_output[self.index.value] or self.is_discont_to_output
                        self.array.is_dependency_to_raymarching[self.index.value] = self.array.is_dependency_to_raymarching[self.index.value] or self.is_dependency_to_raymarching
                    else:
                        # assumption: any vector that are NOT ArgumentArray or Compound will be treated having one single state of is_discont_to_output and is_dependency_to_raymarching
                        self.array.is_discont_to_output = self.array.is_discont_to_output or self.is_discont_to_output
                        self.array.is_dependency_to_raymarching = self.array.is_dependency_to_raymarching or self.is_dependency_to_raymarching
            else:
                # arbitrary lookup, we don't support it yet
                self.array.is_discont_to_output = np.logical_or(self.array.is_discont_to_output, self.is_discont_to_output)
                self.array.is_dependency_to_raymarching = np.logical_or(self.array.is_dependency_to_raymarching, self.is_dependency_to_raymarching)
    
    def backprop(self, compiler_params):
        """
        Compute dL/dself, then propagate dL to children
        """
        
        # Treat RaymarchingLoop the same as ArgumentArray
        # comparison nodes are already handled at the beginning of the backprop, and they will NOT contribute incorrect dL to the RaymarchingLoop node
        
        #if isinstance(self.array, RaymarchingLoop) and self.is_discont_to_output:
            # if self.is_discont_to_output = True, this indicates we are already inside a condition
            # in this case, the backprop method in RaymarchingLoop will find corresponding comparison node directly
            # the current dL_dself already includes undesirable dcond_diff/du, shoudl discard
            # print('here')
            # return
            
        if compiler_params.backprop_source not in self.dL_dself.keys():
            # skipped
            return
        
        global DEFAULT_WRT_ID
        global DEFAULT_WRT_NAME
        
        DEFAULT_WRT_ID = id(self)
        DEFAULT_WRT_NAME = getattr(self, 'name', None)
        
        if compiler_params.backprop_source not in self.array.dL_dself.keys():
            self.array.dL_dself[compiler_params.backprop_source] = Compound(np.zeros(self.array.ndims).tolist())

        assert isinstance(self.index, ConstExpr), 'Error! we do not support backpropagating array lookup with arbitrary index!'
        
        assert isinstance(self.index.value, int) and self.index.value >= 0 and self.index.value < self.array.ndims, 'Error! invalid index %d!' % self.index.value
        
        self.array.dL_dself[compiler_params.backprop_source].children[self.index.value] = \
        self.array.dL_dself[compiler_params.backprop_source].children[self.index.value] + \
        self.dL_dself[compiler_params.backprop_source]
        
        if compiler_params.gradient_mode == 'implicit_current':
            
            if self.array.dL_mask is None:
                self.array.dL_mask = [None] * self.array.ndims
                
            if isinstance(self.dL_mask, Expr) and self.array.dL_mask[self.index.value] is None:
                self.array.dL_mask[self.index.value] = self.dL_mask
            elif self.dL_mask is False:
                self.array.dL_mask[self.index.value] = False
        
        DEFAULT_WRT_ID = None
        DEFAULT_WRT_NAME = None

    def to_source_expr(self, compiler_params, dummy=None):
        
        (lbrace, rbrace) = ('[', ']')
        
        buffer_read = False
        params_read = False

        if isinstance(self.array, ArgumentArray):
            if compiler_params.backend == 'hl' and self.array.name == DEFAULT_ARGUMENT_ARRAY_NAME:
                array_str = 'params'
                params_read = True
            else:
                if compiler_params.backend == 'glsl':
                    try:
                        param_name = compiler_params.par_name_lookup[self.index.value]
                        return f"""{self.dtype}({param_name})"""
                    except:
                        pass
                
                array_str = self.array.to_source(compiler_params.as_mode(MODE_VARNAME))
                if compiler_params.backend == 'hl':
                    buffer_read = True
                    
                    # Should only reach this branch for dL_dcol
                    if array_str != DL_DCOL_ARRAY:
                        raise
        else:
            array_str = self.array.to_source(compiler_params.as_mode(MODE_VARNAME))
                
        idx_str = self.index.to_source(compiler_params)
        if compiler_params.backend == 'hl':
            idx_str = f"""(int) ({idx_str})"""
        elif compiler_params.backend == 'glsl':
            idx_str = f"""int({idx_str})"""
        
        if params_read:
            ans = f"""read_par(params, {idx_str})"""
        elif buffer_read:
            ans = f"""(*{array_str})(u, v, {idx_str})"""
        else:
            ans = f"""{array_str}[{idx_str}]"""
            
        if compiler_params.backend == 'glsl':
            ans = f"""{self.dtype}({ans})"""
        
                
        return ans

class Func(Expr):
    def __init__(self, name, gradient_lambda=None, tf_name=None, skip_log=False):
        super().__init__()
        self.children = [str(name)]
        if tf_name is None:
            self.tf_name = name
        else:
            self.tf_name = tf_name
        self.skip_log = skip_log
        self.gradient_lambda = gradient_lambda

    (name,) = gen_attrs([to_expr])

    def __call__(self, *args):
        return Call(self.name, *args, gradient_lambda=self.gradient_lambda, tf_name=self.tf_name, skip_log=self.skip_log)
    
    def gradient(self, compiler_params, get_idx):
        raise ValueError('Func should be called before backpropagation')

    def to_source_expr(self, compiler_params, dummy=None):
        raise ValueError('Func should be called before conversion to source')

class Call(Expr):
    def __init__(self, name, *args, gradient_lambda=None, tf_name=None, skip_log=False):
        super().__init__()
        self.children = [str(name)] + [to_expr(a) for a in args]
        self.dtype = REAL_TYPE
        self.tf_name = tf_name
        self.gradient_lambda = gradient_lambda
        
        self.ndims = 0
        for child in self.children[1:]:
            if isinstance(child, Expr):
                if self.ndims > 0 and child.ndims > 0:
                    assert self.ndims == child.ndims, 'Error! Inconsistent Compound vector size for function call'
                else:
                    self.ndims = max(self.ndims, child.ndims)

    (name,) = gen_attrs([to_expr])
    
    def propagate_discont(self, forward=False):
        if forward:
            Expr.propagate_discont(self, forward)
        else:
            if self.name in ['fract', 'floor', 'ceil']:
                assert len(self.children) == 2
                self.children[1].is_discont_to_output = True
                self.children[1].is_dependency_to_raymarching = self.is_dependency_to_raymarching or self.children[1].is_dependency_to_raymarching
            elif self.name == 'select':
                assert len(self.children) == 4
                self.children[1].is_discont_to_output = True
                self.children[1].is_dependency_to_raymarching = self.is_dependency_to_raymarching or self.children[1].is_dependency_to_raymarching
                
                for child in self.children[2:]:
                    child.is_discont_to_output = child.is_discont_to_output or self.is_discont_to_output
                    child.is_dependency_to_raymarching = child.is_dependency_to_raymarching or self.is_dependency_to_raymarching
            else:
                Expr.propagate_discont(self, forward)
                
    def backprop(self, compiler_params):
        
        if compiler_params.backprop_source not in self.dL_dself.keys():
            return
        
        if self.gradient_lambda is None:
            raise 'Error! gradient for function %s is not defined!' % self.name
            
        if self.name == 'select' and self.ndims > 0:
            return self.gradient_lambda(self.children[1:], self, compiler_params)
        else:
            Expr.backprop(self, compiler_params)
    
    def gradient(self, compiler_params, get_idx):
        
            
        input_args = []
        for child in self.children[1:]:
            if get_idx is not None and isinstance(child, Expr) and getattr(child, 'ndims', 0) > 0:
                input_args.append(child[get_idx])
            else:
                assert getattr(child, 'ndims', 0) == 0
                input_args.append(child)

        if get_idx is not None:
            current_self = self[get_idx]
        else:
            current_self = self
            
        if self.name == 'select':
            # Slightly different logic for select
            # We don't necessarily only backprop to the direct children (condition, left, right)
            # If condition is not a direct comparison, but logical operations of comparison primitives
            # we actually backprop to every comparison primitive
            return self.gradient_lambda(input_args, current_self, compiler_params)
        else:
            
            AD_derivs = self.gradient_lambda(input_args, current_self, compiler_params)
                    
            if self.allow_ours and self.is_discont_to_params and compiler_params.gradient_mode == 'ours' and self.params_only <= 2:
                if len(input_args) == 1:
                    derivs = [safe_division(get_partial_trace_coord(current_self, compiler_params.pix_idx_pl), 
                                            get_partial_trace_coord(input_args[0], compiler_params.pix_idx_pl),
                                            AD_derivs[0],
                                            chain_rule_thre)]
                    
                else:
                    print('Our rule cannot handle generic N-ary operators with N > 1, resorting to AD rule')
                    derivs = AD_derivs
            else:
                derivs = AD_derivs

            ans = [(self.children[i+1], derivs[i]) for i in range(len(derivs))]
            
        return ans
    
    def simplify_impl(self):
        """
        Simplifies the given Expr but not its children in place, returning the simplified Expr.
        """
        if self.name == 'select':
            cp = CompilerParams()
            
            cond, left, right = self.children[1:]
            
            if is_all_constant(cond):
                
                cond_str = cond.to_source(cp.as_mode(MODE_ALWAYS_INLINE))
                
                if is_constant(cond_str, True):
                    return left
                elif is_constant(cond_str, False):
                    return right
                else:
                    raise 'Error! constant condition not evaluated as Boolean!'
                    
            if id(left) == id(right):
                return left
            
        return self

    def to_source_expr(self, compiler_params, dummy=None):
        
        if self.name == 'select' and compiler_params.backend == 'glsl':
            # use ternary operator
            cond_str = self.children[1].to_source(compiler_params)
            left_str = self.children[2].to_source(compiler_params)
            right_str = self.children[3].to_source(compiler_params)
            
            # makes sure left and right has the same ndims
            
            if self.children[2].ndims != self.children[3].ndims:
                assert self.children[2].ndims == 0 or self.children[3].ndims == 0
                
                if self.children[2].ndims == 0:
                    left_str = left_str.replace(self.children[2].get_declare_str(compiler_params).replace(' ', ''), 
                                                self.children[3].get_declare_str(compiler_params).replace(' ', ''))
                else:
                    right_str = right_str.replace(self.children[3].get_declare_str(compiler_params).replace(' ', ''), 
                                                  self.children[2].get_declare_str(compiler_params).replace(' ', ''))
            
            ans = f"""bool({cond_str}) ? {left_str} : {right_str}"""
            return ans
        
        collected_ans = []
        
        if compiler_params.backend == 'glsl':
            # most GLSL API supports overloading to vec
            nentries = 1
        else:
            nentries = max(1, self.ndims)
        
        for idx in range(nentries):
            
            if self.ndims == 0 or compiler_params.backend == 'glsl':
                get_str = ''
            else:
                get_str = '[%d]' % idx
            
            return_list = []
            for a in self.children[1:]:
                if isinstance(a, Expr):
                    if a.ndims == 0:
                        current_get_str = ''
                    else:
                        current_get_str = get_str
                    return_list.append(a.to_source(compiler_params) + current_get_str)
                else:
                    if compiler_params.backend in ['tf', 'np', 'torch']:
                        return_list.append(str(a))
                    else:
                        if a is True:
                            return_list.append('true')
                        elif a is False:
                            return_list.append('false')
                        elif compiler_params.backend == 'hl':
                            return_list.append('(float) (' + str(a) + ')')

            if compiler_params.backend == 'tf':
                func_name = self.tf_name
            elif compiler_params.backend == 'torch':
                if hasattr(self, 'torch_name'):
                    func_name = self.torch_name
                else:
                    func_name = self.tf_name
            elif compiler_params.backend == 'np':
                if callable(getattr(np, self.name, None)) and self.name not in ['select']:
                    func_name = compiler_params.backend + '.' + self.name
                else:
                    func_name = self.name
            else:
                func_name = self.name

            ans = func_name + '(' + ','.join(return_list) + ')'
            collected_ans.append(ans)
        
        return ', '.join(collected_ans)

def numerical_promote(a, b):
    if a == REAL_TYPE or b == REAL_TYPE:
        return REAL_TYPE
    else:
        return a

def is_any_constant(b_str):
    try:
        b_val = eval(b_str)
        return True
    except:
        return False

def is_all_constant(e):
    if isinstance(e, (int, float)):
        return True
    return all(isinstance(node, (ConstExpr, BinaryOp, UnaryOp, int, float)) for node in e.all_nodes_generator())

def get_const_val(e):
    if isinstance(e, (int, float)):
        return e
    else:
        cp = CompilerParams()
        e_str = e.to_source(cp.as_mode(MODE_ALWAYS_INLINE))
        try:
            e_eval = eval(e_str)
        except:
            return None
        
        return e_eval

def is_constant(b_str, b_value):
    eps = 1e-15
    try:
        b_val = eval(b_str)
        diff = b_val - b_value
    except:
        return False

    ans = abs(b_val - b_value) <= eps
    return ans

def constant_value(b_str):
    ans = eval(b_str)

    return to_expr(ans)

class BinaryOp(Expr):
    def __init__(self, op, a, b):
        super().__init__()
        self.children = [str(op), to_expr(a), to_expr(b)]
        if op in ['<', '>', '<=', '>=', '==', '!=']:
            self.dtype = BOOL_TYPE
        else:
            self.dtype = numerical_promote(self.children[1].dtype, self.children[2].dtype)
        assert isinstance(self.a, Expr)
        assert isinstance(self.b, Expr)
        
        self.ndims = max(self.a.ndims, self.b.ndims)
        
        if self.a.ndims > 0 and self.b.ndims > 0:
            assert self.a.ndims == self.b.ndims, 'Error! Compound vector dimension inconsistent in binary operation'

    (op, a, b) = gen_attrs([str, to_expr, to_expr])

    def __str__(self):
        return super().__str__() + '(' + self.op + ')'
    
    def propagate_discont(self, forward=False):
        if forward and self.is_comparison():
            for par in self.parents:
                par.is_discont_to_params = True
        else:
            Expr.propagate_discont(self, forward)
    
    def gradient(self, compiler_params, get_idx):
        a = self.a
        b = self.b
        
        if get_idx is not None:
            if a.ndims > 0:
                a = a[get_idx]
            if b.ndims > 0:
                b = b[get_idx]
                
            current_self = None
            for par in self.parents:
                if isinstance(par, GetItem):
                    if is_constant(par.index.to_source(compiler_params.ad_mode(MODE_ALWAYS_INLINE)), get_idx):
                        current_self = par
            if current_self is None:
                current_self = self[get_idx]
        else:
            current_self = self
        
        if self.op == '**':
            
            AD_derivs = [(self.a, b * safe_pow(a, (b - 1))),
                         (self.b, safe_log(a) * current_self)]
            
            if self.allow_ours and compiler_params.gradient_mode == 'ours' and self.is_discont_to_params:
                if not (is_all_constant(a) or is_all_constant(b)):
                    print('Our rule cannot handle generic binary operator, resorting to AD rule')
                    derivs = AD_derivs
                elif is_constant(b.to_source(compiler_params.as_mode(MODE_ALWAYS_INLINE)), 2.0):
                    derivs = [(self.a, a + get_neighbor(a, compiler_params.pix_idx_pl))]
                elif is_all_constant(b):
                    derivs = [(self.a, safe_division(get_partial_trace_coord(current_self, compiler_params.pix_idx_pl),
                                                     get_partial_trace_coord(a, compiler_params.pix_idx_pl),
                                                     AD_derivs[0][1],
                                                     chain_rule_thre))]
                else:
                    derivs = [(self.b, safe_division(get_partial_trace_coord(current_self, compiler_params.pix_idx_pl),
                                                     get_partial_trace_coord(b, compiler_params.pix_idx_pl),
                                                     AD_derivs[1][1],
                                                     chain_rule_thre))]
            else:
                derivs = AD_derivs
            
            return derivs
        elif self.op == '*':
            
            if self.allow_ours and compiler_params.gradient_mode == 'ours' and self.is_discont_to_params and compiler_params.multiplication_rule != 0:
                if id(self.a) == id(self.b):
                    return [(self.a, a + get_neighbor(a, compiler_params.pix_idx_pl))]

                derivs = []
                
                if compiler_params.multiplication_rule in [1, 2]:
                    if self.a.is_discont_to_params and self.b.is_discont_to_params:
                        if compiler_params.multiplication_rule == 1:
                            derivs = [(self.a, b),
                                      (self.b, get_neighbor(a, compiler_params.pix_idx_pl))]
                        else:
                            derivs = [(self.a, get_neighbor(b, compiler_params.pix_idx_pl)),
                                      (self.b, a)]
                    else:
                        derivs = [(self.a, b),
                                  (self.b, a)]
                elif compiler_params.multiplication_rule == 3:
                    # we don't really need it if one of a, b is continuous
                    # should check why it's implemented this way
                    if self.b.is_discont_to_params:
                        derivs.append((self.a, 0.5 * (b + get_neighbor(b, compiler_params.pix_idx_pl))))
                    else:
                        derivs.append((self.a, b))

                    if self.a.is_discont_to_params:
                        derivs.append((self.b, 0.5 * (a + get_neighbor(a, compiler_params.pix_idx_pl))))
                    else:
                        derivs.append((self.b, a))

                else:
                    raise
            else:
                derivs = [(self.a, b),
                          (self.b, a)]
            return derivs
        elif self.op == '-':
            return [(self.a, 1),
                    (self.b, -1)]
        elif self.op == '+':
            return [(self.a, 1),
                    (self.b, 1)]
        elif self.op == '/':
            assert compiler_params.gradient_mode == 'AD', 'Error! this branch should not be taken'
            return [(self.a, 1 / b),
                    (self.b, -a / b ** 2)]
        elif self.op in ['&&', '||']:
            
            if compiler_params.gradient_mode != 'ours':
                return []
            
            comparison_primitives = self.get_comparison_primitives()
            #comparison_primitives = [node for node in comparison_primitives if id(node) not in compiler_params.raymarching_dependent_comparisons]
            
            assert self.ndims == 0, 'Vector mode not supported inside condition, please express them using scalar explicitly'
                
            comparison_valids = []
            partial_cond_diff_coords = []
            neg_prev_comparisons = ConstExpr(True)

            for comparison_idx in range(len(comparison_primitives)):
                
                comparison = comparison_primitives[comparison_idx]
                
                if comparison_idx < len(comparison_primitives) - 1:
                    current_valid = neg_prev_comparisons & (comparison != get_neighbor(comparison, compiler_params.pix_idx_pl))
                    neg_prev_comparisons = neg_prev_comparisons & (~current_valid)
                else:
                    # No need to check whether last condition is flipped or not
                    # if the entire condition combination is flipped and every other clause is NOT flipped, then the last one must have flipped
                    current_valid = neg_prev_comparisons
                    
                comparison_valids.append(current_valid)
                
                
                if comparison.is_comparison() and id(comparison) not in compiler_params.raymarching_dependent_comparisons:
                    partial_cond_diff_coords.append(get_partial_trace_coord(comparison.a - comparison.b, compiler_params.pix_idx_pl))
                else:
                    partial_cond_diff_coords.append(None)
                    
            partial_cond_diff_coord = None
            partial_cond_diff_coord_last = None
            partial_cond_diff = None
            all_update_denum = True
            can_update_denum = False
            
            for comparison_idx in range(len(comparison_primitives) - 1, -1, -1):
                if partial_cond_diff_coords[comparison_idx] is not None:
                    if partial_cond_diff_coord is not None:
                        partial_cond_diff_coord = select(comparison_valids[comparison_idx],
                                                         partial_cond_diff_coords[comparison_idx], 
                                                         partial_cond_diff_coord)
                    elif partial_cond_diff_coord_last is not None:
                        partial_cond_diff_coord = select(comparison_valids[comparison_idx],
                                                         partial_cond_diff_coords[comparison_idx],
                                                         partial_cond_diff_coord_last)
                    else:
                        partial_cond_diff_coord_last = partial_cond_diff_coords[comparison_idx]
                    can_update_denum = can_update_denum | comparison_valids[comparison_idx]
                else:
                    all_update_denum = False
                    

            if partial_cond_diff_coord is None and partial_cond_diff_coord_last is not None:
                partial_cond_diff_coord = partial_cond_diff_coord_last
                
            if partial_cond_diff_coord is not None:
                
                if all_update_denum:
                    dL = self.dL_dself[compiler_params.backprop_source]
                else:
                    dL = cast2f(can_update_denum) * self.dL_dself[compiler_params.backprop_source]
                
                partial_cond_diff = backprop_discont(compiler_params,
                                                     self,
                                                     dL, 
                                                     partial_cond_diff_coord)
                
            ans = []
            for comparison_idx in range(len(comparison_primitives)):
                comparison_node = comparison_primitives[comparison_idx]
                comparison_valid = comparison_valids[comparison_idx]
                if partial_cond_diff_coords[comparison_idx] is not None:
                    assert partial_cond_diff is not None
                    
                    dL = partial_cond_diff * select(comparison_valid, 1, 0)
                    ans += [(comparison_node.a, dL),
                            (comparison_node.b, -dL)]
                else:
                    dL = select(comparison_valid, 1, 0)
                    ans += [(comparison_node, dL)]
                
            return ans
                    
            
            return []
            
            # Limitation to our compiler: cannot correctly handle the intersection of 2 distinct discontinuities
            # We always assume if there're 2 discontinuities, they're derived from the same root discontinuity
            # therefore it's ok to either propagate gradient to the first or the second, but never both
            # here we aways default to first
            
            # because self.a and self.b are Booleans, get_partial_trace_coord returns 0, 1, or -1
            a_has_discont = abs(get_partial_trace_coord(a, compiler_params.pix_idx_pl))
            
            return [(a, a_has_discont),
                    (b, 1 - a_has_discont)]
        elif self.is_comparison():
            
            assert self.ndims == 0 and get_idx is None
                    
            cond_diff = a - b

            if compiler_params.gradient_mode == 'ours':
                
                # gradient will be handled in Raymarching Loop
                if id(current_self) in compiler_params.raymarching_dependent_comparisons:
                    
                    return []

                current_dL_dself = self.dL_dself[compiler_params.backprop_source]
                    
                if self.dL_dself_scalar.get(compiler_params.backprop_source, 1) is not 1:
                    current_dL_dself = current_dL_dself * self.dL_dself_scalar[compiler_params.backprop_source]
                
                # includes an inversion of get_partial_trace_coord(a)
                # and update_discont_dir(dL_dself, get_partial_trace_coord(a))
                partial_cond_diff = backprop_discont(compiler_params,
                                                     current_self,
                                                     current_dL_dself, 
                                                     get_partial_trace_coord(cond_diff, compiler_params.pix_idx_pl))
            else:
                partial_cond_diff = 0

            return [(self.a, partial_cond_diff),
                    (self.b, -partial_cond_diff)]
        else:
            raise ValueError('gradient not implemented:', self.op)
                        

    def simplify_impl(self):
        
        cp = CompilerParams()
        a = self.a_str(cp)

        b = self.b_str(cp)
        if self.op == '*':
            if is_constant(b, 1.0):
                return_val = self.a
                return return_val
            elif is_constant(a, 1.0):
                return_val = self.b
                return return_val
            elif is_constant(b, 0.0) or is_constant(a, 0.0):
                return ConstExpr(0.0)

            def handle_mul_mul_const(a_expr, b_expr, a_str):
                if is_any_constant(a_str) and isinstance(b_expr, BinaryOp) and b_expr.op == '*':
                    b_a = b_expr.a_str(cp)
                    b_b = b_expr.b_str(cp)
                    if is_any_constant(b_a):
                        return_val = to_expr(constant_value(a_str).value * constant_value(b_a).value) * b_expr.b
                        return return_val
                    if is_any_constant(b_b):
                        return_val = to_expr(constant_value(a_str).value * constant_value(b_b).value) * b_expr.a
                        return return_val

            arg_list = [(self.a, self.b, a), (self.b, self.a, b)]
            for (a_expr, b_expr, a_str) in arg_list:
                v1 = handle_mul_mul_const(a_expr, b_expr, a_str)
                if v1 is not None:
                    v1.fix_attr(self)
                    return v1
        elif self.op == '/':
            if is_constant(b, 1.0):
                return_val = self.a
                return return_val
        elif self.op == '+':
            if is_constant(a, 0.0):
                return_val = self.b
                return return_val
            elif is_constant(b, 0.0):
                return_val = self.a
                return return_val
        elif self.op == '-':
            if is_constant(b, 0.0):
                return_val = self.a
                return return_val
        elif self.op == '**':
            if is_constant(b, 1.0):
                return self.a
            if is_constant(b, 0.0):
                return to_expr(1.0)
            if is_constant(b, 0.5):
                ap_L = [self.a]
                for ap in ap_L:
                    if isinstance(ap, BinaryOp) and ap.op == '*' and ap.a.identical(ap.b):
                        return_val = ap.a
                        return return_val
                    if isinstance(ap, BinaryOp) and ap.op == '**' and is_constant(ap.b_str(cp), 2.0):
                        return_val = ap.a
                        return return_val
            if is_constant(a, 0.0):
                return to_expr(0.0)
                
            if is_any_constant(b):
                ap_L = [self.a]
                for ap in ap_L:
                    if isinstance(ap, BinaryOp) and ap.op == '*' and ap.a.identical(ap.b):
                        return_val = to_expr(ap.a ** (self.b * 2))
                        return_val.fix_attr(self)
                        return return_val
                    if isinstance(ap, BinaryOp) and ap.op == '**':
                        return_val = to_expr(ap.a ** (self.b * ap.b))
                        return_val.fix_attr(self)
                        return return_val
        elif self.op == '&&':
            if is_constant(a, False) or is_constant(b, False):
                return ConstExpr(False)
            if is_constant(a, True):
                return self.b
            if is_constant(b, True):
                return self.a
        elif self.op == '||':
            if is_constant(a, True) or is_constant(b, True):
                return ConstExpr(True)
            if is_constant(a, False):
                return self.b
            if is_constant(b, False):
                return self.a
        

        if not is_all_constant(self):
            return self

        self_str = self.to_source(cp.as_mode(MODE_ALWAYS_INLINE))
        try:
            ans = constant_value(self_str)
            if ans is not None:
                return ans

        except:
            pass

        return self

    def a_str(self, compiler_params):
        
        if isinstance(self.a, (int, float)):
            return str(self.a)

        if not is_all_constant(self.a):
            return ''
                
        ans = self.a.to_source(compiler_params.as_mode(MODE_ALWAYS_INLINE))
        return ans

    def b_str(self, compiler_params):
        
        if isinstance(self.b, (int, float)):
            return str(self.b)
        
        if not is_all_constant(self.b):
            return ''

        ans = self.b.to_source(compiler_params.as_mode(MODE_ALWAYS_INLINE))

        return ans

    def to_source_expr(self, compiler_params, dummy=None):
        
        collected_ans = []
        
        if compiler_params.backend == 'glsl':
            nentries = 1
        else:
            nentries = max(1, self.ndims)
        
        for idx in range(nentries):
            
            if self.ndims == 0 or compiler_params.backend == 'glsl':
                get_str = ''
            else:
                get_str = '[%d]' % idx
                
            if isinstance(self.a, (int, float)):
                a = str(self.a)
                if compiler_params.backend == 'hl':
                    a = '(float) (' + a + ')'
                elif compiler_params.backend == 'glsl':
                    a = '%s(%s)' % (self.dtype, a)
                    
            # Really Should NOT create new node inside to_source, these nodes will be garbage collected, so their id might coincide with other newly created nodes, making our cache mess up
            # For workaround, directly call to_source_expr
            elif isinstance(self.a, ArgumentArray) and self.a.ndims > 0:
                get_node = self.a[idx]
                a = get_node.to_source_expr(compiler_params)
            else:
                a = self.a.to_source(compiler_params)
                if self.a.ndims > 0:
                    a = a + get_str
                    
            

            if isinstance(self.b, (int, float)):
                b = str(self.b)
                if compiler_params.backend == 'hl':
                    b = '(float) (' + b + ')'
                elif compiler_params.backend == 'glsl':
                    b = '%s(%s)' % (self.dtype, b)
            # Really Should NOT create new node inside to_source, these nodes will be garbage collected, so their id might coincide with other newly created nodes, making our cache mess up
            # For workaround, directly call to_source_expr
            elif isinstance(self.b, ArgumentArray) and self.b.ndims > 0:
                b = self.b[idx].to_source_expr(compiler_params)
            else:
                b = self.b.to_source(compiler_params)
                if self.b.ndims > 0:
                    b = b + get_str
                                        
            if self.op in ['&&', '||'] and compiler_params.backend in ['tf', 'np', 'torch']:
                if self.op == '&&':
                    func_name = 'logical_and'
                elif self.op == '||':
                    func_name = 'logical_or'
                ans = '%s.%s(%s, %s)' % (compiler_params.backend, func_name, a, b)
            elif self.op in ['==', '!='] and compiler_params.backend in ['tf', 'np']:
                if self.op == '==':
                    func_name = 'equal'
                elif self.op == '!=':
                    func_name = 'not_equal'
                ans = '%s.%s(%s, %s)' % (compiler_params.backend, func_name, a, b)
            elif self.op == '**' and compiler_params.backend in ['hl', 'glsl']:
                ans = f"""pow({a}, {b})"""
            elif self.op == '%' and compiler_params.backend in ['tf', 'glsl', 'torch']:
                if compiler_params.backend == 'tf':
                    ans = 'tf.floormod(' + a + ', ' + b + ')'
                elif compiler_params.backend == 'torch':
                    ans = 'torch.remainder(' + a + ', ' + b + ')'
                else:
                    ans = f"""mod({a}, {b})"""
            else:
                ans = '((' + a + ')' + self.op + '(' + b + '))'
                
            collected_ans.append(ans)
            
            
        return ', '.join(collected_ans)
        
class HL_Func(Expr):
    """
    deprecated
    HL syntax, wrap a single Expr into a function
    """
    def __init__(self, center_val):
        super().__init__()
        self.children = [center_val]
        self.dtype = self.children[0].dtype
        
    def to_source_impl(self, compiler_params, dummy=None):
        if compiler_params.mode == MODE_VARNAME:
            return self.var_name(compiler_params)
        else:
            # TODO: finish logic for tf
            assert compiler_params.backend == 'hl'
            
            self_id = self.statement_id(compiler_params)
            if self_id in compiler_params.statement_ids:
                # avoid duplicate
                return ''
            compiler_params.statement_ids.add(self_id)
            
            lhs = self.to_source_impl(compiler_params.as_mode(MODE_VARNAME))
            
            statement = f"""
Func {lhs};
{lhs}(u, v, Ou, Ov) = {self.children[0].to_source(compiler_params.as_mode(MODE_VARNAME))};
"""
            return self.children[0].to_source(compiler_params) + '\n' + statement
        
def get_geometry_discont(f_node, leaf_nodes, compiler_params):
    # travers f_node to collect is_at_geometry_discont
    all_nodes = f_node.all_nodes_dfs(exclude=set([id(node) for node in leaf_nodes]))

    for node in leaf_nodes + all_nodes:
        node.is_at_geometry_discont = ConstExpr(False)

    # bottom-up
    for node in all_nodes:
        
        stop_recurse = False
        
        if isinstance(node, Call) and getattr(node, 'name', '') == 'select':
            
            # Geometry discont should be generated by CSG operators
            # both sides of the branch should be non-constant, with params_only < 3
            cond, left, right = node.children[1:]
            
            if left.params_only < 3 and right.params_only < 3:
                stop_recurse = True
                node.is_at_geometry_discont = (cond != get_neighbor(cond, compiler_params.pix_idx_pl)) | \
                                              select(cond, left.is_at_geometry_discont, right.is_at_geometry_discont)
        if not stop_recurse:
            for child in node.children:
                if isinstance(child, Expr):
                    node.is_at_geometry_discont = node.is_at_geometry_discont | child.is_at_geometry_discont

class RaymarchingWrapper(Expr):
    """
    Defines a wrapper for raymarching
    Generates a RaymarchingLoop instantce
    if include_derivs, also expand lambda_body one extra time to get converged derivatives
    Seperate from RaymarchingLoop because RaymarchingLoop is treated as blackbox, and special gradient rules are used there
    The derivative computation, however, is treated using our regular rule, with some optimization based on ray marching loop semantics applied
    """
    def __init__(self, lambda_body, ro, rd, init_t, niters, include_derivs=False, ignore_crease=False):
        """
        lambda_body: should take input of 3 Expr input arguments (x, y, z for position), and outputs a list of [distance_to_object, surface_label]
        """
        super().__init__()
        self.body = lambda_body
        self.ro = ro
        self.rd = rd
        self.include_derivs = include_derivs
        
        self.raymarching_loop = RaymarchingLoop(lambda_body, ro, rd, init_t, niters, ignore_crease)
        self.raymarching_loop.wrapper = self
        
        # directly assume dependencies of surface normal (or None) is a subset of the raymarching loop's dependency
        # TODO: should auto-generate surface normal, so that the dependency is guaranteed to be loop body's dependency.
        self.dependencies = self.raymarching_loop.dependencies
        
        self.final_t = self.raymarching_loop[1]
        self.final_pos = []
        for idx in range(3):
            self.final_pos.append(self.ro[idx] + self.rd[idx] * self.final_t)
        self.find_geometry_comparison()
        
        if self.include_derivs:
            self.ndims = self.raymarching_loop.ndims + 3
            assert len(self.derivs) == 3
            self.children = [self.raymarching_loop] + list(self.derivs)
        else:
            self.ndims = self.raymarching_loop.ndims
            self.children = [self.raymarching_loop]
                        
        self.is_discont_to_output = np.zeros(self.ndims).astype(bool)
        self.is_dependency_to_raymarching = np.zeros(self.ndims).astype(bool)
        
    def find_geometry_comparison(self):
        """
        Assume any select() within self.body is realted to geometry, and should be handled in special rules
        """
        
        global DEFAULT_ALLOW_OURS
        DEFAULT_ALLOW_OURS = False
        
        # output to user-defined signed distance field function, search dependency within this subgraph
        ans = self.body(*self.final_pos)
        
        DEFAULT_ALLOW_OURS = True
        
        assert isinstance(ans, (list, tuple))
        
        self.res0 = ans[0]
        
        if self.include_derivs:
            self.derivs = ans[2:]
            assert len(self.derivs) == 3
            
    def __getitem__(self, index):
        """
        Output values of the Raymarching loop:
        0: is_fg (boolean)
        1: t_closest (length of the ray such that distance from ray endpoint to object is smallest)
        2: res0_closest (the smallest distance on any point along the ray to object)
        3: t (length of ray at the end of the raymarching loop)
        4: res0 (distance from ray endpoint to object at the end of the raymarching loop)
        5: label (label of the surface)
        Additional output for converged surface normal
        6: derivs_x
        7: derivs_y
        8: derivs_z
        """
        assert isinstance(index, (ConstExpr, int))
        
        if isinstance(index, int):
            index_val = index
        else:
            index_val = index.value
        
        if index_val <= 5:
            return self.raymarching_loop[index]
        else:
            assert self.include_derivs
            return self.derivs[index_val - 6]
        
class RaymarchingLoop(Expr):
    """
    Defines a raymarching loop
    """
    def __init__(self, lambda_body, ro, rd, init_t, niters, ignore_crease=False):
        """
        lambda_body: should take input of 3 Expr input arguments (x, y, z for position), and outputs a list of [distance_to_object, surface_label]
        """
        super().__init__()
        self.body = lambda_body
        self.ro = [to_expr(a) for a in ro]
        self.rd = [to_expr(a) for a in rd]
        self.init_t = to_expr(init_t)
        self.niters = niters
        self.ignore_crease = ignore_crease
        
        global raymarching_count
        self.raymarching_count = raymarching_count
        raymarching_count += 1

        self.t_invariant = ArgumentScalar('_t_raymarching_%d' % self.raymarching_count)
        self.t_closest_invariant = ArgumentScalar('_t_closest_raymarching_%d' % self.raymarching_count)
        self.res0_closest_invariant = ArgumentScalar('_res0_closest_raymarching_%d' % self.raymarching_count)
        self.tmax = ConstExpr(10.)
        
        self.find_dependencies()
        
        self.children = [self.init_t] + list(self.ro) + list(self.rd) + self.dependencies
        self.dtype = VECTOR_TYPE
        self.element_type = [BOOL_TYPE] + [REAL_TYPE] * (self.ndims - 1)
        
        self.comparison_parents = []
        self.is_at_geometry_discont = None
        self.wrapper = None
        
        if need_animate:
            
            self.orig_cond = Var('raymarching_loop_%d_is_fg' % self.raymarching_count, GetItem(self, 0))
            self.t_closest = Var('raymarching_loop_%d_t_closest' % self.raymarching_count, GetItem(self, 1))
            self.res0_closest = Var('raymarching_loop_%d_res0_closest' % self.raymarching_count, GetItem(self, 2))
            self.final_t = Var('raymarching_loop_%d_final_t' % self.raymarching_count, GetItem(self, 3))
            self.final_res0 = Var('raymarching_loop_%d_final_res0' % self.raymarching_count, GetItem(self, 4))
            self.surface_label = Var('raymarching_loop_%d_surface_label' % self.raymarching_count, GetItem(self, 5))
            
            animate = Animate('animate_raymarching_loop_%d_is_fg' % self.raymarching_count, 
                              inout_ls=[self.orig_cond], 
                              in_ls=[self.t_closest,
                                     self.res0_closest,
                                     self.final_t,
                                     self.final_res0,
                                     self.surface_label])
            
            self.animated_cond, = animate.update()
                                          
        
    def propagate_discont(self, forward=False):
        """
        RaymarchingLoop works as a giant black box, anything passes through it becomes discontinuous
        """
        
        if forward:
            node_ls = getattr(self, 'parents', [])
            discont_attr = ['is_discont_to_params']
        else:
            node_ls = self.children
            discont_attr = ['is_discont_to_output', 'is_dependency_to_raymarching']
            
        for node in node_ls:
            if isinstance(node, Expr):
                for attr_name in discont_attr:
                    setattr(node, attr_name, True)
        
    def backprop(self, compiler_params):
        
        """
        Our gradient rule for Raymarching Expr is only correct under the following restrictions
        1. The body of the loop is a sign distance field, formed by smooth subgraphs connected with min/max operations
        2. Number of iterations is large enough for the ray marching loop to converge
        3. The first return value is always the sign input position's distance to the object
        4. The other return values could be surface properties such as a different labelling for different parts of the object, but they should also conceptually represent the property of the object that converges as the number of iteration approaches infinity.
        5. Whenever the return values of the loop is used inside a condition, we assume the semantic of the condition is to take gradient wrt parameters while the ray stays on the edge or sihoulette. This means we are not able to correctly resolve pathological conditions such as select(t - res0 - t + res0 > 0, 0, 1), or arbitrary compositions of conditions that does not have physical meaning in terms of object intersection.
        6. Whenever the return value of the loop is used at outside a condition, we assume the semantic is to take gradient wrt parameters while the ray stays on surface.
        """
        
        if compiler_params.gradient_mode != 'ours':
            
            #return
            
            # ignore discont backprop when not using our mode
            # but keep continuous graidnet dL_dt
            dL_dt = None
        
            if compiler_params.backprop_source in self.dL_dself.keys():
                # we treat t_closest and t equally as they have the same semantics
                dL_dt = self.dL_dself[compiler_params.backprop_source].children[1] + \
                        self.dL_dself[compiler_params.backprop_source].children[3]
                
            if dL_dt is not None:
                
                current_pos = []
                for idx in range(3):
                    current_pos.append(self.ro[idx] + self.rd[idx] * self[1])
                    
                f_current = self.body(*current_pos)[0]
                f_current.dL_mask = False
                    
                current_source_name = 'raymarching_%d_AD' % self.raymarching_count
                cp = copy.copy(compiler_params)
                
                cp.backprop_source = current_source_name
                cp.gradient_mode = 'AD'
                        
                cp.min_abs_diff = ConstExpr(1e8)
                cp.reset_args_idx = ConstExpr(-1)
                cp.reset_args_type = ConstExpr(False)

                # all nodes within loop body
                all_nodes_current = f_current.all_nodes_dfs(exclude=set([id(node) for node in (self.children + current_pos)]))

                # we only care about dres0/dx or dres0/dtheta, don't care about gradient on other output values because they don't have a clear semantic meaning
                f_current.dL_dself[current_source_name] = 1

                # top-down
                for node in all_nodes_current[::-1]:
                    node.backprop(cp)
                
                df1_dx = []
                
                # there's only one copy of current_pos, because deep_copy stops there
                for node in current_pos:
                    df1_dx.append(node.dL_dself.get(current_source_name, 0))

                df1_dtheta = []

                # there's only one copy of the dependencies, because deep_copy stops there
                for node in self.dependencies:
                    df1_dtheta.append(node.dL_dself.get(current_source_name, 0))

                dot_dfdx_rd = dot(df1_dx, self.rd)
                
                # gradient is NOT related to init_t
                for idx in range(1, len(self.children)):

                    if idx <= 3:
                        # ro
                        num_smooth = df1_dx[idx-1]
                    elif idx <= 6:
                        # rd
                        num_smooth = df1_dx[idx-4] * self[1]
                    else:
                        num_smooth = df1_dtheta[idx-7]

                    deriv_cont = safe_division(-num_smooth, dot_dfdx_rd)

                    deriv = deriv_cont * dL_dt

                    if compiler_params.backprop_source in self.children[idx].dL_dself.keys():
                        self.children[idx].dL_dself[compiler_params.backprop_source] = \
                        self.children[idx].dL_dself[compiler_params.backprop_source] + deriv
                    else:
                        self.children[idx].dL_dself[compiler_params.backprop_source] = deriv
            
            return
        
        # find grandparents who are comparison nodes
        dL_dself_comparisons = self.dL_dself[compiler_params.backprop_source][0]
            
        for node in self.comparison_parents:
            
            assert id(node) not in compiler_params.min_discont_denum_dict.keys()
            assert id(node) in compiler_params.raymarching_dependent_comparisons
            
            if compiler_params.backprop_source in node.dL_dself.keys():
                dL_dself_comparisons = dL_dself_comparisons + node.dL_dself[compiler_params.backprop_source]
            else:
                print("Warning, backprop does not reach this parent comparison node. It is possible that the comparison node is only the children of another comparison node. If this behavior is not expected, should debug.")
        
        t_closest = self[1]
        t_closest_neighbor = get_neighbor(t_closest, compiler_params.pix_idx_pl)
        
        current_t = select(t_closest < t_closest_neighbor, 
                           t_closest, 
                           t_closest_neighbor)
        
        current_pos = []
        for idx in range(3):
            current_pos.append(self.ro[idx] + self.rd[idx] * current_t)
            
        # we only care about the first return value: distance to object
        f_current = self.body(*current_pos)[0]
        f_current.dL_mask = False
        
        cp = copy.copy(compiler_params)
        
        cp.min_abs_diff = ConstExpr(1e8)
        cp.reset_args_idx = ConstExpr(-1)
        cp.reset_args_type = ConstExpr(False)
        
        # all nodes within loop body
        all_nodes_current = f_current.all_nodes_dfs(exclude=set([id(node) for node in (self.children + current_pos)]))
        
        current_source_name = 'raymarching_%d_current' % self.raymarching_count
        neighbor_source_name = 'raymarching_%d_neighbor' % self.raymarching_count
        
        cp.backprop_source = current_source_name
        cp.gradient_mode = 'implicit_current'
        
        # we only care about dres0/dx or dres0/dtheta, don't care about gradient on other output values because they don't have a clear semantic meaning
        f_current.dL_dself[current_source_name] = 1
        
        # top-down
        for node in all_nodes_current[::-1]:
            node.backprop(cp)
            
        # only copy self.body part of f_current
        f_neighbor = copy.deepcopy(f_current, {id(node): node for node in (self.children + current_pos)})
            
        # mutate f_neighbor to change select branch when necessary
        all_nodes_neighbor = f_neighbor.all_nodes_dfs(exclude=set([id(node) for node in (self.children + current_pos)]))
        
        def get_reset_bool(val, op):
            if isinstance(val, Expr) and not is_all_constant(val) and hasattr(val, 'comparison_args_idx'):
                
                ans = cp.reset_args_idx == val.comparison_args_idx
                
                if op == 'max':
                    ans = ans & cp.reset_args_type
                else:
                    ans = ans & (~ cp.reset_args_type)
            else:
                ans = None
            return ans
        
        for node in all_nodes_neighbor:
            if isinstance(node, Call) and node.name == 'select':
                assert hasattr(node, 'op')
                cond, left, right = node.children[1:]
                
                reset_left = get_reset_bool(left, node.op)
                reset_right = get_reset_bool(right, node.op)
                
                if reset_left is not None:
                    if reset_right is not None:
                        new_cond = cond != (reset_left | reset_right)
                    else:
                        new_cond = cond != reset_left
                elif reset_right is not None:
                    new_cond = cond != reset_right
                else:
                    new_cond = cond
                
                node.children[1] = new_cond
                
        # compute all_nodes again after mutation
        all_nodes_neighbor = f_neighbor.all_nodes_dfs(exclude=set([id(node) for node in (self.children + current_pos)]))
            
        cp.backprop_source = neighbor_source_name
        # equivalent to AD
        cp.gradient_mode = 'implicit_neighbor'
        f_neighbor.dL_dself[neighbor_source_name] = 1
        
        # top-down
        for node in all_nodes_neighbor[::-1]:
            node.backprop(cp)
            
        df1_dx = []
        df2_dx = []
        
        # there's only one copy of current_pos, because deep_copy stops there
        for node in current_pos:
            df1_dx.append(node.dL_dself.get(current_source_name, 0))
            df2_dx.append(node.dL_dself.get(neighbor_source_name, 0))
            
        df1_dtheta = []
        df2_dtheta = []
        
        # there's only one copy of the dependencies, because deep_copy stops there
        for node in self.dependencies:
            df1_dtheta.append(node.dL_dself.get(current_source_name, 0))
            df2_dtheta.append(node.dL_dself.get(neighbor_source_name, 0))
            
        dot_dfdx_rd = [dot(df1_dx, self.rd),
                       dot(df2_dx, self.rd)]
        
        cos_tangent = abs(safe_division(dot_dfdx_rd[0], (sum([df1_dx[idx] ** 2 for idx in range(3)]) ** 0.5)))
        
        if is_all_constant(cp.min_abs_diff) or self.ignore_crease:
            is_at_intersection = False
        else:
            is_at_intersection = cos_tangent > 0.1
            
        # compute d(ro + t * rd) / du (or dv)
        # Two possibilities:
        # 1. directly use image gradient
        # 2. use AD
        #    2.1. ideally becuase there are 3 outputs and 2 inputs, forward mode AD is more efficient
        #    2.2. in practice, becuase ro and rd are usually very simple, we directly use reverse mode AD to avoid avoid implementing forward mode AD
        
        # find uv from ro and rd
        u, v = locate_argument_uv(Compound(self.children[1:7]))
        
        t_pl = ArgumentScalar('_raymarching_t_pl_')
        
        deriv_uv = np.zeros((2, 3)).tolist()
            
        for idx in range(3):
            
            cp.backprop_source = '_pos%d_' % idx
            cp.gradient_mode = 'ours'
            
            all_pos_nodes = current_pos[idx].all_nodes_dfs(exclude=set([id(current_t)]))
            current_pos[idx].dL_dself[cp.backprop_source] = 1
            
            # top-down
            for node in all_pos_nodes[::-1]:
                node.backprop(cp)
                
            if u is not None:
                deriv_uv[0][idx] = deriv_uv[0][idx] + u.dL_dself.get(cp.backprop_source, 0)
                
            if v is not None:
                deriv_uv[1][idx] = deriv_uv[1][idx] + v.dL_dself.get(cp.backprop_source, 0)

        ill_posed_thre = 10

        denum_intersection_uv_raw = [dot(df2_dx, deriv_uv[0]) * dot_dfdx_rd[0] - dot(df1_dx, deriv_uv[0]) * dot_dfdx_rd[1],
                                     dot(df2_dx, deriv_uv[1]) * dot_dfdx_rd[0] - dot(df1_dx, deriv_uv[1]) * dot_dfdx_rd[1]]

        denum_intersection_uv = [select(ill_posed_thre * abs(denum_intersection_uv_raw[0]) < abs(denum_intersection_uv_raw[1]),
                                        0,
                                        denum_intersection_uv_raw[0]),
                                 select(ill_posed_thre * abs(denum_intersection_uv_raw[1]) < abs(denum_intersection_uv_raw[0]),
                                        0,
                                        denum_intersection_uv_raw[1])]
        
        
        
        
        denum_intersection = select(compiler_params.pix_idx_pl <= 2, denum_intersection_uv[0], denum_intersection_uv[1])
        
        if self.ignore_crease:
            # extra safeguard if we ignore creases to avoid large magnitude error

            denum_smooth_uv_raw = [dot(df1_dx, deriv_uv[0]),
                                   dot(df1_dx, deriv_uv[1])]

            denum_smooth_uv = [select(ill_posed_thre * abs(denum_smooth_uv_raw[0]) < abs(denum_smooth_uv_raw[1]),
                                      0,
                                      denum_smooth_uv_raw[0]),
                               select(ill_posed_thre * abs(denum_smooth_uv_raw[1]) < abs(denum_smooth_uv_raw[0]),
                                      0,
                                      denum_smooth_uv_raw[1])]
        else:
            denum_smooth_uv = [dot(df1_dx, deriv_uv[0]),
                               dot(df1_dx, deriv_uv[1])]
        denum_smooth = -select(compiler_params.pix_idx_pl <= 2, denum_smooth_uv[0], denum_smooth_uv[1])

        denum = select(is_at_intersection, denum_intersection, denum_smooth)

        denum_to_compare = denum
        
        
        
        
        dL_dedge = dL_dself_comparisons
        dL_dt = 0
        
        if compiler_params.backprop_source in self.dL_dself.keys():
            
            # we treat t_closest and t equally as they have the same semantics
            dL_dt = self.dL_dself[compiler_params.backprop_source].children[1] + \
                    self.dL_dself[compiler_params.backprop_source].children[3]
            
            if dL_dt != 0:
                # determine whether the current position is at a geometry discontinuity, and would need to add dL_dt to dL_dedge
                # note that the rule is heuristic based and cannot handle arbitrary SDF
                # for example, for a concave shaped but C1 SDF, because there is not min/max at all in the SDF, it any ray hitting on the object will be classified as smooth
                # however, it is possible that the ray hits an interior edge
                # The most principled way is to sample the pixels at a high frequency, so that dt/du or dt/dv natually converges to 0 whenever it is at a smooth surface. However this costs too much computation than necessary for our optimization task.
                
                final_t = t_closest
                
                if self.is_at_geometry_discont is None:
                    
                    if self.wrapper is not None:
                        f_final = self.wrapper.res0
                        final_pos = self.wrapper.final_pos
                    else:
                        final_pos = []
                        for idx in range(3):
                            final_pos.append(self.ro[idx] + self.rd[idx] * final_t)

                        # we only care about the first return value: distance to object
                        f_final = self.body(*final_pos)[0]
                    
                    get_geometry_discont(f_final, final_pos + self.dependencies, compiler_params)
                    
                    self.is_at_geometry_discont = (self[0] != get_neighbor(self[0], compiler_params.pix_idx_pl)) | \
                                                  f_final.is_at_geometry_discont

                dL_dedge = dL_dedge + select(self.is_at_geometry_discont, 
                                             dL_dt * get_partial_trace_coord(final_t, compiler_params.pix_idx_pl),
                                             0)
                
                
                # verified: exclude dL_dt contribution on continuous values when it's at an intersection
                # the intuition is to avoid tangent rays causing numerical instable values
                dL_dt = select(self.is_at_geometry_discont, 0, dL_dt)
            
            # label
            skip_label = False
            if is_all_constant(self.dL_dself[compiler_params.backprop_source].children[5]):
                if is_constant(self.dL_dself[compiler_params.backprop_source].children[5].
                               to_source(compiler_params.as_mode(MODE_ALWAYS_INLINE)), 
                               0):
                    skip_label = True
                    
            if not skip_label:
                dL_dedge = dL_dedge + self.dL_dself[compiler_params.backprop_source].children[5] * \
                                     get_partial_trace_coord(self[5], compiler_params.pix_idx_pl)
        
        compiler_params.min_discont_denum_dict[id(self)] = [dL_dedge, denum_to_compare]
        
        
        # gradient is NOT related to init_t
        for idx in range(1, len(self.children)):
            
            if idx <= 3:
                # ro
                num_intersection = df1_dx[idx-1] * dot_dfdx_rd[1] - df2_dx[idx-1] * dot_dfdx_rd[0]
                num_smooth = df1_dx[idx-1]
            elif idx <= 6:
                # rd
                num_intersection = (df1_dx[idx-4] * dot_dfdx_rd[1] - df2_dx[idx-4] * dot_dfdx_rd[0]) * current_t
                num_smooth = df1_dx[idx-4] * current_t
            else:
                num_intersection = df1_dtheta[idx-7] * dot_dfdx_rd[1] - df2_dtheta[idx-7] * dot_dfdx_rd[0]
                num_smooth = df1_dtheta[idx-7]
                
            num = select(is_at_intersection, num_intersection, num_smooth)
            deriv_discont = safe_division(num, denum)
            
            if dL_dt != 0:
                deriv_cont = safe_division(-num_smooth, dot_dfdx_rd[0])
            else:
                deriv_cont = 0
            
            # TODO: figure out what the negative sign means mathematically
            
            deriv = -deriv_discont * dL_dedge + deriv_cont * dL_dt
            
            if compiler_params.backprop_source in self.children[idx].dL_dself.keys():
                self.children[idx].dL_dself[compiler_params.backprop_source] = \
                self.children[idx].dL_dself[compiler_params.backprop_source] + deriv
            else:
                self.children[idx].dL_dself[compiler_params.backprop_source] = deriv
                
                
        # children list:
        # 1 - 3: ro0, ro1, ro2
        # 4 - 6: rd0, rd1, rd2
        # 7 - 9: ax0, ax1, ax2
        # 10, 11: d1_thre, d2_thre,
        # 12 - 14: ce0, ce1, ce2,
        # 15 - 17: major_ax0, major_ax1, major_ax2,
        # 18: ellipse_ratio,
        # 19 - 21: minor_ax0, minor_ax1, minor_ax2,
        # 22: d3_thre

    def find_dependencies(self):
        """
        Because lambda_body might be defined in a way that depends on global or non-local variables
        We cannot assume loop output only depends on the input arguments only
        Execute loop body in scrath scope, and find all nodes that are not created in the scratch scope be dependency nodes
        """
        
        global scratch_scope
        scratch_scope = True
        
        inputs_pl = [self.ro[0] + self.rd[0] * self.t_invariant,
                     self.ro[1] + self.rd[1] * self.t_invariant,
                     self.ro[2] + self.rd[2] * self.t_invariant]
        
        # output to user-defined signed distance field function, search dependency within this subgraph
        # ignore derivs for dependencies
        ans = self.body(*inputs_pl)[:2]
        
        assert isinstance(ans, (list, tuple))
        
        ans = [to_expr(a) for a in ans]
        ans = Compound(ans)
            
        scratch_scope = False
            
        #seen = set([id(node) for node in (self.ro + self.rd + [self.t_invariant])])
        seen = set([id(node) for node in inputs_pl])
        
        # TODO: should fix: dependencies can also include ro and rd
        dependencies = []
        
        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            
            if not node.is_scratch:
                # no need to track grandchildren
                dependencies.append(node)
                return
                
            for child in node.children:
                if isinstance(child, Expr):
                    visit(child)
        
        visit(ans)
        
        self.dependencies = dependencies
        self.inputs_pl = inputs_pl
        
        # mutate the signed distance field function into the entire body of the raymarching loop
        res0 = ans.children[0]
        
        if False:
            cond_closest = res0 < self.res0_closest_invariant

            is_converge = res0 < 0.0004 * self.t_invariant

            t_closest = select(cond_closest, self.t_invariant, self.t_closest_invariant)
            res0_closest = select(cond_closest, res0, self.res0_closest_invariant)

            t = self.t_invariant + select(is_converge, 0., res0)
        else:
            cond_closest = res0 < self.res0_closest_invariant

            t_closest = select(cond_closest, self.t_invariant, self.t_closest_invariant)
            res0_closest = select(cond_closest, res0, self.res0_closest_invariant)

            is_converge = res0 < 0.0004 * t_closest

            t = self.t_invariant + res0
        
        self.outputs_pl = Compound([is_converge, t_closest, res0_closest, t] + ans.children, need_cast=False)
        
        # first four elements: is_converge, t_closest, res0_closet, t
        # followed by all other outputs from the original body
        self.ndims = len(self.outputs_pl.children)
        
    def to_source_impl(self, compiler_params, dummy=None):
        if compiler_params.mode == MODE_VARNAME:
            return self.var_name(compiler_params)
        else:
            
            self_id = self.statement_id(compiler_params)
            if self_id in compiler_params.statement_ids:
                # avoid duplicate
                return ''
            compiler_params.statement_ids.add(self_id)
            
            t_name = self.t_invariant.to_source(compiler_params.as_mode(MODE_VARNAME))
            t_closest_name = self.t_closest_invariant.to_source(compiler_params.as_mode(MODE_VARNAME))
            res0_closest_name = self.res0_closest_invariant.to_source(compiler_params.as_mode(MODE_VARNAME))
            
            lhs = self.to_source_impl(compiler_params.as_mode(MODE_VARNAME))
            return_list = []
            for child in self.children:
                return_list.append(child.to_source(compiler_params))
                
            #self.outputs_pl = self.outputs_pl.remove_pow()
            self.outputs_pl = remove_pow(self.outputs_pl)
            
            # generating inner loop code, parent relationship may be messed up, don't want to apply sanity check to var_name
            old_check_varname = compiler_params.check_varname
            compiler_params.check_varname = False
            
            body_str = self.outputs_pl.to_source(compiler_params)
            
            compiler_params.check_varname = old_check_varname
            
            if compiler_params.backend in ['tf', 'np', 'torch']:
                block = f"""
{t_name} = {self.init_t.to_source(compiler_params.as_mode(MODE_VARNAME))}
{t_closest_name} = {t_name}
{res0_closest_name} = {self.tmax.to_source(compiler_params.as_mode(MODE_VARNAME))}

for raymarching_loop_i in range({self.niters}):
{indent(body_str)}
    {lhs} = {self.outputs_pl.to_source(compiler_params.as_mode(MODE_VARNAME))};
    {t_closest_name} = {lhs}[1];
    {res0_closest_name} = {lhs}[2];
    {t_name} = {lhs}[3];
                """
            elif compiler_params.backend == 'glsl':
                
                block = f"""
float {t_name} = {self.init_t.to_source(compiler_params.as_mode(MODE_VARNAME))};
float {t_closest_name} = {t_name};
float {res0_closest_name} = {self.tmax.to_source(compiler_params.as_mode(MODE_VARNAME))};

float[{self.outputs_pl.ndims}] {lhs};

for (int raymarching_loop_i = 0; raymarching_loop_i < _raymarching_iter_{self.raymarching_count}; raymarching_loop_i++) {{
{indent(body_str)}
    {lhs} = {self.outputs_pl.to_source(compiler_params.as_mode(MODE_VARNAME))};
    {t_closest_name} = {lhs}[1];
    {res0_closest_name} = {lhs}[2];
    {t_name} = {lhs}[3];
}}
                """
            else:
                block = f"""
std::vector<Expr> {lhs};
Expr {t_name} = {self.init_t.to_source(compiler_params.as_mode(MODE_VARNAME))};
Expr {t_closest_name} = {t_name};
Expr {res0_closest_name} = {self.tmax.to_source(compiler_params.as_mode(MODE_VARNAME))};

for (int raymarching_loop_i = 0; raymarching_loop_i < {self.niters}; raymarching_loop_i++) {{
{indent(body_str)}
    
    {lhs} = {self.outputs_pl.to_source(compiler_params.as_mode(MODE_VARNAME))};
    {t_closest_name} = checkpoint({lhs}[1]);
    {res0_closest_name} = checkpoint({lhs}[2]);
    {t_name} = checkpoint({lhs}[3]);
}}
                """
                
            end_str = ';' if compiler_params.backend in ['hl', 'glsl'] else ''
            if hasattr(self, 'bind_scalars'):
                #assert self.root
                assign_str = ''
                
                for idx in range(len(self.bind_scalars)):
                    assign_str += f"""
{self.bind_scalars[idx].to_source(compiler_params.as_mode(MODE_VARNAME))} = {self.bind_values[idx].to_source(compiler_params.as_mode(MODE_VARNAME))}{end_str}
                    """
            else:
                assign_str = ''
                
            block = assign_str + '\n'.join(return_list) + block
            return block

    def __getitem__(self, index):
        """
        Output values of the Raymarching loop:
        0: is_fg (boolean)
        1: t_closest (length of the ray such that distance from ray endpoint to object is smallest)
        2: res0_closest (the smallest distance on any point along the ray to object)
        3: t (length of ray at the end of the raymarching loop)
        4: res0 (distance from ray endpoint to object at the end of the raymarching loop)
        5: label (label of the surface)
        """
        
        if need_animate:
            
            if isinstance(index, Expr):
                if is_all_constant(index):
                    index = eval(index.to_source(CompilerParams()))
            
            if isinstance(index, (int, float)):
                if index == 0:
                    return self.animated_cond
                elif index == 1:
                    return self.t_closest
                elif index == 2:
                    return self.res0_closest
                elif index == 3:
                    return self.final_t
                elif index == 4:
                    return self.final_res0
                elif index == 5:
                    return self.surface_label
        
        return GetItem(self, index)
    
class ChooseU(Expr):
    """
    HL syntax
    defines whether to use the choose_u in compute graph, or use choose_u_pl from global buffer
    """
    def __init__(self, node):
        super().__init__()
        self.children = [node]
        self.name = 'choose_u'
        self.dtype = BOOL_TYPE
        self.ndims = 0
        assert node.ndims == 0
        
    def to_source_impl(self, compiler_params, dummy=None):
        if compiler_params.mode in [MODE_VARNAME, MODE_ALWAYS_INLINE]:
            return Expr.to_source_impl(self, compiler_params)
        else:
            self_id = self.statement_id(compiler_params)
            if self_id in compiler_params.statement_ids:
                # avoid duplicate
                return ''
            compiler_params.statement_ids.add(self_id)
            
            if compiler_params.backend == 'hl':
                end_str = ';'
            else:
                end_str = ''
            
            if hasattr(self, 'bind_scalars'):
                #assert self.root
                assign_str = ''
                
                for idx in range(len(self.bind_scalars)):
                    
                    if isinstance(self.bind_values[idx], Expr):
                        rhs = self.bind_values[idx].to_source(compiler_params.as_mode(MODE_VARNAME))
                    else:
                        rhs = to_expr(self.bind_values[idx]).to_source(compiler_params.as_mode(MODE_VARNAME))
                    
                    assign_str += f"""
{self.bind_scalars[idx].to_source(compiler_params.as_mode(MODE_VARNAME))} = {rhs}{end_str}
                    """
            else:
                assign_str = ''
            
            lhs = self.var_name(compiler_params)
            rhs = self.children[0].var_name(compiler_params)
            
            return_list = []
            for child in self.children:
                return_list.append(child.to_source(compiler_params))
                
            if compiler_params.backend in ['tf', 'np', 'torch']:
                # TODO: fix logic here
                # For now, behave as if choose_u from compute_graph is always used
                block = f"""
{lhs} = {rhs}
"""
            else:
                block = f"""
Expr {lhs};
if (use_choose_u_pl) {{
    {lhs} = (*{CHOOSE_U_PL})(u + Ou, v + Ov, 0);
}} else {{
    {lhs} = {rhs};
}}
                """
                
            return assign_str + '\n'.join(return_list) + block
                
    
class GlobalBuffer(AlwaysInlineExpr):
    """
    HL syntax, defines a global read buffer
    """
    def __init__(self, output_idx, is_binary2float=None, combined_binary_idx=-1):
        super().__init__()
        self.output_idx = output_idx
        self.is_binary2float = is_binary2float
        self.combined_binary_idx = combined_binary_idx
        self.name = 'output%d' % self.output_idx
        self.children = [output_idx]
        
    def __call__(self, read_idx, orig_node, is_buffer_producer=False):
        return GlobalRead(self, read_idx, orig_node, is_buffer_producer)
    
    def to_source_impl(self, compiler_params, dummy=None):
        if compiler_params.mode in [MODE_VARNAME, MODE_ALWAYS_INLINE]:
            if compiler_params.backend == 'hl':
                return '(*%s)' % self.name
            else:
                return '(%s)' % self.name
        else:
            return ''

class GlobalRead(AlwaysInlineExpr):
    """
    HL syntax, read from global input
    """
    def __init__(self, buffer_node, read_idx, orig_node, is_buffer_producer):
        super().__init__()
        self.buffer = buffer_node
        self.read_idx = read_idx
        self.children = [buffer_node, read_idx]
        self.dtype = orig_node.dtype
        self.is_buffer_producer = is_buffer_producer
        
    def to_source_impl(self, compiler_params, dummy=None):
        if compiler_params.mode in [MODE_VARNAME, MODE_ALWAYS_INLINE]:
            if compiler_params.backend == 'hl' and self.dtype == BOOL_TYPE:
                cast_str = 'Halide::cast<bool> '
            else:
                cast_str = ''
                
            is_binary2float_args = ''
            if self.buffer.is_binary2float is not None:
                if self.buffer.is_binary2float[self.read_idx]:
                    assert self.buffer.combined_binary_idx >= 0
                    is_binary2float_args = ', ' + ConstExpr(True).to_source(compiler_params) + ', %d' % self.buffer.combined_binary_idx
            
            return f"""get_neighbor({self.buffer.to_source(compiler_params)}, pix_idx, {self.read_idx}, {ConstExpr(self.dtype == BOOL_TYPE).to_source(compiler_params)}, {ConstExpr(self.is_buffer_producer).to_source(compiler_params)}{is_binary2float_args}) """ 
        else:
            return ''

class Mux(Expr):
    """
    deprecated
    HL syntax, fuse multiple expressions into the a function.
    In TF, this simply concatenates the expressions to a list. 
    """
    def __init__(self, L):
        super().__init__()
        self.children = [to_expr(x) for x in L]
        self.dtype = self.children[0].dtype
        
    def to_source_impl(self, compiler_params, dummy=None):
        
        if compiler_params.mode == MODE_VARNAME:
            return self.var_name(compiler_params)
        else:
            # TODO: finish logic for tf
            assert compiler_params.backend == 'hl'
            
            self_id = self.statement_id(compiler_params)
            if self_id in compiler_params.statement_ids:
                # avoid duplicate
                return ''
            compiler_params.statement_ids.add(self_id)
            
            children_name_list = []
            return_list = []
            
            for e in self.children:
                children_name_list.append(e.to_source(compiler_params.as_mode(MODE_VARNAME)))
                return_list.append(e.to_source(compiler_params))
                
            lhs = self.to_source_impl(compiler_params.as_mode(MODE_VARNAME))
            
            statement = f"""
Func {lhs};
{lhs}(u, v, Ou, Ov, q) = mux(q, {{
"""
            for i in range(len(self.children)):
                statement += f"""
                             Halide::cast<float> ({children_name_list[i]}),"""
            statement = statement[:-1] + '\n});'
            
            return_list.append(statement)
            
            return '\n'.join(return_list)
        
def LetBind(node, scalars, values, compiler_params):
    """
    NOTE: NOT part of the DSL, should only be used internally by the compiler!
    
    All scalars that are (grand)children of node should be assigned to values by generating assignment statement before the source code of node is generated.
    Cannot bind AlwaysInlineExpr node, because they do not generate statement level code.
    In the current implementation, might have side-effect to unrelated nodes whose source code appears after the current node, and who also uses scalars. To ensure correctness, always assume node is root when generating source code (in to_source_impl).
    """
    assert not node.is_inline(compiler_params)
    assert len(scalars) == len(values)
    
    for scalar in scalars:
        assert isinstance(scalar, ArgumentScalar)
    
    node.bind_scalars = scalars
    node.bind_values = values 
    
    return node

class Binary2Float(Expr):
    """
    Combined several binary expression to Integer
    """
    def __init__(self, L, base_idx=0, as_scalar=False):
        super().__init__()
        self.children = [to_expr(x) for x in L]
        self.value = 0
        self.base_idx = base_idx
        self.as_scalar = as_scalar
        
        assert len(self.children) > 0
        
        self.dtype = INT_TYPE
        self.ndims = 0
        
        self.update_value()
        
    def update_value(self):
        self.value = 0

        for child_idx in range(len(self.children)):
            child = self.children[child_idx]
            self.value = self.value + cast2f(child) * (2 ** (child_idx + self.base_idx))
            
    def to_source_impl(self, compiler_params, dummy=None):
                
        if compiler_params.mode == MODE_VARNAME:
            return self.var_name(compiler_params)
        else:
            
            self_id = self.statement_id(compiler_params)
            if self_id in compiler_params.statement_ids:
                # avoid duplicate
                return ''
            compiler_params.statement_ids.add(self_id)
            
            lhs = self.to_source_impl(compiler_params.as_mode(MODE_VARNAME))

            return_list = []
            children_name = []
            for e in self.children:
                children_name.append(e.to_source(compiler_params.as_mode(MODE_VARNAME)))
                return_list.append(e.to_source(compiler_params))
                
            
                
            #value_name = self.value.var_name(compiler_params)
            
            #value_str = self.value.to_source_impl(compiler_params)
            #return_list.append(value_str)
                
            
            if compiler_params.backend in ['tf', 'np', 'torch']:
                end_str = ''
                declare_str = ''
                lbrace, rbrace = '[', ']'
            else:
                end_str = ';'
                
                if self.as_scalar:
                    declare_str = 'Expr '
                    lbrace, rbrace = '', ''
                else:
                    declare_str = 'std::vector<Expr> '
                    lbrace, rbrace = '{', '}'
                    
            
            
            if hasattr(self, 'bind_scalars'):
                assign_str = ''
                
                for idx in range(len(self.bind_scalars)):
                    
                    if isinstance(self.bind_values[idx], Expr):
                        rhs = self.bind_values[idx].to_source(compiler_params.as_mode(MODE_VARNAME))
                    else:
                        rhs = to_expr(self.bind_values[idx]).to_source(compiler_params.as_mode(MODE_VARNAME))
                    
                    assign_str += f"""
{self.bind_scalars[idx].to_source(compiler_params.as_mode(MODE_VARNAME))} = {rhs}{end_str}
                    """
            else:
                assign_str = ''
                
            caller_args = ', '.join(children_name)
            if compiler_params.backend == 'hl':
                caller_args = '{' + caller_args + '}'
            else:
                caller_args = '[' + caller_args + ']'
                
            statement = f"""{declare_str} {lhs} = {lbrace} binary2float({caller_args}, {self.base_idx}) {rbrace} {end_str}
            """
            
                
            #statement = declare_str + lhs + ' = ' + lbrace + value_name + rbrace + end_str + '\n'
               
            return_list.append(statement)
                                
            return assign_str + '\n'.join(return_list)
           

class Compound(Expr):
    """
    Combines several expressions into a compound expression.
    If Compound is involved in any computation (without GetItem), it will only be element-wise,
    i.e. we should NEVER compute Jacobians of d_parent/d_child, such computation should always be expressed using scalar Expr
    """
    def __init__(self, L, need_cast=True, check_ndims=True):
        super().__init__()
        self.children = [to_expr(x) for x in L]
        if len(self.children):
            self.dtype = VECTOR_TYPE
            self.ndims = len(self.children)
            self.element_type = []
            if check_ndims:
                for child in self.children:
                    self.element_type.append(child.dtype)
                    assert child.ndims == 0
        else:
            self.dtype = VOID_TYPE
            
        self.need_cast = need_cast
        
    def as_array(self):
        return np.array(self.children)
            
    def backprop(self, compiler_params):
        """
        Compute dL/dself, then propagate dL to children
        """
        
        if compiler_params.backprop_source not in self.dL_dself.keys():
            return
        
        for i in range(len(self.children)):

            child = self.children[i]
            
            if isinstance(child, RaymarchingLoop) and self.is_discont_to_output:
                # if self.is_discont_to_output = True, this indicates we are already inside a condition
                # in this case, the backprop method in RaymarchingLoop will find corresponding comparison node directly
                # the current dL_dself already includes undesirable dcond_diff/du, shoudl discard
                continue
            
            global DEFAULT_WRT_ID
            global DEFAULT_WRT_NAME
        
            DEFAULT_WRT_ID = id(child)
            DEFAULT_WRT_NAME = getattr(child, 'name', None)
            
            extra_deriv = self.dL_dself[compiler_params.backprop_source][i]
            
            if self.dL_dself_scalar.get(compiler_params.backprop_source, 1) is not 1:
                extra_deriv = extra_deriv * self.dL_dself_scalar[compiler_params.backprop_source]
            
            if compiler_params.backprop_source not in child.dL_dself.keys():
                child.dL_dself[compiler_params.backprop_source] = extra_deriv
            else:
                child.dL_dself[compiler_params.backprop_source] = \
                child.dL_dself[compiler_params.backprop_source] + extra_deriv
                
            if compiler_params.gradient_mode == 'implicit_current':
                if isinstance(self.dL_mask[i], Expr) and child.dL_mask is None:
                    child.dL_mask = self.dL_mask[i]
                elif self.dL_mask[i] is False:
                    child.dL_mask = False
                
            DEFAULT_WRT_ID = None
            DEFAULT_WRT_NAME = None

    def to_source_impl(self, compiler_params, dummy=None):
                
        if compiler_params.mode == MODE_VARNAME:
            return self.var_name(compiler_params)
        else:
            
            self_id = self.statement_id(compiler_params)
            if self_id in compiler_params.statement_ids:
                # avoid duplicate
                return ''
            compiler_params.statement_ids.add(self_id)

            children_name_list = []
            return_list = []
            for e in self.children:
                children_name_list.append(e.to_source(compiler_params.as_mode(MODE_VARNAME)))
                return_list.append(e.to_source(compiler_params))
                
            lhs = self.to_source_impl(compiler_params.as_mode(MODE_VARNAME))
            
            end_str = ';' if compiler_params.backend in ['hl', 'glsl'] else ''
            if hasattr(self, 'bind_scalars'):
                #assert self.root
                assign_str = ''
                
                for idx in range(len(self.bind_scalars)):
                    
                    if isinstance(self.bind_values[idx], Expr):
                        rhs = self.bind_values[idx].to_source(compiler_params.as_mode(MODE_VARNAME))
                    else:
                        rhs = to_expr(self.bind_values[idx]).to_source(compiler_params.as_mode(MODE_VARNAME))
                    
                    assign_str += f"""
{self.bind_scalars[idx].to_source(compiler_params.as_mode(MODE_VARNAME))} = {rhs}{end_str}
                    """
            else:
                assign_str = ''
                
            if compiler_params.backend in ['tf', 'np', 'torch']:
                
                statement = f"""{lhs} = ["""
                
                for i in range(len(self.children)):
                    statement += children_name_list[i] + ', '
                statement = statement[:-2]
                statement += ']\n'
            elif compiler_params.backend == 'glsl':
                
                if self.ndims <= 1:
                    statement = f"""float {lhs} = ("""
                elif self.ndims <= 4:
                    statement = f"""vec{self.ndims} {lhs} = vec{self.ndims}("""
                else:
                    statement = f"""float[{self.ndims}] {lhs} = float[]("""
                # always cast to float
                for i in range(len(self.children)):
                    current_element = f"""float({children_name_list[i]}), """
                    statement += current_element
                statement = statement[:-2] + ');\n'
            else:
                
                statement = f"""std::vector<Expr> {lhs} = {{"""
                
                if compiler_params.need_cast and self.need_cast:
                    cast_str = 'Halide::cast<float>'
                else:
                    cast_str = ''
                
                for i in range(len(self.children)):
                    
                    current_element = cast_str + ' (' + children_name_list[i] + '), \n'
                    
                    if isinstance(self, PythonicList):
                        if isinstance(self.child_indices[i], list):
                            current_element = ''
                            for vec_idx in range(len(self.child_indices[i])):
                                current_element += f"""{cast_str}({children_name_list[i]}[vec_idx]), \n"""
                            
                    statement += current_element
                statement = statement[:-3]
                statement += '};\n'
               
            return_list.append(statement)
                                
            return assign_str + '\n'.join(return_list)

class PythonicList(Compound):
    """
    Additional wrapper for Compound
    In Halide the list of nodes should all have the same shape, e.g. Expr
    Therefore we cannot combind an Expr and a std::vector<Expr> into the same list
    This wrapper solves the problem and correctly handles indexing
    """
    def __init__(self, L):
        super().__init__(L, check_ndims=False)
        self.scalar_nodes = []
        self.vector_nodes = []
        
        for x in L:
            x = to_expr(x)
            if x.ndims == 0:
                self.scalar_nodes.append(x)
            else:
                self.vector_nodes.append(x)
        
        self.children = self.scalar_nodes + self.vector_nodes
        
        # scalar idx is simple, their position in scalar_nodes list is their idx
        self.child_indices = np.arange(len(self.scalar_nodes)).tolist()
        
        # resolve vector idx
        for child in self.vector_nodes:
            current_idx = []
            for n in range(child.ndims):
                current_idx.append(n + len(self.child_indices))
            self.child_indices.append(current_idx)

class UnaryOp(Expr):
    def __init__(self, op, a):
        super().__init__()
        self.children = [str(op), to_expr(a)]
        self.dtype = self.a.dtype
        self.ndims = self.a.ndims

    (op, a) = gen_attrs([str, to_expr])
    
    def gradient(self, compiler_params, get_idx):
        
        a = self.a
        
        if get_idx is not None:
            assert a.ndims > 0
            a = a[get_idx]
        
        if self.op == '-':
            return [(self.a, -1)]
        else:
            raise ValueError('not implemented derivative:', self.op)

    def to_source_expr(self, compiler_params, dummy=None):
        
        collected_ans = []
        
        if compiler_params.backend == 'glsl':
            nentries = 1
        else:
            nentries = max(1, self.ndims)
        
        for idx in range(nentries):
            
            if self.ndims == 0:
                get_str = ''
            else:
                get_str = '[%d]' % idx
                
            a = self.a.to_source(compiler_params)
                
            if compiler_params.backend == 'glsl':
                a = '%s(%s)' % (self.dtype, a)
            else:
                a = a + get_str
        
            if self.op == '!' and compiler_params.backend in ['tf', 'np', 'torch']:
                ans = '%s.logical_not(%s)' % (compiler_params.backend, a)
            else:
                ans = '(' + self.op + '(' + a + '))'
                
            collected_ans.append(ans)
        return ', '.join(collected_ans)
    
class Clone(Expr):
    """
    Clone an identity expression
    Should only be used in GLSL backend
    """
    def __init__(self, node):
        super().__init__()
        self.children = [node]
        self.val = node
        self.dtype = node.dtype
        self.ndims = node.ndims
        if hasattr(node, 'short_name'):
            self.short_name = node.short_name
    
    def repr(self, extra_info=True, cache=None):
        """
        should not be removed by remove_redundant_expr
        """
        return f"""Clone({id(self)}, {self.val.repr()})"""
    
    def to_source_impl(self, compiler_params, dummy=None):
        
        assert compiler_params.backend == 'glsl'
        
        if compiler_params.mode == MODE_VARNAME:
            return self.var_name(compiler_params, check=False)
        else:
            self_id = self.statement_id(compiler_params)
            if self_id in compiler_params.statement_ids:
                # avoid duplicate
                return ''
            compiler_params.statement_ids.add(self_id)
            
        if not isinstance(self.val, ArgumentScalar):
            return_str = self.val.to_source_impl(compiler_params)
        else:
            return_str = ''
            
        declare_str = self.val.get_declare_str(compiler_params)
        
        lhs = self.to_source_impl(compiler_params.as_mode(MODE_VARNAME))
        rhs = self.val.to_source_impl(compiler_params.as_mode(MODE_VARNAME))
        
        return_str += '\n' +  f"""{declare_str} {lhs} = {rhs};"""
        return return_str
    
class Animate(Expr):
    """
    Indicates we should expose an animate interface in the GLSL backend
    """
    def __init__(self, name, inout_ls, in_ls=[]):
        
        super().__init__()
        
        if name not in animate_enum.keys():
            self.name = name
            animate_enum[name] = 1
        else:
            self.name = name + str(animate_enum[name])
            animate_enum[name] += 1
            
        if need_animate:
            self.inout_ls = []
            self.updated = []
            for val in inout_ls:
                if isinstance(val, Expr):
                    self.inout_ls.append(Clone(val))
                    self.updated.append(AnimateUpdate(self, self.inout_ls[-1]))
                else:
                    assert isinstance(val, (list, tuple, np.ndarray))
                    current_ans = []
                    current_update = []
                    for entry in val:
                        assert isinstance(entry, Expr)
                        current_ans.append(Clone(entry))
                        current_update.append(AnimateUpdate(self, current_ans[-1]))
                    self.inout_ls.append(current_ans)
                    self.updated.append(current_update)
        else:
            self.inout_ls = copy.copy(inout_ls)
            self.updated = self.inout_ls
            
        self.in_ls = copy.copy(in_ls)
        
        self.children = self.inout_ls + self.in_ls
        
    def update(self):
        
        return self.updated
        
        if False:
            if not need_animate:
                return self.inout_ls

            ans = []
            for val in self.inout_ls:
                if isinstance(val, Expr):
                    ans.append(AnimateUpdate(self, val))
                else:
                    assert isinstance(val, (list, tuple, np.ndarray))
                    current_ans = []
                    for entry in val:
                        assert isinstance(entry, Expr)
                        current_ans.append(AnimateUpdate(self, entry))
                    ans.append(current_ans)
            return ans
    
    def repr(self, extra_info=True, cache=None):
        return self.name
    
    def to_source_impl(self, compiler_params, dummy=None):
        
        assert compiler_params.backend == 'glsl'
        
        if compiler_params.mode == MODE_VARNAME:
            return ''
        else:
            self_id = self.statement_id(compiler_params)
            if self_id in compiler_params.statement_ids:
                # avoid duplicate
                return ''
            compiler_params.statement_ids.add(self_id)
                        
            animate_declare = '\nvoid %s(' % self.name
            call_str = '%s(' % self.name
            return_list = []
            
            for ls, var_type in [(self.inout_ls, 'inout'), (self.in_ls, 'in')]:
                
                for val in ls:
                
                    if isinstance(val, Expr):
                        entries = [val]
                    else:
                        assert isinstance(val, (list, tuple, np.ndarray))
                        entries = val

                    for entry in entries:
                        
                        assert isinstance(entry, Expr)

                        argname = entry.var_name(compiler_params, check=False)

                        #argname = entry.to_source(compiler_params.as_mode(MODE_VARNAME))
                        short_name = getattr(entry, 'short_name', argname)
                        declare_str = entry.get_declare_str(compiler_params)

                        animate_declare += '\n%s %s %s, ' % (var_type, declare_str, short_name)
                        call_str += '%s, ' % argname

                        return_list.append(entry.to_source(compiler_params))
                    
            animate_declare = animate_declare[:-2] + '){}'
            call_str = call_str[:-2] + ');'
            
            compiler_params.animate_declares.append(animate_declare)
            
            return_list.append(call_str)
            return '\n'.join(return_list)

class AnimateUpdate(Expr):
    def __init__(self, animate, val):
        
        super().__init__()
        
        self.animate = animate
        self.val = val
        
        # Do not put Animate object inside the DAT
        # to avoid complication when resolving gradient or discontinuity parameters etc.
        
        self.children = [self.val]
        self.ndims = val.ndims
        self.dtype = val.dtype
        
    def repr(self, extra_info=True, cache=None):
        return self.animate.name + self.val.repr()
        
    def gradient(self, compiler_params, get_idx):
        return [(self.val, 1)]
    
    def to_source_impl(self, compiler_params, dummy=None):
        
        assert compiler_params.backend == 'glsl'
                
        if compiler_params.mode == MODE_VARNAME:
            # use the same name as self.val
            return self.val.var_name(compiler_params, check=False)
        else:
            return self.animate.to_source(compiler_params)
    
def optimize_dot_select_recurse(a, b):
    # b is the select node
    dot_vals = []
    for node in b.children[2:]:
        if isinstance(node, Call) and getattr(node, 'name', '') == 'select' and node.ndims > 0:
            dot_val = optimize_dot_select_recurse(a, node)
        else:
            dot_elements = a * node
            dot_val = sum([dot_elements[idx] for idx in range(node.ndims)], 0)
        dot_vals.append(dot_val)
    
    return select(b.children[1], dot_vals[0], dot_vals[1])
    
#------------------------------------------------------------------------------------
# Function implementations

sign = Func('sign',        
            gradient_lambda=lambda x, *args: [0],
            tf_name = 'wrapper("sign")')

def gradient_select(input_args, output, compiler_params, nosmooth=False):
    
    cond = input_args[0]
    left = input_args[1]
    right = input_args[2]

    propagate_mask = None

    if compiler_params.gradient_mode == 'ours' and not nosmooth and getattr(output, 'op', '') not in ['min', 'max']:

        # can be scalar or vector, depend on ndims for left and right
        raw_scale = left - right
        raw_scale.propagate_params_dependency(compiler_params)
        
        if compiler_params.select_rule == 1 and output.ndims == 0:
            scale = get_neighbor(raw_scale, compiler_params.pix_idx_pl)
        elif compiler_params.select_rule == 3 and output.ndims == 0:
            scale = 0.5 * (raw_scale + get_neighbor(raw_scale, compiler_params.pix_idx_pl))
        else:
            # If possible, defer get_neighbor after dot product to save number of nodes that needs to be memoized
            scale = raw_scale
            
        scalar_scale = sign(get_partial_trace_coord(cond, compiler_params.pix_idx_pl))
        
        if scale.ndims == 0:
            cond_derivs = [(cond, scale * scalar_scale)]
        else:
            cond_derivs = [None]
    elif compiler_params.gradient_mode == 'implicit_current':
        # Follow AD rule (backprop nothing to cond), but collects min/max statistic
        # in sign distance functions, assume it's always min/max of smooth functions
        # therefore the only time select is accessed is via minimum() or maximum()
        assert hasattr(output, 'op')
        
        assert output.ndims == 0
        
        abs_diff = abs(left - right)
        # always prefer nodes closer to leaf
        is_current_minimum = (abs_diff <= compiler_params.min_abs_diff)
        mask = output.dL_mask

        assert mask is not None

        if mask is not False:
            is_current_minimum = is_current_minimum & mask
            
            propagate_mask = [mask & cond, mask & (~cond)]
        else:
            propagate_mask = [cond, ~cond]
                
        if output.op == 'max':
            choose_left = left >= right
            current_type = ConstExpr(True)
        else:
            choose_left = left <= right
            current_type = ConstExpr(False)
            
        update_min = ConstExpr(False)
        
        if isinstance(left, Expr) and not is_all_constant(left) and hasattr(left, 'comparison_args_idx') and left.params_only < 3:
            reset_left = is_current_minimum & choose_left
            compiler_params.reset_args_idx = select(reset_left, left.comparison_args_idx, compiler_params.reset_args_idx)
            update_min = update_min | reset_left
            
        if isinstance(right, Expr) and not is_all_constant(right) and hasattr(right, 'comparison_args_idx') and right.params_only < 3:
            reset_right = is_current_minimum & (~choose_left)
            compiler_params.reset_args_idx = select(reset_right, right.comparison_args_idx, compiler_params.reset_args_idx)
            update_min = update_min | reset_right
            
        if not is_all_constant(update_min):
            compiler_params.reset_args_type = select(update_min, cast2b(current_type), cast2b(compiler_params.reset_args_type))
            compiler_params.min_abs_diff = select(update_min, abs_diff, compiler_params.min_abs_diff)
        
        cond_derivs = []
    else:
        cond_derivs = []
        
    if propagate_mask is not None:
        # treating everythin as continuous in this mode
        assert len(cond_derivs) == 0
        return [(left, [select(cond, 1, 0), propagate_mask[0]]), (right, [select(cond, 0, 1), propagate_mask[1]])]
    elif output.ndims == 0:
        if compiler_params.select_rule in [0, 1]:
            return cond_derivs + [(left, select(cond, 1, 0)), (right, select(cond, 0, 1))]
        elif compiler_params.select_rule == 2:
            return cond_derivs + [(left, select(get_neighbor(cond, compiler_params.pix_idx_pl), 1, 0)), 
                                  (right, select(get_neighbor(cond, compiler_params.pix_idx_pl), 0, 1))]
        elif compiler_params.select_rule == 3:
            return cond_derivs + [(left, 0.5 * (cast2f(cond) + cast2f(get_neighbor(cond, compiler_params.pix_idx_pl)))), 
                                  (right, 1. - 0.5 * (cast2f(cond) + cast2f(get_neighbor(cond, compiler_params.pix_idx_pl))))]
    else:
        if len(cond_derivs) > 0:
        
            can_defer = False
            
            if compiler_params.select_rule in [0, 2]:
                scale = raw_scale * scalar_scale
            else:
                scale = None

            if compiler_params.select_rule in [1, 3]:
                # Check if output.dL_dself[compiler_params.backprop_source] depends on u or v
                # If not, can safely defer get_neighbor(scale) after dot product
                # This optimization should NOT be applied to tf backend, as get_neighbor implemented in Halide cannot seperate out dL_dcol
                if compiler_params.backend in ['tf', 'np', 'torch'] or output.dL_dself[compiler_params.backprop_source].depend_on_uv(compiler_params):
                    if compiler_params.select_rule == 1:
                        scale = get_neighbor(raw_scale, compiler_params.pix_idx_pl) * scalar_scale
                    else:
                        scale = 0.5 * (raw_scale + get_neighbor(raw_scale, compiler_params.pix_idx_pl)) * scalar_scale
                else:
                    can_defer = True
                    # scalar_scale should be multiplied AFTER get_neighbor
                    scale = raw_scale
            
            if can_defer and \
            ((isinstance(left, Call) and getattr(left, 'name', '') == 'select' and left.ndims > 0) or \
             (isinstance(right, Call) and getattr(right, 'name', '') == 'select' and right.ndims > 0)):
                # dot before select to save select() operation
                if isinstance(left, Call) and getattr(left, 'name', '') == 'select' and left.ndims > 0:
                    dot_left = optimize_dot_select_recurse(output.dL_dself[compiler_params.backprop_source], left)
                else:
                    dot_left_elements = output.dL_dself[compiler_params.backprop_source] * left
                    dot_left = sum([dot_left_elements[idx] for idx in range(output.ndims)], 0)

                if isinstance(right, Call) and getattr(right, 'name', '') == 'select' and right.ndims > 0:
                    dot_right = optimize_dot_select_recurse(output.dL_dself[compiler_params.backprop_source], right)
                else:
                    dot_right_elements = output.dL_dself[compiler_params.backprop_source] * right
                    dot_right = sum([dot_right_elements[idx] for idx in range(output.ndims)], 0)

                comparison_dot_val = dot_left - dot_right
            else:
                comparison_dot_elements = output.dL_dself[compiler_params.backprop_source] * scale
                comparison_dot_val = sum([comparison_dot_elements[idx] for idx in range(output.ndims)], 0)

            if can_defer:
                if compiler_params.select_rule == 1:
                    comparison_dot_val = get_neighbor(comparison_dot_val, compiler_params.pix_idx_pl) * scalar_scale
                elif compiler_params.select_rule == 3:
                    comparison_dot_val = 0.5 * (comparison_dot_val + get_neighbor(comparison_dot_val, compiler_params.pix_idx_pl)) * scalar_scale
                else:
                    raise
                
            if output.dL_dself_scalar.get(compiler_params.backprop_source, 1) is not 1:
                comparison_dot_val = comparison_dot_val * output.dL_dself_scalar[compiler_params.backprop_source]
                
            if compiler_params.backprop_source not in cond.dL_dself.keys():
                cond.dL_dself[compiler_params.backprop_source] = comparison_dot_val
            else:
                cond.dL_dself[compiler_params.backprop_source] = cond.dL_dself[compiler_params.backprop_source] + comparison_dot_val
                
        branch_dot_val = sum([output.dL_dself[compiler_params.backprop_source][idx] for idx in range(output.ndims)], 0)
        if output.dL_dself_scalar.get(compiler_params.backprop_source, 1) is not 1:
            branch_dot_val = branch_dot_val * output.dL_dself_scalar[compiler_params.backprop_source]

        for node in [left, right]:
            
            if compiler_params.select_rule in [0, 1]:
                if node is left:
                    deriv = select(cond, 1, 0)
                else:
                    deriv = select(cond, 0, 1)
            elif compiler_params.select_rule == 2:
                if node is left:
                    deriv = select(get_neighbor(cond, compiler_params.pix_idx_pl), 1, 0)
                else:
                    deriv = select(get_neighbor(cond, compiler_params.pix_idx_pl), 0, 1)
            elif compiler_params.select_rule == 3:
                if node is left:
                    deriv = 0.5 * (cast2f(cond) + cast2f(get_neighbor(cond, compiler_params.pix_idx_pl)))
                else:
                    deriv = 1. - 0.5 * (cast2f(cond) + cast2f(get_neighbor(cond, compiler_params.pix_idx_pl)))
                
            if node.ndims == 0:
                if compiler_params.backprop_source not in node.dL_dself.keys():
                    node.dL_dself[compiler_params.backprop_source] = deriv * branch_dot_val
                else:
                    node.dL_dself[compiler_params.backprop_source] = node.dL_dself[compiler_params.backprop_source] + \
                                                                     deriv * branch_dot_val
            else:
                if compiler_params.backprop_source not in node.dL_dself.keys():
                    # only apply the seperate scalar multiplier optimization if NO gradient has previously been accumulated
                    node.dL_dself[compiler_params.backprop_source] = output.dL_dself[compiler_params.backprop_source]
                    node.dL_dself_scalar[compiler_params.backprop_source] = deriv * output.dL_dself_scalar.get(compiler_params.backprop_source, 1)
                else:
                    # use regular rule
                    node.dL_dself[compiler_params.backprop_source] = node.dL_dself[compiler_params.backprop_source] + deriv * output.dL_dself_scalar.get(compiler_params.backprop_source, 1) * output.dL_dself[compiler_params.backprop_source]

        return

select_f = Func('select',
              gradient_lambda=gradient_select,
              tf_name = 'select_nosmooth')

def select(cond, left, right):
    
    if id(left) == id(right):
        return left
    elif is_all_constant(cond):
        cond_val = get_const_val(cond)
        if cond_val is True:
            return left
        else:
            return right
    elif isinstance(left, Expr) and isinstance(right, Expr):
        if is_all_constant(left) and is_all_constant(right):
        
            left_val = get_const_val(left)
            right_val = get_const_val(right)

            if left_val is not None and right_val is not None and left_val is right_val:
                return left_val        

    return select_f(cond, left, right)

select_nosmooth = Func('select_nosmooth',
                  gradient_lambda=lambda *args: gradient_select(*args, nosmooth=True),
                  tf_name = 'select_nosmooth')

get_scale = Func('get_scale')

get_neighbor_f = Func('get_neighbor')

def get_neighbor(node, pix_idx):

    if is_all_constant(node):
        return node
    elif node.params_only == 4:
        # node only depends on continuous parameters, 
        # because we don't apply random noise to such parameters, node will NEVER have different value in neighbor
        return node
    else:
        return get_neighbor_f(node, pix_idx)
    
get_partial_trace_coord_f = Func('get_partial_trace_coord')

def get_partial_trace_coord(node, pix_idx):

    if is_all_constant(node):
        return ConstExpr(0.0)
    elif node.params_only == 4:
        # node only depends on continuous parameters, 
        # because we don't apply random noise to such parameters, node will NEVER have different value in neighbor
        return ConstExpr(0.0)
    else:
        # remove unnecesary constants
        while True:
            if isinstance(node, BinaryOp) and getattr(node, 'op', '') in ['+', '-']:
                if is_all_constant(node.a):
                    node = node.b
                elif is_all_constant(node.b):
                    node = node.a
                else:
                    break
            else:
                break
                
        return get_partial_trace_coord_f(node, pix_idx)


cast2f = Func('cast2f')
cast2b = Func('cast2b')

def backprop_discont(compiler_params, self, dL, partial_coord):
    
    compiler_params.min_discont_denum_dict[id(self)] = [dL, partial_coord]
    
    # first compute min_discont_denum
    
    # do_update = (dL != 0) & (abs(partial_coord) < compiler_params.min_discont_denum)
    
    # compiler_params.min_discont_denum = select(do_update, abs(partial_coord), compiler_params.min_discont_denum)
    
    # then compute safe_division(1 / partial_coord)
    
    return safe_division(1, partial_coord)

def safe_division(a, b, safe_value=0, thre=0, div_val=None):
    
    
    if True:
        cond = abs(b) <= thre + 1e-8
    else:
        # DOGE: debug only
        cond = abs(b) <= thre
    
    if div_val is None:
        div_val = a / b
        
    ans = select(cond, safe_value, div_val)
    ans.op = 'safe_division'
    return ans

def safe_pow(a, b):
    cond = (abs(a) <= 0) & (b < 0)
    return select(cond, 0, a ** b)

def safe_log(a):
    cond = abs(a) <= 0
    return select(cond, 0, log(a))

def mark_comparison_args(a, b):
    global comparison_args_count
    
    if is_all_constant(a) or is_all_constant(b):
        # Do not give comparison_args_idx if node is only comparing with constant
        # Otherwise we may end up in a constant branch
        return
    
    for node in [a, b]:
        if isinstance(node, Expr):
            if not hasattr(node, 'comparison_args_idx'):
                node.comparison_args_idx = comparison_args_count
                comparison_args_count += 1

def minimum(a, b):
        
    if is_all_constant(a):
        # always put constant node as the second argument
        c = a
        a = b
        b = c
    
    ans = select(a <= b, a, b)
    ans.op = 'min'
    
    mark_comparison_args(a, b)
    
    return ans

def maximum(a, b):
    
    if is_all_constant(a):
        # always put constant node as the second argument
        c = a
        a = b
        b = c
        
    ans = select(a >= b, a, b)
    ans.op = 'max'
    
    mark_comparison_args(a, b)
    
    return ans

SQRT_PI = np.pi ** 0.5

erf = Func('erf',
           gradient_lambda=lambda x, *args: [2 / SQRT_PI * exp(-x[0] ** 2)],
           tf_name = 'wrapper("erf")')

exp = Func('exp',
           gradient_lambda=lambda x, *args: [args[0]],
           tf_name = 'wrapper("exp")')

cos = Func('cos',
           gradient_lambda=lambda x, *args: [-sin(x[0])],
           tf_name = 'wrapper("cos")')

sin = Func('sin',
           gradient_lambda=lambda x, *args: [cos(x[0])],
           tf_name = 'wrapper("sin")')

atan = Func('atan',
            gradient_lambda=lambda x, *args: [1 / (x[0] ** 2 + 1)],
            tf_name = 'wrapper("atan")')

floor = Func('floor',
             gradient_lambda=lambda x, *args: [0],
             tf_name = 'wrapper("floor")')

floor_from_fract = lambda x: (x - fract(x))

ceil = Func('ceil',
            tf_name = 'wrapper("ceil")')

ceil_from_fract = lambda x: (x + fract(-x))

fract = Func('fract',
             gradient_lambda=lambda x, *args: [1],
             tf_name = 'tf_fract')


sqrt = lambda arg: arg**0.5
sqrt_f = Func('sqrt',
              gradient_lambda=lambda x, val, *args: [0.5 / val],
              tf_name = 'wrapper("sqrt")')

power = Func('pow',
             tf_name = 'tf.math.pow')

log = Func('log',
           gradient_lambda=lambda x, *args: [select(x[0] > 0, 1 / x[0], 1e4)], # safeguard out of bounds
           tf_name = 'wrapper("log")')

as_const = Func('copy',
                gradient_lambda=lambda x, *args: [],
                tf_name = 'wrapper("copy")')
            
def dot(u, v):
    """
    Take dot product between two arrays.
    """
    return sum(u[i]*v[i] for i in range(len(u)))

class Object:
    """
    Wrapper that binds multiple nodes with semantic meaning, and give unique name to each object
    This object itself will never appear in the DAG
    """
    def __init__(self, name, **kwargs):
        global object_enum
        current_idx = len(object_enum.get(name, []))
        
        self.name = name + '_' + str(current_idx)
        self.class_name = name
        
        if name not in object_enum:
            object_enum[name] = [self]
        else:
            object_enum[name].append(self)
            
        self.entries = {}
        
        for key, val in kwargs.items():
            
            current_name = self.name + '_' + key
            
            if isinstance(val, (list, np.ndarray)):
                for idx in range(len(val)):
                    if isinstance(val[idx], Expr):
                        val[idx].short_name = current_name + '_%d' % idx
                    else:
                        assert isinstance(val[idx], (float, int))
                #setattr(self, key, Compound(val))
                setattr(self, key, np.array(val))
            else:
                assert isinstance(val, Expr)
                val.short_name = current_name
                setattr(self, key, val)
                
            self.entries[key] = getattr(self, key)
                
def possible_update(vals, update_func, *args):
    """
    vals: list of Expr whose values will be updated after calling update_func
    During regular fw and bw, works as simply calling update_func to update vals
    During pruning stage, the additional flag added allows use to search the possibiliy of skipping some certain updates
    """
    
    assert isinstance(vals, list)
    
    # make a shallow copy to avoid side-effect in update_func directly replaces elements vals
    orig_vals = copy.copy(vals)
    
    new_vals = update_func(vals, *args)
    
    global update_counter
    
    for idx in range(len(vals)):
        vals[idx] = UpdateWrapper(new_vals[idx], orig_vals[idx], update_counter)
        
    update_counter += 1