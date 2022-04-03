
import numpy
import random
import os
import pickle

ERROR = 0
TIME = 1

def machine_process_id():
    """
    Get a string uniquely identifying the machine and process.
    """
    machine_id = os.uname()[1]
    pid = os.getpid()
    return machine_id + '_' + str(pid)

def shuffled(L):
    L = list(L)
    random.shuffle(L)
    return L

def get_indices(L, x):
    return [i for (i, v) in enumerate(L) if v == x]

def trap_exception_or_none(f):
    try:
        f()
        return None
    except Exception as e:
        return e

def get_pareto(error_L, time_L):
    """Get indices of Pareto frontier."""
    maxval = 1e100
    assert len(error_L) == len(time_L)
    INDEX = 2
    error_L = [error_L[idx] if not numpy.isnan(error_L[idx]) else maxval for idx in range(len(error_L))]
    time_L  = [time_L[idx] if not numpy.isnan(time_L[idx]) else maxval for idx in range(len(time_L))]
    
    L = sorted([(error_L[idx], time_L[idx], idx) for idx in range(len(error_L))])
    ans = []
    for i in range(len(L)):
        if len(ans) == 0 and L[i][ERROR] < maxval and L[i][TIME] < maxval:
            ans.append(L[i][INDEX])
        else:
            error_prev = error_L[ans[-1]]
            time_prev = time_L[ans[-1]]
            error_current = L[i][ERROR]
            time_current = L[i][TIME]
            assert error_current >= error_prev
            if time_current < time_prev and L[i][ERROR] < maxval and L[i][TIME] < maxval:
                ans.append(L[i][INDEX])
    return numpy.array(ans)

def get_pareto_rank(error_L, time_L):
    """
    Get ranking within Pareto frontier.
    """
    assert len(error_L) == len(time_L)
    pop = [(error_L[i], time_L[i]) for i in range(len(error_L))]
    
    idx_pop = [(i, pop[i]) for i in range(len(pop))]
    rank = [None for x in range(len(pop))]

    current_rank = 0    
    while len(idx_pop):
        errorL = [x[1][ERROR] for x in idx_pop]
        timeL = [x[1][TIME] for x in idx_pop]
        pareto_i = get_pareto(errorL, timeL)
        
        for i in pareto_i:
            rank[idx_pop[i][0]] = current_rank
        current_rank += 1
        pareto_i = set(pareto_i)
        idx_pop = [x for (i, x) in enumerate(idx_pop) if i not in pareto_i]
    
    assert all([isinstance(x, int) for x in rank])
    return numpy.array(rank)

def replace_line(s, start, replace):
    """
    Given a text file str s, match a line starting with the string start and replace it with replace.
    """
    lines = s.split('\n')
    found = False
    for i in range(len(lines)):
        if lines[i].startswith(start):
            lines[i] = replace
            found = True
            break
    if not found:
        raise ValueError('could not match string %s' % start)
    return '\n'.join(lines)

def change_c_define(filename, define_name, value):
    """
    Look for a #define of the given name str and switch it to have the given value str.
    """
    with open(filename, 'rt') as f:
        s0 = s = f.read()

    start = '#define ' + define_name
    s = replace_line(s, start, start + ' ' + value)

    if s != s0:
        with open(filename, 'wt') as f:
            f.write(s)

def inverse_dict(d):
    """
    Invert a mapping or else raise an error if not one-to-one.
    """
    ans = {}
    for (key, value) in d.items():
        if value in d:
            raise ValueError('not one-to-one')
        ans[value] = key
    return ans

def inverse_dict_multivalued(d):
    """
    Convert multi-valued inverse of mapping d (maps key to list of keys), which is also multi-valued.
    
    Example:
    >>> util.inverse_dict_multivalued({1: ['b'], 3: ['b', 'c'], 4: ['c'], 5: ['d']})
    {'c': [3, 4], 'd': [5], 'b': [1, 3]}
    """
    ans = {}
    for (key, valueL) in d.items():
        for value in valueL:
            ans.setdefault(value, set())
            ans[value].add(key)
    return {key: list(value) for (key, value) in ans.items()}
