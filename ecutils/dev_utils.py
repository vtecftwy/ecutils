# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs-dev/dev_utils.ipynb.

# %% ../nbs-dev/dev_utils.ipynb 2
from __future__ import annotations

import re
import sys
import functools

# %% auto 0
__all__ = ['StackTrace', 'StackTraceJupyter', 'stack_trace', 'stack_trace_jupyter']

# %% ../nbs-dev/dev_utils.ipynb 3
__all__ = ['stack_trace', 'stack_trace_jupyter']

# %% ../nbs-dev/dev_utils.ipynb 4
class StackTrace():
    """Capture and prints information on all stack frame executed"""
    def __init__(self, 
                 with_call:bool=True,      
                 with_return:bool=True, 
                 with_exception:bool=True, 
                 max_depth:int=-1
                ):
        self._frame_dict = {}
        self._options = set()
        self._max_depth = max_depth
        if with_call: self._options.add('call')
        if with_return: self._options.add('return')
        if with_exception: self._options.add('exception')

    def __call__(self, 
                 frame, 
                 event, 
                 arg
                ):
        ret = []
        co_name = frame.f_code.co_name
        co_filename = frame.f_code.co_filename
        co_lineno = frame.f_lineno
        if event == 'call':
            back_frame = frame.f_back
            if back_frame in self._frame_dict:
                self._frame_dict[frame] = self._frame_dict[back_frame] + 1
            else:
                self._frame_dict[frame] = 0

        depth = self._frame_dict[frame]

        if event in self._options and (self._max_depth<0 or depth <= self._max_depth):
            ret.append(co_name)
            ret.append(f'[{event}]')
            if event == 'return':
                ret.append(arg)
            elif event == 'exception':
                ret.append(repr(arg[0]))
            ret.append(f'in {co_filename} line:{co_lineno}')
        if ret:
            self.print_stack_info(co_filename, ret, depth)
        return self

    def print_stack_info(self, 
                         co_filename, 
                         ret, 
                         depth
                        ):
        """This methods can be overloaded to customize what is printed out"""
        text = '\t'.join([str(i) for i in ret])
        print(f"{'  ' * depth}{text}")

# %% ../nbs-dev/dev_utils.ipynb 5
class StackTraceJupyter(StackTrace):
    """Prints stack frame information in Jupyter notebook context (filters out jupyter overhead)"""

    def print_stack_info(self, 
                         co_filename, 
                         ret, 
                         depth
                        ):
        """Overload the base class to filter out those calls to Jupyter overhead functions"""

        EXCL_LIBS = ['encodings.*', 'ntpath.*', 'threading.*', 'weakref.*']
        EXCL_SITE_PACKAGES = ['colorama', 'ipykernel', 'zmq']

        PATH_TO_LIBS_RE = r'^[a-zA-Z]:\\([^<>:\"/\\|?\*]*)\\envs\\([^<>:\"/\\|?\*]*)\\lib'
        LIBS = f"{'|'.join(EXCL_LIBS)}"
        SITE_PACKAGES = f"{'|'.join(EXCL_SITE_PACKAGES)}"
        MODULE_FILTERS_RE = rf"{PATH_TO_LIBS_RE}\\(({LIBS})|(site-packages\\({SITE_PACKAGES}))\\.*)"

        pat = re.compile(MODULE_FILTERS_RE)
        match = pat.match(co_filename)
        
        if match is None:
            """Only print stack frame info for those objects where there is no match"""
            text = '\t'.join([str(i) for i in ret])
            print(f"{'  ' * depth}{text}")

# %% ../nbs-dev/dev_utils.ipynb 6
def stack_trace(**kw):
    """Function for stack_trace decorator"""
    def entangle(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # st = StackTrace(**kw)
            st = StackTrace(**kw)
            sys.settrace(st)
            try:
                return func(*args, **kwargs)
            finally:
                sys.settrace(None)
        return wrapper
    return entangle

# %% ../nbs-dev/dev_utils.ipynb 7
def stack_trace_jupyter(**kw):
    """Function for stack_trace_jupyter decorator"""
    def entangle(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # st = StackTrace(**kw)
            st = StackTraceJupyter(**kw)
            sys.settrace(st)
            try:
                return func(*args, **kwargs)
            finally:
                sys.settrace(None)
        return wrapper
    return entangle
