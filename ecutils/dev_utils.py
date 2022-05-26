"""
Utility Functions that can be used in development phase

Includes all stable utility functions that do not fit in other context.

Thank you to socrateslee for his gist https://gist.github.com/socrateslee/4011475 for stack_trace
"""
import re
import sys
import functools

__all__ = ['stack_trace', 'stack_trace_jupyter']


class StackTrace():
    """Capture and prints information on all stack frame executed"""
    def __init__(self, with_call=True, with_return=True, with_exception=True, max_depth=-1):
        self._frame_dict = {}
        self._options = set()
        self._max_depth = max_depth
        if with_call: self._options.add('call')
        if with_return: self._options.add('return')
        if with_exception: self._options.add('exception')

    def __call__(self, frame, event, arg):
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

        if event in self._options\
          and (self._max_depth<0\
               or depth <= self._max_depth):
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

    def print_stack_info(self, co_filename, ret, depth):
        """This methods can be overloaded to customize what is printed out"""
        text = '\t'.join([str(i) for i in ret])
        print(f"{'  ' * depth}{text}")


class StackTraceJupyter(StackTrace):
    """Prints stack frame information in Jupyter notebook context (filters out jupyter overhead)"""



    def print_stack_info(self, co_filename, ret, depth):
        """Overload the base class to filter out those calls to Jupyter overhead functions"""

        EXCL_LIBS = [
            'encodings.*', 'ntpath.*', 'threading.*', 'weakref.*'
        ]
        EXCL_SITE_PACKAGES = [
            'colorama', 'ipykernel', 'zmq'
        ]

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

def stack_trace_jupyter(**kw):
    """Function for stack_trace decorator"""
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



if __name__ == '__main__':
    from datetime import datetime
    from dateutil import tz
    from finutils.data_source_API_classes import MT4ServerTime

    import pandas as pd


    @stack_trace(with_return=True, with_exception=True, max_depth=3)
    def test():
        print('=== NY ==================================================================')
        NY = tz.gettz('America/New_York')
        print('=== MT4 ==================================================================')
        MT4 = MT4ServerTime()

    @stack_trace_jupyter(with_return=True, with_exception=True, max_depth=3)
    def test_nb():
        print('=== NY ==================================================================')
        NY = tz.gettz('America/New_York')
        print('=== MT4 ==================================================================')
        MT4 = MT4ServerTime()


    print(f"{'*'*100}")
    test()
    print(f"{'*'*100}")
    test_nb()
