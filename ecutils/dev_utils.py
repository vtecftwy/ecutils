"""
Utility Functions that can be used 

Includes all stable utility functions that do not fit in other context.

Thank you to socrateslee for his gist https://gist.github.com/socrateslee/4011475 for stack_trace
"""
import sys
import functools

__all__ = ['stack_trace']

class StackTrace(object):
    def __init__(self, with_call=True, with_return=False,
                       with_exception=False, max_depth=-1):
        self._frame_dict = {}
        self._options = set()
        self._max_depth = max_depth
        if with_call: self._options.add('call')
        if with_return: self._options.add('return')
        if with_exception: self._options.add('exception')

    def __call__(self, frame, event, arg):
        ret = []
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
            ret.append(frame.f_code.co_name)
            ret.append(f'[{event}]')
            if event == 'return':
                ret.append(arg)
            elif event == 'exception':
                ret.append(repr(arg[0]))
            ret.append(f'in {frame.f_code.co_filename} line:{frame.f_lineno}')
        if ret:
            text = '\t'.join([str(i) for i in ret])
            print(f"{'  ' * depth}{text}")

        return self

def stack_trace(**kw):
    """Function for stack_trace decorator"""
    def entangle(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            st = StackTrace(**kw)
            sys.settrace(st)
            try:
                return func(*args, **kwargs)
            finally:
                sys.settrace(None)
        return wrapper
    return entangle



if __name__ == '__main__':
    pass
