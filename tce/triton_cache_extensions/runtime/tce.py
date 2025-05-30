#!/usr/bin/env python3

'''Enhancements to Triton to support compressed and signed caches.'''

# Copyright (c) 2025 RedHat Inc
# Copyright (c) 2025 Cambridge Greys Ltd
#
# Licensed under Apache License 2.0
#
from typing import Callable, Iterable, Optional, TypeVar, Union, overload
#from triton import KernelInterface
from triton import JITFunction

T = TypeVar("T")

class TCEJITFunction(JITFunction[T]):

    '''Wrapper around Triton JIT to support cache enhancements'''

    # these need to become Triton version dependent

    init_filter = []
    run_filter = []

    def __init__(self, func, **kwargs):
        for to_remove in self.init_filter:
            del kwargs[to_remove]
        super().__init__(func, **kwargs)
        if kwargs.get("debug") is not None:
            self.debug = kwargs["debug"]

    def run(self, *args, **kwargs):
        '''Replacement run method separating caching and jit compilation'''
        # note - we can filter only kwargs
        for to_remove in self.run_filter:
            del kwargs[to_remove]

        # Load from signed/secure cache
        # Apply any additional optimizations
        # Fall through to triton

        return super().run(*args, **kwargs)


# -----------------------------------------------------------------------------
# `jit` decorator
# -----------------------------------------------------------------------------


@overload
def jit(func: T) -> TCEJITFunction[T]:
    ...

# same as in the __init__ - repr has been redefined by Triton for no apparent reason
# pylint: disable=redefined-builtin
@overload
def jit(
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
#    this one is version dependent - needs version specific filtering
#    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Callable[[T], TCEJITFunction[T]]:
    ...


def jit(
    func: Optional[T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
#    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Union[TCEJITFunction[T], Callable[[T], TCEJITFunction[T]]]:
    """
    Decorator for JIT-compiling a function using the Triton compiler.

    :note: When a jit'd function is called, arguments are
        implicitly converted to pointers if they have a :code:`.data_ptr()` method
        and a `.dtype` attribute.

    :note: This function will be compiled and run on the GPU. It will only have access to:

           * python primitives,
           * builtins within the triton package,
           * arguments to this function,
           * other jit'd functions

    :param func: the function to be jit-compiled
    :type func: Callable
    """

    def decorator(func: T) -> TCEJITFunction[T]:
        assert callable(func)
        return TCEJITFunction(
                func,
                version=version,
                do_not_specialize=do_not_specialize,
#                do_not_specialize_on_alignment=do_not_specialize_on_alignment,
                debug=debug,
                noinline=noinline,
                repr=repr,
                launch_metadata=launch_metadata,
            )

    if func is not None:
        return decorator(func)

    return decorator
