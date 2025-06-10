#!/usr/bin/env python3

'''Enhancements to Triton to support compressed and signed caches.'''

# Copyright (c) 2025 RedHat Inc
# Copyright (c) 2025 Cambridge Greys Ltd
#
# Licensed under Apache License 2.0
#
import hashlib
import dataclasses
from typing import Callable, Iterable, Optional, TypeVar, Union, overload

from triton import __version__ as triton_version
from triton import JITFunction
from triton.runtime import driver
from triton.runtime.cache import _base32
# pylint: disable=no-name-in-module
from triton._C.libtriton import get_cache_invalidating_env_vars
from triton.compiler.compiler import triton_key
from triton._utils import find_paths_if, get_iterable_path

major, minor, subversion = triton_version.split(".")

if int(major) != 3 or int(minor) < 3:
    raise ImportError("Cannot import a compatible Triton version")


T = TypeVar("T")

def form_key(key, src, backend, options, invalidators):
    '''Format the key and return its hash'''

    return hashlib.sha256(
        f"{key}-{src}-{backend}-{options}-{invalidators}".encode("utf-8")).hexdigest()


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


    # separation of concerns:
    # unentangle load() and compile()

    def disk_hash_key(self, *args, **kwargs):
        '''Compute hash key using the same algorythm as Triton'''

        # parse options
        _, _, backend, binder = self.device_caches[driver.active.get_current_device()]

        # specialization is list[tuple[str, Any]], where first element of tuple is
        # the type and the second parameter is the 'specialization' value.

        bound_args, specialization, _ = binder(*args, **kwargs)

        # options
        options = backend.parse_options(kwargs)

        # signature
        sigkeys = [x.name for x in self.params]
        sigvals = [x[0] for x in specialization]

        for k in kwargs:
            if k not in options.__dict__ and k not in sigkeys:
                raise KeyError(f"Keyword argument {k} was specified but unrecognised")
        # constexprs
        constexprs = find_paths_if(sigvals, lambda _, val: val == "constexpr")
        constexprs = {path: get_iterable_path(
                            list(bound_args.values()), path) for path in constexprs}
        # attributes
        attrvals = [x[1] for x in specialization]
        attrs = find_paths_if(attrvals, lambda _, x: isinstance(x, str))
        attrs = {k: backend.parse_attr(get_iterable_path(attrvals, k)) for k in attrs}

        src = self.ASTSource(
            self,
            dict(zip(sigkeys, sigvals)), # signature
            constexprs,
            attrs)

        if dataclasses.is_dataclass(options):
            options = dataclasses.asdict(options)

        options = backend.parse_options(dict(options or {}, **(src.parse_options())))

        # create cache manager
        return form_key(triton_key(),
                        src.hash(),
                        backend.hash(),
                        options.hash(),
                        str(sorted(get_cache_invalidating_env_vars().items())))

    def run(self, *args, grid, warmup, **kwargs):
        '''Replacement run method separating caching and jit compilation'''
        # note - we can filter only kwargs
        for to_remove in self.run_filter:
            del kwargs[to_remove]

        if kwargs.get("debug", None) is not None:
            print ("metadata key is:", self.disk_hash_key(*args, **kwargs))
            print ("disk dir name is:", _base32(self.disk_hash_key(*args, **kwargs)))

        kwargs["grid"] = grid
        kwargs["warmup"] = warmup

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
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
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
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
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
                do_not_specialize_on_alignment=do_not_specialize_on_alignment,
                debug=debug,
                noinline=noinline,
                repr=repr,
                launch_metadata=launch_metadata,
            )

    if func is not None:
        return decorator(func)

    return decorator
