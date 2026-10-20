"""Load land kernels without CuPy's trailing FTZ override.

Flushing intermediate hydraulic conductivity loses accumulated drainage;
flushing tiny snow changes RUC's branches; flushing humidity loses QSFC.
Use the existing direct-NVRTC image cache, with the actual options recorded.
"""
from __future__ import annotations


class _LoadedModule:
    def __init__(self, module, image, source, options):
        self.module = module
        self.code, self.options = source, options
        # Keep the exact loaded image available to the kernel manifest.
        if isinstance(image, str) or image.startswith(b'//'):
            self.ptx = image
        else:
            self.cubin = image

    def get_function(self, name):
        return _Function(self.module.get_function(name))

    def get_global(self, name):
        import cupy as cp
        pointer = self.module.get_global_var(name)
        memory = cp.cuda.UnownedMemory(pointer, 0, self.module)
        return cp.cuda.MemoryPointer(memory, 0)


class _Function:
    """The launch and attribute interface used by the land runtime."""
    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    @property
    def attributes(self):
        from cupy.cuda import driver
        # CUDA's CUfunction_attribute enum, matching RawKernel.attributes.
        names = ('max_threads_per_block', 'shared_size_bytes',
                 'const_size_bytes', 'local_size_bytes', 'num_regs',
                 'ptx_version', 'binary_version')
        return {name: driver.funcGetAttribute(i, self.function.ptr)
                for i, name in enumerate(names)}


def _effective_options(options):
    options = tuple(options)
    if '--ftz=false' not in options:
        raise ValueError('land kernels require --ftz=false to preserve drainage, snow branches and humidity')
    # Strict controls must enter the persistent cache key, not just the final
    # NVRTC program hook. Otherwise a default image can be reused by strict.
    from gpuwm import wrf_exact
    if wrf_exact.ENABLED:
        options = wrf_exact.effective_options(options)
    return options


def compile_image(source, options):
    """Compile into the direct image cache without loading (a background prebuild).

    Same options, filename and cache entry as :func:`compile_module`, so the
    later foreground ``compile_module`` of the same source reads this image
    from disk instead of compiling it again.
    """
    from gpuwm import nvrtc_ptx_cache as compiler
    from gpuwm.certify.kernel_manifest import record_module, source_sha256
    from types import SimpleNamespace
    options = _effective_options(options)
    image, log = compiler.compile_using_nvrtc(source, options, None, 'land.cu')
    module = SimpleNamespace(**{"ptx" if isinstance(image, str) or image.startswith(b'//') else "cubin": image})
    record_module("land_prebuild/" + source_sha256(source), source=source, options=options, module=module)
    return image, log


def compile_module(source, options, module_key, log_stream=None):
    """Compile at the direct boundary, retaining subnormals in both modes."""
    import cupy as cp
    from gpuwm import nvrtc_ptx_cache as compiler
    from gpuwm.certify.kernel_manifest import record_module
    from gpuwm.kernel_compile_notice import observe_module_compile

    options = _effective_options(options)
    with observe_module_compile(module_key):
        image, _ = compiler.compile_using_nvrtc(
            source, options, None, 'land.cu', log_stream=log_stream)
        loaded = cp.cuda.function.Module()
        loaded.load(image.encode() if isinstance(image, str) else image)
    module = _LoadedModule(loaded, image, source, options)
    record_module(module_key, source=source, options=options, module=module)
    return module
