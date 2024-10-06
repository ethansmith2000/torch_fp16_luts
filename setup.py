# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='exp_lut_cuda',
    ext_modules=[
        CUDAExtension(
            name='exp_lut_cuda',
            sources=['exp_lut.cu'],
            extra_compile_args={
                'cxx': [],
                'nvcc': ['-O2']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
