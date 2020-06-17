'''
@author:  Minye Wu
@contact: wuminye.x@gmail.com
'''

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pcpr',
    version="0.2",
    ext_modules=[
        CUDAExtension('pcpr', [
            'CloudProjection/pcpr_cuda.cpp',
            'CloudProjection/point_render.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })