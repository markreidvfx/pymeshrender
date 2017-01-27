from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import os

os.environ['ARCHFLAGS'] ="-arch x86_64"
#os.environ['CFLAGS'] = "-Og"

sourcefiles = [
"meshrender/core.c",
"meshrender/meshrender.pyx",
]
extensions = [Extension("meshrender",
                        sourcefiles,
                        language="c",
                        extra_compile_args=["-O3"],
                        # extra_link_args=["-g"],

)]

setup(
    name='PyMeshRender',
    version='0.1.0',
    description='Rasterizes 3D meshe',
    author="Mark Reid",
    author_email="mindmark@gmail.com",
    url="https://github.com/markreidvfx/pymeshsmooth",
    license='MIT',
    ext_modules = cythonize(extensions),
)
