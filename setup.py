from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Pybind11Extension(
        "kalman_filter",
        ["kalman_filter.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "/usr/include/eigen3",
        ],
        extra_compile_args=['-std=c++14'],
    ),
]

setup(
    name="kalman_filter",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=["pybind11>=2.5.0"],
)
