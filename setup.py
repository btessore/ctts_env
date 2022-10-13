from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os

from ctts_env import __version__


setup(
    name="ctts_env",
    description="Library for generating analytical models of the environment of CTTs",
    version=__version__,
    url="https://github.com/btessore/ctts_env",
    author="Benjamin Tessore",
    license="MIT",
    packages=["ctts_env"],
    zip_safe=False,
)

"""
TO DO: compile and import properluy external fortran modules in ctts_env/src
"""

# FC = "intelem"


# class f2py_Extension(Extension):
#     def __init__(self, name, sourcedirs):
#         Extension.__init__(self, name, sources=[])
#         self.sourcedirs = [os.path.abspath(sourcedir) for sourcedir in sourcedirs]
#         self.dirs = sourcedirs


# class f2py_Build(build_ext):
#     def run(self):
#         for ext in self.extensions:
#             self.build_extension(ext)

#     def build_extension(self, ext):
#         # compile
#         for ind, to_compile in enumerate(ext.sourcedirs):
#             module_loc = os.path.split(ext.dirs[ind])[0]
#             module_name = os.path.split(to_compile)[1].split(".")[0]
#             os.system(
#                 "cd %s;f2py3 -c %s -m %s --fcompiler=%s"
#                 % (module_loc, to_compile, module_name, FC)
#             )


# setup(
#     name="ctts_env",
#     description="Library for generating analytical models of the environment of CTTs",
#     version=__version__,
#     url="https://github.com/btessore/ctts_env",
#     author="Benjamin Tessore",
#     license="MIT",
#     packages=["ctts_env", "ctts_env/src"],
#     zip_safe=False,
#     ext_modules=[
#         f2py_Extension(
#             "fortan_modules", ["ctts_env/src/_mag_mod.f90", "ctts_env/src/_io_mod.f90"]
#         )
#     ],
#     cmdclass=dict(build_ext=f2py_Build),
# )
