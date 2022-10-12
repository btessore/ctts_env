from setuptools import setup
from ctts_env import __version__


setup(name='ctts_env',
      description='Library for generating analytical models of the environment of CTTs',
		version=__version__,
      url='https://github.com/btessore/ctts_env',
      author='Benjamin Tessore',
      license='MIT',
      packages=['ctts_env'],
      zip_safe=False)
