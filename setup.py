from setuptools import setup
from setuptools import find_packages


setup(name='Eyesight',
      version='0.1.0',
      description='Computer Vision at the Edge',
      author='Chan Kha Vu',
      url='https://github.com/hav4ik/eyesight',
      install_requires=['colorlog>=4.1.0',
                        'pyyaml',
                        'readerwriterlock>=1.0.6',
                        'wget>=3.2',
                        'numpy>=1.17.0',
                        'imutils>=0.5.3',
                        'Flask>=1.1.1',
                        'scikit-image>=0.17.0'],
      packages=find_packages())
