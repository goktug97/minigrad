#!/usr/bin/env python

import os
from setuptools import setup
from setuptools.extension import Extension

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='kiwigrad',
      version='0.28',
      description='Autograd Engine written in Python C-API',
      author='Göktuğ Karakaşlı, Marco Salvalaggio',
      author_email='karakasligk@gmail.com, mar.salvalaggio@gmail.com',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/marcosalvalaggio/kiwigrad',
      packages = ['kiwigrad'],
      ext_modules = [Extension('kiwigrad.engine', sources = ['kiwigrad/engine.c'])],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      python_requires='>=3.6',
      install_requires=[
          'graphviz',
      ]
)