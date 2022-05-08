#!/usr/bin/env python

# usage: python setup.py develop

from os import path

import setuptools

def read(fname):
    return open(path.join(path.dirname(__file__), fname)).read()

setuptools.setup(
    name='swadist',
    version='0.1',
    description='Codistilled Stochastic Weight Averaging',
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    author='James Duncan, Keyan Nasseri',
    author_email='jpduncan@berkeley.edu',
    url='https://github.com/jpdunc23/swadist',
    packages=setuptools.find_packages(exclude=['tests']),
    install_requires=[
        'torch>=1.11',
        'tensorboard>=2.8.0',
    ],
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
