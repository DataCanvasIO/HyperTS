# -*- coding:utf-8 -*-

from __future__ import absolute_import

from setuptools import find_packages
from setuptools import setup
import os
from os import path as P


try:
    import tensorflow

    tf_installed = True
except ImportError:
    tf_installed = False


def read_requirements(file_path='requirements.txt'):
    if not os.path.exists(file_path):
        return []

    with open(file_path, 'r')as f:
        lines = f.readlines()

    lines = [x.strip('\n').strip(' ') for x in lines]
    lines = list(filter(lambda x: len(x) > 0 and not x.startswith('#'), lines))

    return lines


try:
    execfile
except NameError:
    def execfile(fname, globs, locs=None):
        locs = locs or globs
        exec(compile(open(fname).read(), fname, "exec"), globs, locs)

HERE = P.dirname((P.abspath(__file__)))

version_ns = {}
execfile(P.join(HERE, 'hyperts', '_version.py'), version_ns)
version = version_ns['__version__']

print("__version__=" + version)

MIN_PYTHON_VERSION = '>=3.6.*'

long_description = open('README.md', encoding='utf-8').read()

requires = read_requirements()
if not tf_installed:
    requirements = ['tensorflow>=2.0.0,<2.5.0', ] + requires

setup(
    name='hyperts',
    version=version,
    description='',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='',
    author='DataCanvas Community',
    author_email='yangjian@zetyun.com',
    license='Apache License 2.0',
    install_requires=requires,
    python_requires=MIN_PYTHON_VERSION,
    classifiers=[
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(exclude=('docs', 'tests')),
    package_data={
        'hyperts': ['examples/*', 'examples/**/*', 'examples/**/**/*', 'datasets/*.pkl', 'datasets/*.csv'],
    },
    zip_safe=False,
    include_package_data=True,
)
