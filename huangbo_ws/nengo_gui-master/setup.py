#!/usr/bin/env python
import imp
import io
import os

try:
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import find_packages, setup


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

root = os.path.dirname(os.path.realpath(__file__))
version_module = imp.load_source(
    'version', os.path.join(root, 'nengo_gui', 'version.py'))

setup(
    name="nengo_gui",
    version=version_module.version,
    author="Applied Brain Research",
    author_email="info@appliedbrainresearch.com",
    packages=find_packages(),
    scripts=[],
    include_package_data=True,
    url="https://github.com/nengo/nengo_gui",
    license="Free for non-commercial use",
    description="Web-based GUI for building and visualizing Nengo models.",
    long_description=read('README.rst', 'CHANGES.rst'),
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'nengo_gui = nengo_gui:old_main',
            'nengo = nengo_gui:main',
        ]
    },
    install_requires=[
        "nengo",
    ],
    tests_require=[
        "pytest",
        "selenium",
        "pyimgur",
    ],
    classifiers=[  # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: Free for non-commercial use',
        'Operating System :: OS Independent',
        'Programming Language :: JavaScript',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)
