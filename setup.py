"""
Setup: Main Project Setup
-----------------------------------
"""

import os
from setuptools import find_packages, setup

install_requires = []
requirements_path = 'requirements.txt'
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

setup(
    name='PantanalFireDetection',
    version='1.0',
    packages=find_packages(where='src', exclude=('test',)),
    package_dir={'': 'src'},
    test_suite='test',
    license='GNU GENERAL PUBLIC LICENSE',
    author='Bruna Zamith Santos',
    description='PantanalFireDetectionService',
    python_requires='==3.9.6',
    install_requires=install_requires
)
