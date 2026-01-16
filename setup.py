"""A setuptools-based script for installing the package."""

from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='tripso',
    packages=find_packages(),
    version='0.1.0',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    extras_require={
        'docs': [
            'sphinx~=8.1.3',  # last version to support Python 3.10
            'sphinx-autobuild~=2024.10.3',  # last version to suppport Python 3.10
            'sphinx-rtd-theme~=3.0.2',
            'myst-nb~=1.3.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'tripso = tripso.__main__:main',
        ],
    },
)
