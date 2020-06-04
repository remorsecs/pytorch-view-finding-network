from setuptools import setup, find_packages

setup(
    name='pytorch-view-finding-network',
    version="0.1",
    author="remorsecs",
    url="https://github.com/remorsecs/pytorch-view-finding-network",
    license="MIT",
    packages=find_packages(exclude=("configs", "tests")),
)