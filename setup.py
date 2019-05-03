from setuptools import setup, find_packages

setup(
    name='viewfinder_benchmark',
    version="0.1",
    author="yiling-chen,remorsecs",
    url="https://github.com/yiling-chen/faster-view-finding-network",
    license="MIT",
    packages=find_packages(exclude=("configs", "tests")),
)
