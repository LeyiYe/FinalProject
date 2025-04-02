from setuptools import setup, find_packages

setup(
    name="defgraspsim",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'hydra-core',
        'omegaconf',
        'pysph',
        'numpy'
    ],
)