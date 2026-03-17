from setuptools import setup, find_packages

setup(
    name="affseg",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'open3d',
        'torch',
        'tqdm',
    ],
) 