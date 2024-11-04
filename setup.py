from setuptools import setup, find_packages

setup(
    name="netam",
    version="0.1.0",
    url="https://github.com/matsengrp/netam.git",
    author="Matsen Group",
    author_email="ematsen@gmail.com",
    description="Neural network models for antibody affinity maturation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "biopython",
        "natsort",
        "optuna",
        "pandas",
        "pyyaml",
        "tensorboardX",
        "torch",
        "tqdm",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
    ],
    entry_points={
        "console_scripts": ["netam=netam.cli:main"],
    },
)
