#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="puddle_world",
    version="0.0.1",
    install_requires=[
        "gym >= 0.2.3",
        "numpy",
        "scipy",
        "matplotlib",
        "explicit_env @ git+https://github.com/aaronsnoswell/explicit-env.git",
    ],
    packages=find_packages(),
)
