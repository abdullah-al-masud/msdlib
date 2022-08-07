"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

import setuptools


def parse_requirements():
    pkg = open('requirements.txt', 'r').read().split('\n')
    pkg = [s.strip() for s in pkg if s.strip() != '']
    return pkg


def parse_version():
    file = open('msdlib/__init__.py', 'r').read().split('\n')
    _version = [f for f in file if '__version__=' in f.replace(
        ' ', '')][0].split('=')[-1].strip()[1: -1]
    return _version


with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="msdlib",
    version=parse_version(),
    author="Abdullah Al Masud",
    author_email="abdullahalmasud.buet@gmail.com",
    description="msdlib is meant for making life easier of a common data scientist/data analyst/ML enginner.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abdullah-al-masud/msdlib",
    packages=setuptools.find_packages('msdlib'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=parse_requirements()
)
