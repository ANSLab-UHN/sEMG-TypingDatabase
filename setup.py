from setuptools import setup, find_packages
from pathlib import Path

# Read dependencies from requirements.txt
def parse_requirements(filename):
    return Path(filename).read_text().splitlines()

setup(
    name="keypressemg",
    version="0.1.1",
    description="A package for managing the sEMG Typing Database",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="ANSLab-UHN",
    url="https://github.com/ANSLab-UHN/sEMG-TypingDatabase",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9"
)
