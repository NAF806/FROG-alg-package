"""Setup script for the FROG package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="frog",
    version="0.1.0",
    author="Nihal FAIZ",
    author_email="nihalfaiz21@gmail.com",
    description="FROG (Frequency-Resolved Optical Gating) algorithm implementations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/frog",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "scipy>=1.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "frog-pcgp=frog.examples.pcgp_example:main",
            "frog-vanilla=frog.examples.vanilla_example:main",
        ],
    },
)