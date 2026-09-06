#!/usr/bin/env python
"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

# with open("HISTORY.md") as history_file:
#     history = history_file.read()

# NOTE: Update pinned_reqs whenever install_requires or extras_require changes.
install_requires = [
    "numpy>=1.17.0",  # >=1.17.0 that is when default_rng becomes available.
    "scipy>=1.4.0",  # Primarily used in CVTArchive.

]

extras_require = {
    "all": ["matplotlib>=3.0.0",],
    # Dependencies for examples (NOT tutorials -- tutorial notebooks should
    # install deps with cell magic and only depend on ribs and ribs[all]).
    "dev": [
        "pip>=20.3",
        "pylint",
        "yapf",

        # Testing
        "tox==3.14.0",
        "pytest==6.1.2",
        "pytest-cov==2.10.1",
        "pytest-benchmark==3.2.3",
        "pytest-xdist==2.1.0",

        # Documentation
        "Sphinx==3.2.1",
        "sphinx-material==0.0.32",
        "sphinx-autobuild==2020.9.1",
        "sphinx-copybutton==0.3.1",
        "myst-nb==0.10.1",

        # Distribution
        "bump2version==0.5.11",
        "wheel==0.36.2",
        "twine==1.14.0",
        "check-wheel-contents==0.2.0",
    ]
}

setup(
    author="Nathan Dennler",
    author_email="dennler@usc.edu",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    description=
    "A Python library for learning user preferences from various inputs.",
    install_requires=install_requires,
    extras_require=extras_require,
    license="MIT license",
    # long_description=readme + "\n\n" + history,
    # long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="irl",
    name="irlpreference",
    packages=find_packages(include=["irlpreference", "irlpreference.*"]),
    python_requires=">=3.6.0",
    test_suite="tests",
    # url="https://github.com/icaros-usc/pyribs",
    version="0.3.1",
    zip_safe=False,
)