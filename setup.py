from setuptools import setup, find_namespace_packages

with open("requirements.txt", "r") as f:
    requirements = [package.replace("\n", "") for package in f.readlines()]

setup(
    name="haiku_hnn",
    url="https://github.com/Raffaelbdl/hyperbolic-nn-haiku",
    author="Raffael Bolla Di Lorenzo",
    author_email="raffaelbdl@gmail.com",
    packages=find_namespace_packages(),
    install_requires=requirements[:],
    version="0.0.1",
    license="MIT",
    description="dm-haiku implementation of hyperbolic neural networks",
)