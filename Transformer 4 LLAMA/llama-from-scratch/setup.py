
from setuptools import setup, find_packages

setup(
    name="llama_from_scratch",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "tokenizers",
    ],
    author="Ton Nom",
    description="LLaMA-like model from scratch in PyTorch",
)
