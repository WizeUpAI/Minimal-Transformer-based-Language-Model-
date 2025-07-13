from setuptools import setup, find_packages

setup(
    name="bert_llm",
    version="0.1.0",
    description="Minimal BERT-like transformer language model from scratch",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0"
    ],
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)