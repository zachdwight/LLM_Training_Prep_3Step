#!/usr/bin/env python
"""Setup for llm-training-prep package."""

from setuptools import setup, find_packages

setup(
    name="llm-training-prep",
    version="1.0.0",
    description="Convert PDFs into fine-tuning datasets for LLMs with human-in-the-loop review",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "unstructured[pdf]",
        "unstructured-pytesseract",
        "transformers>=4.36.0",
        "torch",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "llm-training-prep-step1=scripts.step1:main",
            "llm-training-prep-step2=scripts.step2:main",
            "llm-training-prep-step3=scripts.step3:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
