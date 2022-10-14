from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = [line for line in f.read().splitlines() if not line.startswith("git")]

setup(
    name="nlp-uncertainty-zoo",
    version="0.9.0",
    author="Dennis Ulmer",
    description="PyTorch Implementation of Models used for Uncertainty Estimation in Natural Language Processing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kaleidophon/nlp-uncertainty-zoo",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    license="GPL",
    python_requires=">=3.5.3",
    keywords=[
        "machine learning",
        "deep learning",
        "nlp",
        "uncertainty",
        "uncertainty estimation" "pytorch",
    ],
    packages=find_packages(exclude=["docs", "dist"]),
    install_requires=required,
)
