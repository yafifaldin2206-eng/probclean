from setuptools import setup, find_packages

setup(
    name="probclean",
    version="0.1.0",
    description="Lightweight categorical typo correction for tabular data",
    author="probclean contributors",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ]
    },
)
