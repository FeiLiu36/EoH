from setuptools import setup, find_packages

setup(
    name="eoh",
    version="0.2",
    author="Fei Liu",
    description="Evolutionary Computation + Large Language Model for automatic algorithm design",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "joblib",
        "requests"
    ],
    test_suite="tests"
)