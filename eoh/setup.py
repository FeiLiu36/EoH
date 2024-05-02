from setuptools import setup, find_packages

setup(
    name="eoh",
    version="0.1",
    author="MetaAI Group, CityU",
    description="Evolutionary Computation + Large Language Model for automatic algorithm design",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "numba",
        "joblib"
    ],
    test_suite="tests"
)