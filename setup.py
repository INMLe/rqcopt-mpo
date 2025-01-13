from setuptools import setup

setup(
    name="rqcopt_mpo",
    version="1.0.0",
    author="Isabel Nha Minh Le",
    author_email="isabel.le@tum.de",
    packages=["rqcopt_mpo"],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "psutil",
        "PyYAML",
        "jax",
        "jaxlib",
    ],
)
