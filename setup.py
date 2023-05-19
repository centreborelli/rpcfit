from setuptools import setup

setup(
    name="rpcfit",
    version="0.9.9",
    author="Roland Akiki, Roger Marí",
    description="Robust Rational Polynomial Camera Modelling for SAR and Pushbroom Imaging",
    url="https://github.com/centreborelli/rpcfit",
    license="BSD",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "rasterio",
        "rpcm",
    ],
    packages=["rpcfit"],
    classifiers=[
        "License :: OSI Approved :: BSD License",
    ],
)
