from setuptools import find_packages, setup
from setuptools.command.build_py import build_py


setup(
    name="libquant",
    version="0.1.0.dev0",
    description="Low-bit Model Quantization for Efficient Training and Inference",
    keywords="deep learning",
    license="Apache",
    author="arthurinruc",
    author_email="arthurinruc@gmail.com",
    url="https://github.com/ArthurinRUC/libquant",
    packages=find_packages(),
    package_data={"": ["*.h", "*.cpp", "*.cu", "*.json"]},
    entry_points={
        "console_scripts": [
            # Add script tools
        ]
    },
    python_requires=">=3.8.0",
    install_requires=[
        # Add requirements here
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    cmdclass={"build_py": build_py},
)
