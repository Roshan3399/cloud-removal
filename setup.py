from setuptools import setup, find_packages

setup(
    name="cloudclear-liss4",
    version="0.1.0",
    description="Generative AI for cloud removal in LISS-IV satellite imagery",
    author="[Your Name]",
    author_email="[your email]",
    url="https://github.com/[your-username]/cloudclear-liss4",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "rasterio>=1.3.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "matplotlib>=3.7.0",
        "scikit-image>=0.21.0",
        "albumentations>=1.3.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
