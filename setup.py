"""Setup script for BSMamba2."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / 'README.md'
if readme_file.exists():
    with open(readme_file, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = 'BSMamba2: Music Source Separation with Mamba2'

# Read requirements
requirements_file = Path(__file__).parent / 'requirements.txt'
with open(requirements_file, 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='bsmamba2',
    version='0.1.0',
    description='BSMamba2: Music Source Separation with Mamba2',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/bsmamba2',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
    ],
    keywords='music source separation, mamba2, state space models, deep learning',
    entry_points={
        'console_scripts': [
            'bsmamba2-train=training.train:main',
            'bsmamba2-separate=inference.separate:main',
        ],
    },
)
