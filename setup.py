from distutils.core import setup
from setuptools import find_packages

setup(
    name='PACE',
    version='1.0.0b4',
    packages=find_packages(),
    url='https://github.com/Dispersive-Hydrodynamics-Lab/PACE',
    license='LGPL3',
    author='William Farmer',
    author_email='willzfarmer@gmail.com',
    description='Data Quality of Experimental Data',
    long_description=('A tool for the Dispersive Hydrodynamics Lab at the University of Colorado, Boulder '
                      'to use to determine data quality using conduit straightness algorithms and '
                      'image noise values.'),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3.5'
    ],
    keywords='dispersive hydrodynamics data quality straightness',
    install_requires=[
        'typing',
        'enforce',
        'tqdm',
        'numpy',
        'matplotlib',
        'scipy',
        'scikit-learn'
    ],
    entry_points={
        'console_scripts': [
            'PACE = PACE:main'
        ]
    }

)
