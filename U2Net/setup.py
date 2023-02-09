from setuptools import find_packages
from setuptools import setup
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./thesis-372616-5c0364c4da5f.json"
REQUIRED_PACKAGES = [
    # 'gcsfs==0.7.1',
    # 'six==1.15.0'
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(), # Automatically find packages within this directory or below.
    # include_package_data=True, # if packages include any data files, those will be packed together.
    description='Model for saliency detection   '
)