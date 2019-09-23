import sys

from setuptools import setup, find_packages
from os.path import dirname, join
from os import environ

# this smells, but I don't know how to do better than this right now.
VERSION=environ.get("VERSION") or "0.99.dev0"

install_requires = [
    'mlrose>=1.2.0',
]

if sys.version_info[0] < 3 or sys.version_info[1] < 7:
    raise ValueError("Requires Python >= 3.7")

setup(
    name='cs7641p2',
   packages=find_packages(),
    install_requires=install_requires,
    entry_points='''
        [console_scripts]
    ''',
    license="Apache-2.0",
    include_package_data=True,
    zip_safe=False
)

