# -*- coding: utf-8 -*-

"""
    Enjoy machine learning!

    :copyright: (c) 2015 by Chi-En Wu <jason2506@somewhere.com.tw>.
    :license: BSD.
"""

import uuid

from pip.req import parse_requirements
from setuptools import setup, find_packages

import mhml

def requirements(path):
    return [str(r.req) for r in parse_requirements(path, session=uuid.uuid1())]


setup(
    name='mhml',
    version=mhml.__version__,
    author=mhml.__author__,
    author_email=mhml.__email__,
    url='http://bridgewell.com',
    description='ML rocks',
    long_description=__doc__,
    packages=find_packages(),
    setup_requires=['numpy','scipy'],
    install_requires=requirements('requirements.txt'),
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Bridgewellers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7'
    ]
)
