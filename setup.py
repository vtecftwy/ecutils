from setuptools import setup

 setup(
   name='ec-utils',
   version='0.0.1',
   author='EC',
   author_email='bitbucker@procurasia.com',
   packages=['ec-utils'],
   scripts=[],
   url='',
   license='LICENSE.txt',
   description='Set of utility functions used in several contexts',
   long_description=open('README.md').read(),
   install_requires=[
     backtrader
   ],
)