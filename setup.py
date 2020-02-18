from setuptools import setup

# https://pythonhosted.org/an_example_pypi_project/setuptools.html

setup(
 name='ecutils',
 version='0.0.1',
 author='EC',
 author_email='bitbucker@procurasia.com',
 packages=['ecutils'],
 scripts=[],
 url='',
 license='LICENSE.txt',
 description='Set of utility functions used in several contexts',
 long_description=open('README.md').read(),
 classifiers=[
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Operating System :: Microsoft :: Windows :: Windows 10",
    "Programming Language :: Python :: 3.6",
    "Topic :: Utilities",
],
 install_requires=[
   "bokeh",
   "IPython",
   "matplotlib",
   "numpy",
   "pandas",
   "scipy"
 ],
)
