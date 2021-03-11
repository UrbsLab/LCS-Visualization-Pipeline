from setuptools import setup

with open("README.md","r") as fh:
  ld = fh.read()

setup(
  name = 'LCS-DIVE',
  packages = ['LCSDIVE'],
  version = '1.0',
  license='License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
  description = 'Learning Classifier System Discovery and Visualization Environment',
  long_description_content_type="text/markdown",
  author = 'Robert Zhang, Ryan J. Urbanowicz',
  author_email = 'robertzh@seas.upenn.edu,ryanurb@upenn.edu',
  url = 'https://github.com/UrbsLab/LCS-Visualization-Pipeline',
  keywords = ['machine learning','data analysis','data science','learning classifier systems','dive'],
  install_requires=['numpy','pandas','scikit-learn','seaborn','skrebate','scikit-ExSTraCS','networkx','matplotlib','scipy','sklearn','pygame','fastcluster'],
  classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Information Technology',
    'Intended Audience :: Science/Research',
    'Topic :: Utilities',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7'
  ],
  long_description=ld
)