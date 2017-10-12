from setuptools import setup
from setuptools import find_packages


setup(name='action-provider-rl',
      version='0.1.0',
      description='Deep Reinforcement Learning using Action Providers',
      author='André Reichstaller',
      author_email='reichstaller@isse.de',
      install_requires=[
	  'keras>=1.0.7',
	  'keras_rl>=0.3.0'
	  ],
      extras_require={
          'gym': ['gym'],
      })
