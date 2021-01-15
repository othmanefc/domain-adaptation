from setuptools import setup, find_packages  # type: ignore

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(name='domain-adaptation',
      version='0.1.0',
      description='simplified domain adaptation',
      author='Othmane Hassani',
      author_email='othmane.hassani1@gmail.com',
      license='Apache',
      packages=find_packages('domain-adaptation'),
      python_requires='>=3.7',
      install_requires=requirements)
