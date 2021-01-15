from setuptools import setup  # type: ignore

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(name='domain-adaptation',
      version='0.1.0',
      description='simplified domain adaptation',
      author='Othmane Hassani',
      author_email='othmane.hassani1@gmail.com',
      license='Apache',
      packages=['domain_adaptation'],
      python_requires='>=3.7',
      install_requires=requirements)
