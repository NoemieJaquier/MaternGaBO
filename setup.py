from setuptools import setup, find_packages

# get description from readme file
with open('README.md', 'r') as f:
    long_description = f.read()

# setup
setup(
    name='GaBO Matern kernels',
    version='',
    description='',
    long_description = long_description,
    long_description_content_type="text/markdown",
    author='Noemie Jaquier, Viacheslav Borovitskiy, Andrei Smolensky, Alexander Terenin, Tamim Asfour, Leonel Rozo ',
    author_email='noemie.jaquier@kit.edu ',
    maintainer='Noemie Jaquier',
    maintainer_email='noemie.jaquier@kit.edu',
    license='MIT license',
    url=' ',
    platforms=['Linux Ubuntu'],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)
