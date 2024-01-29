from setuptools import setup, find_packages

setup(
    name='openwfs',
    version='0.1.0a',
    author='Ivo Vellekoop',
    author_email='i.m.vellekoop@utwente.nl',
    description='A libary for performing wavefront shaping experiments and simulations.',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
