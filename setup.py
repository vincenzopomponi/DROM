from setuptools import setup, find_packages

setup(
    name='DROM',
    version='0.01',
    packages=find_packages(),
    install_requires=[
        "robosuite==1.4.1",
        'numpy==1.26.4',
        'experiment_launcher',
        'einops',
        'jinja2',
        'setuptools',
        'typeguard',
        'matplotlib==3.9',
        'sympy==1.13.3',
        'pandas==2.1.4',
        'diffusers',
        'h5py',
        'scipy',
        'PyQt6',
        'egl_probe',
    ],
    author='Vincenzo Pomponi',
    author_email='vincenzo.pomponi@supsi.ch',
    description='DROM: Diffusion for RObotics Manipulation',
    url="https://github.com/vincenzopomponi/DROM",
    python_requires='>=3.8',
)