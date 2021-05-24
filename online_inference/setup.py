"""
Project build:
    python setup.py sdist

Install built project:
    pip install . -U

Project usage example:
    run_service
    run_service 127.0.0.1 7000
"""
from setuptools import setup


setup(
    name='ml_project_inference',
    packages=[
        'src',
        'src.utils'
    ],
    version='0.1.0',
    description='ML project (online inference)',
    author='Lubov Kudrenok',
    entry_points={
        'console_scripts': [
            'run_service = src.app:run_service'
        ]
    },
    install_requires=[
        'click==7.1.2',
        'numpy==1.20.2',
        'pandas==1.2.4',
        'scikit-learn==0.24.2',
        'PyYAML==5.4.1',
        'marshmallow-dataclass==8.4.1',
        'pydantic==1.8.1',
        'fastapi==0.64.0',
        'uvicorn==0.13.4',
    ],
    license='MIT'
)
