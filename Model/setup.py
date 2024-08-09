from setuptools import setup, find_packages

setup(
    name='GazeCorrectionApp',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[i.strip() for i in open("requirements.txt").readlines()],
    entry_points={
        'console_scripts': [
            'gaze_correction=gaze_corr_sys.core:main',  # Adjust the import path and function
        ],
    },
)
