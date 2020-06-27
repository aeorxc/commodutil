import setuptools
import os

if os.environ.get('CI_COMMIT_TAG'):
    version = os.environ['CI_COMMIT_TAG']
else:
    version = os.environ['CI_JOB_ID']

setuptools.setup(
    name="commodutil",
    version=version,
    author="aeorxc",
    author_email="author@example.com",
    description="common commodity/oil analytics utils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)

