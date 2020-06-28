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
    url="https://github.com/aeorxc/commodutil",
    project_urls={
        'Source': 'https://github.com/aeorxc/commodutil',
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['pandas', 'cufflinks'],
    python_requires='>=3.6',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)

