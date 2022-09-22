import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


from setuptools import find_packages

setuptools.setup(
    name="icarogw", # Replace with your own username
    version="1.0.5",
    author="Simone Mastrogiovanni",
    author_email="mastrogiovanni.simo@gmail.com",
    description="A package for gravitational waves population inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="gihub hutl",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
