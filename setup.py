from setuptools import setup, find_packages
import os

setup(
    name="sgs",
    version="0.1.0",
    packages=find_packages(),
    description="Scaling Self-Play with Self-Guidance",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
)
