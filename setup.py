import setuptools
import re

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open("ts_type/version.py", "r", encoding="utf-8") as fh:
    version = re.match("VERSION = '(.*)'", fh.read()).groups()[0]

setuptools.setup(
    name="ts_type",
    version=version,
    author="Ryosuke Sasaki",
    author_email="saryou.ssk@gmail.com",
    description="ts_type generates typescript's type from python code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saryou/ts_type",
    project_urls={
        "Bug Tracker": "https://github.com/saryou/ts_type/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["ts_type"],
    package_dir={"ts_type": "ts_type"},
    python_requires=">=3.9",
)
