from pathlib import Path
from setuptools import setup, find_packages

VERSION = "0.1.0"
DESCRIPTION = "Wavefront sensing and control algorithms and tools developed in UASAL"
README = Path(__file__).resolve().parent / "README.md"

setup(
    name="lina",
    version=VERSION,
    author="Kevin Derby",
    author_email="<derbyk@arizona.edu>",
    description=DESCRIPTION,
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
        "scipy",
        "astropy",
    ],
    keywords=["python", "lina", "wfsc", "wavefront", "sensing", "control"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
    ],
)
