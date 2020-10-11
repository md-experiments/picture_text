import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = "0.1.0"
PACKAGE_NAME = "picture_text"
AUTHOR = "Mihail Dungarov"
AUTHOR_EMAIL = "deeplearnmd@gmail.com"
URL = "https://github.com/md-experiments/picture_text"

LICENSE = "Apache License 2.0"
DESCRIPTION = "Interactive tree-maps for text corpora with SBERT & Hierarchical Clustering (HAC)"
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
        "numpy >= 1.14.0",
        "pandas >= 0.20.0",
        "fastcluster >= 1.1.26",
        "sentence-transformers >= 0.3.4",
        "transformers >= 3.0.2",
        "scipy >= 1.1.0",
        "matplotlib >= 3.0.0",
        "plotly >= 4.10.0"
    ]
CLASSIFIERS = [
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
KEYWORDS = [
        "hierarchical agglomerative clustering", "hac", "treemap", "interactive visualization",
        "data visualization", "deep learning", "machine learning", "transformers",
        "sentence-transformers", "BERT", "SBERT",
        "nlp", "natural language processing", "text", "ai", "ml"
    ]

setup(name=PACKAGE_NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type=LONG_DESC_TYPE,
        author=AUTHOR,
        license=LICENSE,
        author_email=AUTHOR_EMAIL,
        url=URL,
        install_requires=INSTALL_REQUIRES,
        packages=find_packages(),
        classifiers=CLASSIFIERS,
        keywords=KEYWORDS
        )