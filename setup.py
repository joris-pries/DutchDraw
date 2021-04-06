import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "DutchDraw", # Replace with your own username
    version = "0.0.2",
    author = "Etienne van de Bijl, Jan Klein, Joris Pries",
    author_email = "joris.pries@cwi.nl",
    description = "Determine (optimal) baselines for binary classification",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/joris-pries/DutchDraw",
    packages = setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)