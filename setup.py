import os
import setuptools
import sys
import struct

CONFIG = {
    "package_name": "bminf",
    "author": "a710128",
    "author_email": "qbjooo@qq.com",
    "description": "A toolkit for big model inference",
    "version": None
}

def get_readme(path):
    ret = ""
    with open(os.path.join(path, "README.md"), encoding="utf-8") as frd:
        ret = frd.read()
    return ret

def get_requirements(path):
    ret = []
    with open(os.path.join(path, "requirements.txt"), encoding="utf-8") as freq:
        for line in freq.readlines():
            ret.append( line.strip() )
    return ret

def get_version(path):
    if "version" in CONFIG and CONFIG["version"] is not None:
        return CONFIG["version"]
    if "BM_VERSION" in os.environ:
        return os.environ["BM_VERSION"]
    if "CI_COMMIT_TAG" in os.environ:
        return os.environ["CI_COMMIT_TAG"]
    if "CI_COMMIT_SHA" in os.environ:
        return os.environ["CI_COMMIT_SHA"]
    version_path = os.path.join( path, CONFIG["package_name"], "version.py" )
    if os.path.exists(version_path):
        tmp = {}
        exec(open(version_path, "r", encoding="utf-8").read(), tmp)
        if "__version__" in tmp:
            return tmp["__version__"]
    return "test"

def main():
    path = os.path.dirname(os.path.abspath(__file__))

    version = get_version(path)
    open( os.path.join(path, CONFIG["package_name"], "version.py"), "w", encoding="utf-8" ).write('__version__ = "%s"' % version)

    requires = get_requirements(path)

    setuptools.setup(
        name=CONFIG["package_name"],
        version=version,
        author=CONFIG["author"],
        author_email=CONFIG["author_email"],
        description=CONFIG["description"],
        long_description=get_readme(path),
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(exclude=("tools",)),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: C++"
        ],
        python_requires=">=3.6",
        setup_requires=["wheel"],
        install_requires= requires,
    )

if __name__ == "__main__":
    main()
