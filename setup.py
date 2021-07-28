import os
import setuptools
from tools import get_requirements, get_readme, get_version

CONFIG = {
    "package_name": "bigmodels",
    "author": "a710128",
    "author_email": "qbjooo@qq.com",
    "description": "A toolkit for big models"
}

def main():
    path = os.path.dirname(os.path.abspath(__file__))

    version = get_version()

    open( os.path.join(path, CONFIG["package_name"], "version.py"), "w", encoding="utf-8" ).write('version = "%s"' % version)

    setuptools.setup(
        name=CONFIG["package_name"],
        version=version,
        author=CONFIG["author"],
        author_email=CONFIG["author_email"],
        description=CONFIG["description"],
        long_description=get_readme(),
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(exclude=("tools",)),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: C++"
        ],
        python_requires=">=3.6",
        setup_requires=["wheel"],
        install_requires=get_requirements()
    )

if __name__ == "__main__":
    main()
