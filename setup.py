from setuptools import setup
import re


def parse_version(name):
    # http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
    VERSIONFILE = name + "/_version.py"
    verstrline = open(VERSIONFILE, "rt").read()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)

    if mo:
        verstr = mo.group(1)
    else:
        raise RuntimeError("Unable to find version string in %s." %
                           (VERSIONFILE,))

    return verstr

name = 'fleuron'

setup(name=name,
      version=parse_version(name),
      description='Printers ornament extractor',
      url='http://bitbucket.com/dgorissen/ornaments',
      author='Dirk Gorissen',
      author_email='dirk@machinedoing.com',
      license='BSD',
      packages=[name],
      install_requires=[],
      zip_safe=False,
      include_package_data=True,
      entry_points={
          'console_scripts': ['%s=%s.extract:main' % (name, name)]
      })
