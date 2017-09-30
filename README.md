Fleuron: Ornament Extractor
===========================

Given a directory of TIF images from scanned books Fleuron will extract the
printers ornaments.

Prerequisites
------------

* python 2.7
* pip
* opencv 2.4.x

Installation
------------

Note: the use of a virtualenv is recommended

Clone the repository, navigate to the directory and run:

```python
$> pip install -r requirements.txt
$> python setup.py sdist
$> pip install dist/fleuron-x.x.x.tar.gz
```

Alternatively if you want to develop/change the code run

`pip install -e .`

If you want to install it just for your local user (not system wide) add the
--user flag.

To uninstall simply run:

`pip uninstall fleuron`

Usage
-----

A script called *fleuron* will be installed and should be available on your path.
If it is not on your path you just need to add the python script directory to
your $PATH environment variable.

Run `fleuron -h` for full usage information but invocation should be as simple
as:

`$>fleuron /path/to/directory/of/TIFF/files`

Note any extracted ornaments will be created and saved in the same directory!

Caveats
-------

While Fleuron will try to catch all ornaments the performance will depend
strongly on the quality and format of the input images. The approach taken is a
morphological one which means that (degraded) ornaments that resemble the
morphology of text will tend to be missed.

An extension could be to use an OCR engine or rely upon the extracted text to
disambiguate further. Note though that no approach will give perfect accuracy
in all cases.

Useful projects worth exploring for further extension include:

* [gamera](http://gamera.informatik.hsnr.de/)
* [ocropy](https://github.com/tmbdev/ocropy)
* [tesseract](https://github.com/gregjurman/tesserwrap)


License
-------

BSD. See LICENSE.md

Contact
-------

Hazel Wilkinson <hw442@cam.ac.uk>
