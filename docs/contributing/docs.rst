Building the documentation
==========================

To install dependencies for building the docs:

.. code-block:: shell-session

   $ pip install -e .[docs]

To build and serve the HTML docs:

.. code-block:: shell-session

   $ cd docs
   $ make livehtml

.. note::

   If you need to specify the hostname/port manually
   (e.g. you are using a remote machine),
   you can run e.g. ``make livehtml O="--host=$(hostname) --port=8000"``.

Regenerating the API docs
-------------------------

Since |apidoc|_ was only added in Sphinx 8.2,
which also dropped support for Python 3.10,
you must manually regenerate the API docs when necessary:

.. code-block:: shell-session

   $ make apidoc

.. |apidoc| replace:: ``sphinx.ext.apidoc``
.. _apidoc: https://www.sphinx-doc.org/en/master/usage/extensions/apidoc.html
