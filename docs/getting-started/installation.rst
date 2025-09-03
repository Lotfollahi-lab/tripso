Installation
============

Requirements:

- Linux (tripso is not installable on macOS)
- Python 3.10

.. code-block:: shell-session

   $ python3.10 -m venv .venv
   $ source .venv/bin/activate
   $ pip install \
      git+https://huggingface.co/ctheodoris/Geneformer@18a2ca668c0f0239f37e58a34fd8de4ac15b5ed2 \
      -r requirements.txt \
      -e .

.. note::

   This could take several minutes depending on the speed of the filesystem.
   Grab a cup of tea while you wait!
