Installation
============

Requirements:

- Linux (tripso is not installable on macOS)
- Python 3.10

We recommend installing tripso in a virtual environment.
We recommend first installing PyTorch, then Tripso.

.. code-block:: shell-session

   $ python3.10 -m venv .venv
   $ source .venv/bin/activate
   $ pip install torch==2.4.1 torchmetrics==1.7.1
   $ git clone https://github.com/Lotfollahi-lab/tripso.git
   $ cd tripso
   $ pip install -r requirements.txt
   $ pip install .

.. note::

   This could take several minutes depending on the speed of the filesystem.
   Grab a cup of tea while you wait!
