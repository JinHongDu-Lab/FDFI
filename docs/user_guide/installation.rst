Installation
============

FDFI can be installed from source. We recommend using a conda environment
for managing dependencies.

Using Conda (Recommended)
-------------------------

Create and activate a conda environment:

.. code-block:: bash

   conda create -n fdfi python=3.10
   conda activate fdfi

Then install FDFI:

.. code-block:: bash

   git clone https://github.com/jinhongdu-lab/FDFI.git
   cd FDFI
   pip install -e .

From Source with pip
--------------------

.. code-block:: bash

   git clone https://github.com/jinhongdu-lab/FDFI.git
   cd FDFI
   pip install -e .

Optional Dependencies
---------------------

FDFI includes plotting dependencies in the base install. Optional dependency
groups are available for heavier workflows:

**Flow matching models** (PyTorch, torchdiffeq):

.. code-block:: bash

   pip install -e ".[flow]"

**Development tools** (pytest, black, flake8, mypy):

.. code-block:: bash

   pip install -e ".[dev]"

**Documentation** (Sphinx, RTD theme, MyST, nbsphinx):

.. code-block:: bash

   pip install -e ".[docs]"

This extra also installs what is needed to *re-execute* the documentation
notebooks (``ipykernel``, and ``xlrd`` for the CTG case study's spreadsheet).
Building the HTML does not run them — ``conf.py`` sets
``nbsphinx_execute = "never"`` and the stored outputs are rendered as-is.

**All optional dependencies**:

.. code-block:: bash

   pip install -e ".[all]"

Using environment.yml
---------------------

You can also use the provided conda environment file:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate fdfi

Requirements
------------

**Core requirements:**

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0

- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- seaborn >= 0.12.0
- statsmodels >= 0.13.0

**Optional requirements** (``pip install "fdfi[flow]"``, needed only by
``FlowExplainer``):

- torch >= 2.0.0
- torchdiffeq >= 0.2.3

Verifying Installation
----------------------

After installation, verify that FDFI is working:

.. code-block:: python

   import fdfi
   print(fdfi.__version__)

   # Test basic functionality
   import numpy as np
   from fdfi.explainers import OTExplainer

   def model(X):
       return X.sum(axis=1)

   X = np.random.randn(50, 5)
   explainer = OTExplainer(model, data=X, nsamples=20)
   results = explainer(X[:5])
   print("Installation successful!")

Troubleshooting
---------------

**ImportError for torch or torchdiffeq**

Only :class:`~fdfi.explainers.FlowExplainer` requires PyTorch.  If you need it,
install the flow dependencies:

.. code-block:: bash

   pip install -e ".[flow]"

Otherwise use :class:`~fdfi.explainers.OTExplainer` or
:class:`~fdfi.explainers.EOTExplainer`, which never import torch.

**Matplotlib backend issues**

If you encounter issues with matplotlib on headless servers:

.. code-block:: python

   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
