Installation
============

KDExpress needs JAX, which is not installed automatically because the right
build depends on your hardware. Install it first, following the
`JAX instructions <https://github.com/jax-ml/jax#installation>`_:

.. code-block:: bash

   pip install "jax[cpu]"      # CPU only
   pip install "jax[cuda12]"   # NVIDIA GPU (CUDA 12)

Then install KDExpress:

.. code-block:: bash

   pip install KDExpress

To work on the code:

.. code-block:: bash

   git clone https://github.com/mtagliazucchi/KDExpress
   cd KDExpress
   pip install -e ".[test,docs]"
   pytest
   sphinx-build -b html docs docs/_build/html
