# rqcopt-mpo
Repository accompanying the paper "Riemannian quantum circuit optimization based on matrix product operators" (2025)."

A virtual environment for using this repository is given in ``requirements.txt``.
Note that we use a patched version of ``tikzplotlib`` for plotting the results that you can find [here](https://github.com/JasonGross/tikzplotlib).
However, this is only required if you want to save the plots as a ``tex``-file.

The implementation of methods for the Riemannian quantum circuit optimization based on matrix product operators is given in ``rqcopt_mpo``.
You need to install this locally to run the code: ``python3 -m pip install -e .``.

In this project, we optimize one-dimensional (non-)disordered Ising models, Heisenberg models, spinful Fermi-Hubbard models, and the molecule LiH.
The folder ``run`` contains ...
* ... the example configurations to execute each model,
* ... the script to generate the references,
* ... the script to run the Riemannian optimization.

The examplary configurations are chosen such that the execution should not take more than 2 minutes.

You can generate the reference by running ``python generate-reference.py hamiltonian/configs/config reference.yml``.
Similarly, you can run the Riemannian optimization by executing ``python run-optimization.py hamiltonian/configs/config.yml``.
In both cases, ``hamiltonian`` should be replaced by the folder name of the model you want to run, i.e., ``ising-1d``, ``heisenberg``, or ``fermi-hubbard-1d``.

The folder ``plotting-results`` contains the numerical data and the script to generate the figures used in the publication.

The folder ``trotterization`` contains a script to evaluate the (not-optimized) Trotter circuits.

This repository accompanies: I. N. M. Le, S. Sun, and C. B. Mendl, Riemannian quantum circuit optimization based on matrix product operators, In preparation (2025).
