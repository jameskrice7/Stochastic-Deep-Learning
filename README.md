# Stochastic Latent Differential Inference with Predictable Adjoint Control Variates

This repository accompanies James K Rice's revision of [arXiv:2601.05227](https://arxiv.org/abs/2601.05227). It contains the manuscript, synthetic experiments, generated data and results, and a Lean check of selected mathematical identities.

## Build the paper

The editable manuscript is [paper/main.tex](paper/main.tex). From the repository root, run the following in a TeX distribution with pdfLaTeX, BibTeX, AMS LaTeX, natbib, geometry, booktabs, and hyperref.

    cd paper
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    bibtex main
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex

The checked-in bibliography output is sufficient for pdfLaTeX if the citations and bibliography are unchanged.

## Reproduce the experiments

From the repository root, using Python 3.12 and a clean environment:

    python -m pip install -r reproduction/requirements.txt
    python reproduction/reproduce.py --out regenerated

The script generates four CSVs. The committed versions are in [reproduction/results](reproduction/results).

| File | Contents |
| --- | --- |
| ou.csv | Initial-only evidence gap, analytic and simulated OU gradient variances, and pathwise cancellation error |
| contexts.csv | Synthetic observation times and values |
| nonlinear_runs.csv | Matched results for every context, seed, pilot size, and estimator |
| nonlinear_summary.csv | Medians and interquartile ranges used in the paper |

The program checks both parameter and mixed derivatives against centered finite differences and verifies OU cancellation up to floating-point error. It regenerates every observation and simulated path from fixed seeds. Last-digit differences can occur across platforms.

The nonlinear comparison conditions on each observed sequence and evaluates gradients at fixed parameters. Every learned control uses independent pilot paths. Ratios for the learned controls exclude pilot simulation and fitting costs.

## Check the Lean formalization

From the repository root, with Lean installed through Elan:

    cd reproduction
    lean SLDIChecks.lean

The pinned toolchain is Lean 4.29.0 and uses only its standard distribution. The file proves centeredness, unbiasedness and orthogonality to the past for a finite rational product law, a weighted squared-error decomposition under orthogonality, and the pathwise polynomial cancellation in the OU example for an arbitrary finite list of steps. The manuscript provides the analytic arguments for conditional expectation, Gaussian integration by parts and the path-law KL.

[Bibliography audit and mathematical verification scope](VERIFICATION.md)
