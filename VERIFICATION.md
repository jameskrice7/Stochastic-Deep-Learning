# Bibliography and mathematical verification

## Bibliography

All entries used by the revised manuscript were checked against the records below. The bibliography follows the publisher or proceedings record where available and uses version-specific links for the cited preprints.

| Reference | Record checked |
| --- | --- |
| Archambeau et al. (2007), *Variational Inference for Diffusion Processes* | [NeurIPS 20](https://proceedings.neurips.cc/paper_files/paper/2007/hash/818f4654ed39a1c147d1e51a00ffb4cb-Abstract.html) |
| Belomestny et al. (2018), *Variance Reduction for Discretised Diffusions via Regression* | [Journal of Mathematical Analysis and Applications 458(1), 393–418](https://doi.org/10.1016/j.jmaa.2017.09.002) |
| Domingo-Enrich et al. (2025), *Adjoint Matching* | [ICLR proceedings](https://openreview.net/pdf?id=xQBRrtQM8u) and [arXiv author metadata](https://arxiv.org/abs/2409.08861) |
| Girsanov (1960), *On Transforming a Certain Class of Stochastic Processes by Absolutely Continuous Substitution of Measures* | [SIAM English edition 5(3), 285–301](https://doi.org/10.1137/1105027) |
| Kidger et al. (2021), *Efficient and Accurate Gradients for Neural SDEs* | [NeurIPS 34](https://proceedings.neurips.cc/paper/2021/hash/9ba196c7a6e89eafd0954de80fc1b224-Abstract.html) |
| Li et al. (2020), *Scalable Gradients and Variational Inference for Stochastic Differential Equations* | [PMLR 118, 1–28](https://proceedings.mlr.press/v118/li20a.html) |
| Mei and Taghvaei (2026), *Time-Reversed BSDEs for Accurate Gradient Estimation in Diffusion Models* | [arXiv:2603.20455v2](https://arxiv.org/abs/2603.20455v2) |
| Miller et al. (2017), *Reducing Reparameterization Gradient Variance* | [NeurIPS 30](https://papers.neurips.cc/paper_files/paper/2017/hash/325995af77a0e8b06d1204a171010b3a-Abstract.html) |
| Rice (2026), *Stochastic Deep Learning* | [arXiv:2601.05227v1](https://arxiv.org/abs/2601.05227v1) |
| Roeder et al. (2017), *Sticking the Landing* | [NeurIPS 30](https://papers.neurips.cc/paper_files/paper/2017/hash/e91068fff3d7fa1594dfdf3b4308433a-Abstract.html) |
| Rubanova et al. (2019), *Latent Ordinary Differential Equations for Irregularly-Sampled Time Series* | [NeurIPS 32](https://proceedings.neurips.cc/paper/2019/hash/42a6845a557bef704ad8ac9cb4461d43-Abstract.html) |
| Schauer et al. (2017), *Guided Proposals for Simulating Multi-Dimensional Diffusion Bridges* | [Bernoulli 23(4A), 2917–2950](https://doi.org/10.3150/16-BEJ833) |
| Xu et al. (2022), *Infinitely Deep Bayesian Neural Networks with Stochastic Differential Equations* | [PMLR 151, 721–738](https://proceedings.mlr.press/v151/xu22a.html) |

## Mathematical verification

| Manuscript result | Check |
| --- | --- |
| Initial-only optimal ELBO and equality condition | Analytic proof by exponential tilting and strict conditional Jensen. The Gaussian example evaluates the evidence and the bound separately. |
| Path-law KL and finite-grid ELBO | Analytic derivation by change of measure and the chain rule for the Gaussian transitions, using the stated invertibility and finite-energy assumptions. |
| Predictable centering and orthogonality | Lean proves the product-law identities over finite supports with exact rational weights. The paper supplies the general conditional-expectation argument. |
| Residual variance decomposition | Lean verifies the squared-error split when the cross term vanishes. The paper proves the projection and completeness steps. |
| Mixed-adjoint coefficient targets | Analytic derivation by Gaussian integration by parts. The reference implementation checks the parameter gradients and mixed innovation derivatives by finite differences. |
| OU second-order correction | Lean proves the telescope for arbitrary finite step lists over the rationals after clearing the common scale. The simulation checks the cancellation path by path. |

The Lean checks use statements over exact finite sums and polynomials. The measure-theoretic arguments and the SDE model are covered by the manuscript's analytic proofs.
