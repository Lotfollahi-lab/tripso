<div align="center">
  <img src="tripso_logo_cropped.png" alt="Tripso Logo" width="400"/>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.10-brightgreen)
[![Documentation](https://img.shields.io/badge/docs-Sanger%20docs-blue)](https://tripso-docs.cog.sanger.ac.uk/index.html)

**Tripso** (**T**ransformers for learning **R**epresentations of **I**nterpretable gene **P**rograms in **S**ingle-cell transcript**o**mics) is a self-supervised framework that decomposes cellular states into multiple gene program (GP) embeddings. Tripso learns contextualised gene embeddings within each GP, quantifies gene-level contributions to each GP, and the influence of each GP on global cell identity.

## Resources

- **Installation, tutorials, and API reference:** [tripso documentation](https://tripso-docs.cog.sanger.ac.uk/index.html) (built with Sphinx; theme provided by Read the Docs).
- **Bug reports:** use [GitHub Issues](https://github.com/Lotfollahi-lab/tripso/issues).
- **Contributing:** pull requests are welcome. The repo uses checks for PEP 8 and static typing; install [`pre-commit`](https://pre-commit.com/) and run `pre-commit install` before committing. For building the docs locally, see [Building the documentation](https://tripso-docs.cog.sanger.ac.uk/contributing/docs.html).

## Citation

If you use Tripso in your research, please cite our [preprint](https://www.biorxiv.org/content/10.64898/2026.03.24.713961v1.full):

    Moullet, M., Isobe T., Vahidi A., Leonardi C. et al. Self-supervised learning for a gene program-centric view of cell states. bioRxiv 2026.03.24.713961 (2026) doi:10.64898/2026.03.24.713961.
  
