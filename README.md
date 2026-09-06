## CLEA: Contrastive Learning from Exploratory Actions
This is the project page for the 2025 HRI best technical paper nominee "Contrastive Learning from Exploratory Actions: Leveraging Natural Interactions for Preference Elicitation".

[https://interaction-lab.github.io/CLEA/](https://interaction-lab.github.io/CLEA/)

This repository contains the code for re-running the experiments conducted in that paper, the website that provides and overview of the paper, and the various artifacts (dataset, user responses, etc.) that are necessary for training representations. 

### System Requirements

The code currently in this repository was tested on the following system configuration:

| Component | Spec |
|---|---|
| OS | Ubuntu 24.04.4 LTS (Linux 7.0.0-28-generic, x86_64) |
| CPU | Intel Core Ultra 9 285K (24 cores) |
| RAM | 62GB |
| GPU | NVIDIA GeForce RTX 5090 (32GB VRAM), driver 580.173.02 |
| CUDA | 12.0 |
| Python | 3.13.11 |
| PyTorch | 2.9.1+cu128 |
| Package Management | Conda |

This setup is generally quite excessive for this project. Many of the original experiments were conducted on a 2018 laptop with a 4GB GPU and a 2022 Macbook Air. The general requirements are:

**Software**
- Python 3.8+ (see [`src/README.md`](src/README.md) for setup instructions and [`src/requirements.txt`](src/requirements.txt) for the full dependency list)
- Conda recommended for environment management, though you are welcome to port this to uv if you prefer!
- PyTorch; a CUDA-capable GPU is recommended for training but the training scripts fall back to CPU automatically if none is available.

**Hardware**
- GPU: 8GB+ recommended. This is especially for training the visual/auditory models (CPU training works but is much slower)
- RAM: 16GB+ recommended. The dataloaders preload the full raw stimulus set (images/spectrograms) into memory per training run
- Disk: ~1.5GB for the full repository, including the dataset and example results.

### Code Structure

There are two top-level folders in this repository. `static` contains the files served on the website, and `src` contains the code. Please refer to the README.md in the `src` folder for step-by-step instructions for running the paper's code.


### Acknowledgments
Parts of this project page were adopted from the [Nerfies](https://nerfies.github.io/) page and [eliahuhorwitz/Academic-project-page-template](https://github.com/eliahuhorwitz/Academic-project-page-template).

### Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
