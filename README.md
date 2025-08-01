# Portrait Editing with GANs – Hairstyle Transfer with StyleGAN


This repository contains the work from my bachelor’s thesis on hairstyle transfer in portrait images using StyleGAN2 and the Barbershop framework.

## Keywords
Generative Adversarial Networks (GANs), Image Synthesis & Editing, Representation Learning / Latent Space Manipulation, Human-Centered Evaluation

## What I did
- Built a pipeline for hairstyle editing based on StyleGAN2 + Barbershop

- Improved preprocessing to stabilize results:

      - Face alignment for consistent GAN inversion

      - Baldification to reduce interference from source hair

- Implemented GAN inversion to map input portraits into StyleGAN’s latent space

- Applied latent space editing for hairstyle transfer while preserving facial identity

- Evaluated results through a user study on realism and identity preservation


## Key outcome
- Preprocessing steps significantly improved transfer quality

- Produced more realistic and identity‑consistent results compared to the baseline Barbershop approach


## Technologies used
- Python (core implementation)

- PyTorch (deep learning framework)

- StyleGAN2 (generative model for face synthesis)

- Barbershop (framework for hairstyle transfer)

- OpenCV & dlib (image preprocessing and face alignment)
