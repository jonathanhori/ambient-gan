# ambient-gan
Implementation of the AmbientGAN model as a final project for UCLA's Stat 231B course. 


We will reproduce Figure 7. This requires implementing a Wasserstein GAN with gradient penalty (WGANGP) with the Block-Pixels and Convolve+Noise measurement models. 

The AmbientGAN model is benchmarked against alternatives:
- For Block-Pixels:
    1. Ignore - learn the model directly on the measurements 
    2. Unmeasure-blur
    3. Unmeasure-inpaint-tv
- For Convolve+Noise:
    1. Ignore
    2. Unmeasure-weiner

Implementation of the models will build on the code contained from the following sources:
- https://github.com/igul222/improved_wgan_training (WGANGP)
- https://github.com/AshishBora/ambient-gan/blob/master/src/mnist/gen/gan_def.py
- https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
- https://github.com/Zeleni9/pytorch-wgan/blob/master/models/wgan_gradient_penalty.py
