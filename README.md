# mobile-SRGAN

We introduce an innovative super resolution approach to emerging modes of near-field synthetic aperture radar (SAR) imaging. Recent research extends convolutional neural network (CNN) architectures from the optical to the electromagnetic domain to achieve super resolution on images generated from radar signaling. 
Specifically, near-field synthetic aperture radar (SAR) imaging, a method for generating high-resolution images by scanning a radar across space to create a synthetic aperture, is of interest due to its high-fidelity spatial sensing capability, low cost devices, and large application space.
Since SAR imaging requires large aperture sizes to achieve high resolution, super-resolution algorithms are valuable for many applications. Freehand smartphone SAR, an emerging sensing modality, requires irregular SAR apertures in the near-field and computation on mobile devices.
Achieving efficient high-resolution SAR images from irregularly sampled data collected by freehand motion of a smartphone is a challenging task. 
In this paper, we propose a novel CNN architecture to achieve SAR image super-resolution for mobile applications by employing state-of-the-art SAR processing and deep learning techniques. The proposed algorithm is verified via simulation and an empirical study. Our algorithm demonstrates high-efficiency and high-resolution radar imaging for near-field scenarios with irregular scanning geometries.

----

Goals:
 1. Super Resolution approach to emerging modes of near-field synthetic aperture radar (SAR) imaging.
 2. Developed a Generative Adversarial Network (GAN) capable of denoising and clearing data from any kind of distortions. Developed for embedded systems and mobile devices.
 
under scripts/lib/:
> - `ax3dgenperclib`: train a CGAN with perceptual loss and incorporated Adaptive-EXperimentation Platform (AX) for hyperparameters tuning. It trained with 3D SAR data.
> - `ax3dwganlib`: WGAN and incorporated Adaptive-EXperimentation Platform (AX) for hyperparameters tuning. It trained with 3D SAR data.
> - `axdata3dlib`: CGAN and incorporated Adaptive-EXperimentation Platform (AX) for hyperparameters tuning. It trained with 3D SAR data.
> - `data3dlib`: CGAN trained with 3D SAR data.
> - `ganlib`: CGAN trained with 2D SAR data.
> - `mdblib`: CGAN with mini-batch discrimation layer. It trained with 2D SAR data.
> - `mobilelib`: GAN utilized to work in mobile devices. It trained with 2D SAR data.
> - `multiclass3dlib`: 
> - `patchganlib`: CGAN trained with Patch Discriminator. It trained with 2D SAR data.
> - `randpointslib`: CNN trained as a Regressor. It trained with 2D SAR data.
> - `solidlib`: CNN trained as a Regressor. It trained with 2D SAR data.
> - `wganlib`: WGAN trained with 2D SAR data.
 
----

# Publication:

[Efficient CNN-Based Super Resolution Algorithms for Mmwave Mobile Radar Imaging](https://ieeexplore.ieee.org/document/9897190)

# Citation:

C. Vasileiou, J. Smith, S. Thiagarajan, M. Nigh, Y. Makris and M. Torlak, "Efficient CNN-Based Super Resolution Algorithms for Mmwave Mobile Radar Imaging," 2022 IEEE International Conference on Image Processing (ICIP), Bordeaux, France, 2022, pp. 3803-3807, doi: 10.1109/ICIP46576.2022.9897190.
