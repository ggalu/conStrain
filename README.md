# *con*Strain
*con*Strain is an image analysis software which calculates true stress / true strain curves for tensile material tests in engineering. The method computes strain from a specimen's contour, recorded in a bright field approach. See our paper (G. C. Ganzenmüller, P. Jakkula and S. Hiermaier, Strain measurement by contour analysis, https://arxiv.org/abs/2211.14030) for details.

## Description
*con*Strain is applicable to axis-symmetric specimens under tensile loading. Here, it calcualates accurate true stress and true strain. Automated image analysis also yields the curvature of the necking region, such that the effects of stress triaxiality are accounted for. In contrast to Digital Image Correlation (DIC), *con*Strain does not require a contrast pattern, but instead uses a backlight source behind the specimen, such that a shadow image is obtained.

*con*Strain is optimized for analyzing materials with Poisson ratio of 1/2, typically metals undergoing isochoric plastic flow.  It is suboptimal for analyzing small, elastic deformations. In that case, other method ssuch as DIC are better suited. However, *con*Strain really shines when deformation are large as it becomes more accurate with increasing strain.

## Installation
*con*Strain is written in Python 3. You need a working Python 3 installation to use *con*Strain.
- Install the following Python packages
    - scikit-image
    - opencv-python
- extract the contents of the *con*Strain archive into a folder of your choice, termed the *installation directory* henceforth.

## usage
- open a command shell
- Go to the *installation directory*, and to the *sample* directory below that.
- execute:
    `python3 ../conStrain.py`
- this reads in the configuration settings from file `constrain.ini` and iterates over all images present in the sample directory, calculating the cross section diameter and the radius of neck curvature. Finally, the collected data is displayed and saved. The resulting file, `constrain_output.txt`, contains these columns:
        - (1) image index,
        - (2) diameter of the minimum cross section in pixel units
        - (3) the neck radius in pixel units
        - (4) true strain
        - (5) an estimate of the stress triaxiality in the neck
              
## Acknowledgments
*con*Strain uses many open-source packages. In particular, Kanatani's Hyperaccuracy circle fitting package (IEICE TRANSACTIONS on Information and Systems, 2006 Oct;E89-D(10):2653–2660.) made *con*Strain possible.
    





