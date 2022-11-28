# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 01:50:10 2020

@author: Georg Ganzenmueller, Fraunhofer Ernst-Mach Institut, EMI

This code is performs true strain analysis for cylindric specimens based on the 
evaluation of shadow images.
The evolution of the minimal diameter is tracked. Triaxiality of stress is
evaluated using Bridgman's correction [1].
The algorithm works by determining the silhouette of the specimen gauge region
using Canny-edge detection. The minimum distance between the edges constitutes
the current cylindrical diameter. In the vicinity of the minimum location,
the local edge curvature is calculated, which is then used by the Bridgman
correctiion, providing a measure of stress triaxiality.

This file is the user interface. The actual silhouette detection is performed
in the class SilhouetteDiameter.

The user needs to provide a path to a folder containing the images to be analysed,
along with their file type suffix. The images need to be labeled with increasing numbers.
The specimen diameter should be resolved with at least 100 pixels in order to obtain accurate results
with strain errors < 1%.

Two parameters significantly affect the accuracy of the strain computation, and especially the accuracy
of the stress triaxiality factor: The amount of smoothing (nsmooth) applied to the specimen silhouette
detected by the Canny-edge filter, and the order of its spline representation. Good values
for nsmooth are 100 -- 1000, and order = 1 to 3.



[1] P. W. Bridgman, Studies in Large Plastic Flow and Fracture, McGraw-Hill, New York, (1952).
"""

import configparser, triaxiality

cfg = configparser.ConfigParser()
cfg.read("constrain.ini")
suffix = cfg.get("DEFAULT", "suffix")
nMax = cfg.getint("DEFAULT", "nMax")
transpose = cfg.getboolean("DEFAULT", "transpose") # rotate image? Specimen must be vertical in image
notched_specimen = cfg.getboolean("DEFAULT", "notched_specimen")
nueps = cfg.getfloat("DEFAULT", "nueps")

###############################################################################
# *********** Only edit lines above *************
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import glob
from Silhouette import Silhouette

def read_images():
    filenames = []
    for file in glob.glob(suffix):
        filenames.append(file)
        
    from natsort import natsorted
    filenames = natsorted(filenames)
        
    if nMax > 0:
        return filenames[:nMax]
    else:
        return filenames

filenames = read_images()
N = len(filenames)

diameters = np.zeros(N)
notch_radii = np.zeros(N)
valid_flag = np.zeros(N, dtype=bool)

for i in range(0,len(filenames)):
    print("\n-------------- file: %s --------------" %(filenames[i]))
    sil = Silhouette(filenames[i], plot=False, transpose=transpose)
                             
    if sil.status == True:
        if sil.diameter > 0:
            diameters[i] = sil.diameter
            
            if notched_specimen == False: # straight specimen must have D/R = very large initially
                if i == 0:
                    sil.notch_radius = 1000 * sil.diameter
                else:  # i > 0, so we can compute a strain
                    eps_eq = (1./nueps) * np.log(diameters[0]/diameters[i]) # equivalent true strain
                    if eps_eq < 0.2:
                        sil.notch_radius = 1000 * sil.diameter

            if sil.curvature_valid:
                valid_flag[i] = True
                notch_radii[i] = sil.notch_radius
                print("current diameter in pixels is: ", sil.diameter, " notch radius is; ", sil.notch_radius, "sigma is", sil.notch_sigma)
            else:
                if i == 0:
                    notch_radii[i] = 1000 * sil.diameter
                elif i > 0:
                    notch_radii[i] = notch_radii[i-1]
    else:
        print("something went wrong in the analysis, exiting prematurely")
        break

triaxs = triaxiality.triax_function(diameters, notch_radii)
eps_eq = eps_eq = (1./nueps) * np.log(diameters[0]/diameters)
savedata = np.column_stack((np.arange(len(diameters)), diameters, notch_radii, eps_eq, triaxs))
np.savetxt("constrain_output.txt",savedata, header="index, diameter, notch_radius, strain, triax")


plt.style.use('dark_background')
fig, axs = plt.subplots(2)

# --- TOP plt: diameter vs frame number ---
ax = axs[0]
ax.set_xlabel('frame No.')
ax.set_ylabel('diameter [px]')
ax.plot(diameters, "rx", label="diameter")

# --- BOTTOM plt: notch radius vs frame number ---
ax = axs[1]
ax.set_xlabel('frame No.')
ax.set_ylabel('stress triaxiality [-]')
ax.plot(triaxs, "bo", label="stress triax.")



fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
