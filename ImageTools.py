#!/bin/python

# Routines for general ion image analysis using the IPython Notebooks

# Notable entries:
# A class IonImage that defines a set of attributes for ion images,
# plus some easily accessible functions within this class

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import csv
from scipy import optimize
import skimage.filters as filt
import skimage.feature as feat
import skimage.measure as measure
import skimage.draw as draw
import skimage.exposure as exposure
from skimage.transform import hough_circle

############### Classes #################

class IonImage:
    Background = False                     # whether or not it's a background image
    ImageCentre = [ 0., 0.]                # List for image centre
    Reference = " "                        # String holding reference
    PumpWavelength = 0.
    ProbeWavelength = 0.
    DetectionThreshold = 0.                # For edge detection
    ColourMap = "spectral"                 # if not specified, we're staying with spectral
    Comments = []
    def __init__(self, Reference, Image):
        self.Reference = Reference         # Holds the logbook reference
        self.Image = Image                 # np.ndarray holding the ion intensities
    def AddComment(self, Comment):
        self.Comments.append(Comment)      # For scribbling notes
    def ClearComments(self):               # Clears all the comments in instance
        self.Comments = []
    def Show(self):
        DisplayImage(self.Image, ColourMap = self.ColourMap)           # plots the image using matplotlib
    def Report(self):                      # Prints a report of the parameters
        print " Logbook reference:\t" + self.Reference
        print " Background Image?:\t" + str(self.Background)
        print " Image centres:\t" + str(self.ImageCentre)
        print " Comments:\t"
        for Comment in self.Comments:
            print Comment
            
    ###############################################################################################
    #################### Image manipulation
    def BlurImage(self, BlurSize):         # Convolution of image with Gaussian kernel BlurSize
        self.BlurSize = BlurSize
        self.BlurredImage = filt.gaussian_filter(self.Image, BlurSize)
        DisplayImage(self.BlurredImage, ColourMap = self.ColourMap)
        
    def EqualiseContrast(self):            # http://scikit-image.org/docs/dev/auto_examples/plot_equalize.html
        p2, p98 = np.percentile(self.Image, (2, 98))
        self.Contrasted = exposure.rescale_intensity(self.Image, in_range=(p2, p98))
        DisplayImage(self.Contrasted, ColourMap = self.ColourMap)
        
    ###############################################################################################
    #################### Centre finding and edge detection
    def EdgeDetection(self, Sigma):
        self.DetectionThreshold = Sigma
        self.DetectedEdges = feat.canny(self.Image, Sigma)               # Use Canny edge detection
        DisplayImage(self.DetectedEdges, ColourMap = self.ColourMap)
    def FindCentre(self):
        if (self.DetectionThreshold == 0.):
            print " You have to call EdgeDetection first!"
            pass
        else:
            Bubble = measure.regionprops(self.DetectedEdges)[0]
            self.ImageCentre = Bubble.centroid
            print self.ImageCentre
            
# Testing class when I was trying out dynamic creation of instances
class BlankTest:
    name = " "
    def __init__(self, FileName):
        self.name = FileName
        
###############################################################################################
###############################################################################################
###############################################################################################

############### Formatting and file I/O routines #################
# Function for reading a text image from file and converting to numpy array
def LoadImage(Path):
    Image = np.genfromtxt(Path, delimiter = DetectDelimiter(Path))
    #if np.sum(np.isnan(Image)):                             # this checks if there are NaN in image    
    #    Image = np.genfromtxt(Path, delimiter="\s")     # np function generates array from text
    return Image

# This function will "intelligently" detect what delimiter is used 
# in a file, and return the delimiter. This is fed into another
# function that does I/O
def DetectDelimiter(File):
    sniffer = csv.Sniffer()
    f = open(File, "r")                   # open file and read the first line
    fc = f.readline()
    f.close()
    line = sniffer.sniff(fc)
    return line.delimiter

# Pretty self explanatory. Uses matplotlib to show the np.ndarray image
def DisplayImage(Image, ColourMap="spectral"):
    plt.imshow(Image)
    plt.set_cmap(ColourMap)              # set the colourmap
    plt.colorbar()                        # add intensity scale
    plt.show()

# function to extract the filename - usually the logbook reference!
def ExtractReference(File):
    return os.path.splitext(os.path.basename(File))[0]
   
# function that will import all the files (as images) in a folder
# by specifying an extension. I recommend .img for all image files
def MassImportImages(Extension):
    LoadedImages = {}
    FileList = glob.glob(Extension)            # see what files in folder
    for File in FileList:                      # loop over all the files
        FileName = ExtractReference(File)      # rip out the filename
        LoadedImages[FileName] = IonImage(FileName, LoadImage(File))     # dynamically generate instances
    return LoadedImages

def ExportImage(FileName, Image):
    np.savetxt(FileName, Image, delimiter="\t")
