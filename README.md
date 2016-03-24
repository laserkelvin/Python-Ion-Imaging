# Python-Ion-Imaging
Routines that help with Ion Imaging data analysis

## Introduction

This repo contains some Python scripts and modules that I've written to help speed up ion imaging data analysis.
I do all of my analysis these days in IPython Notebook, so a lot of these routines are written with notebook specifically in mind.

Most of the functions included here rely on the fantastic Scikit-Image library, particularly with the Canny edge detection and centre finding algorithms. The only real work that I've done here is tried to make the scripting as neat as possible with regular commenting as well as for use in an IPython notebook setting.

## IonImage class

This class is something I thought of doing after reading some object-oriented programming articles. Essentially, this is allowing me to treat all ion images with the same set of methods and attributes, which should hopefully make storing and analysing lots and lots of ion images easier. The attributes of IonImage are:

Background (boolean) - for telling if it's a background image or not (maybe for some functionality later, but now it's just a boolean)

ImageCentre (tuple) - Two values for storing the x and y centres of an image

Reference (string) - Logbook reference for this image

PumpWavelength (float)

ProbeWavelength (float) - Pretty self-explanatory

DetectionThreshold (float) - Used to store the sigma value used for Canny edge detection (i.e. the blur value), not quite a threshold

ColourMap (string) - Makes imgshow with matplotlib more colourful

Comments (list) - Self-explanatory. Uses the methods AddComment and ClearComments to add and wipe the comments data

BlurSize (float) - Keeps track of the Gaussian kernel size for blurring in the method BlurImage

Image (np.ndarray) - Numpy array that holds the intensity values

BlurredImage (np.ndarray) - Numpy array that holds the blurred image

Contrasted (np.ndarray) - Numpy array that holds the histogram equalised contrast image

DetectedEdges (np.ndarray) - Numpy __boolean__ array that marks where edges are discovered by the method EdgeDetection


## File I/O

The simplest routine in here is the LoadImage function, which will generate a numpy.ndarray to hold a text image. The IonImage class __init__ depends on this function to set an Image attribute.

Some pretty nifty things I've written in here include the delimiter sniffer using OpenCSV. Means you don't have to specify what delimiter you file has, it figures it out from reading in the first line and sniffing it!

I've also written a MassImportImages function that will, as the name suggest import all files in a folder with a given file extension as a dictionary of ion image instances. The idea behind this is to minimise redundancy between notebooks. Often I've made a new notebook to do a specific part of data analysis, it'd be nice and easy to manage a single database (shelf, in fact) containing all the images as well as some "metadata" about them if you will.
