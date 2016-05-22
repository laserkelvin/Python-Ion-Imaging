# Routines for general ion image analysis using the IPython Notebooks

# Notable entries:
# A class IonImage that defines a set of attributes for ion images,
# plus some easily accessible functions within this class

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import glob
import os
import shelve
import csv
import pandas as pd
import skimage.filters as filt
import skimage.feature as feat
import skimage.measure as measure
from skimage.morphology import reconstruction
import skimage.exposure as exposure
from ipywidgets import *
from IPython.display import display  
from IPython.html import widgets
import abel

############### Classes #################

class IonImage:
    Background = False                     # whether or not it's a background image
    ImageCentre = [ 0., 0.]                # List for image centre
    Reference = " "                        # String holding reference
    PumpWavelength = 0.
    ProbeWavelength = 0.
    DetectionThreshold = 0.                # For edge detection
    ColourMap = "paired"                 # if not specified, we're staying with spectral
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
    def SetCalibrationConstant(self, CalConstant):
        self.CalibrationConstant = CalConstant
    def Report(self):                      # Prints a report of the parameters
        print " Logbook reference:\t" + self.Reference
        print " Background Image?:\t" + str(self.Background)
        print " Image centres:\t" + str(self.ImageCentre)
        print " Comments:\t"
        for Comment in self.Comments:
            print Comment
    def ShowQuadrants(self):
        try:
            fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
            axarr[0,0].imshow(self.PreSymmetryQuadrants[0])
            axarr[0,1].imshow(self.PreSymmetryQuadrants[1])
            axarr[1,0].imshow(self.PreSymmetryQuadrants[2])
            axarr[0,1].imshow(self.PreSymmetryQuadrants[3])
        except AttributeError:
            print " You haven't symmetrised the image yet!"
            
    ###############################################################################################
    #################### Image manipulation
    def BlurImage(self, BlurSize):         # Convolution of image with Gaussian kernel BlurSize
        self.BlurSize = BlurSize
        self.BlurredImage = filt.gaussian(self.Image, BlurSize)
        DisplayImage(self.BlurredImage, ColourMap = self.ColourMap)
        
    def SubtractBackground(self, BackgroundFile):
        """ Function to subtract the a background image
            specified as the filepath from the current
            image instance
        """
        BackgroundInstance = IonImage(Reference="Temp",
                                      Image=LoadImage(BackgroundFile))
        self.SubtractedImage = SubtractImages(self, BackgroundInstance)

    def EqualiseContrast(self):            # http://scikit-image.org/docs/dev/auto_examples/plot_equalize.html
        """Enhances the contrast of the image by normalising the intensity histogram
        using example from scikit-image @ 
        http://scikit-image.org/docs/dev/auto_examples/plot_equalize.html

        Sets attribute Contrasted as the normalised image. Displays the image as well.
        """
        p2, p98 = np.percentile(self.Image, (2, 98))
        self.Contrasted = exposure.rescale_intensity(self.Image, in_range=(p2, p98))
        DisplayImage(self.Contrasted, ColourMap = self.ColourMap)

    def SymmetriseCrop(self, x0=None, y0=None, CropSize=651):
        """
        Function that will symmetrise an image using the four quadrants and
        crop the image after symmetrisation

        Input:
        x0 - centre value in x (int)
        y0 - centre value in y (int)
        CropSize - Size of image after cropping (int)

        Returns:
        Sets a few attributes to the Image instance,
        SymmetrisedImage - The image with four symmetrised quadrants in
                           the shape of the original input image (ndarray)
        SymmetrisedQuadrant - A symmetrised quadrant (ndarray)
        CBS - Stands for "Cropped, Blurred and Symmetrised". Returns
              the four symmetrised quadrants in the shape of CropSize (ndarray)

        """
        OriginalImageSize = len(self.Image)
        if x0 and y0 == None:
            x0 = self.ImageCentre[0]
            y0 = self.ImageCentre[1]
        else:
            pass
        # Initialise arrays
        FirstQuarter = np.zeros((OriginalImageSize, OriginalImageSize), dtype=float)
        SecondQuarter = np.zeros((OriginalImageSize, OriginalImageSize), dtype=float)
        ThirdQuarter = np.zeros((OriginalImageSize, OriginalImageSize), dtype=float)
        FourthQuarter = np.zeros((OriginalImageSize, OriginalImageSize), dtype=float)
        # Draw quarters of the original image
        FirstQuarter = AddArrays(FirstQuarter, self.BlurredImage[:x0, :y0]) 
        SecondQuarter = AddArrays(SecondQuarter, np.rot90(self.BlurredImage[:x0, y0:], k=1)).T   # Rotate quadrant
        ThirdQuarter = AddArrays(ThirdQuarter, np.rot90(self.BlurredImage[x0:, y0:], k=2))     # to phase match
        FourthQuarter = AddArrays(FourthQuarter, np.rot90(self.BlurredImage[x0:, :y0], k=3)).T   # first quadrant
        # I keep these for later viewing if needed, to see if the symmetrisation is fucked
        self.PreSymmetryQuadrants = [FirstQuarter, SecondQuarter, ThirdQuarter, FourthQuarter]
        # Calculate symmetrised quadrant by averaging the four quarters
        SymmedQuadrants = AverageArrays([FirstQuarter,
                                         SecondQuarter,
                                         ThirdQuarter,
                                         FourthQuarter])
        FullSymmetrisedImage = np.zeros((OriginalImageSize, OriginalImageSize), dtype=float)
        # Draw a fully symmetrised image
        for angle in range(4):
            if angle % 2 == 0:         # if it's divisible by two, transpose the matrix
                FullSymmetrisedImage = AddArrays(FullSymmetrisedImage,
                                                 np.rot90(SymmedQuadrants, k=angle).T)
            else:
                FullSymmetrisedImage = AddArrays(FullSymmetrisedImage,
                                                 np.rot90(SymmedQuadrants, k=angle))
        self.SymmetrisedImage = np.rot90(FullSymmetrisedImage, k=1)
        #DisplayImage(self.SymmetrisedImage)
        self.SymmetrisedQuadrant = SymmedQuadrants
        # Crop the image now, took the routine from an older set of scripts so it's quite
        # crappily written
        UpperLimit = int(np.ceil(CropSize / 2.))               # Get the upper limit of the image
        LowerLimit = int(np.floor(CropSize / 2.))              # Get the lower limit of the image
        XUp = x0 + UpperLimit
        XDown = x0 - LowerLimit
        YUp = y0 + UpperLimit
        YDown = y0 - LowerLimit
        CroppedImage = np.zeros((CropSize, CropSize), dtype=float)
        # counters for the cropped image array, since it's different from the fullimage
        i = 0
        for row in range(YDown,YUp):
            j = 0      # reset the column counter
            for col in range(XDown,XUp):
                CroppedImage[i,j] = FullSymmetrisedImage[row,col]
                j = j + 1    # increment the column counter by one
            i = i + 1       # increment the row counter by one
        self.CBS = CroppedImage
        #DisplayImage(CroppedImage, ColourMap=self.ColourMap)

        
    ###############################################################################################
    #################### Centre finding and edge detection
    def EdgeDetection(self, Sigma):
        self.DetectionThreshold = Sigma
        self.DetectedEdges = np.array(feat.canny(self.Image, Sigma), dtype=int)     # Use Canny edge detection
        DisplayImage(self.DetectedEdges, ColourMap = self.ColourMap)
    def FindCentre(self):
        if (self.DetectionThreshold == 0.):
            print " You have to call EdgeDetection first!"
            pass
        else:
            Bubble = measure.regionprops(self.DetectedEdges)[0]
            self.ImageCentre = Bubble.centroid
            print self.ImageCentre

    def BASEX(self):
        """ Calculates the inverse Abel transform of the symmetrised
        and cropped image by calling routines in the abel module

        Returns:
        ReconstructedImage - 2D slice of the 3D reconstructed ion image (ndarray)
        PES - Tuple containing the speed distribution vs. pixel radius

        """
        try:
            self.ReconstructedImage = abel.Transform(self.BlurredImage,
                                                direction="inverse",
                                                method="basex",
                                                center=self.ImageCentre).transform
            r, speed = abel.tools.vmi.angular_integration(self.ReconstructedImage,
                                                          dr=self.CalibrationConstant)
            self.PES = pd.DataFrame(data=speed,
                                    columns=["P(s)"],
                                    index=r)
        except AttributeError:
            print " Image has not been centred, blurred or calconstant-less."
            print " Call BlurImage and FindCentre before running."
            pass
            

class Multidimensional:
    """ A class written specifically for analysing
        multidimensional data (e.g. 2D REMPI)

        Image is a 2D numpy array, which here
        we'll specify as the
    """
    def __init__(self, ImagePath):
        self.OriginalData = LoadImage(ImagePath)
        self.ManipulatedData = self.OriginalData
        self.ColourMap = cm.inferno
        self.BlurSize = 0.
        self.Test = 0.
        self.Figure = None
        self.DefaultAxes()
        self.DefaultPlotSettings()

    def BlurImage(self, FilterSize=1.):
        self.ManipulatedData = filt.gaussian(self.ManipulatedData, FilterSize)
        self.BlurSize = FilterSize

    def DefaultAxes(self):
        self.XData = np.arange(0, self.ManipulatedData.shape[1], 1)
        self.YData = np.arange(0, self.ManipulatedData.shape[0], 1)
        self.XMesh, self.YMesh = np.meshgrid(self.XData, self.YData, sparse=True)
        self.Calibrated = False

    def DefaultPlotSettings(self):
        self.PlotSettings = {"rstride": 5,        # Density to plot wavelength in
                             "cstride": 20,            # Density to plot speed in
                             "alpha": 1.,
                             "antialiased": True,
                             "cmap": self.ColourMap,
                             "linewidth": 1}

    def CalibrateAxes(self, Wavelength, CalibrationConstant):
        self.Calibrated = True
        self.XData = np.linspace(Wavelength[0], Wavelength[1], self.ManipulatedData.shape[1])
        self.YData = np.arange(0, self.ManipulatedData.shape[0], 1)
        self.YData = self.YData * CalibrationConstant
        self.XMesh, self.YMesh = np.meshgrid(self.XData, self.YData, sparse=True)

    def ResetImage(self):
        self.ManipulatedData = self.OriginalData
        self.DefaultAxes()
        self.DefaultPlotSettings

    def CropImage(self, WavelengthCrop, SpeedCrop):
        self.ManipulatedData = self.OriginalData[SpeedCrop[0]:SpeedCrop[1],
                                                 WavelengthCrop[0]:WavelengthCrop[1],]
        if self.BlurSize > 0.:     # if it's been blurred before, re-blur it
            self.BlurImage(FilterSize=self.BlurSize)
        self.DefaultAxes()

    def Plot1D(self):
        if self.Figure == None:
            self.Figure = plt.figure(figsize=(12, 10))
        plt.imshow(self.ManipulatedData, cmap=self.ColourMap)
        plt.colorbar()
        plt.draw()

    def Plot2D(self):
        if self.Figure == None:
            self.Figure = plt.figure(figsize=(12, 10))
        self.Axis = self.Figure.gca(projection="3d")
        if self.Calibrated == True:
            plt.xlabel("Two-photon energy / cm$^{-1}$")
            plt.ylabel("Speed / ms$^{-1}$")
        self.SurfPlot = self.Axis.plot_surface(self.XMesh,
                                               self.YMesh,
                                               self.ManipulatedData,
                                               **(self.PlotSettings)
                                               )

    def InteractiveCropFunction(self, WavelengthCrop, SpeedCrop):
        """ Called by InteractiveCrop to do the actual grunt work.
            Basically just calls CropImage then plots the image
            using imshow.
        """
        Wavelength = [WavelengthCrop, -1]
        Speed = [SpeedCrop, -1]
        self.CropImage(Wavelength, Speed)
        self.Figure = plt.figure(figsize=(12,10))
        plt.imshow(self.ManipulatedData, cmap=self.ColourMap)
        plt.colorbar()
        plt.show()

    def InteractiveCrop(self):
        """ Uses IPython widgets to crop a 2D. Makes things a
            bit easier when you can visualise it!
        """
        interact(self.InteractiveCropFunction,
                 WavelengthCrop=(0, self.ManipulatedData.shape[1], 1),
                 SpeedCrop=(0, self.ManipulatedData.shape[0], 1))

    def FilterMaxima(self, Minimum=None):
        if Minimum == None:
            Minimum = self.ManipulatedData.min()
        Seed = self.ManipulatedData - Minimum
        Mask = self.ManipulatedData
        self.Dilated = reconstruction(Seed, Mask, method="dilation")
        self.ManipulatedData = self.ManipulatedData - self.Dilated

    def ExtractSpeed(self, WavelengthRange):
        SpeedSlice = np.zeros((WavelengthRange[1] - WavelengthRange[0]))
        for Index, WavelengthIndex in enumerate(WavelengthRange):
            SpeedSlice[index] = np.sum(self.ManipulatedData[WavelengthIndex,:])
        return SpeedSlice
            
class CorrelatedREMPI:
    def __init__(self, Reference, Image):
        self.Reference = Reference
        self.Image = Image
        self.ColourMap = "Paired"

    def Show(self):
        DisplayImage(self.Image, ColourMap=self.ColourMap)

    def CropImage(self, XRange, YRange):
        self.CroppedImage = self.Image[YRange[0]:YRange[1],
                                       XRange[0]:XRange[1]]

    def IntegrateREMPI(self, Range):
        """ Integrate Y across a range of X, and return
            the average value.
            Array is a np 2D array, and range is given
            as a 2-tuple.
        
            Example of this is to extract the REMPI
            spectrum out of a 2D REMPI.
        """
        X = xrange(Range[0], Range[1])
        SummedArray = np.sum(self.Image[:, Range[0]:Range[1]], axis=0)
        Average = np.sum(SummedArray) / np.shape(SummedArray)
        return X, SummedArray

    def IntegrateSpeed(self, Range):
        """ Integrate X across a range of Y, and return
            the average distribution.
            Array is a np 2D array, and range is given
            as a 2-tuple.
        
            Example of this is to extract an
            averaged speed distribution out of a 2D REMPI.

            Definitely not the most pythonic way of
            coding this up.
        """
        counter = 0
        X = xrange(0, len(self.Image[:,0]))
        Y = np.zeros((len(self.Image[:,0])), dtype=float)
        for index in xrange(Range[0], Range[1]):
            counter = counter + 1
            Y = Y + ExtractSlice(Array, index)
            Y = Y / counter
        return X, Y

    def SubtractBackground(self, Range):
        """ Specify a range to generate
            an average background spectrum
            which will then be subtracted
            from the full 2D REMPI.
        """
        self.BackgroundSlice = IntegrateSpeed(Range)
        self.Background2D = np.array((self.BackgroundSlice,
                                      np.shape(self.Image)[1]),
                                     dtype=float)
        self.SubtractedImage = np.subtract(self.Image, self.Background2D)
        Show(self.SubtractedImage)

# Testing class when I was trying out dynamic creation of instances
class BlankTest:
    name = " "
    def __init__(self, FileName):
        self.name = FileName
     
###############################################################################################
###############################################################################################
###############################################################################################

############### Array Manipulation #################

def AddArrays(A, B):
    """
    Addition of B onto A elementwise, when B is smaller than A

    Returns A, the larger array with B added it it
    """
    ColumnSize = np.shape(B)[0]
    RowSize = np.shape(B)[1]
    for i in range(ColumnSize):
        for j in range(RowSize):
            A[i,j] = A[i,j] + B[i,j]
    return A

def Array2XYZ(Array):
    """ Function to convert a 2D array into
        a set of 1D X,Y,Z column vectors
        for plotting

        Don't think this is efficient or
        pythonic...

        First initialise the three vectors,
        then use the numpy iterator to iterate 
        over the array. The indices of each vector
        is tracked by a separate counter, which is 
        not the smartest way of doing this
    """
    FlatSize = np.shape(Array)[0] * np.shape(Array)[1]
    X = np.zeros((FlatSize))
    Y = np.zeros((FlatSize))
    Z = np.zeros((FlatSize))
    index_counter = 0
    Iterator = np.nditer(Array, flags=["multi_index"])
    while not Iterator.finished:
        Y[index_counter] = Iterator.multi_index[0]
        X[index_counter] = Iterator.multi_index[1]
        Z[index_counter] = Iterator[0]
        Iterator.iternext()
    return X,Y,Z

def SharpnessIndex(Image):
    """ Calculate the sharpness index as described by some paper
    """
def AverageArrays(Arrays):
    """
    Returns the average of a list of arrays by making a 3D array
    This is a good use of 3D arrays.
    """
    return np.mean(np.array(Arrays), axis=0)

def SimulateImage(NIons=100000, Dimension=699):
    """Simulates the pancaking of a 3D ion sphere into two dimensions

    Input:
    NIons - the number of ions to hit the fan (int)
    Dimension - the size of the square image (int)

    Returns:
    A simulated image (ndarray)
    """
    MaxR = 300 # set the maximum R from centre
    MinR = 0
    costhetamin = -1
    costhetamax = 1
    phimin = 0
    phimax = 2 * np.pi

    x0 = 350 # set the x-centre of the image
    y0 = 350 # set the y-centre of the image

    OutputImage = np.zeros((Dimension, Dimension), dtype=float)

    for Ion in range(NIons):
        r = np.random.randint(MinR,MaxR)
        costheta = costhetamin + np.random.random() * (costhetamax - costhetamin)
        phi = phimin + np.random.random() * (phimax - phimin)
        theta = np.arccos(costheta)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        CentredX = np.int(x0 + x)
        CentredY = np.int(y0 + y)
        if CentredX >= 0 and CentredX < Dimension:
            if CentredY >= 0 and CentredY < Dimension:
                OutputImage[CentredX, CentredY] += 1
                #print CentredX, CentredY
    return OutputImage

def ExtractSlice(Array, X):
    """ Extracts a slice of an array """
    return Array[:, X]

def SubtractImages(A, B):
    """ Subtracts two images A - B, where A and B are instances of the IonImage class

    Input:
    A - Instance of IonImage
    B - Instance of IonImage, will be subtracted from A

    Returns:
    Difference of A - B in ndarray
    """
    return A.Image - B.Image

###############################################################################################
###############################################################################################
###############################################################################################

############### Formatting and file I/O routines #################
# Function for reading a text image from file and converting to numpy array
def LoadImage(Path):
    Image = np.rot90(np.genfromtxt(Path, delimiter = DetectDelimiter(Path)), k=1)
    #if np.sum(np.isnan(Image)):                             # this checks if there are NaN in image    
    #    Image = np.genfromtxt(Path, delimiter="\s")     # np function generates array from text
    return Image

def Load2D(Path):
    """ Loads a 2D REMPI/PHOFEX image from file,
        returning the transpose of the resulting array
        since LabView works that way.
    """
    return np.loadtxt(Path).T

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
def DisplayImage(Image, ColourMap=cm.viridis):
    fig = plt.figure(figsize=(8,8))
    plt.imshow(Image)
    plt.set_cmap(ColourMap)              # set the colourmap
    plt.colorbar()                        # add intensity scale
#    plt.show()

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

# Function for exporting mass imported data as a dictionary that
# can be read in and written out. Keep in mind for inter-notebook
# "syncing"
def ShelveData(Shelf, Dictionary):
    ShelfReference = shelve.open(Shelf)
    ShelfReference.update(Dictionary)
    ShelfReference.close()

# Opens the disk data for I/O. Remember to close() it!
def CallShelfData(Shelf):
    ShelfReference = shelve.open(Shelf)
    return ShelfReference

# Save an image (reference as attribute of IonImage) by calling
# numpy save
def ExportImage(FileName, Image):
    """ Saves a ndarray as a text file

    Input:
    FileName - Path to file (string)
    Image - np.ndarray containing the image
    """
    np.savetxt(FileName, Image, delimiter="\t", fmt="%.4f")