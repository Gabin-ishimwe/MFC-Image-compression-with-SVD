'''
Objective:
SVD for image compression example


Prerequisites:
pip install numpy
pip install pillow

'''


import numpy
from PIL import Image


# FUNCTION DEFINTIONS:

# open the image and return 3 matrices, each corresponding to one channel (R, G and B channels)
# decompose the image into 3 matrice
def openImage(imagePath):
    imOrig = Image.open(imagePath)  # opens the image
    im = numpy.array(imOrig)        # represent our image in a form of numpy arrays

    aRed = im[:, :, 0]
    aGreen = im[:, :, 1]
    aBlue = im[:, :, 2]

    return [aRed, aGreen, aBlue, imOrig]


# compress the matrix of a single channel
def compressSingleChannel(channelDataMatrix, singularValuesLimit):

    # this numpy function perfom SVD on those channels
    uChannel, sChannel, vhChannel = numpy.linalg.svd(channelDataMatrix)
    aChannelCompressed = numpy.zeros((channelDataMatrix.shape[0], channelDataMatrix.shape[1]))
    k = singularValuesLimit

    leftSide = numpy.matmul(uChannel[:, 0:k], numpy.diag(sChannel)[0:k, 0:k])
    aChannelCompressedInner = numpy.matmul(leftSide, vhChannel[0:k, :])
    aChannelCompressed = aChannelCompressedInner.astype('uint8')
    return aChannelCompressed


# MAIN PROGRAM:
print('*** Image Compression using SVD - a demo')
aRed, aGreen, aBlue, originalImage = openImage('lena.png')

# image width and height:
imageWidth = 512
imageHeight = 512

# number of singular values to use for reconstructing the compressed image
singularValuesLimit = 1

# compress each color image component by singularvaluesLimit K
aRedCompressed = compressSingleChannel(aRed, singularValuesLimit)
aGreenCompressed = compressSingleChannel(aGreen, singularValuesLimit)
aBlueCompressed = compressSingleChannel(aBlue, singularValuesLimit)

# changes the compressed channels arrays into images
imr = Image.fromarray(aRedCompressed, mode=None)
img = Image.fromarray(aGreenCompressed, mode=None)
imb = Image.fromarray(aBlueCompressed, mode=None)

# merges all the three components of the image into one new image
newImage = Image.merge("RGB", (imr, img, imb))

# prints the original image
originalImage.show()
# prints the new compressed image
newImage.show()

# CALCULATE AND DISPLAY THE COMPRESSION RATIO
mr = imageHeight
mc = imageWidth

# calculate the data size held the original image
originalSize = mr * mc * 3

# calculate the data size held by the compressed image
compressedSize = singularValuesLimit * (1 + mr + mc) * 3

print('original size:')
print(originalSize)

print('compressed size:')
print(compressedSize)

# prints the ratio of data held by the compressed image
print('Ratio compressed size / original size:')
ratio = compressedSize * 1.0 / originalSize
print(ratio)

print('Compressed image size is ' + str(round(ratio * 100, 2)) + '% of the original image ')
print('DONE - Compressed the image! Over and out!')