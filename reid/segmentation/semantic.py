from PIL import Image, ImageOps, ImageDraw
import cv2
import numpy
import sys
import os

maskDir="/home/rdehghani/intra-inter-resnet/Person_ReIdentification/v2-reid/connext_without_Tnorm_AIBN/segmentation/static/masks/"
numpy.set_printoptions(threshold=sys.maxsize)
def generateMask(imgPath,fileName):
    #print(imgPath)
    #print(fileName)
    #exit(0)
    # Load the input image and the binary segmentation mask
    img = cv2.imread(imgPath)
    mask = cv2.imread(maskDir+fileName, cv2.IMREAD_GRAYSCALE)

    # Resize the mask to match the input image size
    # mask_img = mask_img.resize(input_img.size)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    # Save the result as a PNG image
    # masked_img.save()
    
    cv2.imwrite('/home/rdehghani/data/market1501/images/'+fileName,masked_img)






def listFiles(dir):
    folder = os.listdir(dir)
    for fileName in folder:
        imgPath = dir+"/"+fileName
        generateMask(imgPath,fileName)
        
    print("finish")
               

listFiles("/home/rdehghani/intra-inter-resnet/Person_ReIdentification/v2-reid/connext_without_Tnorm_AIBN/segmentation/static/results")


