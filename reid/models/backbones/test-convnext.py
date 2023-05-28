import torch
import torchvision.transforms as transforms
from PIL import Image
from Resnet50_roya import ResNet

"""This file is to test convnext for one RGB image to see the shape of the extracted feature vector """

# Load image
img = Image.open("/home/rdehghani/intra-inter-resnet/Person_ReIdentification/v2-reid/connext_without_Tnorm_AIBN/reid/models/backbones/6715.jpg")


#******************************************Transformation************************************
# Define transform to convert image to PyTorch tensor
transform = transforms.Compose([
    transforms.Resize((256, 128)),# Resize image to fit model input size
    transforms.ToTensor(),# Convert image to PyTorch tensor
])

# Apply transform to image
img_tensor = transform(img)

# Add an extra dimension to the tensor to simulate batch size of 1
img_tensor = img_tensor.unsqueeze(0)

#****************************************load convnext_tiny************************************
# Load model and pass image tensor as input
# my_conv1 = convnext_tiny()
# output = my_conv1(img_tensor)
# print(output.shape)


#********************************To load the resnet to see the shape od extracted vector****************************************
# model=ResNet()
#Load the state dict into the model
# model.load_param('/home/rdehghani/intra-inter-resnet/Person_ReIdentification/v2-reid/connext_without_Tnorm_AIBN/pretrained_weights/resnet50-0676ba61.pth')
# output=model(img_tensor)
# print(output.shape)
# torch.Size([1, 2048, 1, 1])
# x = output.view(output.size(0), output.size(1))
# print(x)






