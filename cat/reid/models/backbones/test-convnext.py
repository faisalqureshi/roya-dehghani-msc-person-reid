import torch
import torchvision.transforms as transforms
from PIL import Image
from  ConvNeXt import ConvNeXt,convnext_xlarge,LayerNorm,Block
from Resnet50_roya import ResNet


# Load image
img = Image.open("/home/roya/foo/Person_ReIdentification/cat/reid/models/backbones/6715.jpg")

# Define transform to convert image to PyTorch tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),# Resize image to fit model input size
    #transforms.Resize((256, 128)),
    transforms.ToTensor(),          # Convert image to PyTorch tensor
])

# Apply transform to image
img_tensor = transform(img)

# Add an extra dimension to the tensor to simulate batch size of 1
img_tensor = img_tensor.unsqueeze(0)

# Load model and pass image tensor as input

my_conv1 = convnext_xlarge()
output = my_conv1(img_tensor)
print(output.shape)

# model=ResNet()
# model.load_param('/home/roya/foo/Person_ReIdentification/cat/pretrained_weights/resnet50-0676ba61.pth')
# # Load the state dict into the model
# print(model)
# #torch.Size([1, 2048, 1, 1])
# output=model(img_tensor)#which means that the module has reduced the spatial dimensions to 1x1 and increased the depth to 2048 channels
# print(output.shape)

