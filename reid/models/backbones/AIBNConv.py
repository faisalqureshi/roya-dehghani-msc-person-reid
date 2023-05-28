import math
import torch
from torch import nn
from .AIBN import AIBNorm2d
from .TNorm import TNorm
import numpy as np
import torch
import io
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


class AIBNBlock(nn.Module):
    
    def __init__(self, dim, resitual_path=0., layer_scale_init_value=1e-6, 
                 adaptive_weight=None,#new
                 generate_weight=True,#new
                 init_weight=0.1,#new
                
                 ):
        super(AIBNBlock, self).__init__()#new
        if adaptive_weight is None:#new
            self.adaptive_weight = nn.Parameter(torch.ones(1) * init_weight)#new
            
        
        #initializes a depthwise convolution with kernel size 7 and padding 3.
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        #self.norm = LayerNorm(dim, eps=1e-6)
        
        #use AIBN normalizaton
        self.aibnNorm= AIBNorm2d(dim,#new
                             adaptive_weight=self.adaptive_weight,#new
                             generate_weight=generate_weight)#new
        
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()# This line initializes the activation function. GELU (Gaussian Error Linear Units) is a non-linear activation function that is used in some deep learning models
        self.pwconv2 = nn.Linear(4 * dim, dim)# initializes a pointwise convolution (1x1 convolution) with 4 times the output channels as input channel
       
        #This line initializes a scaling factor parameter for the block. If layer_scale_init_value is greater than 0, the scaling factor is initialized with that value multiplied by
        # a tensor of ones with dim elements. Otherwise, the scaling factor is set to None.
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        
        
        self.resitual_path = DropPath(resitual_path) if resitual_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # #This line permutes the dimensions of the output tensor of the depthwise convolution to be in the form of (N, H, W, C)
        #x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)#new
        #x = self.norm(x)# This line applies layer normalization to the output tensor
        
        #because the LayerNorm takes the input shape with (N, H, W, C), we change th eshape of input to (N, C, H, W)
        #for aibnNorm
        x=self.aibnNorm(x)#new
        
         
        #the input shape for Pointwise conv should be (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)#This line applies the first pointwise convolution to the output tensor.
        x = self.act(x)#This line applies the activation function to the output tensor.
        x = self.pwconv2(x)#This line applies the second pointwise convolution to the output tensor.
        
        # This line applies the scaling factor parameter to the output tensor if it is not None.
        if self.gamma is not None:
            x = self.gamma * x
        
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)#new
        #residual connection
        x = input + self.resitual_path(x)
        return x


    
    
class ConvNeXt(nn.Module):
    
    def __init__(self, in_chans=3,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], resitual_path_rate=0., 
                 layer_scale_init_value=1e-6,
                 init_weight=0.1,#new
                 adaptive_weight=None#new
                 #domain_number=1#Tnorm
                
                 
                 ):
        super().__init__()
#**************************************************Encoder*********************************************
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        #self.domain_number = domain_number
        
        #stem
        stem = nn.Sequential(
            #4*4, 96, stride 4
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
        
            #LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            AIBNorm2d(dims[0],adaptive_weight=adaptive_weight,#new
                             generate_weight=True)#new
            
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    
                    AIBNorm2d(dims[i],
                    adaptive_weight=adaptive_weight,
                    generate_weight=True,
                    init_weight=init_weight),
                    #LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
            #self.tnorm1 = TNorm(dims[i], domain_number)#TNorm


        #4 stages
        self.stages = nn.ModuleList() #to store all stages
        
        # 4 stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, resitual_path_rate, sum(depths))] 
        cur = 0
        
        for i in range(4):
            stage = nn.Sequential(
                #call AIBNBlock class 
                *[ AIBNBlock(dim=dims[i], resitual_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value,
                 adaptive_weight=None,#new
                 generate_weight=True,#new
                 init_weight=0.1#new
                
                
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            # the cur variable is updated to keep track of the current position in the dp_rates list.
            cur += depths[i]


       
        #final layers
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        
#********************************************************************************************      

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        
        #global average pooling, (N, C, H, W) -> (N, C)  by taking the mean along the height and width dimensions
        #for capturing global information from spatially structured data
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        
        #change the shape of extracted feature vector to use it in out network
        # adds two extra dimensions of size 1 to the tensor x
        x = x.unsqueeze(-1).unsqueeze(-1)
        #x = self.head(x)
        return x
    
    #add by roya 
    def load_param(self, model_path):
            param_dict = torch.load(model_path)
            for k, v in param_dict.items():
                if k in self.state_dict().keys():
                    self.state_dict()[k].copy_(v)    



model_urls = {
  
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth"
}



"""
@register_model is a decorator used to register a model function with PyTorch's hub module.
The hub module is a pre-trained model repository maintained by PyTorch, where users can find and use pre-trained models for various tasks.
By registering a model function with @register_model, users can load the model easily using torch.hub.load()
function with a model name as an argument.
"""

@register_model
def convnext_small(pretrained,in_22k=True, **kwargs):
    #change the default values of input parameters to work in our method correctly
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        #add strict=False to avoid arrors because we removed decoder part
        model.load_state_dict(checkpoint["model"], strict=False)
       
    return model



