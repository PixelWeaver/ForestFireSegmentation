# Forest Fire Segmentation

With the advent of climate change comes the fear wildfires will become a rising concern in the near future as is hinted by several environmental studies. This fear has
already become a reality for some parts of the globe.

This work implements and compares different deep learning architectures for flame semantic segmentation on RGB images from the Corsican Fire Database.

Results are compared in terms of the intersection over union (IoU), the mean squared error (MSE), the binary accuracy and
the recall metrics as well as their number of network parameters. 

The implemented architectures are:
* FLAME U-Net
* DeepLabv3+ with ResNet-50 backbone
* DeepLabv3+ with EfficientNet-B4 backbone
* Squeeze U-Net 
* ATT Squeeze U-Net

![Results](/figures/prediction_plate.png)
