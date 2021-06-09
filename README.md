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

## Results
Architecture | recall | IoU | accuracy | MSE | # parameters
------------ | -------|-----|----------|-----|-------------
FLAME U-Net | 0.94 | 0.892 | 0.943 | 0.043 | 2M
DLV3+ w/ ResNet50 | 0.968 | 0.926 | 0.962 | 0.031 | 40M
DLV3+ w/ EfficientNetB4 | 0.967 | 0.93 | 0.964 | 0.028 | 22M
Squeeze U-Net | 0.930 | 0.897 | 0.946 | 0.042 | 2.5M
ATT Squeeze U-Net | 0.928 | 0.893 | 0.944 | 0.042 | 885K

![Results](/figures/prediction_plate.png | width=500)
