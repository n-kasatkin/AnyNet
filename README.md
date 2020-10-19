# Anytime Stereo Image Depth Estimation on Mobile Devices
Thanks [mileyan/AnyNet](https://github.com/mileyan/AnyNet) for their great work.

`predict.sh` has been added to generate disparities of two images and save them. Note that this model assume you to have **rectified images**.

The output of the model is disparities between two images. If you want to translate it into point cloud, see [generate_lidar.py](https://github.com/mileyan/pseudo_lidar/blob/master/preprocessing/generate_lidar.py).

This model has been tested on custom images set with resolution `600x1200` cropped to `576x1200`. Average time of prediction stage is `0.019s` on TeslaV100.
