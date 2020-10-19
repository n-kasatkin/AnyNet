# Anytime Stereo Image Depth Estimation on Mobile Devices
Это копия репозитория [mileyan/AnyNet](https://github.com/mileyan/AnyNet).

Добавлен скрипт для генерации карт диспаратностей и сохранения их на диск.
Пример запуска скрипта см. в `predict.sh`.

The output of the model is disparities between two images. If you want to translate it into point cloud, see [generate_lidar.py](https://github.com/mileyan/pseudo_lidar/blob/master/preprocessing/generate_lidar.py).
