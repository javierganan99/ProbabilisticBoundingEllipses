# ProbabilisticBoundingEllipses
 This repo is part of the Master Thesis of Francisco Javier Gañán Onieva for the Máster Universitario en Lógica, Computación e Inteligencia Artificial of Universidad de Sevilla, named **Efficient Event-based Moving Object Localisation and Tracking using Probabilistic Distributions**.

## Event-by-event Probabilistic Tracking
| ![Daylight outdoor scenario](https://github.com/javierganan99/ProbabilisticBoundingEllipses/blob/main/videos/method.gif) | ![Indoor scenario](https://github.com/javierganan99/ProbabilisticBoundingEllipses/blob/main/videos/GIF_room.gif) |
|:--:|:--:|
| *Daylight outdoor scenario* | *Indoor scenario* |
| ![Pitch dark conditions](https://github.com/javierganan99/ProbabilisticBoundingEllipses/blob/main/videos/GIF_night.gif) | ![General purpose tracking](https://github.com/javierganan99/ProbabilisticBoundingEllipses/blob/main/videos/GIF_cat.gif) |
| *Pitch dark conditions* | *General purpose tracking* |

The method tracks moving objects using a probabilistic distribution that is updated event by event, but the processing of each event involves few low-cost operations, enabling online execution on resource-constrained onboard computers. The method has been experimentally validated in several real scenarios under different lighting conditions, evidencing its accurate performance, and some of the recorded bag files are available in the bag folder of this respository.

The content of this source code is detailed in the Master Thesis.

This code has been tested in Python 3.8.10 version. Yolov5 Pytorch Version publicly available in https://github.com/ultralytics/yolov5 has been adapted and included in this repo for comparison purposes.

In order to download the bag folder to test the method click in the following link: https://drive.google.com/file/d/1VV7J2cGZaovW0RXHPXAffqXKBp1-Yw4m/view?usp=sharing
