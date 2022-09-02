# ProbabilisticBoundingEllipses
 This repo is part of the Master Thesis of Francisco Javier Gañán Onieva for the Máster Universitario en Lógica, Computación e Inteligencia Artificial of Universidad de Sevilla, named **Application of machine learning techniques in event-based vision**.

## Illustration of event-by-event Probabilistic Tracking of the Method
![](https://github.com/javierganan99/ProbabilisticBoundingEllipses/blob/main/method.gif)

The method tracks a single moving-object using a probabilistic distribution that is updated event by event, but the processing of each event involves few low-cost operations, enabling online execution on resource-constrained onboard computers. The method has been experimentally validated in several real scenarios under different lighting conditions, evidencing its accurate performance, and some of the recorded bag files are available in the bag folder of this respository.

The content of this source code is detailed in the Master Thesis.

This code has been tested in Python 3.8.10 version. Yolov5 Pytorch Version publicly available in https://github.com/ultralytics/yolov5 has been adapted and included in this repo for comparison purposes.
