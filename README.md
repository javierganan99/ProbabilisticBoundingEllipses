# Efficient Event-based Intrusion Monitoring using Probabilistic Distributions

This repository is part of the additional material of the paper _Efficient Event-based Intrusion Monitoring using Probabilistic Distributions_.

> Autonomous intrusion monitoring in unstructured complex scenarios using aerial robots requires perception systems capable to deal with problems such as motion blur or changing lighting conditions, among others. Event cameras are neuromorphic sensors that capture per-pixel illumination changes, providing low latency and high dynamic range. This paper presents an efficient event-based processing scheme for intrusion detection and tracking onboard strict resource-constrained robots. The method tracks moving objects using a probabilistic distribution that is updated event by event, but the processing of each event involves few low-cost operations, enabling online execution on resource-constrained onboard computers. The method has been experimentally validated in several real scenarios under different lighting conditions, evidencing its accurate performance.

| ![Daylight outdoor scenario](https://github.com/javierganan99/ProbabilisticBoundingEllipses/blob/main/videos/method.gif) |     ![Indoor scenario](https://github.com/javierganan99/ProbabilisticBoundingEllipses/blob/main/videos/GIF_room.gif)     |
| :----------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------: |
|                                               _Daylight outdoor scenario_                                                |                                                    _Indoor scenario_                                                     |
| ![Pitch dark conditions](https://github.com/javierganan99/ProbabilisticBoundingEllipses/blob/main/videos/GIF_night.gif)  | ![General purpose tracking](https://github.com/javierganan99/ProbabilisticBoundingEllipses/blob/main/videos/GIF_cat.gif) |
|                                                 _Pitch dark conditions_                                                  |                                                _General purpose tracking_                                                |

## Citation

If you use this work in an academic context, please cite the following publications:

> F.J. Gañán, J.A. Sanchez-Diaz, R. Tapia, J.R. Martínez-de Dios, A. Ollero
> **Efficient Event-based Intrusion Monitoring using Probabilistic Distributions**,
> IEEE International Symposium on Safety, Security, and Rescue Robotics, 2022.

```bibtex
@inproceedings{ganan2022efficient,
  author={Gañán, F.J. and Sanchez-Diaz, J.A. and Tapia, R. and Martínez-de Dios, J.R. and Ollero, A.},
  booktitle={2022 IEEE International Symposium on Safety, Security, and Rescue Robotics},
  title={Efficient Event-based Intrusion Monitoring using Probabilistic Distributions},
  year={2022},
  doi={10.1109/SSRR56537.2022.10018655}
}
```

## License

Distributed under the GPLv3 License. See [`LICENSE`](https://github.com/javierganan99/ProbabilisticBoundingEllipses/tree/main/LICENSE) for more information.
