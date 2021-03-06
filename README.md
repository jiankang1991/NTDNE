﻿# Noise-Tolerant Deep Neighborhood Embedding for Remotely Sensed Images with Label Noise  

[Jian Kang](https://github.com/jiankang1991), [Ruben Fernandez-Beltran](https://scholar.google.es/citations?user=pdzJmcQAAAAJ&hl=es),  [Xudong Kang](http://xudongkang.weebly.com/), [Jingen Ni](https://scholar.google.com/citations?hl=en&user=hqZB5wQAAAAJ&view_op=list_works), [Antonio Plaza](https://www.umbc.edu/rssipl/people/aplaza/) 

This repo contains the main codes for the JSTARS paper: [Noise-Tolerant Deep Neighborhood Embedding for Remotely Sensed Images with Label Noise](https://ieeexplore.ieee.org/document/9345336) We develop a new loss function called noise-tolerant deep neighborhood embedding (NTDNE) which can accurately encode the semantic relationships among RS scenes. Specifically, we target at maximizing the leave-one-out K-NN score for uncovering the inherent neighborhood structure among the images in feature space. Moreover, we down-weight the contribution of potential noisy images by learning their localized structure and pruning the images with low leave-oneout K-NN scores.

## Usage

`./train/main.py` is the training script for NTDNE.

`utils/metrics.py`contains the NTDNE loss implementation.

## Citation

```
@article{kang2021NTNDE,
  title={{Noise-Tolerant Deep Neighborhood Embedding for Remotely Sensed Images with Label Noise}},
  author={Kang, Jian and Fernandez-Beltran, Ruben and Kang, Xudong and Ni, Jingen and Plaza, Antonio},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2021},
  note={DOI:10.1109/JSTARS.2021.3056661}
  publisher={IEEE}
}
```
