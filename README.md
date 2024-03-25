# Physics Informed Contour Selection for Rapid Image Segmentation
PICS combines Physics Informed Neural Networks (PINNs) and an active contour model called Snake to help human experts create 3D image segmentations rapidly. The link of the paper is: TBA . The current repository contains representative MATLAB codes to understand the methodology in the original paper. Please keep in mind that PICS is still under development (effect of number of control knots, position of initialization, etc has not been studied) and could be improved further (inverse parameter estimation). 

# Citation
If you use PICS, please cite our work:
```
﻿@Article{Dwivedi2024,
author={Dwivedi, Vikas
and Srinivasan, Balaji
and Krishnamurthi, Ganapathy},
title={Physics informed contour selection for rapid image segmentation},
journal={Scientific Reports},
year={2024},
month={Mar},
day={24},
volume={14},
number={1},
pages={6996},
abstract={Effective training of deep image segmentation models is challenging due to the need for abundant, high-quality annotations. To facilitate image annotation, we introduce Physics Informed Contour Selection (PICS)---an interpretable, physics-informed algorithm for rapid image segmentation without relying on labeled data. PICS draws inspiration from physics-informed neural networks (PINNs) and an active contour model called snake. It is fast and computationally lightweight because it employs cubic splines instead of a deep neural network as a basis function. Its training parameters are physically interpretable because they directly represent control knots of the segmentation curve. Traditional snakes involve minimization of the edge-based loss functionals by deriving the Euler--Lagrange equation followed by its numerical solution. However, PICS directly minimizes the loss functional, bypassing the Euler Lagrange equations. It is the first snake variant to minimize a region-based loss function instead of traditional edge-based loss functions. PICS uniquely models the three-dimensional (3D) segmentation process with an unsteady partial differential equation (PDE), which allows accelerated segmentation via transfer learning. To demonstrate its effectiveness, we apply PICS for 3D segmentation of the left ventricle on a publicly available cardiac dataset. We also demonstrate PICS's capacity to encode the prior shape information as a loss term by proposing a new convexity-preserving loss term for left ventricle. Overall, PICS presents several novelties in network architecture, transfer learning, and physics-inspired losses for image segmentation, thereby showing promising outcomes and potential for further refinement.},
issn={2045-2322},
doi={10.1038/s41598-024-57281-x},
url={https://doi.org/10.1038/s41598-024-57281-x}
}

```
