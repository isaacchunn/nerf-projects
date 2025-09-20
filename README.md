# NeRF Projects - Final Year Project

This repository contains implementations and experiments with various Neural Radiance Field (NeRF) methods for my Final Year Project. The project explores different approaches to neural scene representation and novel view synthesis.

## Project Structure

- **`nerf/`** - Original NeRF implementation
- **`plenoctree/`** - PlenOctrees for real-time NeRF rendering
- **`svox2/`** - Plenoxels implementation (radiance fields without neural networks)

## Methods Overview

### NeRF (Neural Radiance Fields)
The original neural radiance fields method that represents scenes as continuous volumetric functions using multilayer perceptrons.

### PlenOctrees
An acceleration structure for NeRF that enables real-time rendering by using octree-based spatial representations.

### Plenoxels
A method that achieves competitive results to NeRF without using neural networks, instead using sparse voxel grids with spherical harmonics.

## Citations

If you use any of these methods in your work, please cite the original papers:

### NeRF (Neural Radiance Fields)
**Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020).** NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. *European Conference on Computer Vision (ECCV)*.

```bibtex
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Mildenhall, Ben and Srinivasan, Pratul P and Tancik, Matthew and Barron, Jonathan T and Ramamoorthi, Ravi and Ng, Ren},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

### PlenOctrees
**Yu, A., Li, R., Tancik, M., Li, H., Ng, R., & Kanazawa, A. (2021).** PlenOctrees for Real-time Rendering of Neural Radiance Fields. *IEEE/CVF International Conference on Computer Vision (ICCV)*.

```bibtex
@inproceedings{yu2021plenoctrees,
  title={PlenOctrees for Real-time Rendering of Neural Radiance Fields},
  author={Yu, Alex and Li, Ruilong and Tancik, Matthew and Li, Hao and Ng, Ren and Kanazawa, Angjoo},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

### Plenoxels
**Yu, A., Fridovich-Keil, S., Tancik, M., Chen, Q., Recht, B., & Kanazawa, A. (2022).** Plenoxels: Radiance Fields without Neural Networks. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

```bibtex
@inproceedings{yu2022plenoxels,
  title={Plenoxels: Radiance Fields without Neural Networks},
  author={Yu, Alex and Fridovich-Keil, Sara and Tancik, Matthew and Chen, Qinhong and Recht, Benjamin and Kanazawa, Angjoo},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
