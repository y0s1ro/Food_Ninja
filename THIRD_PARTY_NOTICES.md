# Third-Party Notices

This repository contains or vendors third-party components used to build the school canteen food-waste prototype.

## FoodSAM

- Project: FoodSAM: Any Food Segmentation
- Upstream repository: https://github.com/jamesjg/FoodSAM
- Authors: Xing Lan, Jiayi Lyu, Hanyu Jiang, Kun Dong, Zehai Niu, Yi Zhang, Jian Xue
- License: Apache License 2.0
- Local usage: semantic food segmentation pipeline and related supporting assets under the `FoodSAM/` directory, with project-specific integration into the canteen capture workflow.

## Additional bundled upstream components

The vendored FoodSAM codebase also includes or depends on bundled upstream components in this repository, including:

- `mmseg/` from OpenMMLab mmsegmentation
- `UNIDET/` and bundled `UNIDET/detectron2/` components

These components retain their respective copyright and license notices in their source trees where present.

## License handling

- The top-level `LICENSE` file in this repository is Apache License 2.0.
- Upstream FoodSAM is published under Apache License 2.0.
- Original copyright and attribution notices in vendored source files should be preserved.

If you redistribute this repository, review the bundled third-party directories and preserve any applicable license and attribution notices from the upstream projects.