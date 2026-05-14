# Floor Plan Segmentation Benchmark

Comparing segmentation architectures for multi-class floor plan parsing on the [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) dataset. We benchmark CubiCasa's original hourglass pipeline (ResNet-152) against U-Net and SegFormer to evaluate how different backbone designs handle the unique challenges of architectural drawings: thin walls, overlapping annotations, and a long tail of rare element classes.

**Team:** Barath Ganesh, Devyani Gopal Toshniwal, Shaun Ting

## Models

**CubiCasa Hourglass (baseline):** The original CubiCasa5K pipeline, built on a ResNet-152 hourglass architecture with multi-task heads for room segmentation, icon segmentation, and heatmap regression. Uses uncertainty-based loss weighting to balance tasks. See [Kalervo et al., 2019](https://arxiv.org/abs/1904.01920).

**U-Net:** Encoder-decoder architecture with skip connections, using a ResNet-50 backbone. Trained on the same CubiCasa5K segmentation targets.

**SegFormer:** Transformer-based segmentation model that replaces convolutions with a hierarchical vision transformer encoder and a lightweight MLP decoder.

## Dataset

[CubiCasa5K](https://zenodo.org/record/2613548) contains 5,000 floor plan images annotated into 80+ object categories (rooms, walls, doors, windows, icons). The dataset is split into 4,200 training, 400 validation, and 400 test images.

Download the dataset and extract it to the `data/` folder. The dataset is not included in this repository.

## Setup

```bash
# Clone the repo
git clone git@github.com:Shaunting/floorplan-segmentation-benchmark.git
cd floorplan-segmentation-benchmark

# Install dependencies
pip install -r requirements.txt
```

### Database creation

The CubiCasa pipeline uses an LMDB database for faster data loading during training (~105 GB):

```bash
python create_lmdb.py --txt val.txt
python create_lmdb.py --txt test.txt
python create_lmdb.py --txt train.txt
```

## Training

```bash
python train.py
```

Training options can be found in the script. Logs are saved to `runs/` and can be visualized with TensorBoard:

```bash
tensorboard --logdir runs/
```

## Evaluation

```bash
python eval.py --weights <path_to_weights>
```

## Acknowledgments

This project builds heavily on the [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) codebase and dataset by Kalervo et al. (2019), licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). We use their data pipeline, preprocessing, and post-processing components. Our contributions are the U-Net and SegFormer integration and the comparative analysis.

## References

Kalervo, A., Ylioinas, J., Häikiö, M., Karhu, A., & Kannala, J. (2019). CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis. *arXiv:1904.01920*.

Liu, C., Wu, J., Kohli, P., & Furukawa, Y. (2017). Raster-to-Vector: Revisiting Floorplan Transformation. *ICCV*.

Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J.M., & Luo, P. (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. *NeurIPS*.

## License

The CubiCasa5K dataset and original pipeline code are licensed under CC BY-NC 4.0. Our additions to the codebase are available under the MIT License.
