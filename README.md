# Hound: Locating Cryptographic Primitives in Desynchronized Side-Channel Traces Using Deep Learning

Hound is a tool for locating cryptographic primitives in desynchronized side-channel traces using deep learning. This repository contains the code and resources needed to run experiments and demos.

<div align="center">
  <img src="./images/hound_logo.png" alt="Hound Logo" width="110">
</div>
<br>

To run an experiment, open the Jupyter Notebook `Hound.ipynb`.

## Repository Organization

The repository is organized as follows:

- **`/CNN`**: Contains modules and configuration files for the Convolutional Neural Network used for classifying the start of cryptographic primitives.
- **`/inference_pipeline`**: Contains functions for classifying, segmenting, and aligning cryptographic primitives in a side-channel trace.
- **`Hound.ipynb`**: A Jupyter Notebook for running a demo.

```plaintext
.
├── CNN
│   ├── configs
│   │   ├── common
│   │   │   └── neptune_configs.yaml
│   │   └── exp
│   │       ├── data.yaml
│   │       ├── experiment.yaml
│   │       └── module.yaml
│   ├── datasets
│   │   └── cp_class_dataset.py
│   ├── models
│   │   ├── custom_layers.py
│   │   ├── resnet.py
│   │   └── resnet_time_series_classifier.py
│   ├── modules
│   │   ├── cp_class_datamodule.py
│   │   └── cp_class_module.py
│   ├── train.py
│   └── utils
│       ├── data.py
│       ├── logging.py
│       ├── module.py
│       ├── trainer.py
│       └── utils.py
├── inference_pipeline
│   ├── alignment.py
│   ├── debug.py
│   ├── screening.py
│   └── sliding_window_classification.py
└── Hound.ipynb
```

## Dataset

The dataset for the AES cryptosystem is available [here]() on Zenodo.

The dataset is organized as follows:

- **`/training`**: Contains three subsets: `train`, `valid`, and `test`. Each subset consists of two `.npy` files:
  - **`_set`**: Contains the preprocessed side-channel traces.
  - **`_labels`**: Contains the target labels for training the CNN, labeling each data as `CP start`, `CP spare`, or `noise`.
- **`/inference`**: Contains files for two demos: consecutive AES executions and AES executions interleaved with noisy applications. Each demo consists of two `.npy` files:
  - **`aes_`**: Contains the side-channel traces to input into Hound.
  - **`gt_`**: Contains the ground truth for checking the correctness of Hound segmentation.

## Note

This work is part of [1], available [online](https://arxiv.org/abs/2408.06296).

This repository is protected by copyright and licensed under the [Apache-2.0 license](https://github.com/hardware-fab/Hound/blob/main/LICENSE).

© 2024 hardware-fab

> [1] Galli, D., Chiari, G., and Zoni, D. (2024). Hound: Locating Cryptographic Primitives in Desynchronized Side-Channel Traces Using Deep-Learning. Accepted for presentation at the 42nd IEEE International Conference on Computer Design (ICCD). arXiv preprint arXiv:2408.06296.