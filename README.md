# PUSL3123 Assessment 1: CIFAR-10 Set Exercises

PyTorch notebooks for the CIFAR-10 set exercises. Each exercise is a separate notebook:

| Notebook | Exercise | What it does |
| --- | --- | --- |
| `Exercise 1 MLP.ipynb` | 1: Multi-Layer Perceptron | Trains a three-layer MLP, picks the optimiser on the validation set, and saves the shared experiment configuration. |
| `Exercise 2 CNN.ipynb` | 2: Convolutional Neural Network | Trains a three-layer CNN under the same conditions and compares it with the MLP (accuracy, training/validation loss, parameter count). |
| `Exercise 3 Regularisation.ipynb` | 3: Regularisation and generalisation | Evaluates dropout and data augmentation, each independently against the Exercise 2 CNN baseline, plus both combined. |
| `Exercise 4 Width and Depth.ipynb` | 4: Model width and depth | Trains three CNN widths and three CNN depths and compares accuracy, training loss, overfitting, parameters, and compute (MACs). |

All notebooks use the standard CIFAR-10 partitions: the 50,000 training images are split into a seeded 45,000 / 5,000 train/validation split, and the 10,000 test images are used only for final evaluation. They are never used for training, optimiser selection, or checkpoint selection.

## Environment setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The versions in `requirements.txt` are the ones used to produce the saved outputs. On a CPU-only machine you can install the smaller CPU build of PyTorch first:
`pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu`.

## Running the notebooks

**Run them in order.** Each notebook reads files saved by the one before it:

1. `Exercise 1 MLP.ipynb` saves `outputs/exercise1 config.json` (seed, batch size, epochs, normalisation statistics, validation indices) and the MLP metrics and history.
2. `Exercise 2 CNN.ipynb` reads that configuration and the MLP results, and saves `outputs/exercise2 config.json` (the CNN's selected optimiser).
3. `Exercise 3 Regularisation.ipynb` reads both configuration files, so its baseline is trained exactly like the Exercise 2 CNN.
4. `Exercise 4 Width and Depth.ipynb` reads the same two files. Exercises 3 and 4 do not depend on each other and can be run in either order.

If a required file is missing, the notebook stops with a message naming the notebook to run first.

- **Locally:** open each notebook with `jupyter notebook`, then use *Kernel > Restart Kernel and Run All*. You can also run them headless:

  ```bash
  jupyter nbconvert --to notebook --execute --inplace "Exercise 1 MLP.ipynb"
  jupyter nbconvert --to notebook --execute --inplace "Exercise 2 CNN.ipynb"
  jupyter nbconvert --to notebook --execute --inplace "Exercise 3 Regularisation.ipynb"
  jupyter nbconvert --to notebook --execute --inplace "Exercise 4 Width and Depth.ipynb"
  ```

- **Google Colab:** upload the notebooks, choose *Runtime > Change runtime type > GPU*, and run them in the order above in the same session, so the `outputs/` files are shared between them.

CIFAR-10 (about 170 MB) is downloaded into `data/` on the first run. That folder is ignored by git.

## Experimental setup

Settings shared by all four exercises (from `exercise1 config.json`): seed 42, batch size 128, cross-entropy loss, 30 training epochs, the same 5,000 validation indices and normalisation statistics. In every run the checkpoint with the lowest validation loss is kept and evaluated.

- **Exercise 1 (MLP):** `Flatten -> Linear(3072, 512) -> ReLU -> Linear(512, 256) -> ReLU -> Linear(256, 10)`, 1,707,274 parameters. Three optimisers (Adam lr=1e-3, Adam lr=1e-4, SGD lr=0.01 momentum=0.9) are each trained for 10 epochs, and the one with the lowest validation loss is used for the main run.
- **Exercise 2 (CNN):** `Conv(3,32,3)-ReLU-MaxPool -> Conv(32,64,3)-ReLU-MaxPool -> Conv(64,10,3) -> global average pooling`, 25,162 parameters. The final convolution has no ReLU because its pooled outputs are the class logits. The optimiser is chosen with the same procedure and candidates as Exercise 1.
- **Exercise 3 (regularisation):** the Exercise 2 CNN and optimiser, trained four times from the same initialisation and batch order:
  - baseline;
  - `Dropout2d(p=0.2)` after the first two convolutional blocks;
  - data augmentation: an independent random translation of up to ±4 px, a horizontal flip (p=0.5) and a rotation of up to ±10° for every training image, applied on the GPU;
  - dropout and augmentation combined.

  Training accuracy is always measured on the clean, unaugmented training images.
- **Exercise 4 (width and depth):** six variations of the Exercise 2 CNN, trained with its optimiser from the same initialisation seed and batch order. Each is a stack of 3×3 convolutions with ReLU, max pooling, a final 10-channel convolution and global average pooling. The baseline (W2 = D2) is trained once and reused for both experiments.
  - **Widths** (3 conv layers, kernels per hidden layer increasing): W1 16→32, W2 32→64, W3 64→128.
  - **Depths** (baseline channel scale): D1 2 conv layers (32), D2 3 conv layers (32→64), D3 5 conv layers (32→32→64→64, two convolutions per pooling stage).
  - **Complexity** is reported as trainable parameters, multiply–accumulate operations (MACs) per image (counted analytically from the layer shapes), and median epoch time.

## Results (saved outputs)

| Model | Train acc. (%) | Val acc. (%) | Test acc. (%) | Train − test gap (pp) | Parameters |
| --- | --- | --- | --- | --- | --- |
| MLP (Exercise 1) | 66.05 | 52.72 | 53.64 | +12.41 | 1,707,274 |
| CNN (Exercise 2 = Exercise 3 baseline) | 70.51 | 68.22 | 68.17 | +2.34 | 25,162 |
| CNN + dropout (p=0.2) | 62.12 | 61.24 | 60.90 | +1.22 | 25,162 |
| CNN + data augmentation | 65.82 | 64.84 | 65.58 | +0.24 | 25,162 |
| CNN + dropout + augmentation | 57.65 | 58.00 | 57.78 | −0.13 | 25,162 |

Exercise 4 (W2 and D2 are the same baseline model as Exercise 2):

| Variation | Conv layers | Hidden kernels | Parameters | MACs / image | Train loss | Val acc. (%) | Test acc. (%) | Train − test gap (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1 narrow | 3 | 16-32 | 7,978 | 1.81 M | 0.973 | 64.60 | 64.48 | +1.55 |
| W2 baseline | 3 | 32-64 | 25,162 | 5.97 M | 0.850 | 68.22 | 68.17 | +2.34 |
| W3 wide | 3 | 64-128 | 87,178 | 21.38 M | 0.762 | 70.70 | 69.98 | +3.48 |
| D1 shallow | 2 | 32 | 3,786 | 1.62 M | 1.352 | 52.20 | 51.49 | +1.38 |
| D2 baseline | 3 | 32-64 | 25,162 | 5.97 M | 0.850 | 68.22 | 68.17 | +2.34 |
| D3 deep | 5 | 32-32-64-64 | 71,338 | 24.85 M | 0.516 | 76.08 | 76.30 | +6.06 |

Both the MLP and the CNN selected SGD (lr=0.01, momentum=0.9) on the validation set. All results come from a single seed (42). With `torch.backends.cudnn.deterministic = True`, re-running on the same hardware reproduces these numbers. Different GPUs or a CPU may differ slightly in the last decimal places.

## Expected runtime

Measured on a laptop with an NVIDIA RTX 4050 GPU, after the dataset has downloaded:

| Notebook | Epochs trained | GPU time |
| --- | --- | --- |
| Exercise 1 | 60 (3 × 10 optimiser selection + 30 main run) | about 5 minutes |
| Exercise 2 | 60 (3 × 10 optimiser selection + 30 main run) | about 5 minutes |
| Exercise 3 | 120 (4 runs × 30) | about 8 to 10 minutes |
| Exercise 4 | 150 (5 distinct runs × 30; the baseline is reused) | about 8 to 10 minutes |

All four notebooks run on a CPU without changes. The CNN notebooks are much slower there; an earlier CPU run of the CNN took about 30 seconds per epoch, so expect roughly 30 minutes for Exercise 2 and over an hour each for Exercises 3 and 4 (the wide and deep models in Exercise 4 are the slowest).

## Outputs

All outputs go to `outputs/`. Figures are saved at 300 dpi in `outputs/figures/`.

### Exercise 1 outputs

| File | Contents |
| --- | --- |
| `exercise1 config.json` | Seed, batch size, epochs, normalisation statistics, chosen optimiser, validation indices (reused by Exercises 2 to 4) |
| `exercise1 history.csv` | Per-epoch train/validation loss and accuracy, and the epoch time |
| `exercise1 metrics.json` | Final train/validation/test loss and accuracy, accuracy gap, best epoch, parameter count, per-class accuracy, top confused pairs |
| `exercise1 best model.pt` | `state_dict` of the epoch with the lowest validation loss |
| `figures/exercise1 *.png` | Loss curves, accuracy curves, optimiser comparison, confusion matrix, sample images, class distribution |

### Exercise 2 outputs

| File | Contents |
| --- | --- |
| `exercise2 config.json` | CNN architecture and the optimiser selected on validation (read by Exercises 3 and 4) |
| `exercise2 history.csv` | Per-epoch train/validation loss and accuracy, and the epoch time |
| `exercise2 metrics.json` | Final train/validation/test loss and accuracy, accuracy gap, best epoch, parameter count, optimiser, training time, per-class test accuracy |
| `exercise2 best model.pt` | `state_dict` of the epoch with the lowest validation loss |
| `exercise1 vs exercise2 comparison.csv` | MLP and CNN side by side: accuracies, losses, gap, best epoch, parameter count |
| `figures/exercise2 optimiser comparison.png` | Validation loss of the three optimiser candidates |
| `figures/exercise2 training curves.png` | CNN loss and accuracy curves |
| `figures/exercise2 confusion matrix.png` | CNN test-set confusion matrix |
| `figures/exercise1 vs exercise2 comparison.png` | Test accuracy and parameter count (log scale) bar charts |
| `figures/exercise1 vs exercise2 loss curves.png` | Training and validation loss curves of the MLP and the CNN |

### Exercise 3 outputs

| File | Contents |
| --- | --- |
| `exercise3 comparison.csv` | One row per configuration: dropout rate, augmentation, best epoch, train/validation/test loss and accuracy, train − test gap, parameter count |
| `exercise3 metrics.json` | The same results in full, plus the train − validation gap and training time per run |
| `exercise3 history.csv` | Per-epoch train/validation loss and accuracy for all four runs |
| `exercise3 per_class_accuracy.csv` | Per-class test accuracy for each run, and the change caused by augmentation and by dropout |
| `exercise3 baseline model.pt`, `exercise3 dropout model.pt`, `exercise3 augmentation model.pt`, `exercise3 combined model.pt` | `state_dict` of each run's lowest-validation-loss epoch |
| `figures/exercise3 augmentation samples.png` | Original training images next to augmented versions |
| `figures/exercise3 training curves.png` | Training/validation loss and accuracy of all four runs |
| `figures/exercise3 comparison.png` | Train/validation/test accuracy bars and the train − test gap per configuration |

### Exercise 4 outputs

| File | Contents |
| --- | --- |
| `exercise4 comparison.csv` | One row per variation: conv layers, hidden kernels, parameters, MACs per image, best epoch, train/validation/test loss and accuracy, train − test gap, median epoch time |
| `exercise4 metrics.json` | The same results in full, plus the final-epoch training loss, train − validation gap, and mean/median/total training time per variation |
| `exercise4 history.csv` | Per-epoch train/validation loss and accuracy for all six variations |
| `exercise4 per_class_accuracy.csv` | Per-class test accuracy for each variation, and the W3 − W1 and D3 − D1 gains |
| `exercise4 width1_narrow model.pt`, `exercise4 width2_baseline model.pt`, `exercise4 width3_wide model.pt`, `exercise4 depth1_shallow model.pt`, `exercise4 depth2_baseline model.pt`, `exercise4 depth3_deep model.pt` | `state_dict` of each variation's lowest-validation-loss epoch |
| `figures/exercise4 training curves.png` | Training loss and validation accuracy per epoch for the width and depth experiments |
| `figures/exercise4 width depth comparison.png` | Validation accuracy, test accuracy and training loss against the three widths and the three depths |
| `figures/exercise4 complexity tradeoff.png` | Validation accuracy and train − test gap against parameter count, and MACs per image with median epoch time |
