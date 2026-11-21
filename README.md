# PUSL3123 Assessment 1: Exercise 1 (MLP on CIFAR-10)

`Exercise 1 MLP.ipynb` trains and evaluates a three-layer Multi-Layer Perceptron on CIFAR-10 with PyTorch. It uses a seeded 45,000 / 5,000 train/validation split and picks the optimiser on the validation set. The test set is evaluated once, at the end.

## Environment setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The versions in `requirements.txt` are the ones used to produce the saved outputs. On a CPU-only machine you can install the smaller CPU build of PyTorch first:
`pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu`.

## Running the notebook

- **Locally:** run `jupyter notebook "Exercise 1 MLP.ipynb"`, then use *Kernel > Restart Kernel and Run All*. You can also run it headless:
  `jupyter nbconvert --to notebook --execute --inplace "Exercise 1 MLP.ipynb"`
- **Google Colab:** upload the notebook, choose *Runtime > Change runtime type > GPU*, then *Runtime > Run all*. Colab already has the required packages.

CIFAR-10 (about 170 MB) is downloaded into `data/` on the first run. That folder is ignored by git.

## Expected runtime

The notebook trains for 60 epochs in total: 3 optimiser candidates for 10 epochs each, then the 30-epoch main run.

- **CPU (tested on a laptop):** about 8 minutes for the whole notebook (about 7 seconds per epoch), after the dataset has downloaded.
- **GPU (for example, a Colab T4):** about 5 to 10 minutes. With a model this small, data loading is the main bottleneck.

## Outputs

All outputs go to `outputs/`:

| File | Contents |
| --- | --- |
| `exercise1 config.json` | Seed, batch size, epochs, normalisation statistics, chosen optimiser, validation indices (reused by Exercises 2 to 4) |
| `exercise1 history.csv` | Per-epoch train/validation loss and accuracy, and the epoch time |
| `exercise1 metrics.json` | Final train/validation/test loss and accuracy, accuracy gap, best epoch, parameter count, per-class accuracy, top confused pairs |
| `exercise1 best model.pt` | `state_dict` of the epoch with the lowest validation loss |
| `figures/*.png` | Loss curves, accuracy curves, optimiser comparison, confusion matrix, sample images, class distribution (300 dpi) |
