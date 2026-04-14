# Digit Recognition with MLP (MNIST)

This project investigates the use of a Multilayer Perceptron (MLP) for handwritten **digit recognition** on the MNIST dataset, which consists of 60,000 training grayscale images.

The goal of this work is to systematically compare different hyperparameter configurations and training strategies through quantitative evaluation and visualization. The best-performing models are further assessed using standard performance metrics and integrated into a simple interactive interface.

All experiments are implemented in **Python** using the PyTorch framework.


---

## Repository Structure
```
src/
├── main.py
├── tests/
├── data/
└── plot/
```

---

### main.py

This file contains the core implementation of the neural network.

It allows configuration of key hyperparameters such as learning rate, batch size, and activation functions, as well as the integration of different training strategies.

Inside the `test()` function, it is possible to enable different evaluation modes, including:
- visualization of the confusion matrix
- display of random test samples
- analysis of misclassified samples
- computation of full evaluation statistics
- activation of a simple user-input interface (by uncommenting `draw_interface(model)`)

---

### tests/

This folder contains the code used to run the different experimental setups, including plotting utilities and evaluation scripts.

It also includes helper modules such as:
- `utils.py` (utility functions)
- `dataloader.py` (dataset loading)
