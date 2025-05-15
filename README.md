---

# CIFAR10 Classification: One-vs-All vs. Softmax

Repository: `mark_classification_cifar10`  
File: [`mark_classification_cifar10.ipynb`](https://github.com/shlomias1/mark_classification_cifar10/blob/main/mark_classification_cifar10.ipynb)

## Objective

This notebook addresses **Exercise 1** from a machine learning course, focusing on multi-class classification of the well-known **CIFAR10 dataset** using **logistic regression**.

The goal is to implement and compare two strategies:
- **One-vs-All (OvA)** classification
- **Softmax (multinomial logistic regression)**

Both methods are evaluated on classification accuracy, runtime, loss function value, F1 score, and confusion matrix analysis.

---

## Dataset Description

Two `.npy` files were provided:
- `X.npy`: 50,000 samples, each with 16 pre-extracted features (from CIFAR10 images)
- `y.npy`: Corresponding class labels (integers from 0 to 9)

More on the original dataset: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## Tasks Implemented

1. **Train Logistic Regression models** using:
   - `multi_class='ovr'`
   - `multi_class='multinomial'` with `solver='lbfgs'`
2. **Compare**:
   - Classification **accuracy**
   - **Runtime**
   - **Loss function** values (log-loss)
   - **F1-macro** score
3. For OvA:
   - **Confusion matrix analysis**
   - Detect hardest-to-distinguish class pairs
4. **Binary classifier** trained on most-confused class pair
   - Used to **refine predictions** of main model
   - Compare improvements

---

## Evaluation Function

A custom test function is implemented:

```python
def testmymodel(model, X, y):
    ...
````

This function receives a trained model along with `.npy` files and returns classification accuracy (and can include additional metrics as desired). It will be used for grading and external evaluation.

---

## ⚙Technologies Used

* Python 3.x
* NumPy
* scikit-learn
* matplotlib (for visualization)
* Jupyter Notebook (`.ipynb`)

---

## File Structure

```
.
├── mark_classification_cifar10.ipynb     # Main notebook with all code, analysis, and results
├── X.npy                                 # Feature file (provided)
├── y.npy                                 # Labels file (provided)
└── README.md                             # Project documentation
```

---

## Results Summary

| Metric                         | One-vs-All           | Softmax |
| ------------------------------ | -------------------- | ------- |
| Accuracy                       | \~XX.X%              | \~XX.X% |
| Runtime (seconds)              | X.XXs                | X.XXs   |
| F1-macro                       | XX.X                 | XX.X    |
| Hardest class pair             | Class A vs. B        | --      |
| Improvement after binary model | +X% in pair accuracy | --      |

> 💡 *Detailed results and plots are available inside the notebook.*

---

## Author

Developed by **Shlomi Asayag**
Course: Machine Learning – Classification Exercise #1
Institution: \[Your University or Department]

---

## License

This repository is for academic and educational use.
You may reuse code snippets with attribution.

```
