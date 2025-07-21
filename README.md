

# CIFAR10 Classification: One-vs-All vs. Softmax
---
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
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc
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
├── data/
│ ├── cifar10_features.npy             # Feature file: 50,000 samples × 16 features
│ └── cifar10_labels.npy               # Label file: class numbers 0–9
├── mark_classification_cifar10.ipynb  # Main notebook with code, analysis, and results
├── README.md                          # Project documentation
```

---

## Results Summary

| Metric                         | One-vs-All             | Softmax |
| ------------------------------ | ---------------------- | ------- |
| Accuracy                       | \~96%                  |\~96.2%  |
| Runtime (seconds)              | 1.04s                  | 1.63s   |
| F1-macro                       | 0.961                  | 0.963   |
| Hardest class pair             | Class A vs. B          |   --    |
| Improvement after binary model | +4.1% in pair accuracy |  --     |

>  *Detailed results and plots are available inside the notebook.*

---

## Author

Developed by **Shlomi Asayag**

---

## License

This repository is for academic and educational use.
You may reuse code snippets with attribution.

```
