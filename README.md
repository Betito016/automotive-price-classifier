#  Automotive Price Segment Classifier

Binary classification of used vehicles into **Economy** or **Premium** market segments using machine learning.  
Comparison of 6 supervised learning algorithms on a dataset of ~40,000 European used-car listings adapted to the Mexican market.

---

##  Overview

The used-car market suffers from significant information asymmetry between buyers and sellers. This project addresses that by building a model that automatically classifies a vehicle as **Economy (Class 0)** or **Premium (Class 1)** based solely on its technical characteristics — no subjective assessment required.

**Cut-off point:** the market median price (MXN $96,212.50), chosen over the mean to avoid distortion from luxury outliers.

---

##  Dataset

- **Source:** European used-car listings (eBay Kleinanzeigen)
- **Adaptation:** Currency converted to MXN (×21.5), labels translated to Spanish, outliers filtered
- **Final size:** ~40,000 vehicles after cleaning
- **Features used:** Brand, Vehicle Type, Transmission, Fuel Type, Registration Year, Mileage, Horsepower, Unrepaired Damage
- **Target:** Binary class (0 = Economy, 1 = Premium) split at the price median

---

##  Models Compared

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| **Random Forest** ✅ | **0.9249** | **0.9322** | 0.9172 | **0.9246** | **0.9808** |
| Gradient Boosting | 0.9235 | 0.9304 | 0.9163 | 0.9233 | 0.9797 |
| Neural Network | 0.9220 | 0.9280 | 0.9160 | 0.9219 | 0.9796 |
| Logistic Regression | 0.9092 | 0.9138 | 0.9048 | 0.9092 | 0.9670 |
| SVM | 0.9087 | 0.9194 | 0.8970 | 0.9081 | 0.9698 |
| LDA | 0.9069 | 0.9211 | 0.8910 | 0.9058 | 0.9644 |

**Random Forest selected** as the final model: best performance across all metrics simultaneously, interpretable feature importances, and computationally efficient.

---

##  Key Findings

- **Top predictors:** Registration Year > Horsepower > Mileage — vehicle age explains nearly 2× more than any other single feature
- **All models surpassed 90% accuracy**, confirming that the selected features carry genuine predictive power regardless of algorithm
- PCA analysis showed that 95% of variance is retained with far fewer components than the original 55-feature space

---

##  Repository Structure

```
automotive-classifier/
├── AP_P2_604775.ipynb   # Full notebook: EDA, PCA, 6 models, comparison, simulator
├── autos.csv            # Cleaned dataset (European listings adapted to MXN)
└── README.md
```

---

##  Tech Stack

- **Python 3** — pandas, numpy, matplotlib, seaborn
- **scikit-learn** — LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, LinearDiscriminantAnalysis, SVC, Pipeline, StandardScaler, PCA, cross_validate
- **Keras / TensorFlow** — Sequential Neural Network (4 layers, 17,537 parameters)
- **ipywidgets** — interactive classification simulator (bonus)
- **Google Colab** — development environment

---

##  How to Run

1. Clone the repo and open `AP_P2_604775.ipynb` in Google Colab or Jupyter
2. Upload `autos.csv` to `/content/`
3. Run all cells in order — each section is self-contained with comments

---

##  Interactive Simulator

The notebook includes a **bonus interactive widget** built with `ipywidgets`. Users can input vehicle specs (brand, year, mileage, horsepower, damage status) via sliders and dropdowns and receive an instant classification with confidence probability — no coding knowledge required.

---

##  Academic Context

Developed for *Inteligencia Artificial I* — 8th semester, Ingeniería en Tecnologías Computacionales  
Universidad de Monterrey (UDEM) · March 2026

---

##  Author

**Humberto Vargas Sánchez** · [@Betito016](https://github.com/Betito016) *(update with your actual handle)*
