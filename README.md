![UTA-DataScience-Logo](https://github.com/user-attachments/assets/36b0607e-06da-485c-97a1-34a4f0552141)

# ICU Mortality and Readmission Prediction Capstone 1

* **IMPORTANT:** This project uses the publicly available eICU Collaborative Research Database Demo (v2.0.1) provided by © PhysioNet. Due to data use agreements and size constraints, raw source files are not redistributed here. Users must download the dataset directly from PhysioNet and follow the provided preprocessing and merging steps to reproduce the analysis

## OVERVIEW

This repository contains a machine learning project focused on predicting poor outcomes among Intensive Care Unit (ICU) patients using demographic, clinical, and physiological data from the PhysioNet eICU Collaborative Research Database. The goal of this project is to evaluate how class imbalance affects predictive performance and to compare modeling strategies under unbalanced, balanced, and threshold-optimized training conditions.

## SUMMARY OF WORK DONE

### Data

  * **Type:**
    * Input: Demographic, clinical, and physiological variables collected during ICU stays (age, gender, ethnicity, vital signs, laboratory measurements)
    * Output: Binary outcome variable (bad_outcome), where 0 indicates a good outcome and 1 indicates ICU mortality or readmission
  * **Source:**
    * PhysioNet eICU Collaborative Research Database (eICU-CRD) Demo v2.0.1
    * Data merged from multiple tables, including patient, apachepatientresult, apacheapsvar, apachepredsvar, intakeoutput, and vitalperiodic
  * **Size of Classes:**
    * 2,520 ICU patients total
    * 2,227 good outcomes (majority class)
    * 293 bad outcomes (minority class), indicating severe class imbalance
  * **Splits:**
    * 80% training and 20% validation with default threshold = 0.5
   
#### Compiling Data and Image Pre-processing

* **Data Collection and Cleaning:**
    * Merged multiple eICU tables into a single analytic dataset (mergeICU_db) containing patient-level records
    * Removed irrelevant, redundant, or non-informative features (e.g., aids, teachingstatus, numbedscategory)
    * Identified and corrected inappropriate data types (e.g., converting age from categorical to numeric)
    * Addressed clinically meaningful missing values by replacing placeholder values (–1) with NaN
    * Removed features containing post-outcome information to prevent data leakage (e.g., dischargeweight, acutephysiologyscore)
* **Pre-processing:**
* Applied median imputation for numerical variables to remain robust to outliers
* Filled missing categorical values with an explicit “Unknown” category to avoid fabricating medical data
* Created missingness indicator flags for variables with high proportions of missing values
* One-hot encoded categorical features such as ethnicity, gender, and admission source
* Removed highly multicollinear features using Variance Inflation Factor (VIF) analysis
* Generated both scaled and unscaled datasets to support different model requirements
 
#### Data Visualization

<img width="1119" height="372" alt="Screenshot 2026-01-11 at 3 53 22 PM" src="https://github.com/user-attachments/assets/4740b48f-0ab7-4f7b-9ac2-e090bf2ff2fe" />

### Problem Formulation

* **Models Used:**
  * Logistic Regression (baseline, interpretable model)
  * Support Vector Machine (SVM)
  * Random Forest
  * XGBoost
  * Stacked Ensemble Model (LR+XGB as base models, LR as meta learner)
 
### Training

Model training was conducted locally on a custom-built **Windows PC equipped with an AMD Ryzen 7 7700X CPU, RTX 4060 GPU, and 64 GB of DDR5 RAM**, using Jupyter Notebook. The training utilized TensorFlow/Keras along with key libraries such as numpy, matplotlib, and pandas.

**Challenges:** Severe class imbalance posed a major challenge, as standard accuracy metrics caused models to favor the majority class and overlook high-risk patients. Clinically meaningful missing values required careful preprocessing to avoid distorting medical information, while high multicollinearity among physiological variables complicated feature selection. Additionally, preventing data leakage necessitated removing post-outcome features that artificially improved performance. Throughout the project, balancing predictive performance with clinical interpretability and real-world prevalence remained a key challenge.

### Performance Comparison

#### **Classification Reports, Classification Matrices, and ROC-AUC Curves for XGBoost (Unbalanced) vs. Stacked (Balanced) Model**
<img width="696" height="195" alt="Screenshot 2026-01-11 at 3 56 43 PM" src="https://github.com/user-attachments/assets/6c461460-20e3-469f-a4fd-2efbbb14fdab" />

<img width="322" height="201" alt="Screenshot 2026-01-11 at 3 56 15 PM" src="https://github.com/user-attachments/assets/0b847cab-8b4f-4b8a-8945-13fa4ca7ebb0" />
<br>
<img width="466" height="188" alt="Screenshot 2026-01-11 at 3 56 04 PM" src="https://github.com/user-attachments/assets/1a5723e7-7acb-43e5-aa4a-e30f3900f172" />

#### **Best Model Interpretability and Demonstration**
<img width="280" height="167" alt="Screenshot 2026-01-11 at 3 58 03 PM" src="https://github.com/user-attachments/assets/04df05f0-f6ed-45b9-84ef-4bb532de8a94" />
<br>
<img width="742" height="410" alt="Screenshot 2026-01-11 at 3 58 51 PM" src="https://github.com/user-attachments/assets/4b6ffe71-ecd9-4353-a19a-230a2deb00f9" />

### Conclusions

**WHAT THIS MEANS:** This project demonstrates that predicting poor ICU outcomes is feasible using demographic, clinical, and physiological data, but model performance is highly dependent on how class imbalance is handled. While unbalanced training achieved high overall accuracy, it consistently failed to identify high-risk patients due to poor sensitivity for the minority class. Balanced training strategies improved recall, ROC-AUC, and PR-AUC, resulting in fewer false negatives and more reliable detection of patients at risk of mortality or readmission. Both balanced and unbalanced models identified clinically meaningful severity and organ function markers, supporting the medical relevance of the features. However, balanced training can distort real-world prevalence and inflate false positives, making it less suitable for probability estimation. Ultimately, the optimal modeling approach depends on the clinical goal: balanced training for early detection and screening, unbalanced training with threshold optimization for realistic risk estimation, and hybrid approaches for clinical decision support.

### Future Work

* Threshold optimization for unbalanced training
* Model fine-tuning
* Multiple iteration (common technique used in medical data)
* Improve stacked model architecture
* Expand feature engineering
* Dimensionality reduction (PCA)
* Create a simulation dashboard

## HOW TO REPRODUCE RESULTS

### Overview of Files in Repository

The list below follows the chronological order in which each component of the project was developed:

* **creating_mergeICU_db.ipynb** weee 

### Software Setup

This project was developed and executed in Google Colab Jupyter Notebook. If you don’t already have it installed, you can download it as part of the Anaconda distribution or install it via pip "pip install notebook".

* **Data Handling:** sqlite3, pandas, numpy
* **Visualization:** matplotlib.pyplot, seaborn
* **Statistics and Multicollinearity:** statsmodels.stats.outliers_influence.variance_inflation_factor, statsmodels.tools.tools.add_constant
* **Preprocessing and Pipelines:** sklearn.model_selection, sklearn.compose.ColumnTransformer, sklearn.preprocessing, sklearn.pipeline.Pipeline, sklearn.impute.SimpleImputer
* **Metrics and Evaluation:** sklearn.metrics – classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score, recall_score, precision_score, f1_score
* **Models:** sklearn.linear_model.LogisticRegression, sklearn.svm.SVC, sklearn.ensemble.RandomForestClassifier, StackingClassifier, xgboost.XGBClassifier, lightgbm.LGBMClassifier
* **Feature Importance:** sklearn.inspection.permutation_importance

### Data

* **Websites Used:**
    * **PhysioNet — eICU Collaborative Research Database (eICU-CRD) Demo v2.0.1:** https://physionet.org/content/eicu-crd-demo/2.0.1/
    * From here, you will download the "eicu_v2_0_1.sqlite3" file provided and create the data frame used for this pipeline
 ***For reference, see creating_mergeICU_db.ipynb***
 
### Training
* Install required packages in notebook
* Download and prepare the data using the creating_mergeICU_db.ipynb notebook as a guideline
* Train and test each model with default threshold=0.5 and Stratified k-fold cross-validation for all models
***For reference, see icu_pipeline_mnc.ipynb***

#### Performance Evaluation
* After training, model performance can be evaluated
***For reference, see icu_pipeline_mnc.ipynb***

## CITATIONS

[1] Armstrong, R. A., Kane, A. D., Kursumovic, E., Oglesby, F. C., & Cook, T. M. (2021). Mortality in patients admitted to intensive care with COVID-19: an updated systematic review and meta-analysis of observational studies. Anaesthesia, 76(4), 537–548. https://doi.org/10.1111/anae.15425

[2] Lai, J. I., Lin, H. Y., Lai, Y. C., Lin, P. C., Chang, S. C., & Tang, G. J. (2012). Readmission to the intensive care unit: a population-based approach. Journal of the Formosan Medical Association = Taiwan yi zhi, 111(9), 504–509. https://doi.org/10.1016/j.jfma.2011.06.012

