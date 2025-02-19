# Anomaly-Detection-research-project
# Deep Learning for Network Anomaly Detection


## Overview

This project explores the use of deep learning techniques for enhancing network anomaly detection in supervised datasets.  Traditional methods often struggle with novel or complex threats. This research proposes a novel framework that leverages deep learning to capture intricate patterns and relationships within network traffic data to identify anomalous behavior indicative of potential attacks. The goal is to improve the accuracy and efficiency of attack detection, leading to more robust and adaptive security systems.

This README provides details on the methodology, implementation, results, and how to reproduce the findings. It's based on research towards improving cybersecurity defense mechanisms against complex digital cyberattacks.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Related Work](#related-work)
3.  [Methodology](#methodology)
    *   [Data Acquisition & Preparation](#data-acquisition--preparation)
    *   [Feature Selection](#feature-selection)
    *   [Model Training & Evaluation](#model-training--evaluation)
4.  [Implemented Models](#implemented-models)
5.  [Dataset](#dataset)
6.  [Results](#results)  (To be added after experimentation)
7.  [Usage](#usage)
    *   [Dependencies](#dependencies)
    *   [Installation](#installation)
    *   [Running the Code](#running-the-code)
8.  [Future Work](#future-work)
9.  [Contributing](#contributing)
10. [License](#license)
11. [Acknowledgements](#acknowledgements)
12. [Contact](#contact)

## 1. Introduction

The increasing reliance on digital technologies makes networks and systems vulnerable to a wide range of cyber threats. Anomaly detection is a crucial component of cybersecurity, but conventional approaches often struggle to identify sophisticated and evolving threats.  This project investigates the application of deep learning to improve attack detection models using supervised datasets, focusing on real-world data characteristics like imbalanced class representation and noise.  The aim is to develop a framework that can effectively identify anomalous behavior and enhance overall cybersecurity defenses.

## 2. Related Work

In the ever-evolving digital landscape, cybersecurity faces significant challenges due to the constant adaptation of cyber-attacks. Traditional cybersecurity measures often fall short in protecting against complex digital cyberattacks. Machine learning (ML) approaches, particularly anomaly detection techniques, have gained attention as valuable tools for strengthening cybersecurity defenses. However, the effectiveness of these techniques in real-world datasets can be compromised by factors such as imbalanced class representation, noise, and moving concepts.
This project aims to address these challenges by leveraging guided machine learning and deep learning to enhance attack discovery models in real-world datasets, as well as the development of a new approach.

## 3. Methodology

This project follows a structured methodology for identifying network attacks.  The key steps are:

*   **Data Acquisition and Preparation:** Collecting and preparing the network traffic data.
*   **Feature Selection:** Selecting the most relevant features for anomaly detection.
*   **Model Training and Evaluation:** Training and evaluating machine learning models to identify attacks.

![Proposed Methodology](link_to_your_diagram_image.png)  *(Replace `link_to_your_diagram_image.png` with a link to a diagram of your methodology.  You can create one and host it on GitHub or elsewhere.)*

### Data Acquisition & Preparation

1.  **Raw Data Acquisition:** Obtain raw network traffic data from sources like network connection logs and intrusion detection systems.  A well-designed supervised dataset with labeled instances (normal vs. malicious activity) is used.
2.  **Data Cleaning:**  Remove noise, handle missing values, and correct inconsistencies in the raw data to improve model performance.
3.  **Data Preprocessing:** Transform the data into a suitable format for deep learning algorithms:
    *   **Normalization:** Scale numerical features to a standard range to ensure equal contribution.
    *   **Encoding:** Convert categorical features into numerical values (e.g., using one-hot encoding).
    *   **Feature Engineering:** Create new, relevant features from existing data to enhance model performance.
4.  **Data Splitting:** Divide the preprocessed data into training, validation, and testing sets:
    *   **Training Set:** Used to train the deep learning models.
    *   **Validation Set:** Used to tune hyperparameters and prevent overfitting.
    *   **Testing Set:** Used to evaluate the final model's performance.

### Feature Selection

1.  **PCA (Principal Component Analysis):** Apply PCA to reduce dimensionality by identifying the principal components (features) that explain most of the variance in the data.
2.  **Scree Plot:** Use a scree plot to visualize the variance explained by each principal component, helping to determine the optimal number of components to retain.

### Model Training & Evaluation

1.  **Model Training:** Train the specified machine learning models using the training data.
2.  **Model Comparison:** Train and test multiple models to identify the most effective one.
3.  **Cross-Validation:** Perform cross-validation on the best-performing model to ensure its generalizability to unseen data.
4.  **Prediction:** Use the best-performing model to make predictions on unseen data.

## 4. Implemented Models

The following machine learning models were implemented and evaluated:

*   **Decision Tree Classifier:** A non-parametric model that partitions the feature space based on decision rules.
*   **Random Forest Classifier:** An ensemble learning technique that combines multiple decision trees to improve accuracy and robustness.
*   **GaussianNB Classifier:** A probabilistic classifier based on Bayes' theorem, assuming feature independence.
*   **LGBM Classifier:** (LightGBM) A gradient boosting framework known for its efficiency and accuracy.
*   **CatBoost Classifier:** A gradient boosting algorithm that excels at handling categorical features.

(Add more details about the specific deep learning architecture you used – e.g., CNN, RNN, Autoencoder, etc.  Include information on the layers, activation functions, loss function, and optimizer.)

## 5. Dataset

The dataset used in this project contains network traffic data with labeled instances of normal and malicious activities. It consists of 84 columns and 555278 rows, including a timestamp indicating the recording time of the data.

*   **Description:** The dataset provides a multifaceted insight into various metrics such as flow duration, packet counts, and window sizes.
*   **Features:** The columns offer a wide array of network traffic measurements encompassing packet exchanges, data packet dimensions, and flow durations.
*   **Statistical Measures:** Each column includes statistical measurements such as count, mean, standard deviation, minimum, 25th percentile, median, 75th percentile, and maximum.
*   **Missing Values:** The dataset does not appear to have any missing or null values, as each column has all of the statistical measures.

(Provide a link to download the dataset, if publicly available.  If it's not publicly available, describe how to obtain it or create a similar dataset.  Clearly state any data usage restrictions.)

## 6. Results

*   Accuracy, precision, recall, and F1-score for each model.
*   ROC curves and AUC values.
*   Comparison of deep learning model performance with traditional ML models.
*   Analysis of the effectiveness of feature selection techniques.

Analyzed five supervised learning models: Decision Tree Classifier, Random Forest Classifier, Gaussian NB, LGBM Classifier, and CatBoost Classifier, with experimental outcomes. An methodology was developed for each model by combining the best encoding methods, PCA (Principal Component Analysis) for dimension reduction, optimal feature selection, and different hyperparameter tuning strategies. The Decision Tree Classifier, Random Forest Classifier, and CatBoost Classifier have been identified as the models with the highest accuracy and performance; they are the top three out of five, and their accuracy is fairly similar. Nonetheless, the Random Forest Classifier achieved the highest accuracy score of 99.97%. These three top-performing models underwent cross-validation using the k-fold approach (k = 5). Random Forest Classifier was found to be underperforming, as it was on the left in the first, fourth, and fifth folds when compared to the other two. However, during the five folds, the Decision Tree Classifier and CatBoost Classifier consistently produced the same results.

Analyzed, an exploration was also undertaken, deep learning models like ANN, CNN and RNN. In addition, the accuracy of RNN was higher from rest two with 99.95% while model processing time was higher in respect to RNN whose accuracy was 99.95% which was Execution Time:
453.9617456730002 seconds in respect to ANN which was Execution Time: 273.9491973869999 seconds with accuracy was 99.94%.

*  Precision Scores: All models have very high precision rates, namely 1.00 and 0.99, 1.00, 0.99, and 1.00 for Decision Tree Classifier, RandomForest Classifier, LGBM Classifier, and Cat Boost Classifier. However, the GaussianNB model is less accurate as its recall for class 0 is only 0.98.
* Recall Scores: The Decision Tree Classifier, RandomForestClassifier, LGBM Classifier, and Cat Boost Classifier also have very good Recall (1.00) for both groups of Class 0 and Class 1. The GaussianNB model achieved a higher level of recall for Class 1 and a lower level for Class 0 which are 0.88 and 0.81 respectively.
* F1-Score Comparison: The F1-Score, which integrates Precision and Recall, exhibits a nearly identical trend. The Decision Tree Classifier, RandomForestClassifier, LGBM Classifier, and Cat Boost Classifier have F1- Scores of 1.00, 1.00, 1.00, and 1.00 for Class 0 respectively and 0.99, 1.00, 0.99, and 1.00 for Class 1.
Nevertheless, the GaussianNB model has a somewhat lower F1-Score of 0.93 for Class 0 and 0.46 for Class 1.
* Overall Performance: According to the Precision, Recall, and F1-Score metrics, the Decision Tree Classifier, the RandomForestClassifier, the Light Gradient Boosting Machine (LGBM) Classifier, and the Cat Boost Classifier seem to have almost the same and pretty good classification abilities,while the GaussianNB model seems to have the least efficient performance, especially for Class
1.

## 7. Usage

### Dependencies

List the required Python libraries and their versions.  Example:

*   Python 3.8+
*   TensorFlow 2.5+
*   Scikit-learn 0.24+
*   Pandas 1.2+
*   NumPy 1.20+
*   Matplotlib 3.4+
*   Seaborn 0.11+

