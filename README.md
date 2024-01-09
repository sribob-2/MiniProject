# D7041E, Applied Artificial Intelligence-miniproject-group 19

Srinivas Bobba(sribob-2@student.ltu.se)

## Project :Machine Learning Approach for Early Breast Cancer Detection: Insights from Wisconsin Diagnostic Dataset, Coimbra Dataset and Mammographic Mass Dataset

YouTube link: https://youtu.be/RUoWwP-kR3w

## Project Introduction
*The objective is to predict malignancy presence based on the features, with a focus on evaluating the performance of machine learning models in breast cancer detection utilizing Wisconsin (Diagnostic), Coimbra, and mammographic mass data.

*Machine learning approach is important here because it can analyze complex patterns in the  datasets and helps create models that make breast cancer detection more accurate and efficient using smart algorithms.

## Data Collection
The Datasets(Vectorized) below are collected from the UCI machine learning repository which is accessible to the public for research purposes.
Breast Cancer Wisconsin(Diagnostic)  Dataset available at : https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
Breast Cancer Coimbra Dataset  avalibale at :
https://archive.ics.uci.edu/dataset/451/breast+cancer+coimbra
Breast Cancer Mammographic Mass Dataset available at :
https://archive.ics.uci.edu/dataset/161/mammographic+mass

## Methodology
Building Models using supervised Machine learning Algorithms
Logistic Regression: Linear model for binary classification. 
Decision Trees: Non-linear model for classification and regression. 
Random Forest: Ensemble learning method using multiple decision trees. 
k-Nearest Neighbors (KNN): Instance-based learning algorithm. 
Support Vector Machines (SVM): Linear and non-linear classification algorithm. 
Multi-Layer Perceptron (MLP):Feedforward neural network with multiple layers.
 Artificial Neural Networks (ANN):General term for interconnected nodes arranged in layers.
 Convolutional Neural Networks (CNN):Specialized neural network architecture for processing grid-like data.
## Model Evaluation
The Performance metrics are chosen for our analysis because due to  imbalanced datasets
1. Sensitivity (Recall): True positive rate.
2. Specificity: True negative rate.
3. F1 Score: Harmonic mean of precision and recall.
4. Accuracy: Overall correctness.
5. AUC (Area Under the Curve): Discrimination ability.

 ## Implementation:
![image](https://github.com/sribob-2/MiniProject/assets/154096358/ce1f75ce-0075-4015-b21c-40ba306ae330)

## Experiment-1: Breast Cancer Prediction Analysis on Wisconsin Dataset
https://github.com/sribob-2/MiniProject/blob/main/D7041E_MiniProject_BreastCancer_WisconsinData.ipynb

Data Preprocessing:
1. Feature Grouping:
   - Grouped features into 'worst,' 'mean,' and 'se' categories.

2. Diagnostic Result Visualization:
   - Plotted histograms for each feature group by diagnostic result.

3. Correlation Analysis:
   - Calculated correlation matrix for all features (excluding 'id' and 'diagnosis').

4. Feature Filtering:
   - Removed highly correlated features (correlation >= 0.9).

5. Final Feature Selection:
   - Selected a subset of 20 features for further analysis.

Data Transformation (Wisconsin Dataset):
1. Standardization
   - Scaled numerical features using `StandardScaler`.

2. PCA Dimensionality Reduction:
   - Applied PCA to reduce dimensions to 10 components.
   - Selected features explaining over 95% of variance.

3. Visualization:
   - Visualized data distribution and confirmed linearity.

4. Feature Selection:
   - Identified most important features in each principal component.

5. New Dataset:
   - Created a 569x11 dataset for  final analysis.

6. Resulting Data:
- 10 principal components and a 'Response' column.
- 'Response' indicates cancer (1) or non-cancer (0).
Result : 
If our primary concern is accuracy, Decision Tree, Random Forest, k-Neighbors, and Support Vector Machine all achieved perfect training accuracy. However, training accuracy alone might not represent the model's generalization to new data.

Additionally, it's crucial to consider metrics like recall, precision, and specificity, especially in medical applications where false negatives (missing cancer cases) can be critical. Based on these metrics, Support Vector Machine and k-Neighbors seem to have high recall and specificity.

Based on this analysis,the Logestic regression, Support Vector Machine and k-Neighbors models seem to be strong contenders for cancer prediction. 

## Experiment-2: Breast Cancer Prediction Analysis on Coimbra Dataset
https://github.com/sribob-2/MiniProject/blob/main/D7041E_MiniProject_BreastCancer_CoimbraData.ipynb
Data Preprocessing:
Feature Identification:
   - Originally, the classes were represented as 1 and 2. we converted with a more conventional representation, where 0 often signifies one class (e.g., benign), and 1 signifies another class (e.g., malignant).
2. Diagnostic Result Visualization:
   - Plotted histograms for each feature group by diagnostic result.

3. Correlation Analysis:
   - Calculated correlation matrix for all features (excluding 'id' and 'diagnosis').

4. Feature Filtering:
   - Removed highly correlated features (correlation >= 0.9).

5. Final Feature Selection:
   - Selected all 10 features for further analysis.

Data Transformation (Coimbra Dataset):
1. Standardization
   - Scaled numerical features using `StandardScaler`.

2. PCA Dimensionality Reduction:
   - Applied PCA to reduce dimensions.  No apparent Clustering.

3. Visualization:
   - Visualized data distribution and confirmed linearity.

4. Feature Selection:
   - Identified most important features(Variable Importance) using Extra tree classifier.

Result :
Logistic Regression: Training Accuracy: 75.86% Test Metrics: AUC: 0.864 Recall: 0.929 Specificity: 0.800 Analysis: The Logistic Regression model shows decent performance on the test set. It has a good AUC and recall, indicating its ability to discriminate between classes. However, specificity could be improved.

Decision Tree Classifier: Training Accuracy: 100.0% Test Metrics: AUC: 0.695 Recall: 0.857 Specificity: 0.533 Analysis: The Decision Tree Classifier achieves perfect accuracy on the training set but shows some overfitting on the test set. The AUC is moderate, and while recall is good, specificity is relatively low.

Random Forest Classifier: Training Accuracy: 97.70% Test Metrics: AUC: 0.764 Recall: 0.929 Specificity: 0.600 Analysis: The Random Forest Classifier demonstrates good generalization with high accuracy on the training set. It performs well on the test set, with a high AUC and recall, but specificity could be improved.

k-Nearest Neighbors: Training Accuracy: 100.0% Test Metrics: AUC: 0.793 Recall: 0.786 Specificity: 0.800 Analysis: The k-Nearest Neighbors model achieves perfect training accuracy but shows some variation on the test set. It performs reasonably well, with good AUC and specificity.

Support Vector Machine: Training Accuracy: 81.61% Test Metrics: AUC: 0.831 Recall: 0.929 Specificity: 0.733 Analysis: The Support Vector Machine model demonstrates good performance on the test set, with a decent AUC and recall. Specificity is also at an acceptable level.

Neural Network Models: Artificial Neural Network (ANN):

Test Metrics: Accuracy: 59% Precision, Recall, F1-score: Varying Analysis: The ANN shows relatively low accuracy and mixed precision-recall performance.

Convolutional Neural Network (CNN):
Test Metrics: Accuracy: 79% Precision, Recall, F1-score: Balanced Analysis: The CNN performs well with balanced accuracy, precision, recall, and F1-score.

Based on the above results,We observed that 
**Decision Tree and k-Nearest Neighbors models exhibit potential overfitting due to perfect training accuracy.**
**Random Forest and Support Vector Machine models show good generalization.**
**Neural network models, especially the CNN, demonstrate competitive performance, outperforming the ANN.**

## Experiment-3: Breast Cancer Prediction Analysis on Mammographic Mass
https://github.com/sribob-2/MiniProject/blob/main/D7041E_MiniProject_BreastCancer_Mammographic%20Mass-final.ipynb

Data Preprocessing:
Impute Function:
  - Converts a column to numeric, replacing non-numeric values with the median.

2. Replace Function:
  - Replaces outliers in a numeric column with an upper limit.

3. ZNorm Function:
  - Z-normalizes numeric values in a column.

4. Decode Function:
  - Decodes numeric values into categorical names.

5. Consolidate Function:
  - Consolidates categorical variables in a given column.

6. OneHotEncode Function:
   - One-hot encodes categorical data, creating binary values in a new column based on a specific categorical variable.

Data Transformation (Mammographic Mass Dataset):

1.Dataset Splitting:
  - Utilized `train_test_split` from `sklearn.model_selection` to split data into training and test sets (75% training, 25% test).

2.Feature Preprocessing:
  - Used `SimpleImputer` for missing values (mean for numeric, most frequent for categorical).
  - Applied `StandardScaler` for numeric features to standardize them.

3. Column Transformation:
  - Used `ColumnTransformer` to independently preprocess numeric and categorical features.

4.Pipeline for Transformation:
  - Constructed pipelines for numeric and categorical transformations.

5. One-Hot Encoding:
  - Applied `OneHotEncoder` to handle categorical variables, ignoring unknown values.

6. Feature Scaling:
  - Standardized features using `StandardScaler` independently for training and test sets.
7. Feature Selection:
   - Identified most important features(Variable Importance) using Extra tree classifier.
Result :
Logistic Regression:

Training Accuracy: 83.89%
Testing AUC: 81.8%
Recall: 78.8%
Specificity: 84.7%
The Logistic Regression model demonstrates a good balance between sensitivity and specificity, making it suitable for binary classification tasks. The AUC value of 81.8% indicates a decent discriminatory power.

Decision Tree Classifier:

Training Accuracy: 94.44%
Testing AUC: 79.3%
Recall: 76.9%
Specificity: 81.8%
The Decision Tree Classifier achieves high training accuracy, suggesting it might be prone to overfitting. The AUC value of 79.3% on testing data indicates good but not excellent discriminatory performance.

Random Forest Classifier:

Training Accuracy: 93.19%
Testing AUC: 81.1%
Recall: 79.8%
Specificity: 82.5%
Random Forest shows robust performance with a good balance between sensitivity and specificity. The ensemble nature helps mitigate overfitting observed in individual decision trees.

k-Nearest Neighbors:

Training Accuracy: 94.44%
Testing AUC: 81.5%
Recall: 79.8%
Specificity: 83.2%
The k-Nearest Neighbors model demonstrates high accuracy and a competitive AUC, making it a reliable choice for classification tasks.

Support Vector Machine (SVM):

Training Accuracy: 85.97%
Testing AUC: 82.8%
Recall: 81.7%
Specificity: 83.9%
SVM achieves good accuracy and AUC, indicating its effectiveness in discriminating between classes. The balance between sensitivity and specificity is notable.

Artificial Neural Network (ANN):

Accuracy: 82%
Precision: 82%
Recall: 82%
The ANN model achieves a balanced accuracy and precision-recall values, indicating its capability for binary classification tasks.

Convolutional Neural Network (CNN):

Accuracy: 82%
Precision: 82%
Recall: 82%
The CNN model exhibits performance similar to the ANN, suggesting that the convolutional layers may not significantly improve the model's performance in this context.

Feature Importance:

Among the features, "BI-RADS" and "Density" appear to be the most important predictors according to their respective importance scores.
The ensemble models (Random Forest, k-Nearest Neighbors) outperform individual models, showcasing the power of combining multiple learners.
Support Vector Machine and Logistic Regression provide a good balance between sensitivity and specificity.
Neural networks (ANN and CNN) offer competitive performance, but their interpretability may be limited compared to traditional machine learning models.

## Refection on Experiment-1 (Wisconsin Dataset)
Hypothesis testing suggested that the mean radius of benign tumors is less than malignant tumors. 
Features like ['radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean','concavity_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'smoothness_se', 'compactness_se', concavity_se', 'concave points_se', 'symmetry_se','fractal_dimension_se', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'symmetry_worst', 'fractal_dimension_worst'],,' were identified as indicative of tumor malignancy. 
Machine learning algorithms (Logistic Regression, Decision Tree, Random Forest, k-Neighbors, SVM, MLP, CNN, ANN) were employed. 
High sensitivity (minimizing false negatives) was crucial.
Recommendations: 
 For high sensitivity: Logistic regression, Support Vector Machine and k-Neighbors models seem to be strong contenders for cancer prediction. 

## Refelection on Experiment-2 (Coimbra Dataset)
Features like Glucose, Age, Resistin, BMI, and Insulin were indicative of tumor nature. 
Machine learning algorithms (Decision Tree, Random Forest, k-Neighbors, SVM, MLP, CNN, ANN) were applied. 
Observations: 
Based on the results, We observed that 
Decision Tree and k-Nearest Neighbors models exhibit potential overfitting due to perfect training accuracy.
Random Forest and Support Vector Machine models show good generalization.
Neural network models, especially the CNN, demonstrate competitive performance, outperforming the ANN.
## Reflection on Experiment-3 (Mammography Mass Dataset)
Features such as BI-RADS, Density and Age were important. 
Machine learning algorithms (Logistic Regression, Decision Tree, Random Forest, k-Neighbors, SVM, MLP, CNN, ANN) were employed. 
Observations.
The ensemble models (Random Forest, k-Nearest Neighbors) outperform individual models, showcasing the power of combining multiple learners.
Support Vector Machine and Logistic Regression provide a good balance between sensitivity and specificity.
Neural networks (ANN and CNN) offer competitive performance, but their interpretability may be limited compared to traditional machine learning models
## General Recommendations-Technical Perspective!
Consider the task-specific requirements for model selection. 
Prioritize high sensitivity for early detection in breast cancer diagnosis. 
Evaluate the trade-off between model complexity and interpretability. 
Ensemble models, such as Random Forest, can offer improved performance. 
Regularly validate models on independent datasets to ensure generalizability. 
Collaborate with  medical experts for a more comprehensive understanding of feature importance and model outputs.

## General Recommendations- Medical Perspective
Regular Monitoring:
 - Monitor individuals with malignancy-indicative features regularly.
Focused Assessments:
 - Assess 'Glucose', 'Age', 'Resistin', 'BMI', and 'Insulin' for tumor nature.
Tailored Screening:
- Customize breast cancer screenings based on 'BI-RADS', 'Density', and 'Age'.
Patient Education:
- Educate patients about feature significance and encourage regular check-ups.
Consult healthcare professionals for personalized advice
