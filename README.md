# Brain_Tumor_Detection
### Introduction

This notebook explores the task of classifying brain tumors from MRI images. It utilizes a Convolutional Neural Network (CNN) as the primary model, employing data augmentation techniques to improve robustness and address potential data limitations. Furthermore, the performance of the CNN is compared against several traditional machine learning models (SVM, Random Forest, and XGBoost) trained on features extracted from the CNN, providing insights into the effectiveness of different approaches for this image classification problem. The following steps were taken:

###### Data Loading and Exploration: 
Downloads a brain tumor MRI dataset from Kaggle using kagglehub, explores the class distribution, and visualizes sample images.
###### Data Preprocessing and Augmentation:
Uses ImageDataGenerator for normalizing and augmenting the training images. It also calculates class weights to handle potential class imbalance.
###### CNN Model Building and Training: 
Constructs a Convolutional Neural Network (CNN) model using Keras, compiles it, and trains it on the augmented data with callbacks for early stopping and learning rate reduction.
###### CNN Model Evaluation: 
Evaluates the trained CNN model using a confusion matrix, classification report, and AUC-ROC score. It also visualizes the model architecture.
###### Comparison with Traditional ML Models: 
Extracts features from the trained CNN and uses them to train and evaluate Support Vector Machine (SVM), Random Forest, and XGBoost classifiers. The performance of these models is compared to the CNN.

### Conclusion
The evaluation metrics for each model, including accuracy, precision, recall, F1-score, confusion matrix, and AUC-ROC, were computed and displayed. By analyzing these results, the performance of the CNN could be assessed with other traditional models (SVM, Random Forest, or XGBoost) to determine which yields the most promising results for this specific dataset and task. This presented the CNN model to have achieved a higher ability to detect brain tumor with a 98% accuracy.
