# Wine-Quality-Prediction-Using-ML
Here is a well-structured and detailed description of the project:  

---

## **Wine Quality Prediction Using Machine Learning**  

### **Introduction**  
This project focuses on predicting the quality of red wine using machine learning techniques. The dataset used for this purpose is the **Wine Quality - Red** dataset, which consists of **1,599 rows** and **12 columns**, available in **CSV format**. The target variable, **quality**, is originally labeled with values **3, 4, 5, 6, 7, and 8**. To simplify the classification task, label binarization is applied:  
- Wines with a quality score **≥ 7** are categorized as **good quality (1)**.  
- Wines with a quality score **< 7** are categorized as **bad quality (0)**.  
This transformation is implemented using a **lambda function**.  

### **Technologies and Dependencies Used**  
Several Python libraries have been used for data processing, visualization, and model training:  
- **NumPy**: For numerical computations.  
- **Pandas**: For handling and processing structured data.  
- **Seaborn & Matplotlib**: For data visualization, including **heatmaps** to analyze correlations between features.  
- **Scikit-Learn (sklearn)**:  
  - **Train-Test Split**: To split the dataset into training and testing subsets.  
  - **Random Forest Classifier**: As the machine learning model for prediction.  
  - **Accuracy Score**: To evaluate model performance.  

### **Data Processing & Feature Engineering**  
- The dataset is preprocessed by handling missing values (if any) and normalizing numerical values where necessary.  
- **Heatmap visualization** is used to study the correlation between different features.  
- The **target column (quality)** is binarized using a lambda function to convert it into a binary classification problem.  

### **Model Implementation**  
- The dataset is split into **training** and **testing** subsets.  
- A **Random Forest Classifier** is applied to train the model on the training data.  
- Model performance is evaluated using the **accuracy score**.  

### **Results & Accuracy**  
- The model achieves an **accuracy score of 1.0 on the training data**, indicating that it perfectly fits the training set.  
- The **test accuracy is 0.928**, demonstrating that the model generalizes well to unseen data.  

### **Conclusion**  
This project successfully predicts the quality of red wine using machine learning, achieving high accuracy. The **Random Forest Classifier** proves to be an effective model for this classification task, offering high predictive performance. Further improvements can be explored by fine-tuning hyperparameters or experimenting with other classification models.  

---
