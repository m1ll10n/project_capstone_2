![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
# Multi-class Text Classification

A Natural Language Processing (NLP) problem where the task required to sort into five categories : Sport, Tech, Business, Entertainment, Politics and each categories contains text associated with it.

A deep learning model which is limited to LSTM was created to solve the following task.

## Applications
Below are the steps taken on solving the task.
### 1. Exploratory Data Analysis
1.1 Check the first five(5) data in dataset.
```python
print(df.head())
```
![image](https://user-images.githubusercontent.com/49486823/220275540-c36b84da-82fd-471a-9da8-7a8c9d730c67.png)

1.2 Check for text data on one row. There is anomalies such as 'worldcom s problems' and '(£5.8bn)'.
```python
print(df['text'][1])
```
![image](https://user-images.githubusercontent.com/49486823/220282646-6cae0647-11ad-4621-8e75-8bf0c3000c51.png)

### 2. Data Cleaning
Data cleaning are done by:\
2.1 Removing everything in between () brackets.
```python
temp = re.sub('\(.*?\)', ' ', data)
```
2.2 Replacing all non-characters to spaces.
```python
temp = re.sub('[^a-zA-Z]', ' ', temp)
```
2.3 Removing singular characters such as 'worldcom s problems' to 'worldcom problems'.
```python
temp = re.sub('\s[a-z]\\b', '', temp)
```
2.4 Changing all characters to lowercase.
```python
text[i] = temp.lower()
```
### 3. Data Preprocessing
There are four(4) steps of data preprocessing:\
3.1 Text tokenization\
3.2 Text padding & truncation\
3.3 OneHotEncoder for target variable\
3.4 Train-test split with test size of 0.2

### 4. Model Development
This is the model architecture. Few notable settings not included in the screenshot:\
4.1 Dropout rate is set to 0.4\
4.2 Vocabulary size is set to 5000\
4.3 Activation function is Softmax Activation Function\
4.4 Optimizer is Adam Optimization\
4.5 Loss function is Categorical Cross-Entropy Function\
4.6 No early stopping implemented\
![Wan_Umar_Farid_model](https://user-images.githubusercontent.com/49486823/220282835-37e77735-0408-42f5-bc93-5e0135ecf3c1.png)
## Results
This section shows all the performance of the model and the reports.
### Training Logs
The model shows signs of overfitting as training accuracy is higher than validation accuracy.\
![Wan_Umar_Farid_Accuracy_Train](https://user-images.githubusercontent.com/49486823/220283233-c1e41871-3a1d-45d5-84fb-05f0a4601d1d.jpg)
![Wan_Umar_Farid_Loss_Train](https://user-images.githubusercontent.com/49486823/220283252-c6b5d59f-8115-4c7f-8d03-128a621a5f95.jpg)

### Accuracy & F1 Score
The model recorded an accuracy of 0.8 and f1 score of 0.8023.\
![Wan_Umar_Farid_Accuracy_F1](https://user-images.githubusercontent.com/49486823/220283383-2885dfc0-e92d-4c4b-8730-d999bb59832e.jpg)

### Classification Report
The model shows high f1 score on class 3 and 4 but rather low f1 score on class 1.\
![Wan_Umar_Farid_Classification_Report](https://user-images.githubusercontent.com/49486823/220283631-0ff07824-fc28-44b4-9784-1837f72f3f30.jpg)

### Confusion Matrix
The confusion matrix showed the model are good at classifying 'business', 'entertainment', and 'politics' classes but struggles to classify between 'tech' and 'sport' classes.\
![Wan_Umar_Farid_Confusion_Matrix](https://user-images.githubusercontent.com/49486823/220283676-cae165db-b9d5-4096-9d57-0ba7cf96ed64.png)

## Credits
Data can be obtained from https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv .
