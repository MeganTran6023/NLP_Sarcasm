# NLP Sarcasm Dataset

## 🌟 Highlights

- KNN has the **highest** accuracy **(71%)** after determining best k - value.
- Improved accuracy of Random Forest Model from **59% to 62%** through rigorous hyperparameter tuning of Random Forest Model.
- XGBoost has **lowest** acccuracy **(59%)** after hyperparameter tuning via grid search.

## ℹ️ Overview

Sarcasm is something we use in face-to-face conversations to make light of a situation or joke around with friends. We know something/someone is sarcastic from their facial expressions, tone of voice, and perhaps the way they phrase their sentiments. However, this is dificult to detect over text or other forms of media since there is only the physical text we have. Because of this, people mistakenly misunderstand one another which leads to uneccesary conflicts.

By creating this NLP project, I hope to identify a effective way to determine sarcastic comments by identifying what words are common in sarcastic comments with the highest accuracy possible.


## 🚀 Usage

**a) Requirements**
* python 3.X (I used Python 3.10.9)

**b) Preprocessing text dataset**

* steps included removing unnecessary words (stop words) and changing words to their root form for all text data points.

```
corpus[0]

#output
#'podcast make solemn promis improv sound qualiti next episod'

```
**c) Tf-IDF vectorizer**
*  I used TF-IDF (Term Frequency-Inverse Document Frequency), which gives a value to each word in a text from 0 to 1 to determine its importance. 1 is high importance, 0 is no importance 

```
X = tfidf_vec.fit_transform(corpus).toarray()

#output
#TF-IDF:
['ab' 'abandon' 'abba' 'abomin' 'abort' 'abram' 'absenc' 'absolut' 'abu'
 'academ']
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

```
**d) Random Forest Model**
Best hyperparameters were determined using two methods:</br>

Grid search:</br>
</br>
* Runs every parameter combinations to determine best ones
* Takes up too much time and computationally expensive
</br>
Random search:</br>

* Runs random parameter combinations to determine best ones
* More efficient

Final results:

```
print(classification_report(y_test, y_pred))

Classification report
              precision    recall  f1-score   support

           0       0.55      0.95      0.70        96
           1       0.86      0.30      0.44       104

    accuracy                           0.61       200
   macro avg       0.71      0.62      0.57       200
weighted avg       0.71      0.61      0.57       200

```
**e) XGBoost**

Best hyperparameters along with gini score:

```
 Best normalized gini score for 3-fold search with 3 parameter combinations:
0.2894234737911132

 Best hyperparameters:
{'subsample': 0.6, 'min_child_weight': 1, 'max_depth': 3, 'gamma': 1.5, 'colsample_bytree': 0.8}

```
Results:

```
Classification report
              precision    recall  f1-score   support

           0       0.58      0.77      0.66       103
           1       0.62      0.41      0.50        97

    accuracy                           0.59       200
   macro avg       0.60      0.59      0.58       200
weighted avg       0.60      0.59      0.58       200


```

**f) KNN**

Best hyperparameters (k-value):

![image](https://github.com/user-attachments/assets/74952d87-9428-4f5d-a86f-33951f21e878)

![image](https://github.com/user-attachments/assets/79cb6f39-bf33-426c-a442-e6fa67a544eb)

In the project, k = 43

Results:

```
Classification report
              precision    recall  f1-score   support

           0       0.66      0.88      0.76       102
           1       0.81      0.53      0.64        98

    accuracy                           0.71       200
   macro avg       0.74      0.71      0.70       200
weighted avg       0.74      0.71      0.70       200


```

## ⬇️ Installation

type 
```
pip install {package}
```

**packages:**

* pandas
* numpy
* nltk
* scikit-learn
* matplotlib.pyplot
* xgboost

## 💭 Future Improvements

* Increase number of training samples
* More steps to text feature preprocessing step

