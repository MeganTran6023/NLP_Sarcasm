# NLP Sarcasm Dataset

## 🌟 Highlights

- Improved accuracy of Random Forest Model from 59% to 62% through rigorous hyperparameter tuning of Random Forest Model.
- Quickened the hyperparameter tuning and training process by reducing dataset from 27,000 items to 1000 items


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

## 💭 Future Improvements

* test dataset on other machine learning models to compare accuracies

