from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import spacy
import tensorflow as tf
import numpy as np


corpus = [
    "Thor eating pizza, Loki is eating pizza, Ironman ate pizza already",
    "Apple is announcing new iphone tomorrow",
    "Tesla is announcing new model-3 tomorrow",
    "Google is announcing new pixel-6 tomorrow",
    "Microsoft is announcing new surface tomorrow",
    "Amazon is announcing new eco-dot tomorrow",
    "I am eating biryani and you are eating grapes"
]

vec = TfidfVectorizer()
result = vec.fit_transform(corpus)

vec.vocabulary_

all_feature_names = vec.get_feature_names_out()

for word in all_feature_names:
    indx = vec.vocabulary_.get(word)
    print(f'{word} {vec.idf_[indx]}')

corpus[:2]

result.toarray().shape

df = pd.read_csv('data/Ecommerce_data.csv')
df.shape
df.head()

df['label'].value_counts()

CLASS_NAMES = ['Household', 'Books', 'Electronics', 'Clothing & Accessories']
df['label_num'] = df.label.map({
    'Household': 0,
    'Books': 1,
    'Electronics': 2,
    'Clothing & Accessories': 3
})
df.head()

nlp = spacy.load('en_core_web_sm')

def preprocess(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)

    return " ".join(filtered_tokens) 

preprocess(df.iloc[0].Text)

text = nlp('Low Back')

for token in text:
    print(f'{token.text} {token.is_stop}')

df['ProcessedText'] = df.Text.apply(lambda x: preprocess(x))
df.head()

X = df['ProcessedText']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2022, stratify=y)
y_train.value_counts()
y_test.value_counts()

clf = Pipeline([('vectorizer_tfidf', TfidfVectorizer()),
                ('KNN', KNeighborsClassifier())
                ])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

y_test[:5]

y_pred[:5]

X_test.iloc[4]


clf = Pipeline([('vectorizer_tfidf', TfidfVectorizer()),
                ('Multi NB', MultinomialNB())
                ])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


clf = Pipeline([('vectorizer_tfidf', TfidfVectorizer()),
                ('Random Forest', RandomForestClassifier())
                ])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

X_train.head()

vec = TfidfVectorizer(max_features=300, dtype=np.float32)
vec.fit(X_train)
X_train_scaled = vec.transform(X_train)
X_test_scaled = vec.transform(X_test)

X_train_scaled.toarray()[:2]
X_train_scaled[0].shape
X_test_scaled.shape
X_test_scaled[0].toarray()[:2]

    
model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, input_shape=(300,), activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(4, activation='sigmoid'),
])

model.compile(
    optimizer = 'adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

model.fit(X_train_scaled, y_train, epochs=5, batch_size=100)

model.evaluate(X_test_scaled, y_test)

y_pred = model.predict(X_test_scaled)
CLASS_NAMES[np.argmax(y_pred[1:2][0])]
X_test.iloc[1:2]

from sklearn.preprocessing import LabelEncoder

y_label = df['label']

lb = LabelEncoder()
y_enc = lb.fit_transform(y_label)


def predict(text):
    text = preprocess(text)
    l_text = list([text])
    len(l_text)
    r = vec.transform(l_text)
    print(r.toarray())
    pred = model.predict(r)
    return CLASS_NAMES[np.argmax(pred[0])]
    #return "failed"
    
text="Set of 3 Hosley® Highly Fragranced Lavender Fields Filled Glass Candles, 1.6 Oz Wax Each Flavor name:Lavender   Our designer Hosley Candles are made from natural wax with using the highest quality of imported fragrance oils. Delicate and entrancing, the scented candles promise to bring warmth to your home."
text2 = "Urbaano Belo Lace Bikini Set - Black Urbaano Belo bikini Lace set in a simple stunning triangle makes this bikini standout, halter bikini top & bikini panty style that can be tied at both the sides according to your size. Be a Stunner."
text3 = "Lenovo Ideapad 330 Core i5 8th Gen 15.6-inch FHD Laptop (8GB/2TB HDD/2GB Graphics/Platinum Grey/2.2kg), 81DE0048IN PC-ABS finish. Sleek uni-body chassis with special protective finish . Dolby Audio. Better sound quality. 180 degree hinge. Prevents accidental hinge damages. FHD Antiglare Display. Visibly clear visual output. USB Type-C and 1X1 AC Wi-fi. Faster data transfer with reversible port and faster WiFi protocol."
print(f'Text={text}, Prediction={predict(text)}')
print(f'Text={text3}, Prediction={predict(text3)}')
