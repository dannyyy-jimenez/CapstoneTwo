import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, accuracy_score
import re
import string
from LemmaTokenizer import LemmaTokenizer
from sklearn.linear_model import SGDClassifier


stop_words = set([*stopwords.words('english'), 's', 'know', 't', 'don', 'need', 'yeah', 'm', 're', "'d", "'ll", "'re", "'s", "'ve", 'could', 'doe', 'ha', 'might', 'must', "n't", 'sha', 'wa', 'wo', 'would', 'going', 'like', 'got', 'na', 'im', 'abov', 'ani', 'becaus', 'befor', 'dure', 'go', 'hi', 'onc', 'onli', 'ourselv', 'themselv', 'thi', 'veri', 'whi', 'yourselv'])

mcu_dialogue = pd.read_csv('../data/mcu.csv')

mcu_dialogue

mcu_dialogue = mcu_dialogue[['character', 'line']]


mcu_dialogue['character'] = mcu_dialogue['character'].apply(lambda x: 'HULK' if x == 'BRUCE BANNER' else x)
mcu_dialogue.groupby('character').count().sort_values(by='line', ascending=False)[:20]


characters = ['STEVE ROGERS', 'THOR', 'PETER PARKER']
mcu_dialogue['is_main'] = mcu_dialogue['character'].apply(lambda x: x in characters)

mcu_dialogue = mcu_dialogue[mcu_dialogue['is_main'] == True].drop(columns=["is_main"])
mcu_dialogue['character'] = mcu_dialogue['character'].apply(lambda x: characters.index(x))

mcu_dialogue.set_index('character', inplace=True)

mcu_dialogue
# EDA


def GetAmountOfLines(df, character):
    return len(df.loc[character])


for character in range(len(characters)):
    print(f'AMOUNT OF LINES FOR {characters[character]}: {GetAmountOfLines(mcu_dialogue, character)}')

X = pd.Series(mcu_dialogue['line'].values, name='line')
y = pd.Series(mcu_dialogue.index)

print(string.punctuation)


def CleanLine(line):
    lowered = line.lower()
    return re.sub(r"[0-9!\"#$%&'()*+,-.\/:;\’`<=>?@[\]^_{|}~]", '', lowered)


X = X.apply(lambda x: CleanLine(x))

X

X_train, X_test, y_train, y_test = train_test_split(X, y)

X = pd.concat([X_train, y_train], axis=1)
X

peter = X[X.character == characters.index('PETER PARKER')]
steve = X[X.character == characters.index('STEVE ROGERS')]
#hulk = X[X.character == characters.index('HULK')]
thor = X[X.character == characters.index('THOR')]
#natasha = X[X.character == characters.index('NATASHA ROMANOFF')]

# tony is the one with the most lines so we must resample to achieve that state

X_indexed = X.copy().set_index('character')

for character in range(len(characters)):
    print(f'AMOUNT OF LINES FOR {characters[character]}: {GetAmountOfLines(X_indexed, character)}')


peter_upsampled = sklearn.utils.resample(peter,replace=True, n_samples=len(steve))
#hulk_upsampled = sklearn.utils.resample(hulk,replace=True, n_samples=len(steve))
thor_upsampled = sklearn.utils.resample(thor,replace=True, n_samples=len(steve))
#natasha_upsampled = sklearn.utils.resample(natasha,replace=True, n_samples=len(steve))


def GetTopNWords(data, n=None):
    vec = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words=stop_words).fit(data)
    bag_of_words = vec.transform(data)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq,  key=lambda x: x[1], reverse=True)

    return words_freq[:n]


peter_words = GetTopNWords(peter['line'], 10)
steve_words =  GetTopNWords(steve['line'], 10)
#hulk_words =  GetTopNWords(hulk['line'], 10)
thor_words =  GetTopNWords(thor['line'], 10)
#natasha_words =  GetTopNWords(natasha['line'], 10)


def PlotHistOfWords(data, labels):

    fig, ax = plt.subplots(figsize=(20, 8))

    for idx, person in enumerate(data):
        values = [words[1] for words in person]
        keys = [words[0] for words in person]
        ax.bar(keys, values, label=labels[idx], alpha=0.3)

    fig.tight_layout()
    fig.legend()
    fig.savefig('../plots/words_bar.png')


PlotHistOfWords([thor_words, peter_words, steve_words], ['Thor', 'Peter', 'Steve'])


upsampled = pd.concat([steve, peter, thor])

characters

upsampled['character'].value_counts()

X_train = upsampled['line']
y_train = upsampled['character']


X_train


comment_words = ''

for val in mcu_dialogue['line']:
    val = str(val)
    tokens = val.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 800, height = 800, background_color ='white', stopwords = stop_words, min_font_size = 10).generate(comment_words)

fig, ax = plt.subplots(figsize=(12, 12), facecolor=None);
ax.axis('off')
ax.imshow(wordcloud)
fig.tight_layout(pad=0)
fig.savefig('../plots/wordcloud.png')

X_indexed_upsample = upsampled.set_index("character")

for characteridx in range(len(characters)):
    comment_words = ''

    for val in X_indexed_upsample.loc[characteridx]['line']:
        val = str(val)
        tokens = val.split()
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        comment_words += " ".join(tokens)+" "

    wordcloud = WordCloud(width = 800, height = 800, background_color ='white', stopwords = stop_words, min_font_size = 10).generate(comment_words)
    fig, ax = plt.subplots(figsize=(12, 12), facecolor=None);
    ax.axis('off')
    ax.imshow(wordcloud)
    fig.tight_layout(pad=0)
    fig.savefig(f'../plots/wordcloud-{characters[characteridx]}.png')


cv = CountVectorizer(stop_words='english', tokenizer=LemmaTokenizer())

X_train_counts = cv.fit_transform(X_train)

X_train_counts.toarray()

counts_df = pd.DataFrame(data=X_train_counts.toarray(), columns=cv.get_feature_names())
counts_df
cv.get_feature_names()

with open('../logs/features_.txt', 'w') as features_file:
    for feature in cv.get_feature_names():
        features_file.writelines(f'{feature}\n')

# Get the term frequencies for the words so we can start conducting our model

tfidf_vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words='english')
X_train_freq = tfidf_vectorizer.fit_transform(X_train)


clf = MultinomialNB()

clf.fit(X_train_freq, y_train)

preds = clf.predict(tfidf_vectorizer.transform(X_test))

clf.score(tfidf_vectorizer.transform(X_test), y_test)

cm = confusion_matrix(y_test, preds)

cm


fig, ax = plt.subplots(figsize=(8,  8))

plot_confusion_matrix(clf, tfidf_vectorizer.transform(X_test), y_test, display_labels=characters, cmap='viridis', ax=ax)

set(steve['line'])

fig

# Using a pipeline

text_clf = Pipeline([
    ('vect', CountVectorizer(stop_words='english', tokenizer=LemmaTokenizer())),
    ('tdidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])

text_clf.fit(X_train, y_train)

predicted = text_clf.predict(X_test)

text_clf.score(X_test, y_test)

wrong = np.where((predicted == y_test) == False)[0]

characters
y_test.iloc[wrong][:10]
X_test.iloc[wrong][:10]


fig, ax = plt.subplots(figsize=(8,  8))

plot_confusion_matrix(text_clf,  TfidfTransformer().fit_transform(CountVectorizer(stop_words='english', tokenizer=LemmaTokenizer()).fit_transform(X_test)), y_test, display_labels=characters, normalize='true', cmap='viridis', ax=ax)

set(steve['line'])

fig

predicted

print('accuracy %s' % accuracy_score(predicted, y_test))
print(classification_report(y_test, predicted, target_names=characters))


sgd = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, tol=None))
])

sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, target_names=characters))
