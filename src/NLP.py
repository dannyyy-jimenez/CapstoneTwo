import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import plot_confusion_matrix
from wordcloud import WordCloud
from sklearn.metrics import classification_report, accuracy_score
import re
from LemmaTokenizer import LemmaTokenizer
from nltk.corpus import stopwords


# Default MCU Starter class to bring in all of our data

class MCU():
    def __init__(self, characters=['PETER PARKER', 'TONY STARK'], stop_words='english'):
        # Clean up data according to the characters we are going to be using for each class
        self.characters = characters
        self.stop_words = stop_words
        self.data = pd.read_csv('../data/mcu.csv')
        self.data = self.data[['character', 'line']]

        # even if we dont use hulk wed like to have the data have bruce = hulk
        self.data['character'] = self.data['character'].apply(lambda x: 'HULK' if x == 'BRUCE BANNER' else x)
        self.data['is_main'] = self.data['character'].apply(lambda x: x in self.characters)
        self.data = self.data[self.data['is_main'] == True].drop(columns=["is_main"])

        self.LineDist(True)

        # replace the character with its respective index
        self.data['character'] = self.data['character'].apply(lambda x: characters.index(x))
        self.data.set_index('character', inplace=True)

    def LineDist(self, index=False):
        baseline_accuracy = 0
        counts_lines_per_char = self.data.groupby('character').count().sort_values(by='line', ascending=False)
        filename = '../plots/line_dist'

        if index:
            counts_lines_per_char = counts_lines_per_char.loc[self.characters]
            filename += '-'
            filename += '-'.join(self.characters)

            for character in self.characters:
                baseline_accuracy += (len(counts_lines_per_char.loc[character]) / len(counts_lines_per_char)) ** 2

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"# of Lines Between {' & '.join(self.characters)}")
        ax.bar(counts_lines_per_char.index, counts_lines_per_char['line'])
        fig.savefig(filename)

        print(f"Baseline Accuracy: {baseline_accuracy * 100}%")

    def LowerPunc(self, line):
        lowered = line.lower()
        return re.sub(r"[0-9!\"#$%&'()*+,-.\/:;\’`<=>?@[\]^_{|}~]", '', lowered)

    def fit(self, upsample=0, lemmatize=False):
        self.X = pd.Series(self.data['line'].values, name='line')
        self.y = pd.Series(self.data.index)
        self.X = self.X.apply(lambda x: self.LowerPunc(x))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)

        self.X = pd.concat([self.X_train, self.y_train], axis=1)

        if upsample > 0:
            upsamples = []

            for character in self.characters:
                character_lines = self.X[self.X.character == self.characters.index(character)]
                character_upsampled = sklearn.utils.resample(character_lines, replace=True, n_samples=upsample)
                upsamples.append(character_upsampled)

            self.X = pd.concat(upsamples)
            self.X_train = self.X['line']
            self.y_train = self.X['character']

        self.cv = CountVectorizer(stop_words=self.stop_words)
        if lemmatize:
            self.cv = CountVectorizer(stop_words=self.stop_words, tokenizer=LemmaTokenizer())
        self.X_train_counts = self.cv.fit_transform(self.X_train)
        self.X_train_counts = pd.DataFrame(data=self.X_train_counts.toarray(), columns=self.cv.get_feature_names())

        self.tfidf_transformer = TfidfTransformer()
        self.X_train_freq = self.tfidf_transformer.fit_transform(self.X_train_counts.to_numpy())

        self.clf = MultinomialNB()
        self.clf.fit(self.X_train_freq, self.y_train)
        self.y_pred = self.clf.predict(self.tfidf_transformer.transform(self.cv.transform(self.X_test)))
        self.accuracy = self.clf.score(self.tfidf_transformer.transform(self.cv.transform(self.X_test)), self.y_test)

        self.PlotConfusionMatrix()
        self.WriteFeatures()
        self.PlotFreqWords()
        self.Report()

    def Report(self):
        print('accuracy %s' % accuracy_score(self.y_pred, self.y_test))
        print(classification_report(self.y_test, self.y_pred, target_names=self.characters))

    def PlotConfusionMatrix(self):
        fig, ax = plt.subplots(figsize=(8,  8))
        plot_confusion_matrix(self.clf, self.tfidf_transformer.transform(self.cv.transform(self.X_test)), self.y_test, display_labels=self.characters, cmap='viridis', ax=ax)
        fig.tight_layout()
        fig.savefig(f'../plots/confusion_matrix-{"-".join(self.characters)}')
        return fig

    def WriteFeatures(self):
        with open('../logs/features_.txt', 'w') as features_file:
            for feature in self.cv.get_feature_names():
                features_file.writelines(f'{feature}\n')

    def PlotWordCloud(self, characters=True):
        comment_words = ''

        for val in self.X['line']:
            val = str(val)
            tokens = val.split()
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()

            comment_words += " ".join(tokens)+" "

        wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=self.stop_words, min_font_size=10).generate(comment_words)

        fig, ax = plt.subplots(figsize=(12, 12), facecolor=None);
        ax.axis('off')
        ax.imshow(wordcloud)
        fig.tight_layout(pad=0)
        fig.savefig(f'../plots/wordcloud-{"-".join(self.characters)}.png')

        if characters:
            X_indexed_upsample = self.X.set_index("character")

            for characteridx in range(len(self.characters)):
                comment_words = ''

                for val in X_indexed_upsample.loc[characteridx]['line']:
                    val = str(val)
                    tokens = val.split()
                    for i in range(len(tokens)):
                        tokens[i] = tokens[i].lower()

                    comment_words += " ".join(tokens)+" "

                wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=self.stop_words, min_font_size=10).generate(comment_words)
                fig, ax = plt.subplots(figsize=(12, 12), facecolor=None);
                ax.axis('off')
                ax.imshow(wordcloud)
                fig.tight_layout(pad=0)
                fig.savefig(f'../plots/wordcloud-{self.characters[characteridx]}.png')

    def PlotFreqWords(self, n=10):
        top_words_per_character = self.TopNWords(n)

        fig, ax = plt.subplots(figsize=(20, 8))

        ax.set_title(f'Top {n} Words')

        for idx, character in enumerate(top_words_per_character):
            values = [words[1] for words in character]
            keys = [words[0] for words in character]
            ax.bar(keys, values, label=self.characters[idx], alpha=0.3)

        fig.tight_layout()
        fig.legend()
        fig.savefig(f'../plots/words_bar-{"-".join(self.characters)}.png')

    def TopNWords(self, n=None):
        top_words_per_character = []

        for character in self.characters:
            character_lines = self.X[self.X.character == self.characters.index(character)]['line']
            vec = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words=self.stop_words).fit(character_lines)
            bag_of_words = vec.transform(character_lines)
            sum_words = bag_of_words.sum(axis=0)
            words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
            words_freq = sorted(words_freq,  key=lambda x: x[1], reverse=True)
            top_words_per_character.append(words_freq[:n])

        return top_words_per_character

    def wrongs(self, n=10):
        wrongs = np.where((self.y_pred == self.y_test) == False)[0]
        to_display = np.random.choice(wrongs, size=n)

        print("Here's some insight as to what the model is getting incorrectly\n")
        print("---------------------------------------------------------------\n\n")

        for wrong in to_display:
            actual = self.y_test.iloc[wrong]
            predicted = self.y_pred[wrong]
            phrase = self.X_test.iloc[wrong]
            print(phrase)
            print(f'Model Predicted: {self.characters[predicted]}')
            print(f'Actually Said By: {self.characters[actual]}')
            print('\n\n\t---------------------------------\n\n')


if __name__  == "__main__":
    stop_words = set([*stopwords.words('english'), 's', 'know', 't', 'don', 'need', 'yeah', 'm', 're', "'d", "'ll", "'re", "'s", "'ve", 'could', 'doe', 'ha', 'might', 'must', "n't", 'sha', 'wa', 'wo', 'would', 'going', 'like', 'got', 'na', 'im', 'abov', 'ani', 'becaus', 'befor', 'dure', 'go', 'hi', 'onc', 'onli', 'ourselv', 'themselv', 'thi', 'veri', 'whi', 'yourselv', 'dont', 'get', 'right', 'im', 'just', 'got', 'know'])
    ironman_spiderman = MCU(stop_words='english')
    ironman_spiderman.fit(1000)
    ironman_spiderman.Wrongs()
