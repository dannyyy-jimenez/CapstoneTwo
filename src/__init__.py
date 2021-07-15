import NLP
from ImageClassification import MCU, DefaultNNModel, DefaultCNNModel, CustomCNNModel, LogisticRegressionCLF, PlotConfusionMatrix, PlotAccuracy, PlotLoss
from nltk.corpus import stopwords

# Image Classification

mcu = MCU()
mcu.fit()

mcu_noevans = MCU(characters=['chris_evans', 'mark_ruffalo', 'robert_downey_jr', 'scarlett_johansson'])
mcu_noevans.fit()

nnclf = DefaultNNModel(mcu)
print(nnclf)
nnclf.predictions()
nnclf.model.evaluate(nnclf.X_test, nnclf.y_test)
PlotConfusionMatrix(nnclf)
PlotAccuracy(nnclf)

PlotLoss(nnclf)

def_cnnclf = DefaultCNNModel(mcu)

print(def_cnnclf)
def_cnnclf.predictions()
def_cnnclf.model.evaluate(def_cnnclf.X_test, def_cnnclf.y_test)

PlotConfusionMatrix(def_cnnclf)
PlotAccuracy(def_cnnclf)
PlotLoss(def_cnnclf)

custom_cnnclf = CustomCNNModel(mcu)
print(custom_cnnclf)
custom_cnnclf.predictions()
custom_cnnclf.wrongs()
custom_cnnclf.model.evaluate(custom_cnnclf.X_test, custom_cnnclf.y_test)

PlotConfusionMatrix(custom_cnnclf)
PlotAccuracy(custom_cnnclf)
PlotLoss(custom_cnnclf)

logistic = LogisticRegressionCLF(mcu)
logistic.predictions()
PlotConfusionMatrix(logistic)
print(logistic)


logistic = LogisticRegressionCLF(mcu_noevans)
logistic.predictions()
PlotConfusionMatrix(logistic)
print(logistic)


# NLP
stop_words = set([*stopwords.words('english'), 's', 'know', 't', 'don', 'need', 'yeah', 'm', 're', "'d", "'ll", "'re", "'s", "'ve", 'could', 'doe', 'ha', 'might', 'must', "n't", 'sha', 'wa', 'wo', 'would', 'going', 'like', 'got', 'na', 'im', 'abov', 'ani', 'becaus', 'befor', 'dure', 'go', 'hi', 'onc', 'onli', 'ourselv', 'themselv', 'thi', 'veri', 'whi', 'yourselv', 'dont', 'get', 'right', 'im', 'just', 'got', 'know', 'youre', 'let', 'want', 'one', 'thats', 'come', 'well'])

ironman_spiderman = NLP.MCU()
ironman_spiderman.fit()


ironman_steve = NLP.MCU(['TONY STARK', 'STEVE ROGERS'])
ironman_steve.fit()

thor_steve = NLP.MCU(['THOR', 'STEVE ROGERS'], stop_words=stop_words)
thor_steve.fit()

ironman_steve_thor = NLP.MCU(['THOR', 'STEVE ROGERS', 'TONY STARK'], stop_words=stop_words)
ironman_steve_thor.fit(1000)
ironman_steve_thor.wrongs()

imgclf_five = NLP.MCU(['THOR', 'STEVE ROGERS', 'TONY STARK', 'HULK', "NATASHA ROMANOFF"], stop_words='english')
imgclf_five.fit()
imgclf_five.wrongs()
