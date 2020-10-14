pip install pyLDAvis
#Don't need all these, gensim and lda are for topic model
import gensim
import os
from gensim.models.wrappers import LdaMallet
import re
import numpy as np
import pandas as pd
from pprint import pprint
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
#import polyglot
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from urllib import request
import matplotlib.pyplot as plt
#from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#from PIL import Image
from wordcloud import WordCloud
#some downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
sid= SentimentIntensityAnalyzer()
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from google.colab import files
uploaded = files.upload()
text= open("TOP2.txt", encoding= "latin-1").read()
text_lower= text.lower()
import spacy 
nlp= spacy.load('en_core_web_sm')
#prepare stopwords
#NLTK stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
#trying this without the word essential: 8/14/19 9:23 am
stop_words.append('essential')
stop_words.append('\s')
stop_words.append('\'s')
stop_words.append('\'\'')
stop_words.append('$')
stop_words.append('__')
stop_words.append('____-__-__-_____')
stop_words.append('_________')
#need to add gutenberg to stop words
tokens= word_tokenize(text_lower)
tokens = [token for token in tokens if token not in '.,:;<>!?[]()`"\'']
tokens = [token for token in tokens if token not in stop_words]
stemmer= SnowballStemmer("english")
print(" ".join(SnowballStemmer.languages))
stems= [stemmer.stem(word) for word in tokens]
#print(stems)
#fdist= FreqDist(stems)
#print(fdist.most_common(200))
#scores= sid.polarity_scores(text)
#print(scores)
#generate a wordcloud
#wordcloud= WordCloud().generate(text)
#plt.imshow(wordcloud, interpolation= 'bilinear')
#plt.axis("off")
#plt.show()
#sentiment1= TextBlob(text)
#print(sentiment1.sentiment)
######### THIS IS FOR THE TOPIC MODELING ######
#try to split the data
dataset= [d.split() for d in tokens]
new_dict= corpora.Dictionary(dataset)
#Create Corpus
texts= dataset
corpus = [new_dict.doc2bow(text) for text in texts]
#build the LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=new_dict,
                                           num_topics=10, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
## Compute Coherence Score
##coherence_model_lda = CoherenceModel(model=lda_model, texts=dataset, dictionary=new_dict, coherence='c_v')
##coherence_lda = coherence_model_lda.get_coherence()
##print('\nCoherence Score: ', coherence_lda)
#pyLDAvis.enable_notebook()
#vis = pyLDAvis.gensim.prepare(lda_model, corpus, new_dict)
#pyLDAvis.show(vis)      
print(lda_model.print_topics())
#try to make an LDA model with mallet
#os.environ['MALLET_HOME']= 'C:\\mallet'
#mallet_path= 'C:\\Users\\Administrator\\Documents\\mallet-2.0.8\\bin\\mallet'
#ldamallet= gensim.models.wrappers.LdaMallet(mallet_path, corpus= corpus, num_topics= 10, id2word= new_dict)