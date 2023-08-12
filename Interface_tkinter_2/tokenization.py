import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def segmenter(texte):
    #segmentation du texte en phrases
    phrases = sent_tokenize(texte)
    return phrases

def dataframe(texte):
    phrases=segmenter(texte)
    df=pd.DataFrame(columns=['premise','hypotesysis'])
    for i in range(len(phrases)):
        for j in range(len(phrases)):
            if i!=j:
                df=df.append({'premise':phrases[i],'hypotesysis':phrases[j]},ignore_index=True)
    return df    

