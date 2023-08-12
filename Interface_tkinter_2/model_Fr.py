from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer 
import os
import pickle

filename = 'pretrained_model_fr.sav'
filename_tokenizer = 'tokenizer_fr.sav'
model = None
tokenizer=None

def saveModel_fr(model, tokenizer):
    pickle.dump(model, open(filename, 'wb'))
    pickle.dump(tokenizer, open(filename_tokenizer, 'wb'))

def downloadModel_fr():
    model = AutoModelForSequenceClassification.from_pretrained("BaptisteDoyen/camembert-base-xnli")
    tokenizer = AutoTokenizer.from_pretrained("BaptisteDoyen/camembert-base-xnli")
    saveModel_fr(model, tokenizer)
    return model, tokenizer


def loadModel_fr():
    if os.path.exists(filename):
        return pickle.load(open(filename, 'rb')), pickle.load(open(filename_tokenizer, 'rb'))
    else :
        return downloadModel_fr()

def predict_fr(model, tokenizer, premise, hypothesis):
    x = tokenizer.encode(premise, hypothesis, return_tensors='pt')
    logits = model(x)[0]
    prob = logits[:,::1]
    prob = prob.softmax(dim=1)
    prob=prob.detach().numpy()
    return prob

def traiter():
    return 0




"""
test

model,tokenizer=downloadModel_fr()
premise = "le score pour les bleus est élevé"
hypothesis = "les bleu ont gagné"

prob=predict_fr(premise, hypothesis)
print(prob)
"""


"""
L'ordre des probabilité selon le label est le suivant: [entailment,neutral,contradiction]
ce lui qui l'intègre doit faire attention à l'ordre, sinon il faut changer l'ordre des probabilité 
 dans la fonction predict
"""

