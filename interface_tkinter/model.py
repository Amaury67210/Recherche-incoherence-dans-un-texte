from allennlp_models.pretrained import load_predictor
import os
import pickle

filename = 'pretrained_model.sav'
model = None

def downloadModel():
    model = load_predictor("pair-classification-roberta-snli")
    saveModel(model)
    return model

def saveModel(model):
    pickle.dump(model, open(filename, 'wb'))

def loadModel():
    if os.path.exists(filename):
        return pickle.load(open(filename, 'rb'))
    else :
        return downloadModel()

def predict(model, premise, hypothesis):
    return model.predict(premise,hypothesis)

def traiter():
    return 0