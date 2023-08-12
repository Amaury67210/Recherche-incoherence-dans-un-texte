from tkinter import *
from model import *
from tokenization import *
from tkinter.ttk import *
import re
import numpy as np
import platform
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# pd.set_option('display.max_colwidth', None)

def segmenter(texte):
    phrases = sent_tokenize(texte)
    return phrases

def fenetre_coulissante(texte, longueur):
    phrases=segmenter(texte)
    df=pd.DataFrame(columns=['premise','hypotesysis'])
    for i in range(len(phrases)):
        for j in range(i+1, i+longueur+1):
                  try:
                    df=df.append({'premise':phrases[i],'hypotesysis':phrases[j]},ignore_index=True)
                  except IndexError: 
                    j = i+longueur+1 
    return df    

def command_function():

    data = fenetre_coulissante(txt.get("1.0", "end-1c"), int(taille_fenetre.get()))

    model = downloadModel()

    for i in range(0,(data.shape[0]-1)):
        analyse = model.predict(data.premise[i],data.hypotesysis[i])

        maximum = max(analyse['probs'])

        txt_2.insert(INSERT, data.premise[i])
        txt_2.insert(INSERT, "\n")
        txt_2.insert(INSERT, data.hypotesysis[i])
        txt_2.insert(INSERT, "\n")

        txt_2.insert(INSERT, maximum)

        if(maximum == analyse['probs'][0]):
            if(analyse['probs'][0] > float(valeur_seuil.get())):
                txt_2.insert(INSERT," ==> Implication !", "implication")
                txt_2.insert(INSERT, "\n")
            else:
                txt_2.insert(INSERT, " ==> Aucun changement ne sapplique !", "seuil_non_atteint")
                txt_2.insert(INSERT, "\n")    

        if(maximum == analyse['probs'][1]):
            if(analyse['probs'][1] > float(valeur_seuil.get())):
                txt_2.insert(INSERT," ==> Contradiction !", "contradiction")
                txt_2.insert(INSERT, "\n")  
            else:
                txt_2.insert(INSERT, " ==> Aucun changement ne sapplique !", "seuil_non_atteint")
                txt_2.insert(INSERT, "\n")      

        if(maximum == analyse['probs'][2]):
            if(analyse['probs'][2] > float(valeur_seuil.get())):
                txt_2.insert(INSERT," ==> Neutre !", "neutral")
                txt_2.insert(INSERT, "\n")
            else:
                txt_2.insert(INSERT, " ==> Aucun changement ne sapplique !", "seuil_non_atteint")
                txt_2.insert(INSERT, "\n")

    return 0

# Interface
window = Tk()

window.title("Application analyse de texte")

window.configure(bg='lightblue')

if platform.system() == 'Linux':
    window.attributes('-zoomed', True)
else:
    window.state('zoomed')

width, height = window.maxsize()[0], window.maxsize()[1]

_canvas = Canvas(window, bg="lightblue", highlightthickness=0, width=width, height=height)
_canvas.pack()

titre = _canvas.create_text(0, 0, text="Analyse Texte", fill="black", font=('Helvetica 32 bold underline'))
sous_titre_gauche = _canvas.create_text(300, 125, text="Texte de base :", fill="black", font=('Helvetica 24 bold underline'))
sous_titre_droit = _canvas.create_text(1250, 125, text="Texte modifie :", fill="black", font=('Helvetica 24 bold underline'))

txt = Text(_canvas, highlightthickness = 0, borderwidth=0)

txt_2 = Text(_canvas, highlightthickness = 0, borderwidth=0)

txt_2.tag_config("implication", foreground="green", underline="true")
txt_2.tag_config("neutral", foreground="yellow", underline="true")
txt_2.tag_config("contradiction", foreground="red", underline="true")
txt_2.tag_config("seuil_non_atteint", foreground="grey", underline="true")

btn = Button(_canvas, text='Analyser',command=command_function)

red_rect = _canvas.create_rectangle(1050, 800, 1075, 825,outline="red",fill="red")
yellow_rect = _canvas.create_rectangle(1050, 835, 1075, 860,outline="yellow",fill="yellow")
green_rect = _canvas.create_rectangle(1050, 870, 1075, 895,outline="green",fill="green")

red_rect_text = _canvas.create_text(1212, 812, text=": Contradiction/Redondance", fill="black", font=('Helvetica 15 bold'))
yellow_rect_text = _canvas.create_text(1117, 847, text=": Neutre", fill="black", font=('Helvetica 15 bold'))
green_rect_text = _canvas.create_text(1140, 882, text=": Implication", fill="black", font=('Helvetica 15 bold'))


# Taille de la fenêtre de décalage choisit par notre utilisateur

taille_fenetre_text = _canvas.create_text(200, 200, text="Taille de la fenêtre coulissante :", fill="black", font=('Helvetica 4 bold'))

taille_fenetre = StringVar()
taille_fenetre.set("1")

fenetre_1 = Radiobutton(_canvas, text="1", variable=taille_fenetre, value="1")
fenetre_2 = Radiobutton(_canvas, text="2", variable=taille_fenetre, value="2")
fenetre_3 = Radiobutton(_canvas, text="3", variable=taille_fenetre, value="3")
fenetre_4 = Radiobutton(_canvas, text="4", variable=taille_fenetre, value="4")
fenetre_5 = Radiobutton(_canvas, text="5", variable=taille_fenetre, value="5")

# Valeur seuil de validité pour la détection pour notre modèle

valeur_seuil_text = _canvas.create_text(200, 200, text="valeur seuil acceptable pour le modèle :", fill="black", font=('Helvetica 4 bold'))

valeur_seuil = StringVar()
valeur_seuil.set("0.80")

seuil_1 = Radiobutton(_canvas, text="0.80", variable=valeur_seuil, value="0.80")
seuil_2 = Radiobutton(_canvas, text="0.85", variable=valeur_seuil, value="0.85")
seuil_3 = Radiobutton(_canvas, text="0.90", variable=valeur_seuil, value="0.90")
seuil_4 = Radiobutton(_canvas, text="0.95", variable=valeur_seuil, value="0.95")


# Placements :

width_txt, height_txt = width*(45/100), height*(65/100)
txt.place(relx=0.025, rely=0.2, width = width_txt, height = height_txt)
txt_2.place(relx=0.525, rely=0.2, width = width_txt, height = height_txt)
btn.place(relx=0.23, rely=0.95)

_canvas.coords(titre, width*0.5, height*0.05)
_canvas.itemconfigure(titre, font=('Helvetica %i bold underline' % int(17+width/100)))
_canvas.coords(sous_titre_gauche, width*0.1, height*0.15)
_canvas.itemconfigure(sous_titre_gauche, font=('Helvetica %i bold underline' % int(9+width/100)))
_canvas.coords(sous_titre_droit, width*0.6, height*0.15)
_canvas.itemconfigure(sous_titre_droit, font=('Helvetica %i bold underline' % int(9+width/100)))

_canvas.coords(red_rect, width*0.6, height*0.87, width*0.6+20, height*0.87+20)
_canvas.coords(yellow_rect, width*0.6, height*0.91, width*0.6+20, height*0.91+20)
_canvas.coords(green_rect, width*0.6, height*0.95, width*0.6+20, height*0.95+20)

_canvas.coords(red_rect_text, width*0.71, height*0.88)
_canvas.itemconfigure(red_rect_text, font=('Helvetica %i bold underline' % int(width/100)))
_canvas.coords(yellow_rect_text, width*0.65, height*0.92)
_canvas.itemconfigure(yellow_rect_text, font=('Helvetica %i bold underline' % int(width/100)))
_canvas.coords(green_rect_text, width*0.6623, height*0.96)
_canvas.itemconfigure(green_rect_text, font=('Helvetica %i bold underline' % int(width/100)))

_canvas.coords(valeur_seuil_text, width*0.15, height*0.92)
_canvas.itemconfigure(valeur_seuil_text, font=('Helvetica %i bold underline' % int(width/100)))
_canvas.coords(taille_fenetre_text, width*0.15, height*0.88)
_canvas.itemconfigure(taille_fenetre_text, font=('Helvetica %i bold underline' % int(width/100)))

fenetre_1.place(relx = 0.28, rely = 0.870)
fenetre_2.place(relx = 0.32, rely = 0.870)
fenetre_3.place(relx = 0.36, rely = 0.870)
fenetre_4.place(relx = 0.40, rely = 0.870)
fenetre_5.place(relx = 0.44, rely = 0.870)

seuil_1.place(relx = 0.28, rely = 0.910)
seuil_2.place(relx = 0.32, rely = 0.910)
seuil_3.place(relx = 0.40, rely = 0.910)
seuil_4.place(relx = 0.44, rely = 0.910)

def size(event):
    width, height = window.winfo_width(), window.winfo_height()
    width_txt, height_txt = width*(45/100), height*(65/100)
    items = _canvas.find_withtag('all')
    for item in items:
        txt.place(relx=0.025, rely=0.2, width = width_txt, height = height_txt)
        txt_2.place(relx=0.525, rely=0.2, width = width_txt, height = height_txt)

        _canvas.coords(titre, width*0.5, height*0.05)
        _canvas.itemconfigure(titre, font=('Helvetica %i bold underline' % int(width/45)))
        _canvas.coords(sous_titre_gauche, width*0.1, height*0.15)
        _canvas.itemconfigure(sous_titre_gauche, font=('Helvetica %i bold underline' % int(width/65)))
        _canvas.coords(sous_titre_droit, width*0.6, height*0.15)
        _canvas.itemconfigure(sous_titre_droit, font=('Helvetica %i bold underline' % int(width/65)))

        _canvas.coords(red_rect, width*0.6, height*0.87, width*0.6+20, height*0.87+20)
        _canvas.coords(yellow_rect, width*0.6, height*0.91, width*0.6+20, height*0.91+20)
        _canvas.coords(green_rect, width*0.6, height*0.95, width*0.6+20, height*0.95+20)

        _canvas.coords(red_rect_text, width*0.71, height*0.88)
        _canvas.itemconfigure(red_rect_text, font=('Helvetica %i bold underline' % int(width/100)))
        _canvas.coords(yellow_rect_text, width*0.65, height*0.92)
        _canvas.itemconfigure(yellow_rect_text, font=('Helvetica %i bold underline' % int(width/100)))
        _canvas.coords(green_rect_text, width*0.6623, height*0.96)
        _canvas.itemconfigure(green_rect_text, font=('Helvetica %i bold underline' % int(width/100)))

        _canvas.coords(valeur_seuil_text, width*0.15, height*0.92)
        _canvas.itemconfigure(valeur_seuil_text, font=('Helvetica %i bold underline' % int(width/100)))
        _canvas.coords(taille_fenetre_text, width*0.15, height*0.88)
        _canvas.itemconfigure(taille_fenetre_text, font=('Helvetica %i bold underline' % int(width/100)))
        pass

window.bind('<Configure>', size)

window.mainloop()
