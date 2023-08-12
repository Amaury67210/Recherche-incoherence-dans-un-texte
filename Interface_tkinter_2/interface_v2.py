from tkinter import *
from model import *
from model_Fr import *
from tokenization import *
from tkinter.ttk import *
import re
import numpy as np
import platform
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
nltk.download('punkt')

#fonctions 
def segmenter(texte):
    phrases = sent_tokenize(texte)
    return phrases

def fenetre_coulissante(texte, longueur):
    
    phrases=segmenter(texte)
    pairs_sentences = pd.DataFrame(columns=['premise','hypothesis','target_NLI'])
    for i in range(len(phrases)):
        for j in range(i+1, i+longueur+1):
                  try:
                    pairs_sentences = pairs_sentences.append({'premise':phrases[i],'hypothesis':phrases[j],'id_1':i,'target_NLI':"E",'id_2':j},ignore_index=True)
                  except IndexError: 
                    j = i+longueur+1 
                    
    pairs_sentences['id_1'] = pairs_sentences['id_1'].astype('int')
    pairs_sentences['id_2'] = pairs_sentences['id_2'].astype('int')
     

    sentences = pd.DataFrame (segmenter(texte), columns = ['sentences'])
    sentences['NLI_V'] = 'E'
    sentences['NLI_With'] = '_'

    return pairs_sentences ,sentences  

def command_button():
    txt_2.delete('1.0', END)
    langage_selected = langage.get()
    if langage_selected == "English":
        command_function_english()
    else:
        command_function_francais()

def command_function_english():
    #load the model from disk
    loaded_model = loadModel()

    pairs_sentences, sentences = fenetre_coulissante(txt.get("1.0", "end-1c"), int(taille_fenetre.get()))

    for i in range(0,(pairs_sentences.shape[0]-1)):
        # résultat du model NLI
        analyse = predict(loaded_model, pairs_sentences.premise[i],pairs_sentences.hypothesis[i])
        maximum = max(analyse['probs'])
#         txt_2.insert(INSERT, pairs_sentences.premise[i])
#         txt_2.insert(INSERT, "\n")
#         txt_2.insert(INSERT, pairs_sentences.hypothesis[i])
#         txt_2.insert(INSERT, "\n")
#         txt_2.insert(INSERT, maximum)
        
#         #Implication
#         if(maximum == analyse['probs'][0]):
#             if(analyse['probs'][0] > float(valeur_seuil.get())):
#                 txt_2.insert(INSERT," ==> Implication !", "implication")
#                 txt_2.insert(INSERT, "\n")
#                 pairs_sentences.at[i,'target_NLI'] = 'I' 
#             else:
#                 txt_2.insert(INSERT, " ==> Aucun changement ne sapplique !", "seuil_non_atteint")
#                 txt_2.insert(INSERT, "\n")    
                
        #Contradiction
        if(maximum == analyse['probs'][1]):
            if(analyse['probs'][1] > float(valeur_seuil.get())):
#                 txt_2.insert(INSERT," ==> Contradiction !", "contradiction")
#                 txt_2.insert(INSERT, "\n")
                pairs_sentences.at[i,'target_NLI'] = 'C'  
                
                sentences.at[pairs_sentences.at[i,'id_1'],'NLI_V'] = 'C' 
                sentences.at[pairs_sentences.at[i,'id_2'],'NLI_V'] = 'C' 
                
                 # pour la 1ere phrase 
                # la 1ere contradition 
                if sentences.NLI_With[pairs_sentences.at[i,'id_1']] == '_': 
                    sentences.at[pairs_sentences.at[i,'id_1'],'NLI_With'] = pairs_sentences.at[i,'id_2']
                # la N eme contradition 
                else : 
                    strr = str(sentences.NLI_With[pairs_sentences.at[i,'id_1']]) + ',' + str(pairs_sentences.at[i,'id_2'])
                    sentences.at[pairs_sentences.at[i,'id_1'],'NLI_With'] = strr
                    
                # pour la 2eme phrase 
                # la 1ere contradition 
                if sentences.NLI_With[pairs_sentences.at[i,'id_2']] == '_': 
                    sentences.at[pairs_sentences.at[i,'id_2'],'NLI_With'] = pairs_sentences.at[i,'id_1']
                # la N eme contradition 
                else : 
                    strr = str(sentences.NLI_With[pairs_sentences.at[i,'id_2']]) + ',' +  str(pairs_sentences.at[i,'id_1'])
                    sentences.at[pairs_sentences.at[i,'id_2'],'NLI_With'] = strr
#             else:
#                 txt_2.insert(INSERT, " ==> Aucun changement ne sapplique !", "seuil_non_atteint")
#                 txt_2.insert(INSERT, "\n")      
        #Neutre
#         if(maximum == analyse['probs'][2]):
#             if(analyse['probs'][2] > float(valeur_seuil.get())):
#                 txt_2.insert(INSERT," ==> Neutre !", "neutral")
#                 txt_2.insert(INSERT, "\n")
#                 pairs_sentences.at[i,'target_NLI'] = 'N' 
#             else:
#                 txt_2.insert(INSERT, " ==> Aucun changement ne sapplique !", "seuil_non_atteint")
#                 txt_2.insert(INSERT, "\n")
                
    update_text(sentences)
    pairs_sentences.to_csv('data.csv', sep='\t')
    return 0

def command_function_francais():
    #load the model from disk
    loaded_model, loaded_tokenizer = loadModel_fr()

    pairs_sentences, sentences = fenetre_coulissante(txt.get("1.0", "end-1c"), int(taille_fenetre.get()))

    for i in range(0,(pairs_sentences.shape[0]-1)):
        # résultat du model NLI
        analyse = predict_fr(loaded_model, loaded_tokenizer, pairs_sentences.premise[i],pairs_sentences.hypothesis[i])
        maximum = max(analyse[0])
        # txt_2.insert(INSERT, pairs_sentences.premise[i])
        # txt_2.insert(INSERT, "\n")
        # txt_2.insert(INSERT, pairs_sentences.hypothesis[i])
        # txt_2.insert(INSERT, "\n")
        # txt_2.insert(INSERT, maximum)
        
        # #Implication
        # if(maximum == analyse[0][0]):
        #     if(analyse[0][0] > float(valeur_seuil.get())):
        #         txt_2.insert(INSERT," ==> Implication !", "implication")
        #         txt_2.insert(INSERT, "\n")
        #         pairs_sentences.at[i,'target_NLI'] = 'I' 
        #     else:
        #         txt_2.insert(INSERT, " ==> Aucun changement ne sapplique !", "seuil_non_atteint")
        #         txt_2.insert(INSERT, "\n")    
        # #Neutre
        # if(maximum == analyse[0][1]):
        #     if(analyse[0][1] > float(valeur_seuil.get())):
        #         txt_2.insert(INSERT," ==> Neutre !", "neutral")
        #         txt_2.insert(INSERT, "\n")
        #         pairs_sentences.at[i,'target_NLI'] = 'N' 
        #     else:
        #         txt_2.insert(INSERT, " ==> Aucun changement ne sapplique !", "seuil_non_atteint")
        #         txt_2.insert(INSERT, "\n")
        #Contradiction
        if(maximum == analyse[0][2]):
            if(analyse[0][2] > float(valeur_seuil.get())):
                # txt_2.insert(INSERT," ==> Contradiction !", "contradiction")
                # txt_2.insert(INSERT, "\n")
                pairs_sentences.at[i,'target_NLI'] = 'C'  
                
                sentences.at[pairs_sentences.at[i,'id_1'],'NLI_V'] = 'C' 
                sentences.at[pairs_sentences.at[i,'id_2'],'NLI_V'] = 'C' 
                
                 # pour la 1ere phrase 
                # la 1ere contradition 
                if sentences.NLI_With[pairs_sentences.at[i,'id_1']] == '_': 
                    sentences.at[pairs_sentences.at[i,'id_1'],'NLI_With'] = pairs_sentences.at[i,'id_2']
                # la N eme contradition 
                else : 
                    strr = str(sentences.NLI_With[pairs_sentences.at[i,'id_1']]) + ',' + str(pairs_sentences.at[i,'id_2'])
                    sentences.at[pairs_sentences.at[i,'id_1'],'NLI_With'] = strr
                    
                # pour la 2eme phrase 
                # la 1ere contradition 
                if sentences.NLI_With[pairs_sentences.at[i,'id_2']] == '_': 
                    sentences.at[pairs_sentences.at[i,'id_2'],'NLI_With'] = pairs_sentences.at[i,'id_1']
                # la N eme contradition 
                else : 
                    strr = str(sentences.NLI_With[pairs_sentences.at[i,'id_2']]) + ',' +  str(pairs_sentences.at[i,'id_1'])
                    sentences.at[pairs_sentences.at[i,'id_2'],'NLI_With'] = strr
            # else:
            #     txt_2.insert(INSERT, " ==> Aucun changement ne sapplique !", "seuil_non_atteint")
            #     txt_2.insert(INSERT, "\n")
                
    update_text(sentences)
    pairs_sentences.to_csv('data.csv', sep='\t')
    return 0

def onclick1(sentence,i,sentences ,event):
    newWindow = Toplevel(window)
    newWindow.title("New Window")
    newWindow.geometry("700x700")
    
 
    Label(newWindow,text ="        ").pack()
    Label(newWindow,text =sentence , underline = 2, wraplength = 700,   relief = "groove").pack()
    
    Label(newWindow,text = "is contradictory " ,foreground='green').pack()
    Label(newWindow,text = "with ",foreground='green').pack()
    d = str(sentences.iloc[i,2])
    for i in d.split(','): 
        Label(newWindow,text =sentences.iloc[int(i),0], underline = 2, wraplength = 700,   relief = "groove").pack()
        Label(newWindow,text ="        ").pack()
def update_text(sentences):
    for i in range  (len(sentences)):
        if sentences.NLI_V[i] == 'C':
            txt_2.insert(INSERT,sentences.sentences[i], i)
            txt_2.tag_config(i, background="white", foreground="red")
            txt_2.tag_bind(i, '<ButtonRelease-1>', lambda event,a=sentences.sentences[i],b=i,c = sentences : onclick1(a,b,c, event))
        else :
            txt_2.insert(INSERT,sentences.sentences[i],i)
            txt_2.tag_config(i, background="white", foreground="black")
            

def update_text1(sentences):

    print(sentences)
    
    for i in range  (len(sentences)):
        if sentences.v[i] == 'c':
            txt_2.insert(INSERT,sentences.stn[i], 'warning')
        else :
            txt_2.insert(INSERT,sentences.stn[i])

def openNewWindow():

        data = pd.read_csv("data.csv",sep = '\t', error_bad_lines=False)
        newWindow = Toplevel(window)
        newWindow.geometry("800x600")
        
        data=data.drop(data.columns[[0,  4, 3,5]], axis=1)
        # Extract number of rows and columns
        n_rows = data.shape[0]
        n_cols = data.shape[1]
        # Extracting columns from the data and
        # creating text widget with some
        # background color
        column_names = data.columns
        i=0
        for j, col in enumerate(column_names):
            text = Text(newWindow, width=46, height=1, bg = "#9BC2E6")
            text.grid(row=i,column=j)
            text.insert(INSERT, col)
        # adding all the other rows into the grid
        for i in range(n_rows):
            for j in range(n_cols):
                text = Text(newWindow, width=46, height=3)
                text.grid(row=i+1,column=j)
                text.insert(INSERT, data.loc[i][j])

def size(event):
    width, height = window.winfo_width(), window.winfo_height()
    width_txt, height_txt = width*(45/100), height*(65/100)
    items = _canvas.find_withtag('all')
    for item in items:
        txt.place(relx=0.025, rely=0.2, width = 500, height = height_txt)
        txt_2.place(relx=0.64, rely=0.2, width = 500, height = height_txt)
        _canvas.coords(titre, width*0.5, height*0.05)
        _canvas.itemconfigure(titre, font=('Helvetica %i bold underline' % int(width/45)))
        _canvas.coords(sous_titre_gauche, width*0.1, height*0.15)
        _canvas.itemconfigure(sous_titre_gauche, font=('Helvetica %i bold underline' % int(width/65)))
        _canvas.coords(sous_titre_droit, width*0.7, height*0.15)
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

# Define a function
def myClick():
   greet= name.get()
   parentlist.append(greet)
   monthchoosen['values'] = parentlist
   
def Analy():
    loaded_model = loadModel()
    txt_2.delete(1.0,END)
    stn = str(n.get())   
    sentences =segmenter(txt.get("1.0", "end-1c"))
    v = [None] * len(sentences)
    df = pd.DataFrame(list(zip(sentences, v)),
               columns =['stn', 'v'])
    for index, row in df.iterrows():
        analyse = predict(loaded_model, stn,row['stn'])
        maximum = max(analyse['probs'])
        #Contradiction
        if(maximum == analyse['probs'][1]):
            row['v'] = "c"
    update_text1(df)      

######################################################################################
###
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

parentlist =  []
titre = _canvas.create_text(0, 0, text="Analyse Texte", fill="black", font=('Helvetica 32 bold underline'))
sous_titre_gauche = _canvas.create_text(300, 125, text="Texte de base :", fill="black", font=('Helvetica 24 bold underline'))
sous_titre_droit = _canvas.create_text(1250, 125, text="Texte modifie :", fill="black", font=('Helvetica 24 bold underline'))

######################################################################
_canvas.create_text(750, 180, text="Add senetence", fill="black")
# Create an entry widget
name=Entry(_canvas, width=20, font=('Arial 24'))
_canvas.create_window(760, 230, window=name)

# Create a button
button=Button(_canvas, text="Add", command=myClick)
_canvas.create_window(760, 280, window=button)
_canvas.create_text(750, 320, text="choice sentence:", fill="black")

# Adding combobox drop down list

n = StringVar()
monthchoosen = Combobox(_canvas, width = 57, textvariable = n)
monthchoosen['state'] = 'readonly'
monthchoosen['values'] = parentlist
_canvas.create_window(760, 360, window=monthchoosen)


button = Button(_canvas, text="Analays with text ", command=Analy)
_canvas.create_window(760, 400, window=button)
######################################################################
txt = Text(_canvas, highlightthickness = 0, borderwidth=0)
txt_2 = Text(_canvas, highlightthickness = 0, borderwidth=0)
 
txt_2.tag_config("implication", foreground="green", underline="true")
txt_2.tag_config("neutral", foreground="yellow", underline="true")
txt_2.tag_config("contradiction", foreground="red", underline="true")
txt_2.tag_config("seuil_non_atteint", foreground="grey", underline="true")
txt_2.tag_config("contradiction_sentence", foreground="red", underline="false")
txt_2.tag_config('warning',foreground="red")
btn = Button(_canvas, text='Analyser',command=command_button)
btn1 = Button(_canvas, text='More',command=openNewWindow)
btn2 = Button(_canvas, text='Reset', command=lambda: [txt.delete(1.0,END), txt_2.delete(1.0,END)])


red_rect = _canvas.create_rectangle(1050, 800, 1075, 825,outline="red",fill="red")
yellow_rect = _canvas.create_rectangle(1050, 835, 1075, 860,outline="yellow",fill="yellow")
green_rect = _canvas.create_rectangle(1050, 870, 1075, 895,outline="green",fill="green")

red_rect_text = _canvas.create_text(1212, 812, text=": Contradiction/Redondance", fill="black", font=('Helvetica 15 bold'))
yellow_rect_text = _canvas.create_text(1117, 847, text=": Neutre", fill="black", font=('Helvetica 15 bold'))
green_rect_text = _canvas.create_text(1140, 882, text=": Implication", fill="black", font=('Helvetica 15 bold'))

# Langage du texte de l'utilisateur

langage = StringVar()
langage.set("English")

langage_1 = Radiobutton(_canvas, text="English", variable=langage, value="English")
langage_2 = Radiobutton(_canvas, text="Français", variable=langage, value="Francais")

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
btn1.place(relx=0.83, rely=0.95)
btn2.place(relx=0.10, rely=0.95)
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

langage_1.place(relx = 0.3, rely = 0.950)
langage_2.place(relx = 0.35, rely = 0.950)

fenetre_1.place(relx = 0.28, rely = 0.870)
fenetre_2.place(relx = 0.32, rely = 0.870)
fenetre_3.place(relx = 0.36, rely = 0.870)
fenetre_4.place(relx = 0.40, rely = 0.870)
fenetre_5.place(relx = 0.44, rely = 0.870)

seuil_1.place(relx = 0.28, rely = 0.910)
seuil_2.place(relx = 0.32, rely = 0.910)
seuil_3.place(relx = 0.40, rely = 0.910)
seuil_4.place(relx = 0.44, rely = 0.910)

window.bind('<Configure>', size)
window.mainloop()
