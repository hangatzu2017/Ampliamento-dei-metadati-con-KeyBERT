# -*- coding: utf-8 -*-
from langchain.schema.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from keybert import KeyBERT
import datetime
import warnings
warnings.filterwarnings(action='ignore')

# Selezione del modello
kw_model = KeyBERT(model='intfloat/multilingual-e5-base')
logfile = 'KeyBert.log'

# Scrittura del file di log
def writehistory(text):
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

# Estrazione delle keywords
def extract_keys(text, ngram,dvsity):

    a = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, ngram), stop_words='english',
                              use_mmr=True, diversity=dvsity, highlight=True)     #highlight=True
    tags = []
    for kw in a:
        tags.append(str(kw[0]))
    timestamped = datetime.datetime.now()
   
    # Log del testo e dei metadati
    logging_text = f"Log del: {str(timestamped)}\nMETADATI: {str(tags)}\nimpostazioni: keyphrase_ngram_range (1,{str(ngram)})  Diversity {str(dvsity)}\n---\nTesto Originale:\n{text}\n---\n\n"
    writehistory(logging_text)
    return tags

## Primo esempio con un estratto di testo ##
text = """
Titolo: Sulla decifratura di Enigma
pubblicato su ArXiv
Autori: Fabio S. Priuli, Claudia Violante
url: https://arxiv.org/pdf/2008.03122.pdf
Con la desecretazione di numerosi documenti risalenti al periodo della Seconda Guerra Mondiale, avvenuta nei primi anni del XXI secolo, si `e cominciato a comprendere il ruolo determinante giocato da un
gruppo di matematici, informatici, enigmisti e scacchisti britannici nella decifratura del “Codice Enigma”
e quindi nella determinazione dell’esito della Seconda Guerra Mondiale sul fronte occidentale. Nel corso
dell’ultimo decennio vi sono stati vari libri e documentari sul tema, e particolare attenzione `e stata posta
sul ruolo decisivo che ebbe il matematico Alan Turing, condannato nel 1952 alla castrazione chimica a
causa della sua omosessualit`a e morto, forse suicida, nel 1954. La riabilitazione di Turing `e avvenuta
solo negli ultimi anni, con le scuse ufficiali da parte del governo britannico nel 2009 e la grazia postuma
nel 2013.
Il punto su cui vogliamo fissare la nostra attenzione in questo scritto `e l’importante ruolo che ricopr`ı
nella vicenda un teorema del ’700, proposto dal reverendo Bayes per risolvere alcuni problemi sull’equit`a
di lotterie e scommesse. Si tratta della cosiddetta regola di Bayes, che verr`a presentata e discussa in
maggior dettaglio nel paragrafo 5.1: tale regola permette di quantificare correttamente come debba essere
aggiornata la valutazione di una probabilit`a1 alla luce di nuove informazioni o evidenze.
La regola permette quindi di effettuare nel modo pi`u appropriato il cosiddetto processo di inferenza2 per
valutare le cause pi`u probabili che abbiano provocato un fenomeno osservato, pesando opportunamente
la probabilit`a che tali cause avevano prima dell’osservazione (probabilit`a a priori) con la verosimiglianza
(o likelihood) che il fenomeno osservato sia effettivamente conseguenza della causa considerata.
Proprio in virt`u di questa specifica capacit`a di supporto all’inferenza, la regola di Bayes `e utilizzata
ormai da decenni in numerosissime attivit`a scientifiche ed accademiche (dall’analisi dati nell’ambito dei
grandi esperimenti della fisica di frontiera [1, 2], sino alle ricerche in ambito medico [24] e alle scienze
forensi [13]), al punto di essere divenuta un requisito indispensabile per la valutazione dei rischi nei
progetti NASA [10, 25]. Negli ultimi anni, complice lo sviluppo di algoritmi di previsione ed intelligenza
artificiale basati su di essa, le reti bayesiane, questa regola ha trovato applicazione con enorme successo.
"""

# Estrazione delle keywords
a = extract_keys(text, 1,0.32)

print(f"Keywords: {a}")

## Secondo esempio partendo dall'intero documento in formato Pdf. ##

# Conversione del Pdf in testo semplice
loader = PyPDFLoader("Sulla_decifratura_di_Enigma.pdf")
pages = loader.load_and_split()

# Divisione del testo in piccole parti (chunk) con sovrapposizione (overlap)
text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=100)

title = 'Sulla decifratura di Enigma'
filename = 'Sulla_decifratura_di_Enigma.txt'
author = 'Fabio S. Priuli, Claudia Violante'
url = 'https://arxiv.org/pdf/2008.03122.pdf'

keys = []

for page in pages:
  splitted_text = text_splitter.split_text(page.page_content)
  page_meta     =  page.metadata["page"]

  for i in range(0,len(splitted_text)):
    text = splitted_text[i]
    keys.append({
      'document' : filename,
      'title' : title,
      'author' : author,
      'url' : url,
      'doc': text,
      'page': page_meta + 1,
      'keywords' : extract_keys(text, 1, 0.34)
    })

# Visualizza un elemento di keys
print(keys[10])

# Riconversione in formato Document per langchain
goodDocs = []
for i in range(0,len(keys)):
  goodDocs.append(Document(
    page_content = keys[i]['doc'],
    metadata = {
      'source': keys[i]['document'],
      'type': 'chunk',
      'title': keys[i]['title'],
      'author': keys[i]['author'],
      'url' : keys[i]['url'],
      'page': keys[i]['page'],
      'keywords' : keys[i]['keywords']
    }
  ))

# Stampa di un Document
print(goodDocs[10])
