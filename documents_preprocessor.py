import nltk

def remove_stopwords(tokens):
  stop_words = nltk.corpus.stopwords.words('english')
  return [token for token in tokens if token not in stop_words]

def stem(tokens):
  stemmer = nltk.stem.snowball.SnowballStemmer('english')
  return [stemmer.stem(token) for token in tokens]

# Recebe o mapa contendo os documentos, como definido no sgml_parser, e retorna lista de pares de treinamento (texto, classes)
def preprocess_training_documents(documents):
  text_wrapper = []
  punctuation = [',', ';', '.', ':', '.', '!', '?', '\"', '*', '\'', '(', ')', '-', '>', '<']
  for document_id, document_info in documents.items():
    if document_info['split'] == "TRAIN" and 'body' in document_info['text']:
      text = stem(remove_stopwords(nltk.word_tokenize(document_info['text']['body'])))
      text = list(filter(lambda word: word not in punctuation, text))
      classes = document_info['topics']
      if len(classes) > 0:
        text_wrapper.append((text, classes))
  return text_wrapper
