import nltk

def remove_stopwords(tokens):
  stop_words = set(nltk.corpus.stopwords.words('english'))
  return [token for token in tokens if token not in stop_words]

def stem(tokens):
  stemmer = nltk.stem.snowball.SnowballStemmer('english')
  return [stemmer.stem(token) for token in tokens]

# Recebe o mapa contendo os documentos, como definido no sgml_parser, e retorna lista de pares de treinamento (texto, classes)
def preprocess_training_documents(documents):
  test_set = []
  id_to_text = {}
  class_to_ids = {}

  punctuation = set([',', ';', '.', ':', '.', '!', '?', '\"', '*', '\'', '(', ')', '-', '>', '<'])
  for document_id, document_info in documents.items():

    if 'body' in document_info['text']:
      text = stem(remove_stopwords(nltk.word_tokenize(document_info['text']['body'])))
      text = list(filter(lambda word: word not in punctuation, text))
      classes = document_info['topics']

      if document_info['split'] == "TRAIN":
        id_to_text[document_id] = text
        for topic in classes:
          if topic in class_to_ids:
            class_to_ids[topic].add(document_id)
          else:
            class_to_ids[topic] = set([document_id])

      elif document_info['split'] == "TEST":
        test_set.append((text, set(classes)))

  return { 'test_set': test_set, 'id_to_text': id_to_text, 'class_to_ids': class_to_ids }