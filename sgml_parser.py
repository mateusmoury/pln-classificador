from bs4 import BeautifulSoup
import os

def relevant_topic(topic):
  return topic in ["acq", "earn", "money-fx", "grain", "crude", "trade", "interest", "ship", "wheat", "corn"]

def get_documents_from_sgml():
  documents = {}
  file_names = os.listdir('./reuters21578/')
  for file_name in file_names:
    if file_name[:5] == "reut2":
      with open("./reuters21578/" + file_name, "r", encoding="utf-8", errors="ignore") as current_file:
        soup = BeautifulSoup(current_file.read(), 'html.parser')
        for document in soup.find_all('reuters'):
          current_document = {}
          topics = []
          for topic in document.find('topics').find_all('d'):
            if relevant_topic(topic.string):
              topics.append(topic.string)
          current_document['topics'] = topics
          current_document['split'] = document.get('lewissplit')
          current_document['text'] = {}
          document_text = document.find('text')
          for child in document_text.children:
            if child.name is not None:
              current_document['text'][child.name] = child.string
          documents[document.get('newid')] = current_document
  return documents
