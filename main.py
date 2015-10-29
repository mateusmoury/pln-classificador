from sgml_parser import get_documents_from_sgml
from documents_preprocessor import preprocess_training_documents
from NaiveBayesClassifier import NAryClassifier

if __name__ == '__main__':
  print("begin parser")
  documents = get_documents_from_sgml()
  print("end parser")
  print("begin preprocessing")
  training_documents = preprocess_training_documents(documents)
  print("end preprocessing")
  print("begin training")
  classifier = NAryClassifier(training_documents['class_to_ids'], training_documents['id_to_text'])
  print("end training")
  print("begin metrics")
  print(classifier.get_metrics(training_documents['test_set']))