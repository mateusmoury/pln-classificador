from sgml_parser import get_documents_from_sgml
from documents_preprocessor import preprocess_training_documents
from NaiveBayesClassifier import NAryClassifier
import time

if __name__ == '__main__':
  print("beginning parsing")
  parser_begin = time.time()
  documents = get_documents_from_sgml()
  parser_end = time.time()

  print("beginning preprocessing")
  preprocess_begin = time.time()
  training_documents = preprocess_training_documents(documents)
  preprocess_end = time.time()

  print("beginning trainning")
  trainning_begin = time.time()
  classifier = NAryClassifier(training_documents['class_to_ids'], training_documents['id_to_text'])
  trainning_end = time.time()

  print("calculating metrics")
  metrics_begin = time.time()
  metrics = classifier.get_metrics(training_documents['test_set'])
  metrics_end = time.time()

  print()
  for kind in metrics:
    print(kind + "averaging")
    for a, b in metrics[kind].items():
      print(str(a) + " = " + str(round(b * 100, 5)) + "%")
    print("-------------------------")

  print("Time:")
  print("Parser: " + str(parser_end - parser_begin) + " seconds")
  print("Preprocess: " + str(preprocess_end - preprocess_begin) + " seconds")
  print("Training: " + str(trainning_end - trainning_begin) + " seconds")
  print("metrics: " + str(metrics_end - metrics_begin) + " seconds")
