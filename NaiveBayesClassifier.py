'''
text: abbreviation for [tokens]
id: integer
class_to_ids: map {'class_name' : [id]}
id_to_text: map {'id' : text}
pos_examples: [id]
neg_examples: [id]
test_set: [(text, [class_names])]
'''

def BinaryClassifier:

  def __init__(self, class_name, id_to_text, pos_examples, neg_examples):
    self._class_name = class_name
    self.train(id_to_text, pos_examples, neg_examples)
    pass


  def train(self, id_to_text, pos_examples, neg_examples):
    ''' Trains this classifier. '''
    pass


  def classify(self, text):
    ''' Returns True if this text belongs to this class and False otherwise. '''
    pass

def NAryClassifier:

  def __init__(self, class_to_ids, id_to_text):
    self._classifiers = {}
    self._class_names = []
    for class_name in class_to_ids:
      self._class_names.append(class_name)
      pos_examples = class_to_ids[class_name]
      neg_examples = [id_exmp for id_exmp, _ in id_to_text.items(): if id_exmp not in pos_examples]
      self._classifiers[class_name] = BinaryClassifier(class_name, id_to_text, pos_examples, neg_examples)


  def get_classes_for_text(self, text):
    ''' Returns a list containing the classes' names to which this text belongs. '''
    ret = []
    for classifier in self._classifiers:
      if classifier.classify(text):
        ret.append(classifier._class_name)
    return ret


  def get_metrics(self, test_set):
    ''' Returns the metrics: precision, recall, accuracy and f1, for both micro and macro averaging. '''
    tp = {class_name: 0.0 for class_name in self._class_names}
    fp = {class_name: 0.0 for class_name in self._class_names}
    tn = {class_name: 0.0 for class_name in self._class_names}
    fn = {class_name: 0.0 for class_name in self._class_names}

    for (text, classes) in test_set:
      for classifier in self._classifiers:
        if classifier.classify(text):
          if classifier._class_name in classes:
            tp[classifier._class_name] += 1.0
          else:
            fp[classifier._class_name] += 1.0
        else:
          if classifier._class_name in classes:
            fn[classifier._class_name] += 1.0
          else:
            tn[classifier._class_name] += 1.0

    # macro average
    precision, recall, accuracy = 0.0, 0.0, 0.0
    for class_name in self._class_names:
      precision += tp[class_name] / (tp[class_name] + fp[class_name])
      recall += tp[class_name] / (tp[class_name] + fn[class_name])
      accuracy += (tp[class_name] + tn[class_name]) / (tp[class_name] + tn[class_name] + fn[class_name] + fp[class_name])

    precision /= len(self._class_names)
    recall /= len(self._class_names)
    accuracy /= len(self._class_names)
    f1 = 2.0 * precision * recall / (precision + recall)

    macro = {'precision' : precision,
             'recall'    : recall,
             'accuracy'  : accuracy,
             'f1'        : f1}

    # micro average
    tp_sum = sum(tp.values())
    fp_sum = sum(fp.values())
    tn_sum = sum(tn.values())
    fn_sum = sum(fn.values())

    precision = tp_sum / (tp_sum + fp_sum)
    recall = tp_sum / (tp_sum + fn_sum)
    accuracy = (tp_sum + tn_sum) / (tp_sum + fn_sum + tn_sum + fp_sum)
    f1 = 2.0 * precision * recall / (precision + recall)

    micro = {'precision' : precision,
             'recall'    : recall,
             'accuracy'  : accuracy,
             'f1'        : f1}

    return {'micro' : micro, 'macro' : macro}