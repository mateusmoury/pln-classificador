class BinaryClassifier:

  def __init__(self, id_to_text, pos_examples, neg_examples):
    '''
        id_to_text: {text_id : [token]}
        pos_examples: set(text_id)
        neg_examples: set(text_id)
    '''
    self._word_freq = {'pos' : {}, 'neg' : {}}
    self._vocabulary = set()
    self._prob_class = {'pos' : len(pos_examples) / (len(pos_examples) + len(neg_examples)),
                        'neg' : len(neg_examples) / (len(pos_examples) + len(neg_examples))}
    self.train(id_to_text, pos_examples, neg_examples)


  def train(self, id_to_text, pos_examples, neg_examples):
    ''' Trains this classifier.

        id_to_text: {text_id: [token]}
        pos_examples: set(text_id)
        neg_examples: set(text_id)
    '''
    for id_exmp in pos_examples:
      for token in id_to_text[id_exmp]:
        self._vocabulary.add(token)
        if token in self._word_freq['pos']:
          self._word_freq['pos'][token] += 1
        else:
          self._word_freq['pos'][token] = 1

    for id_exmp in neg_examples:
      for token in id_to_text[id_exmp]:
        self._vocabulary.add(token)
        if token in self._word_freq['neg']:
          self._word_freq['neg'][token] += 1
        else:
          self._word_freq['neg'][token] = 1


  def classify(self, text):
    ''' Returns True if this text belongs to this class and False otherwise.

        text: [token]
    '''
    from math import log
    prob = {'pos' : log(self._prob_class['pos']),
            'neg' : log(self._prob_class['neg'])}
    for c in prob:
      denominator = log(sum(self._word_freq[c].values()) + len(self._vocabulary))
      for token in text:
        try:
          f = self._word_freq[c][token]
        except:
          f = 0
        prob[c] += log(f + 1) - denominator
    return prob['pos'] > prob['neg']


class NAryClassifier:

  def __init__(self, class_to_ids, id_to_text):
    '''
        class_to_ids: {class_name: set(text_id)}
        id_to_text: {text_id: [token]}
    '''
    self._classifiers = {}
    self._class_names = set()
    for class_name in class_to_ids:
      self._class_names.add(class_name)
      pos_examples = class_to_ids[class_name]
      neg_examples = set([id_exmp for id_exmp in id_to_text.keys() if id_exmp not in pos_examples])
      self._classifiers[class_name] = BinaryClassifier(id_to_text, pos_examples, neg_examples)


  def get_classes_for_text(self, text):
    ''' Returns a list containing the classes' names to which this text belongs.

        text: [token]
    '''
    ret = []
    for class_name, classifier in self._classifiers.items():
      if classifier.classify(text):
        ret.append(class_name)
    return ret


  def get_metrics(self, test_set):
    ''' Returns the metrics: precision, recall, accuracy and f1, for both micro and macro averaging.

        test_set: [([token], set(class_name))]
    '''
    tp = {class_name: 0 for class_name in self._class_names}
    fp = {class_name: 0 for class_name in self._class_names}
    tn = {class_name: 0 for class_name in self._class_names}
    fn = {class_name: 0 for class_name in self._class_names}

    for (text, classes) in test_set:
      for class_name, classifier in self._classifiers.items():
        if classifier.classify(text):
          if class_name in classes:
            tp[class_name] += 1
          else:
            fp[class_name] += 1
        else:
          if class_name in classes:
            fn[class_name] += 1
          else:
            tn[class_name] += 1

    # macro average
    precision, recall, accuracy = 0.0, 0.0, 0.0
    for class_name in self._class_names:
      if tp[class_name] + fp[class_name] != 0:
        precision += tp[class_name] / (tp[class_name] + fp[class_name])
      else:
        precision += 1

      if tp[class_name] + fn[class_name] != 0:
        recall += tp[class_name] / (tp[class_name] + fn[class_name])
      else:
        recall += 1

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

    if tp_sum + fp_sum != 0:
      precision = tp_sum / (tp_sum + fp_sum)
    else:
      precision = 1.0

    if tp_sum + fn_sum != 0:
      recall = tp_sum / (tp_sum + fn_sum)
    else:
      recall = 1.0

    accuracy = (tp_sum + tn_sum) / (tp_sum + fn_sum + tn_sum + fp_sum)

    f1 = 2.0 * precision * recall / (precision + recall)

    micro = {'precision' : precision,
             'recall'    : recall,
             'accuracy'  : accuracy,
             'f1'        : f1}

    return {'micro' : micro, 'macro' : macro}


if __name__ == '__main__':
  class_to_ids = {'c' : set([1, 2, 3]), 'j' : set([4])}
  id_to_text = {1: ["chinese", "beijing", "chinese"],
                2: ["chinese", "chinese", "shangai"],
                3: ["chinese", "macao"],
                4: ["tokyo", "japan", "chinese"]}
  txt = ["chinese", "chinese", "chinese", "tokyo", "japan"]

  classifier = NAryClassifier(class_to_ids, id_to_text)
  print(classifier.get_classes_for_text(txt))
  test_set = [(txt, ["c"])]
  print(classifier.get_metrics(test_set))
