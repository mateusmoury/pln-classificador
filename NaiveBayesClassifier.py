class NaiveBayesClassifier:

  def __init__(self, trainset):
    self.trainset = trainset
    self.classes = []
    self.train()

  def train(self):
    '''
      Trains this classifier.
    '''
    pass

  def prob_classify(self, text):
    '''
      Returns a dictionary in which the key is the class and the value
      is the probability of the argument belong to this class.
    '''
    pass

  def classify(self, text):
    '''
      Returns which class the argument has the highest probability of
      belonging to.
    '''
    probs = self.prob_classify(text)
    return max(probs, key = probs.get)

  def class_id(self, c):
    return self.classes.index(c)

  def get_metrics(self, testset):
    '''
      Returns precision, recall, accuracy and f1 metrics of the classifier
      regarding this test set.
    '''
    matrix = []
    for i in range(0, len(self.classes)):
      matrix.append([0] * len(self.classes))

    for (text,real_class) in testset:
      c = self.classify(text)
      matrix[class_id(real_class)][class_id(c)] += 1

    precision, recall, accuracy = 0, 0, 0
    tpall, fpall, tnall, fnall = 0, 0, 0, 0

    for i in range(0, len(self.classes)):
      tp = matrix[i][i]
      fp = 0
      tn = 0
      fn = 0
      
      for j in range(0, len(self.classes)):
        if i != j:
          fp += matrix[j][i]
          tn += matrix[j][j]
          fn += matrix[i][j]
      
      precision += tp / (tp + fp)
      recall += tp / (tp + fn)
      accuracy += (tp + tn) / (tp + tn + fp + fn)

      tpall += tp
      fpall += fp
      tnall += tn
      fnall += fn

    precision /= len(self.classes)
    recall /= len(self.classes)
    accuracy /= len(self.classes)
    f1 = 2.0 * precision * recall / (precision + recall)

    macro = {'precision' : precision,
            'recall' : recall,
            'f1' : f1,
            'accuracy' : accuracy}

    precision = tpall / (tpall + fpall)
    recall = tpall / (tpall + fnall)
    accuracy = (tpall + tnall) / (tpall + tnall + fpall + fnall)
    f1 = 2.0 * precision * recall / (precision + recall)

    micro = {'precision' : precision,
            'recall' : recall,
            'f1' : f1,
            'accuracy' : accuracy}

    return {'micro' : micro, 'macro' : macro}
