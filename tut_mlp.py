from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import gzip
import cPickle as pkl
import math
import os
import theano
import logging
import time
logging.basicConfig(format='%(levelname)s: [%(asctime)s] %(message)s', datefmt='%d-%b-%Y %I:%M:%S %p', level=logging.INFO)

import utils
from MLP import MLP
from sklearn.metrics import accuracy_score

image_size = 28
num_labels = 10
force_training = False
model_file = 'theano_MLP.model'
beta = 0.01
batch_size = 100
num_epochs = 30
report_freq = 1


train_dataset, train_labels, \
  valid_dataset, valid_labels, \
  test_dataset, test_labels = \
    utils.load_data(
      num_labels=num_labels,
      image_size=image_size
    )

def load_model():
  clf = None
  try:
    if os.path.isfile(model_file):
      fp = gzip.open(model_file)
      clf = pkl.load(fp)
      if not isinstance(clf, MLP):
        logging.warning('Found model file %s which had bad data'%model_file)
        clf = None
  except IOError:
    clf = None
  except KeyError:
    clf = None

  if clf is None:
    logging.info('Could not load model from disk. Will need to run training')
  else:
    logging.info('Successfully loaded model from disk.')
  return clf

def train_model(clf):
  start_time = time.time()
  num_chunks = train_labels.shape[0] / batch_size
  assert(train_labels.shape[0] % batch_size == 0)
  logging.info('Batch size: %d, Num chunks %d, Num epochs %d'%(batch_size, num_chunks, num_epochs))
  abort = False
  avg_costs = []
  for iepoch in range(num_epochs):
    epoch_costs = []
    for ichunk in range(num_chunks):
      chunk_base = ichunk * batch_size
      chunk_offset = chunk_base + batch_size
      cost = clf.fit(train_dataset[chunk_base:chunk_offset], train_labels[chunk_base:chunk_offset])
      epoch_costs.append(cost)
      
      if math.isnan(cost):
        abort = True
        logging.error('Epoch [%d] Chunk [%d] Cost %f'%(iepoch, ichunk, cost))
        logging.error('Cost is nan. Aborting...')
        break
    if abort:
      break
    avg_cost = 0.
    if epoch_costs:
      avg_cost = sum(epoch_costs) / float(len(epoch_costs))
    else:
      logging.critical('No costs for epoch %d'%iepoch)
    avg_costs.append(avg_cost)
    if epoch_costs and iepoch % report_freq == 0:
      valid_acc = accuracy_score(
        valid_labels,
        clf.predict(valid_dataset)
      )
      logging.info('Epoch [%d] Cost %f, Validation Accurary %.2f%%'%(iepoch, cost, valid_acc * 100))
  end_time = time.time()
  logging.info('Model training completed in %.0fs'%(end_time - start_time))
  plt.plot(np.arange(0, num_epochs), avg_costs)
  plt.show()
  fp = gzip.open(model_file, 'wb')
  pkl.dump(clf, fp)
  return

clf = load_model()

if clf is None:
  clf = MLP(beta=beta, n_in=image_size * image_size,
    n_hidden=1024,
    n_out=num_labels)
  train_model(clf)
elif force_training:
  logging.info('force_training is set. Model will be retrained')
  train_model(clf)

acc = accuracy_score(test_labels, clf.predict(test_dataset)) * 100
logging.info('Model accuracy %f%%'%acc)


