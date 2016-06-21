import cPickle as pickle
import gzip
import logging
import numpy as np
import theano

def __reformat(dataset, labels, num_labels, image_size):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

def load_data(filename='notMNIST.pkz', num_labels=10, image_size=28):
  try:
    with gzip.open(filename) as fp:
      logging.info('Loading data from %s'%filename)
      save = pickle.load(fp)
      train_dataset = save['train_dataset']
      train_labels = save['train_labels']
      valid_dataset = save['valid_dataset']
      valid_labels = save['valid_labels']
      test_dataset = save['test_dataset']
      test_labels = save['test_labels']

      train_dataset, train_labels = __reformat(train_dataset, train_labels, num_labels, image_size)
      valid_dataset, valid_labels = __reformat(valid_dataset, valid_labels, num_labels, image_size)
      test_dataset, test_labels = __reformat(test_dataset, test_labels, num_labels, image_size)

      logging.info('Loaded data successfully')

      return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels
  except IOError as ierr:
    logging.error('Failed to load %s due to IOError %s'%(filename, str(ierr)))
    return None, None, None, None, None, None

def weights(shape, dev=0.1):
  return theano.shared(
    value=np.random.normal(scale=dev, size=shape)
  )

def bias_variable(shape, avg=0.1):
  return theano.shared(
      valud=np.zeros(shape=shape)
    )