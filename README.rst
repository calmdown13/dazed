.. |linting| image:: https://github.com/calmdown13/dazed/workflows/Linting/badge.svg
.. |tests| image:: https://github.com/calmdown13/dazed/workflows/Tests/badge.svg
.. |pypi| image:: https://img.shields.io/pypi/v/dazed.svg
   :target: https://pypi.org/project/dazed/
.. |rtd| image:: https://readthedocs.org/projects/dazed/badge/
   :target: https://dazed.readthedocs.io/

|linting| |tests| |pypi| |rtd|

*************************************
💫 Dazed - A Confusion Matrix Package
*************************************

Dazed is little confusion matrix package designed to make your life easier.
Its key features are:

-  support for lots of different data formats (sparse integers, sparse strings, one-hot arrays, dataframes)
-  support for multilabel data
-  ability to list most confused labels
-  ability to index sample information by confused label names
-  prints out nicely


************
Installation
************
For the basic installation:

.. code-block:: console

   $ pip install dazed

To include pandas dataframe support:

.. code-block:: console

   $ pip install dazed[pandas]


***********
Basic Usage
***********
To give you an idea of why you might want to use dazed, here is a toy example
demonstrating the kind of investigation it was designed to help with. Note: I
am using sparse string labels here but dazed's interfaces can cope with integers,
onehot encoded arrays and dataframes as well (refer to the
`API Reference <https://dazed.readthedocs.io/en/latest/api_reference.html>`_
for more information).

Imagine your building a machine learning model to catalogue a pet store's
inventory (primarily cats, dogs and fish). The owner has given you an image of
each animal and you've trained your model and made some predictions. Your data
looks like:

.. code-block::

   filenames = [
      "img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg"
   ]
   truth = ["cat", "dog", "cat", "dog", "fish", "dog"]
   pred = ["cat", "dog", "dog", "cat", "fish", "cat"]

In order to understand how your model is doing, you make a quick confusion
matrix:

.. code-block::

   from dazed import ConfusionMatrix

   cm = ConfusionMatrix.from_sparse(truth, pred, info=filenames)
   print(cm)

.. code-block:: console

     | 0 1 2     index | label
   ---------     -------------
   0 | 1 1 0         0 |   cat
   1 | 2 1 0         1 |   dog
   2 | 0 0 1         2 |  fish
   ---------     -------------

From the confusion matrix it looks like the model might be prone to thinking that
dogs are actually cats. To double check:

.. code-block::

   cm.most_confused()

.. code-block:: console

   [('dog', 'cat', 2), ('cat', 'dog', 1)]

Ah yes, dogs were predicted to be cats twice and cats to be dogs
once. To try and find out what the problem might be you decide that you should
check the images. To get the appropiate images:

.. code-block::

   cm.label_pair_info("dog", "cat")

.. code-block:: console

   ['img3.jpg', 'img5.jpg']

Upon investigating the images you notice that both dogs are white. You
decide to go back through and label your images for animal colour.

.. code-block::

   truth = [
      ["cat", "white"],
      ["dog", "brown"],
      ["cat", "brown"],
      ["dog", "white"],
      ["fish", "orange"],
      ["dog", "white"]
   ]
   pred = [
      ["cat", "white"],
      ["dog", "brown"],
      ["dog", "brown"],
      ["cat", "white"],
      ["fish", "orange"],
      ["cat", "white"]
   ]
   cm = ConfusionMatrix.from_sparse(
      truth, pred, info=filenames, multilabel=True
   )
   print(cm)

.. code-block:: console

     | 0 1 2 3 4     index |        label
   -------------     --------------------
   0 | 0 0 1 0 0         0 |   cat, brown
   1 | 0 1 0 0 0         1 |   cat, white
   2 | 0 0 1 0 0         2 |   dog, brown
   3 | 0 2 0 0 0         3 |   dog, white
   4 | 0 0 0 0 1         4 | fish, orange
   -------------     --------------------

Hmm looks like all white dogs were miss classified as white cats.

.. code-block::

   cm.most_confused()

.. code-block:: console

   [('dog, white', 'cat, white', 2), ('cat, brown', 'dog, brown', 1)]

Ah yes looks like your model might be basing much of its prediction on animal
colour, maybe time to go collect some more data.

To find out more about dazed take a look at the `API Reference <https://dazed.readthedocs.io/en/latest/api_reference.html>`_.
