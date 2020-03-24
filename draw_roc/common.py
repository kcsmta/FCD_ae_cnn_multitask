import numpy as np
import re
# from sklearn.preprocessing import label_binarize

def readPredictedResult(file,whitespace='[,]', remove_header = True, dttype= int):
    # read data
    f = open(file, 'r')
    # read data
    if remove_header:
        f.readline()
    y_true =[]
    y_pred =[]
    for line in f.readlines():
        line = re.sub(whitespace,'',line)
        line = line.strip()
        items = line.split(' ')

        y_true.append(dttype (items[0]))
        y_pred.append(dttype (items[1]))

    f.close()
    return y_true, y_pred
def readROCFile(results='', keepFormat =False):
   # file result contains Gold_Label and Scores
   # the first row is label of curves
   # keep format: group scores by rows
   # else: group scores by each column
   f = open(results,'r')
   # read data
   y_test =[]
   scores = []

   line = f.readline().strip()
   curve_labels = line.split(',')[1:]
   for line in f.readlines():
       line = line.strip()
       items = line.split(' ')
       if len(items)<2:
           continue
       items =[float(i) for i in items]
       y_test.append(items[0])
       scores.append(items[1:])

   f.close()

   if keepFormat:
       y_scores = scores
   else:
       n_results = len(curve_labels)
       y_scores=[]
       y_values = np.array(scores)
       for idx in range(0, n_results):
           y_scores.append(y_values[:,idx])

   return curve_labels, y_test, y_scores

def saveArray(array, filename, separator=', '):
       np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
       with open(filename, 'w') as f:
           f.write(np.array2string(array, separator=separator))

def loadArray(filename, dtype=None):
   array = np.loadtxt(filename, dtype=dtype, delimiter=', ')
   # if type == 'float':
   #     array = np.apply_along_axis(lambda y: [float(i) for i in y], 0, array)
   # if type == 'int':
   #     array = np.apply_along_axis(lambda y: [int(i) for i in y], 0, array)
   return array

def readTextFile(filename=''):
   # read a file and an arrays of texts of lines
   file = open(filename, "r")
   texts = []
   for line in file:
       line = line.rstrip()
       if line != '':
           texts.append(line)
   return texts
def getFileName(fullname):
    idx = fullname.rindex('/')
    return fullname[idx+1:]
# curve_labels, y_test, y_scores = readResultFile(results='flow_multiclass.txt', keepFormat=True)
# classes = np.unique(y_test)
# print 'classes: ', classes
# y_test = label_binarize(y_test, classes=classes)
# y_scores = np.array(y_scores)
# print '\n',y_test[0:2]
# print '\n', y_scores[0:2]
# n_classes = 5
# for i in range(n_classes):
#     y = y_test[:, i]
#     score = y_scores[:, i]