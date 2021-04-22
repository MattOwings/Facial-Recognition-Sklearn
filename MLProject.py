
# In[IMPORTANT Declarations and Imports]

#IMPORTANT

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.datasets import fetch_lfw_people

people = fetch_lfw_people(min_faces_per_person=90, resize=0.7)
image_shape = people.images[0].shape

import warnings
warnings.filterwarnings("ignore")

mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
    
X_people = people.data[mask]
y_people = people.target[mask]

# scale the grey-scale values to be between 0 and 1
# instead of 0 and 255 for better numeric stability:
X_people = X_people / 255.


X_train, X_test, y_train, y_test = train_test_split(
    X_people, y_people, stratify=y_people, random_state=0)



# In[Plots and Print-Outs]

fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                         subplot_kw={'xticks': (), 'yticks': ()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])
    
    
# count how often each target appears
counts = np.bincount(people.target)
# print counts next to target names:
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count), end='   ')
    if (i + 1) % 3 == 0:
        print()
        
#mglearn.plots.plot_pca_faces(X_train, X_test, image_shape)

# Sample Counter
num = 0
Sample_total = 0
for x in zip(people.target_names):
    Sample_total = Sample_total  + int(counts[num])
    num = num+1
print("Number of Sample: ",Sample_total )
Sample_total = 0

print()
num = 0
total = 0
for x in zip(people.target_names):
    total = total + int(counts[num])
    num = num+1
print("Number of Sample: {}  Number of People: {}".format(total, num))
total = 0

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB #GaussianNB from naive_bayes
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
# The six classifiers we use throughout all the code

# In[N-Nearest Neighbors Classifier]

# KNN

# split the data in training and test set

# build a KNeighborsClassifier with using one neighbor:
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
#print("Test set score of 1-nn: {:.2f}".format(knn.score(X_test, y_test)))

knn.fit(X_train, y_train)
print("KNN Default: {:.2f}".format(knn.score(X_test, y_test)))

pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

knn.fit(X_train_pca, y_train)
print("KNN Default with PCA: {:.2f}".format(knn.score(X_test_pca, y_test)))

for x in range(4,6):
    knn = KNeighborsClassifier(n_neighbors=x, 
                               weights="distance")
    knn.fit(X_train_pca, y_train)
    print("KNN with PCA n-neighbors={}: {:.2f}".format(x, knn.score(X_test_pca, y_test)))

# Higher k-neighbors lowers the accuracy

# Using the weights='distance' parameter, it raises accuracy on n_neighbors =
# 5, 6, 7, 8, and 9 at 33% accurate
# weights='distance' makes the classifier prioritize cells over clusters

# Most parameters aren't helpful as they do not affect data that has been fitted

# IMPROVEMENTS: 
# DEFAULT = 31% accuracy
# WITH n_neighbors=20 AND weights="distance" = 33% accuracy


# In[Random Forest Classifier]

# RF


#play around with parameters
forest = RandomForestClassifier(max_depth=20, 
                                random_state=0)
forest.fit(X_train, y_train)

print("RF DEFAULT Accuracy: {:.2f}".format(forest.score(X_test, y_test)))



for x in range(20,21):
    forest = RandomForestClassifier(max_depth=x, 
                                    random_state=0, 
                                    class_weight="balanced")
    forest.fit(X_train, y_train)
    print("RF max-depth={}: {:.2f}".format(x, forest.score(X_test, y_test)))
    
# There is a sweet spot at max_depth=20 for RF.

# Using the parameter Class_weight="balanced" increases accuracy across the board
# class_weight="balanced" 

# IMPROVEMENT: 
# DEFAULT = 34% accuracy
# WITH max_depth=20 AND class_weight="balanced" = 38% accuracy 



# In[Naive Bayes Classifier]

# NAIVE BAYES


nb = GaussianNB()
nb.fit(X_train, y_train)

print("Naive Bayes Accuracy: {:.2f}".format(nb.score(X_test, y_test)))

# DEFAULT = 24% accuracy.

# In[Decision Trees Classifier]

# DECISION TREE


tree = DecisionTreeClassifier(max_depth=None, random_state=0)
tree.fit(X_train, y_train)

print("Default Decision Trees Accuracy: {:.2f}".format(tree.score(X_test, y_test)))

for x in range(15,16):
    it_skips = 1
    tree = DecisionTreeClassifier(max_depth=x*it_skips, 
                                  random_state=0)
    tree.fit(X_train, y_train)
    print("Decision Trees max_depth={} Accuracy: {:.2f}".format(x*it_skips, tree.score(X_test, y_test)))


# DEFAULT = 11% accuracy
# max_depth=15 12% accuracy
# class_weight="balanced" lowers accuracy


# In[Multi Layer Perceptron Classifier]

# Multi Layer Perceptron


MLP_clf = MLPClassifier()
MLP_clf.fit(X_train,y_train)
print("Default MLP Accuracy: {:.2f}".format(MLP_clf.score(X_train, y_train)))

pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
MLP_clf.fit(X_train,y_train)
print("MLP with PCA Accuracy: {:.2f}".format(MLP_clf.score(X_train, y_train)))

MLP_clf = MLPClassifier(solver='adam', alpha=1e-5,
                        hidden_layer_sizes=(26, 8), random_state=1)
MLP_clf.fit(X_train,y_train)


print("Boosted MLP Accuracy: {:.2f}".format(MLP_clf.score(X_train, y_train)))

# MLP with PCA Accuracy: Randomized


# In[Support Vector Classifier SVC]

# SVC


vector = LinearSVC()
vector.fit(X_train, y_train)


print("Default SVC Accuracy: {:.2f}".format(vector.score(X_test, y_test)))


# 50% at max_iter=50 class_weight="balanced"



for x in range(125,126):
    it_skips = 1
    vector = LinearSVC(random_state=0, 
                   max_iter=x*it_skips,
                   class_weight="balanced")
    vector.fit(X_train, y_train)
    print("max_iter={} SVC Accuracy: {:.2f}".format(x*it_skips, vector.score(X_test, y_test)))

# max_iter=50 gives 50% accuracy
# max_iter=52 gives 51% accuracy
# max_iter=125 gives 52% accuracy

# Dinc meeting 4/8
# play around with dataset, remove sets with low amounts of samples
# values with less samples less accurate less training

# Use regular SVC Classifier (SVM)
# Try Kernal parameter on both

# In[Removing Features with Few Samples]

for x in range(3,10):
    min_value = x * 10
    
    
    people = fetch_lfw_people(min_faces_per_person=min_value, resize=0.7)
    
    image_shape = people.images[0].shape
    
    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
        
    X_people = people.data[mask]
    y_people = people.target[mask]
    
    # scale the grey-scale values to be between 0 and 1
    # instead of 0 and 255 for better numeric stability:
    X_people = X_people / 255.
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_people, y_people, test_size=.25, stratify=y_people, random_state=0)
    ####### 
    ### CHANGE TEST SIZE HERE TO ALTER TRAIN TEST SPLIT
    #######
    
    
    #KNN
    knn = KNeighborsClassifier(n_neighbors=6, weights="distance")
    knn.fit(X_train, y_train)
    print("Min_Faces: {}, KNN Accuracy: {:.2f}".format(min_value, knn.score(X_test, y_test)))
    
    #RF
    forest = RandomForestClassifier(max_depth=20, random_state=0, class_weight="balanced")
    forest.fit(X_train, y_train)
    print("Min_Faces: {}, RF Accuracy: {:.2f}".format(min_value, forest.score(X_test, y_test)))
    
    #NB
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    print("Min_Faces: {}, Naive Bayes Accuracy: {:.2f}".format(min_value, nb.score(X_test, y_test)))
    
    #DT
    tree = DecisionTreeClassifier(max_depth=15, random_state=0)
    tree.fit(X_train, y_train)
    print("Min_Faces: {}, Decision Trees Accuracy: {:.2f}".format(min_value, tree.score(X_test, y_test)))

    #MLP
    MLP_clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(26, 8), random_state=1)
    MLP_clf.fit(X_train, y_train)
    print("Min_Faces: {}, MLP Accuracy: {:.2f}".format(min_value, MLP_clf.score(X_train, y_train)))

    #SVC
    vector = LinearSVC(random_state=0, max_iter=125, class_weight="balanced")
    vector.fit(X_train, y_train)
    print("Min_Faces: {}, SVC Accuracy: {:.2f}".format(min_value, vector.score(X_test, y_test)))
