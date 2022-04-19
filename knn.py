import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from collections import Counter
import matplotlib.pyplot as plt

#TO RUN THE CODE PLEASE USE- "python knn.py" in the command terminal

#Load the dataset
df = pd.read_csv ('iris.csv')


#create K-NN Model
def knn_algo(X, X_train, y_train, k):
    distance_lst = []
    a = [X[0], X[1], X[2], X[3]]

    for index, row in X_train.iterrows():
        b = [row[0], row[1], row[2], row[3]]
        distance_lst.append((distance.euclidean(a,b), index))

    distance_lst.sort(key = lambda x: x[0])
    distance_lst = distance_lst[:k]
    y_lst = []
    for elem in distance_lst:
        y_lst.append(y_train[elem[1]])
    
    counter = Counter(y_lst)
    y = counter.most_common(1)
    return y[0][0]

#Accuracy of training dataset
accuracy_matrix = []
for i in range(0,20):
    #Shuffle the dataset
    shuffled_df = shuffle(df)

    #Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(shuffled_df[shuffled_df.columns[0:4]], shuffled_df[shuffled_df.columns[4]], test_size=0.2)
    accuracy_lst = []

    #Normalize the attributes 
    for i in range(1,5):
        col_vals = X_train[str(i)]
        max_val = col_vals.max()
        min_val = col_vals.min()
        for index in X_train.index:
            val = (X_train.loc[index].at[str(i)]-min_val)/(max_val-min_val)
            X_train.at[index, str(i)] = val

        for index in X_test.index:
            val = (X_test.loc[index].at[str(i)]-min_val)/(max_val-min_val)
            X_test.at[index, str(i)] = val

    for k in range(1,52,2):
        train_counter = 0
        for index, row in X_train.iterrows():
            y = knn_algo(row, X_train, y_train, k)
            if y == y_train[index]:
                train_counter += 1
        accuracy = train_counter/len(y_train)
        accuracy_lst.append(accuracy)
    accuracy_matrix.append(accuracy_lst)


#Accuracy of testing dataset
# accuracy_matrix = []
# for i in range(0,20):
#     #Shuffle the dataset
#     shuffled_df = shuffle(df)

#     #Split the data into training and testing
#     X_train, X_test, y_train, y_test = train_test_split(shuffled_df[shuffled_df.columns[0:4]], shuffled_df[shuffled_df.columns[4]], test_size=0.2)
#     accuracy_lst = []
#     #Normalize the attributes 
#     for i in range(1,5):
#         col_vals = X_train[str(i)]
#         max_val = col_vals.max()
#         min_val = col_vals.min()
#         for index in X_train.index:
#             val = (X_train.loc[index].at[str(i)]-min_val)/(max_val-min_val)
#             X_train.at[index, str(i)] = val

#         for index in X_test.index:
#             val = (X_test.loc[index].at[str(i)]-min_val)/(max_val-min_val)
#             X_test.at[index, str(i)] = val    

#     for k in range(1,52,2):
#         test_counter = 0
#         for index, row in X_test.iterrows():
#             y = knn_algo(row, X_train, y_train, k)
#             if y == y_test[index]:
#                 test_counter += 1
#         accuracy = test_counter/len(y_test)
#         accuracy_lst.append(accuracy)
#     accuracy_matrix.append(accuracy_lst)

a = np.array(accuracy_matrix)
average_lst = a.mean(axis=0)
std_lst = a.std(axis=0)
k_lst = list(range(1,52,2))
k_lst = np.array(k_lst)
plt.errorbar(k_lst, average_lst, std_lst)
plt.xlabel("Value of k")
plt.ylabel("Accuracy over training data")
plt.show()