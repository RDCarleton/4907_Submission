#!/usr/bin/env python
# coding: utf-8
# import required libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import silhouette_samples, silhouette_score
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
# for tensorflow
import tensorflow as tf
# for gui
from tkinter import *

# Read the input directory for a .CSV dataset specified by the user 
def readInput():
    global df_data
    global input_text
    # Get textbox value without end character
    df_location = input_text.get("1.0",'end-1c')
    # Import dataset with pandas (CSV)
    df = pd.read_csv (df_location)
    df_removed_header = df 
    # Remove the first column which should be student numbers (or any other index scheme)
    # Obtain list of columns indices 
    df_column_numbers = [x for x in range(df.shape[1])]
    # Remove the column at index 0 (index scheme)
    df_column_numbers.remove(0)
    # Return original data without the first column 
    df_removed_header.iloc[:, df_column_numbers]
    # Check number of missing entries then sum both axis for total
    df_check_error = df_removed_header.isnull().sum().sum()
    # Check against any missing data within the file and replace with the mean of the feature
    df_filled_data = df_removed_header.fillna(df_removed_header.mean())
    df_arr = df_filled_data.to_numpy()
    df_row = len(df_arr)
    df_col = len(df_arr[0])
    df_total = df_row*df_col
    df_err = df_check_error/df_total
    df_data = df_filled_data
    print(str(df_err*100)+'% error from missing data.')
    print('Data file has been loaded.')
    
# Iterate through every column for calculating mean
# Formula follows - sum of values/number of values
def calc_mean(dataset):
    input_col = len(dataset[0]) # Number of columns
    input_r = len(dataset) # Number of rows 
    mean_matrix = [0 for i in range(input_col)]
    for i in range(input_col):
        # Add every sliced row value to variable 
        col_values = dataset[:,i]
        # Calculate mean using float64 precision digits 
        mean_matrix[i] = np.sum(col_values)/input_r
        
    return mean_matrix

# Standardization - formula follows score = (value-sample mean)/sample standard deviation
def apply_standardization(input_dataset):
    # Number of columns 
    input_columns = len(input_dataset[0]) 
    # Number of rows
    input_rows = len(input_dataset)
    # Create same size arrays for mean, variance, and standard deviation 
    input_mean = [0 for i in range(input_columns)]
    variance = [0 for i in range(input_columns)]
    input_standard_deviation = [0 for i in range(input_columns)]
    variance_sum = [0 for i in range(input_columns)]
    input_mean = calc_mean(input_dataset)
    # Iterate through every column for calculating standard deviation
    # Formula follows - square root(sum(values-population mean)^2/population size-1)
    for i in range(input_columns):
        # Calculate variance of each row 
        variance = [(row[i]-input_mean[i])**2 for row in input_dataset] 
        # Sum all of the variance values for each row before dividing 
        variance_sum[i] = sum(variance)
        # Divide each value then square root the row 
        input_standard_deviation[i] = (variance_sum[i]/(input_rows-1))**0.5
    # Iterate each row in the input 
    for row in input_dataset:
        # Iterate the number of values in each row
        for i in range(len(row)):
            # Apply standardization formula using calculated values 
            row[i] = (row[i]-input_mean[i])/input_standard_deviation[i]

# Runs functions for standardizing data
def standardize():
    # Reference global variables
    global standardized_data
    global df_data
    # Convert input data to a numbpy array
    dset = np.array(df_data)
    # Set the type of array to a float
    dset1 = dset.astype(np.float)
    # Run standardization on the dataset
    apply_standardization(dset1)
    # Set global value to standardized data 
    standardized_data = dset1

# Utilizes numbpy functions to calculate PCA value
def PCA():
    global standardized_data
    global pca_data
    # Get standardized data
    standardize() 
    # Calculate mean of the standardized dataset
    standardized_mean = calc_mean(standardized_data)
    standardized_mean = np.array(standardized_mean)
    # Center columns by subtracting the mean from the dataset
    centered_vals = standardized_data - standardized_mean 
    centered_rows = len(standardized_data)
    centered_vals = centered_vals.T
    # Find the covariance matrix from the centered dataset
    cov_applied = np.dot(centered_vals-standardized_mean[:, None],(centered_vals-standardized_mean[:, None]).T)/centered_rows
    # Find the eigenvalues and eigenvectors of the covariance matrix
    values, vectors = eig(cov_applied)
    # Project the data using the vectors
    pca_vec = vectors.T.dot(centered_vals)
    # Flip sign to enforce deterministic output 
    pca_vec = pca_vec.T * -1 
    # Slice the vectors to retain the first two principal components
    pca_data = pca_vec[:, :2]
    print('Preprocessing has completed.')

# Output a PCA plot after running the PCA function
def plotPCA():
    global pca_data
    # Plot a graph of the data after PCA
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Component 1', fontsize = 15)
    ax.set_ylabel('Component 2', fontsize = 15)
    ax.set_title('2 Component PCA', fontsize = 20) 
    ax.scatter(pca_data[:,0], pca_data[:,1])
    fig.savefig('Output/PCA_Decomposed_Graph.png')
    plt.close(fig)

def silhouette_calc():
    global pca_data
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10] # Cluster range is for 2 to 10 centroids
    silhouette_score_output = []
    silhouette_scores = [0]
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(pca_data)
        silhouette_avg = silhouette_score(pca_data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        output = "For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg
        silhouette_score_output.append(output)
    # Save data to the appropriate file 
    with open("Output/Silhouette_Scores.txt", "w") as txt_file:
        txt_file.write(str(silhouette_score_output[0])+"\n")
        txt_file.write(str(silhouette_score_output[1])+"\n")
        txt_file.write(str(silhouette_score_output[2])+"\n")
        txt_file.write(str(silhouette_score_output[3])+"\n")
        txt_file.write(str(silhouette_score_output[4])+"\n")
        txt_file.write(str(silhouette_score_output[5])+"\n")
        txt_file.write(str(silhouette_score_output[6])+"\n")
        txt_file.write(str(silhouette_score_output[7])+"\n")
        txt_file.write(str(silhouette_score_output[8])+"\n")

def kmeans():
    global num_clusters 
    global num_iterations
    global cluster_centers
    global cluster_indices
    global pca_data
    global df_data
    # Convert input data(numbpy array) to a tensor
    tf_data = tf.convert_to_tensor(pca_data, dtype=tf.float32)
    # Create empty tensors for holding clusters later
    cluster_assignments = tf.zeros([tf.shape(tf_data)[0], ], dtype=tf.int64)
    centroids_array = tf.TensorArray(tf.float32, size=num_clusters)
    # Create cluster index in range of clusters (1, num_clusters)
    compare_cluster_index = tf.range(num_clusters, dtype=tf.int64)[:, None]
    # Randomly select num_clusters worth of random values from the input dataset
    randomStart = tf.random.shuffle(tf_data)[:num_clusters]
    # Hold the current starting values in a variable
    centroids = randomStart
    # Iterate the number of times specified 
    for i in tf.range(num_iterations):
        # tf.square -> Calculate squared difference between the input and the random centroids
        # Input data has shape [n, 2] matrix, whereas centroid has shape [y,2]
        # tf.reduce_sum -> Calculate sum across elements and square root the result value for euclidean distance
        # Since we are working with an input [n, 2] matrix, where n is the number of features. These shapes are not compatible.
        # To make them compatible we create empty columns in the matrix so that the values become compatible.
        # Input data changes to [1, n, 2], whereas centroids change to [y, 1, 2]. These None columns are the addition of 1.
        # These columns of 1 can then be stretched to match the other shape allowing these to then become compatible. 
        euclidean_distance = tf.reduce_sum(tf.square(tf_data[None, :, :] - centroids[:, None, :]), axis = -1) ** 0.5 
        # Reduce the euclidean distance vector(axis=0) to the index of the smallest value across its axes  
        cluster_assignments = tf.argmin(euclidean_distance, axis=0)
        # Compare the values of the potential cluster index, to the cluster mapping for true/false map
        comparator = tf.math.equal(compare_cluster_index, cluster_assignments[None,:])
        # For each cluster
        for j in tf.range(num_clusters): 
            # Get true/false map of current centroid
            current_map = comparator[j]
            # Get of the original data points that are in the current profile (current_map -> where true)
            # Calculate centroid by taking the mean of all the data points that match the profile 
            centroid_mean = tf.reduce_mean(tf_data[current_map], axis=0)
            # Store centroid position with value
            centroids_array = centroids_array.write(j, centroid_mean)
        # Call stack() to return the tensor array values (dimension-R) as a concatenated tensor (dimension-R+1)
        centroids = centroids_array.stack()
    # Convert from tensor to numpy array to use for graphs     
    cluster_indices = cluster_assignments.numpy()
    cluster_centers = centroids.numpy()
    with open("Output/KMeans_Centroids.txt", "w") as txt_file:
        for x in range(num_clusters):
            txt_file.write(str(cluster_centers[x]))
    input_dataset = np.array(df_data)
    for i in range(num_clusters):
        output_loc = "Output/KMeans_Profile_At_Cluster_" + str(i) + ".txt"
        with open(output_loc, "w") as txt_file:
            current_map = comparator[i]
            profile_indices = input_dataset[current_map]
            for r in range(len(profile_indices)):
                txt_file.write(str(profile_indices[r])+"\n")
    map_indices()

# Arrange scatterplot color coded to nearest centroid
def map_indices(): 
    global pca_data
    global cluster_indices
    fig = plt.figure(figsize = (10,10))
    plt.scatter(pca_data[:,0],pca_data[:,1], c=cluster_indices, cmap='rainbow')
    fig.savefig('Output/KMeans_Clustered_Graph.png')
    plt.close(fig) 
    ax = plt.figure(figsize = (10,10))
    plt.scatter(pca_data[:,0],pca_data[:,1], c=cluster_indices, cmap='rainbow')
    plt.scatter(cluster_centers[:,0] ,cluster_centers[:,1], color='black')
    ax.savefig('Output/KMeans_Clustered_Graph_With_Centroids.png')
    plt.close(ax) 
    print('Kmeans has finished clustering.')

# Get iterations input value
def check_iterations():
    global num_iterations
    num_iterations = iterations_text.get("1.0",'end-1c')
    if num_iterations.isdigit():
        num_iterations = int(num_iterations)
        check_clusters()
    else:
        print("Number of iterations must be a positive integer value!")

# Get cluster input value
def check_clusters():
    global num_clusters
    num_clusters = clusters_text.get("1.0",'end-1c')
    if num_clusters.isdigit():
        num_clusters = int(num_clusters)
    else:
        print("Number of clusters must be a positive integer value!")
        
def run_kmeans():
    check_iterations()
    check_clusters()
    kmeans()
    
# Setup global variables 
root = Tk()
df_data = 0
standardized_data = 0
pca_data = 0
num_clusters = 0
num_iterations = 0
num_components = 2
cluster_centers = 0
cluster_indices = 0
cluster_mapping = 0
df_location = 0

# Set dimensions of the GUI window 
root.geometry("500x500")

# Create buttons, specify which method they run using event listener (command=)
read_input_button = Button(root, text="Fetch Input",command=readInput)
preprocessing_button = Button(root, text="Apply Preprocessing",command=PCA) 
pca_plot_button = Button(root, text="Output PCA Scatter Plot",command=plotPCA)
silhouette_button = Button(root, text="Output Silhouette Scores",command=silhouette_calc)
train_button = Button(root, text="Run Kmeans",command=run_kmeans)
#predict_button = Button(root, text="Predict Profile",command=mapIndices)
#output_model_button = Button(root, text="Output Model",command=exportModel)
#load_model_button = Button(root, text="Load model ",command=kmeansTF)

# Create textbox for inputting dataset file path
input_text = Text(root)
iterations_text = Text(root)
clusters_text = Text(root)
input_text.insert(1.0, "Dataset file location")
iterations_text.insert(1.0, "Number of Iterations")
clusters_text.insert(1.0, "Number of Clusters")

# Add buttons to window
read_input_button.place(x=0, y=50)
preprocessing_button.place(x=0, y=80)
pca_plot_button.place(x=0, y=110)
silhouette_button.place(x=0, y=140)
train_button.place(x=0, y=215)
#predict_button.place(x=0, y=265)
#output_model_button.place(x=0, y=315)
#load_model_button.place(x=0, y=365)

# Add text and scales to window 
input_text.place(x=0, y=25, height=20, width=350)
clusters_text.place(x=0, y=165, height=20, width=350)
iterations_text.place(x=0, y=190, height=20, width=350)
# Run the mainloop of the program
root.mainloop()



