# Data Set is obtained from the research
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.23.3307&rep=rep1&type=pdf

# Preparing the data

url <- "http://s3.amazonaws.com/assets.datacamp.com/production/course_1903/datasets/WisconsinCancer.csv"

# Download the data: 
df <- read.csv(url)

# Convert the features of the data:
data <- as.matrix(df[3:32])

# Set the row names of data
row.names(data) <- df$id

# Create diagnosis vector
diagnosis <- as.numeric(df$diagnosis == "M")

# Data is prepared for exploratory data analysis.

# Principal Component Analysis (PCA) is a dimension reduction technique applied for simplifying the data and for visualizing the most important information in the data set.

# It's important to check if the data need to be scaled before performing PCA

# two common reasons for scaling data:

 # The input variables use different units of measurement.
 # The input variables have significantly different variances.

# Check the mean and standard deviation of the features of the data to determine if the data should be scaled.

colMeans(data)
apply(data, 2, sd)


# Execute PCA, scaling if appropriate:
pr <- prcomp(data, scale = TRUE)

# Look at summary of results
summary(pr)

# Use some visualizations to better understand your PCA model. 

# Create a biplot of pr
biplot(pr)

# Is it easy or difficult to understand ?

# scatter plot each observation by principal components 1 and 2, coloring the points by the diagnosis.

plot(pr$x[, c(1, 2)], col = (diagnosis + 1), 
     xlab = "PC1", ylab = "PC2")

# scatter plot each observation by principal components 1 and 2, coloring the points by the diagnosis.

plot(pr$x[, c(1, 3)], col = (diagnosis + 1), 
     xlab = "PC1", ylab = "PC3")

# Because principal component 2 explains more variance in the original data than principal component 3,
# you can see that the first plot has a cleaner cut separating the two subgroups.

# produce scree plots showing the proportion of variance explained as the number of principal components increases.

# Set up 1 x 2 plotting grid
par(mfrow = c(1, 2))

# Calculate variability of each component
pr.var <- pr$sdev^2

# Variance explained by each principal component: 
pve <- pr.var / sum(pr.var)

# ask yourself if there's an elbow in the amount of variance explained that might lead you to pick a natural number of principal components. 
# If an obvious elbow does not exist, as is typical in real-world datasets, consider how else you might determine the number of principal components to retain based on the scree plot.

# Plot variance explained for each principal component
plot(pve, xlab = "Principal Component", 
     ylab = "Proportion of Variance Explained", 
     ylim = c(0, 1), type = "b")

# Plot cumulative proportion of variance explained
plot(cumsum(pve), xlab = "Principal Component", 
     ylab = "Cumulative Proportion of Variance Explained", 
     ylim = c(0, 1), type = "b")


# Now move on towards hierarchical clustering

# As part of the preparation for hierarchical clustering, distance between all pairs of observations are computed. 
# Furthermore, there are different ways to link clusters together, with single, complete, and average being the most common linkage methods.

# Scale the data
data.scaled <- scale(data)

# Calculate the (Euclidean) distances between all pairs of observations in the new scaled dataset
data.dist <- dist(data.scaled)

# Create a hierarchical clustering model:
hclust <- hclust(data.dist, method = "complete")

# Using the plot() function, what is the height at which the clustering model has 4 clusters?

plot(hclust)

# Answer: 20

# Now compare the outputs from your hierarchical clustering model to the actual diagnoses.

# Normally when performing unsupervised learning like this, a target variable isn't available. 
# We do have it with this dataset,
# however, so it can be used to check the performance of the clustering model.


# Use cutree() to cut the tree so that it has 4 clusters.

wisc.hclust.clusters <- cutree(hclust, k = 4)

# Compare cluster membership to actual diagnoses
table(wisc.hclust.clusters, diagnosis)

# Four clusters were picked after some exploration. 
# Before moving on, you may want to explore how different numbers of clusters affect the ability of the hierarchical clustering to separate the different diagnoses.

# there are two main types of clustering: hierarchical and k-means.

# Create a k-means model on data, Be sure to create 2 clusters, corresponding to the actual number of diagnosis. 
# Also, remember to scale the data and repeat the algorithm 20 times to find a well performing model.

kmean <- kmeans(scale(data), centers = 2, nstart = 20)

# Compare k-means to actual diagnoses
table(kmean$cluster, diagnosis)

# How well does k-means separate the two diagnoses?

# Compare k-means to hierarchical clustering
table(wisc.hclust.clusters, kmean$cluster)

# Looking at the last table, it looks like clusters 1, 2, and 4 from the hierarchical clustering model can be interpreted as the cluster 1 equivalent from the k-means algorithm, 
# and cluster 3 can be interpreted as the cluster 2 equivalent.

