import numpy as np

def bounds(samples):
    res = np.sort(samples, axis=0)
    max = res[0]
    min = res[len(res)-1]
    return max, min


def point_dist(X, Y):
    sum = 0
    for i in range(0,len(X)):
        xpos = X[i]
        ypos = Y[i]
        sum += (xpos-ypos) * (xpos-ypos)
    result = (sum)**(1/2)
    return result



class KMeans():
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.means = None

    def update_assignments(self, features, means):
        
        clusters = []
        for k in range(0,len(means)):
            clusters.append([])

        for i in range(0,len(features)):
            cur_high_index = 0
            cur_low_dist = point_dist(features[i],means[0])
            for m in range(0,len(means)):
                cur_dist = point_dist(features[i],means[m])
                if cur_dist < cur_low_dist:
                    cur_low_dist = cur_dist
                    cur_high_index = m
            clusters[cur_high_index].append(features[i])
        return clusters



    
    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        max_bound, min_bound = bounds(features)
        
        initial_means = np.empty([self.n_clusters,np.shape(features)[1]])
        for i in range(0,self.n_clusters):
            rand_mean = np.empty(np.shape(features)[1])
            for j in range(0,np.shape(features)[1]):
                rand_mean[j] = np.random.uniform(min_bound[j],max_bound[j])
            initial_means[i]=rand_mean
        
        
        self.means = initial_means
        old_means = np.empty(np.shape(initial_means))

        while (self.means != old_means).all():
            old_means = self.means
            
            
            clusters = self.update_assignments(features, old_means)
            #have to check if one of the clusters is empty
            empty = False
            for i in range(0,len(clusters)):
                if len(clusters[i]) == 0:
                    empty = True

            if empty:
                initial_means = np.empty([self.n_clusters,np.shape(features)[1]])
                for i in range(0,self.n_clusters):
                    rand_mean = np.empty([1,np.shape(features)[1]])
                    for j in range(0,np.shape(features)[1]):
                        rand_mean[0,j] = np.random.uniform(min_bound[j],max_bound[j])
                    initial_means[i]=rand_mean
                
                self.means = initial_means
            else:
                new_means = []
                for i in range(0,self.n_clusters): 
                    new_means.append(np.mean(np.array(clusters[i]), axis=0))
                new_means = np.array(new_means)
            self.means = new_means
                    
            
            
        


    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """
        predictions = np.empty([len(features)])
        for i in range(0,len(features)):
            cur_index = 0
            cur_low_dist = point_dist(features[i],self.means[0])
            for j in range(1,len(self.means)):
                cur_dist = point_dist(features[i],self.means[j])
                if cur_dist < cur_low_dist:
                    cur_index = j
                    cur_low_dist = cur_dist
            predictions[i] = cur_index

        return predictions


