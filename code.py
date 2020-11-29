import numpy as np
import math
from scipy.linalg import sqrtm
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt
#import pdb


# Load data
data = np.genfromtxt("example1.txt", delimiter=",")

# Create graph
G = nx.Graph() 
G.add_edges_from(data)

# Get Adjacency matrix
aux = nx.linalg.graphmatrix.adjacency_matrix(G)
Ad = np.array(aux.todense())

# Create data matrix 
sigma = 1
k = 10 # Number of subsets
# Compute affinity matrix
def compute_affinity(Ad):
    A = np.zeros((len(Ad), len(Ad))) # Affinity matrix
    for j in range(len(data)):
        for i in range(len(data)):
            if i!=j:
                a = Ad[i]-Ad[j]
                A[i][j] = math.exp((-(np.linalg.norm(a)**2)/2*sigma))
            else:
                A[i][j] = 0
    return A

# Construct L matrix as a diagonal amtrix
def diagonal_matrix(A):
    D = np.zeros((len(A), len(A)))
    for i in range(0, len(A)):
        D[i][i]=sum(A[i,:])
    sqrtD = sqrtm(D)
    L = np.linalg.inv(sqrtD) @ A @ np.linalg.inv(D)
    return L, D



def compute_eigenvectors(L, k=k):
    evev = np.linalg.eig(L)
    evalues = evev[0]
    evectors = evev[1]
    result = []
    ind = 0
    for evalue in evalues:
        result.append(np.linalg.norm(evalue))
    norm_evalues=np.array(result) #Now we have the modulo of evalues
    indexes = norm_evalues.argsort()[-k:][::-1] # Extract k highest values indexes
    V = np.zeros((len(L), k)) # Type of evectors can be complex
    for value in indexes:
        V[:,ind] = evectors[:,value] # Stack the k highest evectors to V
        ind = ind + 1
    return V

def normalize_vectors(V):
    N = np.zeros((len(V), V.shape[1]))
    for col in range(0, V.shape[1]-1):
        vector = V[:,col]
        vector_norm = vector/np.linalg.norm(vector) # Normalize using modulo
        N[:,col] = vector_norm
    return N

def k_means_clustering(X, k=k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    return kmeans



print(compute_affinity(data))


