import numpy as np
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from sklearn.cluster import KMeans
#from src.inference_EM import EM
from math import lgamma

# Stochastic Block model class
class SBM:
    """	
    Stochastic Block Model class for fitting a clustering model.
    
    n: number of nodes
    q: number of clusters
    a - vector of cluster sizes: a_i is the probability of a node being in cluster i
    P - connectivity matrix cluster-cluster probabilities: P_ij is the probability of an edge between a node in cluster i and a node in cluster j
    """
    def __init__(self, n=None, q=None, a=None, P=None):
        """ initialize the SBM """
        if n == None:
            raise ValueError("n must be specified")
        if q == None:
            raise ValueError("q must be specified")
        if a.any() == None:
            a = np.ones(q)/q
        if P.any() == None:
            P = np.ones((q,q))/q      
        self.a = a
        self.P = P 
        self.generate(n, q)

    def generate(self, n, q=None):
        """ generate a random graph from the SBM """
        if n == None:
            raise ValueError("n must be specified")
        # number of clusters
        if q == None:
             self.q = self.P.shape[0]
        else:
            self.q = q
        # Membership
        self.Z = np.zeros((n,self.q)) # cluster assignments (one-hot)
        # number of nodes
        self.n = n
        # initialize adjacency matrix
        self.X = np.zeros((n,n)) # adjacency matrix
        self.k = np.zeros(n) # degree vector
        # generate cluster assignments
        self.Z = np.random.multinomial(1,self.a,size=self.n) # cluster assignments
        # generate adjacency matrix
        for i in range(self.n):
            for j in range(i+1,self.n):
                self.X[i,j] = np.random.binomial(1, self.Z[i,:]@self.P@self.Z[j,:])
                self.X[j,i] = self.X[i,j]
        self.k = np.sum(self.X,axis=1)

    def display_degree_distribution(self):
        """ display the degree distribution of the graph """
        import matplotlib.pyplot as plt
        plt.hist(self.k)
        plt.show()
          
        
    def get_clusters(self):
        """ return the cluster assignments """
        return self.Z
    
    def get_tau(self):
        '''Return soft assignment'''
        return self.tau
    
    def get_cluster_probs(self):
        """ return the cluster probabilities """
        return self.a
    
    def get_connectivity(self):
        """ return the cluster-cluster probabilities """
        return self.P
    
    def get_node_degrees(self):
        """ return the degree vector """
        return self.k
    
    def get_adjacency(self):
        """ return the adjacency matrix """
        return self.X
    
    def fit(self, X, q, inference_method='variational_EM'):
        """ fit the SBM to a given graph """
        if q == None:
            print("q not specified, using ..." )
        self.X = X
        self.k = np.sum(X,axis=1)
        self.n = X.shape[0]
        self.q = q
        self.P = np.ones((q,q))/q

        if inference_method == 'variational_EM':
            self.variational_EM(self)
        elif inference_method == 'spectral_clustering':
            self.spectral_clustering(self)
        elif inference_method == 'EM':
            em = EM(self.X, self.q)
            em.run()

    def accuracy(self, labels):
        """ compute the accuracy of the clustering """
        return np.sum(np.argmax(self.Z,axis=1) == labels)/self.n

    @staticmethod
    def binary_cross_entropy(x, y):
        return x*np.log(y) + (1-x)*np.log(1-y)
    
    @staticmethod
    def update_beta(self, beta):
        ' update the variational parameters beta '
        for i in range(self.n):
            for j in range(self.n):
                for q in range(self.q):
                    for l in range(self.q):
                        beta[i,j,q,l] = self.binary_cross_entropy(self.X[i,j], self.P[q,l])
        return beta	

        
    
    @staticmethod
    def b(x, pi):
        return (pi**x) * ((1-pi)**(1-x))
    
    @staticmethod
    def update_tau(self, tau_t):
        '''update the variational parameters tau '''

        tau = tau_t.copy()
        count = 0

        # Fix point iteration
        while True:
            old_tau = tau.copy()
            for i in range(self.n):
                #norm_const = np.exp(np.sum([tau[i,r] for r in range(self.q)]))
                for q in range(self.q):
                    tau[i,q] = self.a[q] * np.prod([
                        self.b(self.X[i,j], self.P[q,l]) ** tau[j,l] 
                        for j in range(self.n)
                        for l in range(self.q) if j != i
                    ])
            for i in range(self.n):
                for q in range(self.q):
                    c = np.sum(tau[i,:]) - tau[i,q]
                    tau[i,q] = tau[i,q] * (1 - c) / np.sum(tau[i,:])

            print(count)
            count += 1
            if (np.abs(tau - old_tau) < 1e-3).all():
                break

                
        return tau
    
    @staticmethod
    def spectral_clustering(self):
        ''' Spectral clustering algorithm '''
        # Adjacency matrix
        A = self.X

        # Number of clusters
        k = self.q

        # Identity matrix
        I = eye(A.shape[0])

        # Inverse of diagonal degree matrix
        D_inv = diags([1/self.k], [0])

        # Random walk normalized Laplacian matrix
        L_rw = I - (D_inv @ A)          

        _, eig_vecs = eigs(L_rw, k, which='SR')
        eig_vecs = eig_vecs.real

        kmeans = KMeans(n_clusters=k).fit(eig_vecs)

        return kmeans.labels_
    
    @staticmethod
    def initialize_clusters(self):
        '''Spectral clustering initialization'''
        # Spectral clustering 
        self.Z = np.eye(self.q)[self.spectral_clustering(self)]
        # Initialize a
        self.a = np.zeros(self.q)
        for i in range(self.q):
            self.a[i] = np.sum(self.Z == i) / self.n
        # Intialize tau
        tau = self.Z.copy() + np.random.uniform(0, 0.1, (self.n, self.q))
        for i in range(self.n):
            tau[i,:] = tau[i,:] / np.sum(tau[i,:])
        return tau

    @staticmethod
    def variational_EM(self):
        ''' perform variational EM on the SBM '''
        # initialize variational parameters

        tau = self.initialize_clusters(self)

        while True:

            # Update alpha
            self.a = np.mean(tau, axis=0)
            
            # Update pi
            for q in range(self.q):
                for l in range(self.q):
                    self.P[q,l] = np.sum([
                        tau[i,q] * tau[j,l] * self.X[i,j]
                        for i in range(self.n)
                        for j in range(self.n) if j != i
                    ]) / np.sum([
                        tau[i,q] * tau[j,l]
                        for i in range(self.n)
                        for j in range(self.n) if j != i
                    ])
            
            # Update tau
            tau_new = self.update_tau(self, tau)
            
            if (np.abs(tau - tau_new) < 1e-3).all():
                break
            else:
                tau = tau_new
        # binaryzise the cluster assignments
        self.tau = tau
        self.Z = np.eye(self.q)[np.argmax(tau, axis=1)]

  



class EM(SBM):
    '''Class for the inference of the SBM using the EM algorithm'''

    def __init__(self, X, K_up):
        '''Initialize the class
        
        X: adjacency matrix
        K_up: upper bound on the number of clusters
        '''
        SBM.__init__(self, X.shape[0], K_up)
        self.Z = np.zeros((self.n, self.q))
        self.a = np.zeros(self.q)
        self.P = np.zeros((self.q, self.q))
        self.edges_between_communities = np.zeros((self.q, self.q))
        self.non_edges_between_communities = np.zeros((self.q, self.q))
        self.nodes_in_comunity = np.zeros(self.q)
        self.ICL = 0

    @staticmethod
    def gamma(n):
        '''Compute the gamma function of n'''
        return np.math.factorial(n-1)

    @staticmethod
    def ICL_ex_criterion(self):
        '''Compute the ICL criterion for the current partition'''

        ebc0 =  1
        nebc0 = 1
        nic0 = 1

        ICL = 0

        for q in range(self.q):
            
            for l in range(self.q):
                ebc = self.edges_between_communities[q,l] + ebc0
                nebc = self.non_edges_between_communities[q,l] + nebc0

                ICL += lgamma(ebc) + lgamma(nebc) - lgamma(ebc + nebc)

            ICL += lgamma(nic0 + self.nodes_in_comunity[q]) 
        
        ICL += lgamma(self.q) - lgamma(self.q + self.n)

        return ICL
   # @staticmethod
   # def ICL_ex_criterion(self):
        '''Compute the ICL criterion for the current partition'''

        ebc0 =  np.ones((self.q, self.q), dtype = int) # IDK exactly what are the 0 matrices
        nebc0 = np.ones((self.q, self.q), dtype=int)
        nic0 = np.ones(self.q, dtype=int)

        self.update_matrices(self)

        ebc = (self.edges_between_communities + ebc0).astype(int)
        nebc = (self.non_edges_between_communities + nebc0).astype(int)
        nic = (self.nodes_in_comunity + nic0).astype(int)

        ICL = 0
        for q in range(self.q):
            for l in range(self.q):
                #ICL += lgamma(ebc0[q,l] + nebc0[q,l]) + lgamma(ebc[q,l]) + lgamma(nebc[q,l]) - lgamma(ebc[q,l] + nebc[q,l])  - lgamma(ebc0[q,l])  - lgamma(nebc0[q,l])
                ICL += lgamma(ebc[q,l]) + lgamma(nebc[q,l]) - lgamma(ebc[q,l] + nebc[q,l]) 
        ICL += np.sum([lgamma(nic[q]) for q in range(self.q)]) - lgamma(np.sum(nic)) 
        
        return ICL

    @staticmethod
    def current_partition(self):
        '''Return the current partition
        return:
                Partition: current partition s.t. Partition[i] = list of nodes in cluster i
        '''
        Partition = []
        for i in range(self.q):
            Partition.append(np.where(self.Z == i)[0])
        return Partition
    
    def num_edges_between_clusters(self, cluster1, cluster2):
        '''Return the number of edges between cluster1 and cluster2'''
        nodes_cluster1 = self.current_partition(self)[cluster1]
        nodes_cluster2 = self.current_partition(self)[cluster2]
        return np.sum(self.X[nodes_cluster1,nodes_cluster2])
    
    def num_non_edges_between_clusters(self, cluster1, cluster2):
        '''Return the number of non edges between cluster1 and cluster2'''
        nodes_cluster1 = self.current_partition(self)[cluster1]
        nodes_cluster2 = self.current_partition(self)[cluster2]
        return np.sum(1 - self.X[nodes_cluster1,nodes_cluster2])
    
    def num_nodes_in_cluster(self, cluster):
        '''Return the number of nodes in cluster'''
        return len(self.current_partition(self)[cluster])

    @staticmethod
    def update_matrices(self):
        '''Update the matrices for the computation of the ICL'''
            # matrices for the computation of the ICL
        self.edges_between_communities = np.zeros((self.q, self.q))
        self.non_edges_between_communities = np.zeros((self.q, self.q))
        Partitions = self.current_partition(self)
        for k, pk in enumerate(Partitions):
            for l, pl in enumerate(Partitions):
                self.edges_between_communities[k,l] = np.sum(self.X[pk,:][:,pl])
                self.non_edges_between_communities[k,l] = np.sum(1 - self.X[pk,:][:,pl])

    @staticmethod
    def B(self, a, b):
        '''Compute the beta function of a and b'''
        return (lgamma(a) * lgamma(b)) / lgamma(a + b)

    @staticmethod
    def delta_ICL(self, Z_test, cluster, q, i):
        '''Compute the difference in ICL criterion when moving node i from cluster cluster to cluster q'''

        g = cluster
        h = q

        ebc0 =  np.ones((self.q, self.q), dtype = int) # IDK exactly what are the 0 matrices
        nebc0 = np.ones((self.q, self.q), dtype=int)
        nic0 = np.ones(self.q, dtype=int)

        ebc = (self.edges_between_communities + ebc0).astype(int)
        nebc = (self.non_edges_between_communities + nebc0).astype(int)
        nic = (self.nodes_in_comunity + nic0).astype(int)

            
        delta_i = np.zeros((self.q, self.q)) # changes in edges_between_communities
        for l in range(self.q):
            delta_i[h, l] += np.sum(self.Z[:,l] * self.X[i,:])
            delta_i[g, l] -= np.sum(self.Z[:,l] * self.X[i,:])
            if l == h:
                delta_i[h, l] += np.sum(self.Z[:,h] * self.X[:, i])
                delta_i[g, l] += np.sum(self.Z[:,g] * self.X[:, i])
            elif l == g:
                delta_i[h, l] -= np.sum(self.Z[:,h] * self.X[:, i])
                delta_i[g, l] -= np.sum(self.Z[:,g] * self.X[:, i])
            
        ro_i = np.zeros((self.q, self.q)) # changes in non_edges_between_communities
        ro_i = - delta_i
        ro_i[h, h] += 2 * (nic[h] - nic0[h] - self.Z[i,h])
        ro_i[g, g] -= 2 * (nic[g] - nic0[g] + self.Z[i,g])
        ro_i[h, g] +=  nic[h] - nic0[h] - self.Z[i,h] - (nic[g] - nic0[g] - self.Z[i,g])
        ro_i[g, h] +=  nic[h] - nic0[h] - self.Z[i,h] - (nic[g] - nic0[g] - self.Z[i,g])


        delta_ICL = 0
        if np.sum(Z_test[:, g]) == 0: ## g empty
            nic0 = np.ones(self.q - 1, dtype=int)
            arg = ((nic[h]/nic0) * lgamma((self.q - 1) * nic0) * lgamma( self.q * nic0 + self.n) /
                    (lgamma(self.q * nic0) * lgamma((self.q - 1) * nic0 + self.n)))
            delta_ICL += np.log(arg)
            for q in range(self.q):
                if q != g:
                    delta_ICL += np.log((self.B(ebc[q,h] + delta_i[q,h], nebc[q,h] + ro_i[q,h]))/(self.B(ebc[q,h], nebc[q,h])))
                    delta_ICL += np.log((self.B(ebc[h,q] + delta_i[h,q], nebc[h,q] + ro_i[h,q]))/(self.B(ebc[h,q], nebc[h,q])))
                else:
                    delta_ICL += np.log(self.B(ebc0[g,q], nebc0[g,q])/self.B(ebc[g,q], nebc[g,q]))
                    delta_ICL += np.log(self.B(ebc0[q,g], nebc0[q,g])/self.B(ebc[q,g], nebc[q,g]))
        
            
            
        else: ## g not empty
            delta_ICL += np.log((nic[h])/(nic[g] - 1))
            for l in range(self.q):
                delta_ICL += ((np.log((self.B(ebc[g,l] + delta_i[g,l], nebc[g,l] + ro_i[g,l]))/(self.B(ebc[g,l], nebc[g,l])))
                            + np.log((B(ebc[h,l] + delta_i[h,l], nebc[h,l] + ro_i[h,l]))/(B(ebc[h,l], nebc[h,l])))))
                if l != g and l != h:
                    delta_ICL += ((np.log((self.B(ebc[l,g] + delta_i[l,g], nebc[l,g] + ro_i[l,g]))/(self.B(ebc[l,g], nebc[l,g])))
                                + np.log((self.B(ebc[l,h] + delta_i[l,h], nebc[l,h] + ro_i[l,h]))/(self.B(ebc[l,h], nebc[l,h])))))
            
        return delta_ICL


    def run(self):
        ''' Performs the EM algorithm for the integrated complete likelihood criterion on the SBM presented in the paper:
            "Model selection and clustering in stochastic block models based on the exact inte- grated complete data likelihood" by Etienne Côme and Pierre Latouche 
            
            K_up: upper bound on the number of clusters
            '''
        
        #self.q = K_up
        _ = self.initialize_clusters(self)
        self.a = np.sum(self.Z, axis=0) / self.n
        self.P = np.zeros((self.q, self.q))

        # matrices for the computation of the ICL
        self.update_matrices(self)
        self.nodes_in_comunity = np.sum(self.Z, axis=0)

        print(self.Z)

        # While the iteration still improves the ICL
        while True:
            ICL = self.ICL_ex_criterion(self)
            print(ICL)
            Z_test = self.Z.copy()
            # Iter for each node randomly selecteds
            for i in np.random.permutation(self.n): 
                cluster = np.argmax(self.Z[i,:])
                ICL_perm = np.zeros(self.q)
                ICL_perm[cluster] = ICL # ICL for the current cluster assignment

                # Compute the ICL for each possible cluster assignment
                Z_test[i,cluster] = 0
                self.Z[i,cluster] = 0 
                for q in range(self.q):
                    if q != cluster:
                        self.Z[i,q] = 1
                        Z_test[i,q] = 1
                        #ICL_perm[q] = self.Delta_ICL(self, Z_test, cluster, q)
                        ICL_perm[q] = self.ICL_ex_criterion(self)
                        Z_test[i,q] = 0
                        self.Z[i,q] = 0
                
                self.Z[i,:] = np.eye(self.q)[np.argmax(ICL_perm)] # Update the cluster assignment
        
                # If the cluster is empty, delete it
                if len(self.current_partition(self)[cluster])== 0: 
                    self.Z = np.delete(self.Z, cluster, axis=1)
                    self.q = self.q - 1
                    self.nodes_in_comunity = np.delete(self.nodes_in_comunity, cluster)
                    self.P = np.delete(self.P, cluster, axis=0)
                    self.P = np.delete(self.P, cluster, axis=1)
                else:
                    self.nodes_in_comunity[cluster] -= 1
                
                # update the matrices for the computation of the ICL
                self.update_matrices(self)
                self.nodes_in_comunity[np.argmax(ICL_perm)] += 1
        


            new_ICL = self.ICL_ex_criterion(self)
            if new_ICL > ICL:
                ICL = new_ICL
            else:
                break
                
        # Update the parameters
        self.a = np.sum(self.Z, axis=0) / self.n
        print(self.a)
        # Update P
        for q in range(self.q):
            for l in range(self.q):
                self.P[q,l] = self.edges_between_communities[q,l] / (self.nodes_in_comunity[q] * self.nodes_in_comunity[l])
        print(self.P)
        print(self.q)
        print(np.argmax(self.Z, axis=1))

            

            







        


