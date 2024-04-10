import numpy as np
import random
from IPython.display import display, SVG
from sknetwork.visualization import svg_graph
from scipy import sparse


class NewmanMixtureModel():
    ''' A Newman mixture model for community detection. '''

    def __init__(self, A, k):
        ''' Initialize the model with a graph and the number of communities. 
            Implemented for directed graphs only.'''
        self.A = A # adjacency matrix of the graph
        self.k = k # number of communities
        self.n = A.shape[0] # number of nodes
        self.m = A.sum() # number of edges
        self.graph = np.where(A)
        self.q = np.random.dirichlet(np.ones(self.k) , size=self.n) # community membership probabilities
        self.labels = np.argmax(self.q, axis=1) # community labels

    def generate(self, n):
        """ generate a random graph from the SBM """
        if n == None:
            raise ValueError("n must be specified")
        # Membership
        self.Z = np.zeros((n,self.k)) # cluster assignments (one-hot)
        # number of nodes
        self.n = n
        # initialize adjacency matrix
        self.A = np.zeros((n,n)) # adjacency matrix
        # generate cluster assignments
        self.Z = np.random.multinomial(1,self.a,size=self.n) # cluster assignments 
        self.labels = np.argmax(self.Z, axis=1) # community labels
        # generate adjacency matrix
        for i in range(self.n):
            for j in range(i+1,self.n):
                self.A[i,j] = np.random.binomial(1, self.Z[i,:]@self.P@self.Z[j,:])
                self.A[j,i] = self.A[i,j]


    def _get_neighbors(self, node):
        ''' Return the neighbors of a node. How graph is directed, both incoming and outgoing neighbors are returned, in dimension 0 and 1 respectively. '''
        return np.where(self.A[node]), np.where(self.A[:, node])
    
    def _get_out_degree(self, node):
        ''' Return the out-degree of a node. '''
        return np.sum(self.A[node])
    
    def _get_in_degree(self, node):
        ''' Return the in-degree of a node. '''
        return np.sum(self.A[:, node])
    
    def _get_community(self, node):
        ''' Return the community of a node. '''
        return np.argmax(self.q[node]) 
    
    def _get_community_size(self, community):
        ''' Return the size of a community. '''
        return sum(self._get_community(node) == community for node in range(self.n))
    
    def _get_community_edges(self, community):
        ''' Return the number of edges within a community. '''
        return sum(self._get_community(node) == community for node in self.graph[0] if self._get_community(node) == community)
    
    def _get_community_edges_to(self, community, other_community):
        ''' Return the number of edges from a community to another community. '''
        return sum(self._get_community(node) == community for node in self.graph if self._get_community(node) == other_community)
    
    def _get_community_edges_from(self, community, other_community):
        ''' Return the number of edges from a community to another community. '''
        return sum(self._get_community(node) == other_community for node in self.graph if self._get_community(node) == community)
    
    def fit(self,  A, k, max_iter=100, tol=1e-3):
        ''' Fit the model to the data. '''
        # initialize the model
        em = EM(A, k, max_iter, tol)
        # run the EM algorithm
        a, theta = em.run()

        self.a = a # community size probabilities
        self.q = em.q
        self.labels = np.argmax(em.q, axis=1)
        
        self.P = np.zeros((self.k, self.k)) # edge probabilities between communities
        for r in range(self.k):
            for s in range(self.k):
                self.P[r, s] = np.sum([theta[r,j] for j in range(self.n) if self.labels[j] == s])
        return theta
    
    def accuracy(self, labels):
        ''' Return the accuracy of the model. '''
        return np.mean(self.labels == labels)
    
    def display(self, node_size=5, edge_width=0.5, labels = None, position = None):
        ''' Display the graph. '''
        clusters = self.labels
        nodes = np.arange(self.n)
        adjacency = sparse.csr_matrix(self.A)
        if labels is None:
            labels = self.labels
        if position is None:
            position = np.zeros( (self.n,2) )
            n_community = len(set(clusters))
            for i in range(len(nodes)):
                c = clusters[i]
                position[i,:] = (np.random.multivariate_normal( [np.round(1+c/n_community)*np.cos(2*np.pi*2*c/n_community),np.round(1+c/n_community)*np.sin(2*np.pi*2*c/n_community)],
                                                            0.05*np.eye(2)))

       
        image = svg_graph(adjacency, position, labels=labels.astype(int), node_size = node_size, edge_width = edge_width)
        display( SVG(image) )


class EM(NewmanMixtureModel):
    ''' EM algorithm for Newman mixture models. '''

    def __init__(self, A, k, max_iter=100, tol=1e-5):
        ''' Initialize the EM algorithm. '''
        super().__init__(A, k)
        self.pi = np.mean(self.q, axis=0) # community size probabilities
        self.theta = np.ones((self.k, self.n)) / self.n # edge probabilities for a node in a community to a specific node
        self.max_iter = max_iter
        self.tol = tol

    def _E_step(self):
        ''' E step of the EM algorithm. '''
        for i in range(self.n):
            for r in range(self.k):
                self.q[i, r] = self.pi[r] * np.prod([self.theta[r,j] for j in self._get_neighbors(i)[0]])
            self.q[i] /= np.sum(self.q[i])
    
    def _M_step(self):
        ''' M step of the EM algorithm. '''
        self.pi = np.mean(self.q, axis=0)
        for r in range(self.k):
            for j in range(self.n):
                self.theta[r, j] = np.sum([self.q[i, r] for i in self._get_neighbors(j)[1]])
            self.theta[r] /= np.sum([self._get_out_degree(i)*self.q[i, r] for i in range(self.n)])
        
    def run(self):
        ''' Run the EM algorithm. '''
         # initial parameters
        X = [1.0+random.random() for i in range(self.k)]
        norm = sum(X)
        self.pi = [x/norm for x in X]
        
        for i in range(self.k):
            Y = [1.0+random.random() for j in range(self.n)]
            norm = sum(Y)
            self.theta[i] = [y/norm for y in Y]
            
        count = 0
        for i in range(self.max_iter):
            q_old = self.q.copy()
            self._E_step()
            self._M_step()
            count += 1
            if np.linalg.norm(self.q - q_old) < self.tol:
                print(f'Norm of difference between q and q_old: {np.linalg.norm(self.q - q_old)}')
                break
        print(f'EM algorithm converged after {count} iterations.')
        
        return self.pi, self.theta