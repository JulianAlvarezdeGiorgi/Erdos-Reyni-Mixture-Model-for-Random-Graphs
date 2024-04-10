import numpy as np
from scipy import sparse
from IPython.display import display, SVG
from sknetwork.visualization import svg_graph
import networkx as nx
from src.Dataset import BaseDataset
import importlib
import random 
import collections

class BaseCommunitiesGraph(BaseDataset):
    """
    Generate a synthetic n-communities graph. Uses scikit-network to display graph.
    """

    def __init__(self, n_community, p, q, n, cf, seed=0):
        """
        Generate a random graph with n_community communities, p intra-community probability, q inter-community probability and n nodes.
        
        Parameters
        ----------
        n_community : int
            Number of communities   
        p : float   
            Probability of an edge within a community
        q : float
            Probability of an edge between communities
        n : int
            Number of nodes
        cf : np.array((n_community)) 
            Community frequency.
        """
        super(BaseDataset, self).__init__()

        np.random.seed(seed)
        
        label = np.zeros(n, dtype=int)
        position = np.zeros( (n,2) )
        adjacency = np.zeros( (n,n) )
        
        if cf.shape[0] != n_community:
            print('Community frequencies shape must match n_community')
            return 1
        elif np.sum(cf) != 1:
            print('community frequencies must sum to 1')
            return 2
        community_labels = np.arange(n_community)
        for i in range(n):
            label[i] = np.random.choice(community_labels, size=1, replace=False, p=cf)
            c = label[i]
            position[i] = np.random.multivariate_normal( [np.round(1+c/n_community)*np.cos(2*np.pi*2*c/n_community),np.round(1+c/n_community)*np.sin(2*np.pi*2*c/n_community)],
                                                        0.05*np.eye(2) )

        for i in range(n): 
            for j in range(i+1,n):
                if label[i] == label[j]:
                    adjacency[i,j] = int(np.random.binomial(1, p)) 
                else:
                    adjacency[i,j] = int(np.random.binomial(1, q))
                adjacency[j,i] = adjacency[i,j]
        
        self.adjacency = adjacency
        self.features = np.ones( (self.adjacency.shape[0],1) )
        self.labels = label
        self.positions = position

        self.n_community = n_community
        self.p = p
        self.q = q

        self.n = n
        self.n_edges = np.count_nonzero(self.adjacency)
        self.n_features = self.features.shape[1]
        self.n_classes = self.n_community

        self.name = 'Communities'
        self.description = 'Communities dataset'

    def display(self, labels=None):
        """
        Display the graph.
        """
        if labels is None:
            labels=self.labels
        image = svg_graph(sparse.csr_matrix(self.adjacency), self.positions, labels=labels, node_size=2, edge_width=0.05)
        display( SVG(image) )

        
class SBM(BaseCommunitiesGraph):
    def __init__(self, n_community, p, q, n, cf, seed=0):
        """
        Generate a random graph with n_community communities, p intra-community probability, q inter-community probability and n nodes.
        
        Parameters
        ----------
        n_community : int
            Number of communities   
        p : float   
            Probability of an edge within a community
        q : float
            Probability of an edge between communities
        n : int
            Number of nodes
        cf : np.array((n_community)) 
            Community frequency. n_community-size-vector with componentes P(community = i).
        """
        super(BaseCommunitiesGraph, self).__init__()
        
        
        if cf.shape[0] != n_community:
            print('Community frequencies shape must match n_community')
            return 1
        elif ~np.isclose(np.sum(cf),1.):
            print('community frequencies must sum to 1')
            return 2
        if n_community > 2:
            if q.shape[0] != q.shape[1]:
                print('q must be squared')
                return 3
            elif q.shape[0] != n_community:
                print('q must be n_community by n_community')
                return 4
        if p.shape[0] != n_community:
            print('p shape must equal the number of communities')
            return 5
        
        np.random.seed(seed) # Set random seed
        
        label = np.zeros(n, dtype=int) # Initialize vector of labels
        position = np.zeros( (n,2) ) # Initialize vector of positions (for viz purposes)
        adjacency = np.zeros( (n,n) ) # Init. adjacency matrix
        
        community_labels = np.arange(n_community) # Vector with community labels
        label = np.random.choice(community_labels, size=n, replace=True, p=cf) # Randomly assign points to labels, following cf distrib.
        
        mu = np.random.multivariate_normal( mean=np.array([0,0])
                                           , cov=4*np.eye(2)
                                           , size=n_community) # Rample randomly for 2D-positions mean of the communities
        
        # Loop through communities and assign positions to the points in each community.
        # They are "close" in euclidean terms, but just for viz. purposes
        for community in community_labels:
            position[label == community] = np.random.multivariate_normal(
                                                                        mean = mu[community]
                                                                        ,cov = 0.1*np.eye(2)
                                                                        ,size = np.sum(label==community)  
                                                                        )
        
        # Generate edges between nodes
        for i in range(n):
            for j in range(i+1,n):
                # Intra class
                if label[i] == label[j]:
                    adjacency[i,j] = int(np.random.binomial(1, p[label[i]]))
                # Extra class
                else:
                    adjacency[i,j] = int(np.random.binomial(1, q[label[i]][label[j]]))
                # Symmetry
                adjacency[j,i] = adjacency[i,j]
        
        # Save the computed information in the object's attributes
        self.adjacency = adjacency
        self.features = np.ones( (self.adjacency.shape[0],1) )
        self.labels = label
        self.positions = position

        self.n_community = n_community
        self.p = p
        self.q = q

        self.n = n
        self.n_edges = np.count_nonzero(self.adjacency)
        self.n_features = self.features.shape[1]
        self.n_classes = self.n_community

        self.name = 'SBM Graph'
        self.description = 'SBM Graph'
        # Generate the networkx format graph so we can interact with other APIs easily
        self.networkx_format = nx.Graph(incoming_graph_data=self.adjacency)
        
    def get_labels(self):
        return self.labels
    
    def get_nx_format(self):
        '''Gets the networkx formated graph.'''
        return self.networkx_format
    
class BaseEstimator:
    def __init__(self):
        NotImplemented
    def fit(self):
        NotImplemented
        
        
class EM_Estimator(BaseEstimator):
    '''
    Estimator based on the EM algorithm.
    
    G <nx.Graph> Graph to cluster.
    k <int> Number of communities to find.
    max_iter <int> Maximum number of iterations of the algorithm.
    '''
    def __init__(self, G, k, max_iter = 100):
        self._G = G # NetworkX Graph
        self._n = len(self._G.nodes) # Nb of nodes
        self._k = k # Nb of classes
        self._pi = [] # P(class = i)
        self._theta = [] # ?
        self._max_iter = max_iter # Max nb of iter
        
    def e_step(self, q):
        for i in range(self._n):
            q.append([])
            norm = 0.0
            for g in range(self._k):
                x = self._pi[g]
                for j in self._G.neighbors(i):
                    x *= self._theta[g][j]
                q[i].append(x)
                norm += x
            for g in range(self._k):
                q[i][g] /= norm
    
    def m_step(self, q):
        for g in range(self._k):
            sum1 = 0.0
            sum3 = 0.0
            for i in range(self._n):
                sum1 += q[i][g]
                sum2 = 0.0
                for j in self._G.neighbors(i):
                    sum2 += q[j][g]
                self._theta[g][i] = sum2  # update theta
                sum3 += q[i][g]*len(list(self._G.neighbors(i)))
            self._pi[g] = sum1/self._n  # update pi
            for i in range(self._n):
                self._theta[g][i] /= sum3 # norm
        
    def estimate_VEM(self):
        
        # initial parameters
        X = [1.0+random.random() for i in range(self._k)]
        norm = sum(X)
        self._pi = [x/norm for x in X] # Proba of a given class
        
        for i in range(self._k):
            Y = [1.0+random.random() for j in range(self._n)]
            norm = sum(Y)
            self._theta.append([y/norm for y in Y])
        
        q_old = []
        for iter_time in range(self._max_iter):
            q = []
            # E-step
            self.e_step(q)
            # M-step
            self.m_step(q)
                    
            if(iter_time != 0):
                deltasq = 0.0
                for i in range(self._n):
                    for g in range(self._k):
                        deltasq += (q_old[i][g]-q[i][g])**2
                #print "delta: ", deltasq
                if(deltasq < 0.05):
                    #print "iter_time: ", iter_time
                    break
            
            q_old = []
            for i in range(self._n):
                q_old.append([])
                for g in range(self._k):
                    q_old[i].append(q[i][g])
        
        communities = collections.defaultdict(lambda:set())
        for i in range(self._n):
            c_id = 0
            cur_max = q[i][0]
            for j in range(1,self._k):
                if q[i][j] > cur_max:
                    cur_max = q[i][j]
                    c_id = j
            communities[c_id].add(i)
        self.communities = communities.values()
    
class VEM(BaseEstimator):
    def __init__(self, X):
        self.adjacency = X
    
    def fit(self, q, maxiter, seed=12):
        np.random.seed(seed)
        self._n = self.adjacency.shape[0]
        self._q = q

        self._alpha = np.ones((q,1)) +  np.random.random(size=(self._q, 1))# P(class q)
        self._alpha /= np.sum(self._alpha)

        self._Pi = np.ones((q,q)) +  np.random.random(size=(self._q, self._q))# P(link between classes)
        self._Pi /= np.sum(self._Pi, axis=1, keepdims=True)

        self._tau = np.ones((self._n, self._q)) + np.random.random(size=(self._n, self._q)) # P( i in class q)
        self._tau /= np.sum(self._tau, axis=1, keepdims=True)

        for iter in range(maxiter):
            self.mstep()
            self.estep()
            #self._likelihood 
            
    def binom(self,connected,pi):
        return pi**connected * (1-pi)**(1-connected)
    
    def mstep(self):
        self.updateAlpha()
        self.updatePi()
    def estep(self):
        self.updateTau()

    def updateTau(self):
        _tmp_tau = np.empty((self._n,self._q))
        #_tmp_lambdas = np.zeros((self._n, self._q))
#
        #for q in range(self._q):
        #    #_tmp_tau[:,q] = self._alpha[q]
        #    for i in range(self._n):
        #        _placeHolder = 1
        #        # Here I am at a given Tau_iq
        #        for j in range(self._n):
        #            if j != i:
        #                for l in range(self._q):
        #                    beta = (self.binom(self.adjacency[i,j], self._Pi[q,l]))**self._tau[j,l]
        #                    _placeHolder *= beta
        #            else:
        #                continue
        #    _tmp_lambdas[i,q] += _placeHolder    
        for q in range(self._q):
            #_tmp_tau[:,q] = self._alpha[q]
            for i in range(self._n):
                # Here I am at a given Tau_iq
                _tmp_tau[i,q] =  self._alpha[q] #* (-1) * np.log(self._alpha[q] * _tmp_lambdas[i,q])
                for j in range(self._n):
                    if j != i:
                        for l in range(self._q):
                            beta = (self.binom(self.adjacency[i,j], self._Pi[q,l]))**self._tau[j,l]
                            _tmp_tau[i,q] *= beta
                    else:
                        continue
            #_tmp_tau[i,:] /=  np.sum(_tmp_tau[i,:])
        _nrm_tmp_tau = _tmp_tau/np.sum(_tmp_tau, axis=1, keepdims=True)
        self._tau = _nrm_tmp_tau
        
    def updateAlpha(self):
        for q in range(self._q):
            self._alpha[q] = np.sum(self._tau[:,q])/self._n
        
    def updatePi(self):
        for q in range(self._q):
            for l in range(self._q):
                _cur_prod = 0
                _cur_denom = 0
                _cur_num = 0
                for i in range(self._n):
                    for j in range(i+1,self._n):
                        if i != j:
                            cur_prod = self._tau[i,q] * self._tau[j,l]
                            _cur_num += (cur_prod * self.adjacency[i,j])
                            _cur_denom += cur_prod
                        else:
                            continue
                
                self._Pi[q,l] = _cur_num/_cur_denom
    
    def get_communities(self):
        community_assignments = np.argmax(self._tau, axis=1)
        list_of_communities = []
        for q in range(self._q):
            list_of_communities.append([community_assignments[community_assignments==q]])
        return list_of_communities
        