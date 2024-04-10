import numpy as np

from scipy import sparse

from IPython.display import display, SVG
from sknetwork.visualization import svg_graph


class keystones_graph():
    ''' 108 vertices. 100 of those vertices are divided into four groups of 25 each and
         directed edges are placed uniformly at random between them
           such that the mean degree (both in and out) is ten. 
           The remaining eight vertices are denoted keystone vertices and the other vertices link to them depending on their group membership'''
    
    def __init__(self):
        self.vertices = [i for i in range(108)]
        self.edges = []
        self.adjacency = np.zeros((108,108))
        self.degree = np.zeros(108)
        self.in_degree = np.zeros(108)
        self.out_degree = np.zeros(108)
        self.group = np.zeros(108)
        self.keystone = np.zeros(108)
        self.group[0:25] = 1
        self.group[25:50] = 2
        self.group[50:75] = 3
        self.group[75:100] = 4
        self.generate_graph()

    def generate_graph(self):
        '''generate the graph'''
        self.generate_edges()
        self.generate_adjacency()
        self.generate_degree()
        self.generate_in_degree()
        self.generate_out_degree()
        self.generate_keystone()

    def generate_edges(self):
        '''generate edges'''
        # for first group
        for i in range(25):
            for j in np.random.choice(100,10,replace=False):
                if i != j:
                    self.edges.append((i,j))
             # first group connected to keystone {1,2,3,4}
            for j in range(0,4):
                self.edges.append((i,100+j))
            #self.edges.append((i,100)) 
            #self.edges.append((i,101))       
        # for second group
        for i in range(25,50):
            for j in np.random.choice(100,10,replace=False):
                if i != j:
                   self.edges.append((i,j))
            # second group connected to keystone {3,4,5,6}
            for j in range(2,6):
                self.edges.append((i,100+j))
            #self.edges.append((i,102))
            #self.edges.append((i,103))


        # for third group   
        for i in range(50,75):
            for j in np.random.choice(100,10,replace=False):
                if i != j:
                    self.edges.append((i,j))
            # third group connected to keystone {5,6,7,8}
            for j in range(4,8):
            #    print(j)
                self.edges.append((i,100+j))
            #self.edges.append((i,104))
            #self.edges.append((i,105))


        # for fourth group
        for i in range(75,100):
            for j in np.random.choice(100,10,replace=False):
                if i != j:
                   self.edges.append((i,j))
            # fourth group connected to keystone {7,8,1,2}
            for j in range(6,8):
                self.edges.append((i,100+j))
            self.edges.append((i,100))
            self.edges.append((i,101))       
            #self.edges.append((i,106))
            #self.edges.append((i,107))                 

    def generate_adjacency(self):
        '''generate adjacency matrix'''
        for edge in self.edges:
    
            self.adjacency[edge[0],edge[1]] = 1

    def generate_in_degree(self):
        '''generate in degree'''
        self.in_degree = np.sum(self.adjacency, axis=1)

    def generate_out_degree(self):
        '''generate out degree'''
        self.out_degree = np.sum(self.adjacency, axis=0)
    def generate_degree(self):
        '''generate degree'''
        self.degree = self.in_degree + self.out_degree

    def generate_keystone(self):
        '''generate keystone'''
        self.keystone[100:108] = 1

    def display(self):
        '''display graph'''
        position = np.zeros( (108,2) )
        n_community = len(set(self.group))
        for i in range(108):
            c = self.group[i]
            position[i,:] = (np.random.multivariate_normal( [np.round(1+c/n_community)*np.cos(2*np.pi*2*c/n_community),np.round(1+c/n_community)*np.sin(2*np.pi*2*c/n_community)],
                                                            0.05*np.eye(2)))
        
        adjacency = sparse.csr_matrix(self.adjacency)
        image = svg_graph(adjacency, position, labels=self.group.astype(int), node_size=2, edge_width=0.05)
        display( SVG(image) )