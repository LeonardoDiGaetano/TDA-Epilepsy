from networkx.readwrite import json_graph
from scipy.stats import percentileofscore
from traceback import format_exception
import cProfile
import json
import networkx as nx
import numpy as np
import pandas as pd
import pstats
import random
import sys
import matplotlib.pyplot as plt
import itertools

from multiprocessing import Pool
import tqdm
import pickle
from scipy import stats
import scipy.io
from pathlib import Path
plt.rcParams["figure.facecolor"]="white"

from scipy.sparse import lil_matrix

def load_file(file_path):
    with open(file_path, "rb") as f:
        file = pickle.load(f)
    return file
    
def Curv_density(G, verbose=False):
    """Compute nodal curvature (Knill's curvature) based on density
    
    Parameters
    ---------   
    G: networkx network object
        

    Returns
    -------
    
    curv: numpy array
        array with curvature values
        
    """
    
    def DIAGNOSTIC(*params) :
        if verbose : print(*params)
    DIAGNOSTIC('This function run over all nodes and computes the curvature of the nodes in the graph' )
    
    # This is the initial Graph
    #fden, 
    G_den = G#densthr(d,i) 
    
    if len(G_den.nodes()) == 0:
        return []
    # Enumerating all cliques of G up to a certain size
    temp = list(nx.enumerate_all_cliques(G))#Kmax_all_cliques(G_den)
    
    # This lista is a vector V where each v_i is the number of cliques of size i
    lista = {n:None for n in G_den.nodes()}
    
    # We suppose that the size of the cliques are smaller than 50, so we create an 
    # empty list of size 50 for the lista
    for i in G_den.nodes():
        # We start with empty scores for the curvature
        # creating a list of lists for each node - all empty for the 
        # scores for each size for each node
        #print(i)
        lista[i] = [0] * 50
    
    DIAGNOSTIC('These are all cliques of the Network:')
    # THIS WILL PRINT ALL THE CLIQUES
    DIAGNOSTIC(temp)
    
    DIAGNOSTIC('We now print the curvature/clique score of each node in the network')
    
    # Now we run over all nodes checking if the node belongs to one clique or another
    Sc = []
    for node in G_den.nodes(): # now we run the script for each clique
        score = 0 # This is the initial score of the node in the participation rank
        for clique in temp:
            # Checking the size of the clique
            k = len(clique)
            # If this node is in the clique, we update the curvature
            if node in clique:
                score += 1 # If the node is in the clique raises the score
                # Increases the curvature score for a size k with a 
                # different weight due to Gauss-Bonnet theorem - is k-1 since 
                # len>0 and python starts from zero.
                lista[node][k-1] += (-1)**(k+1)*1/k 
        Sc.append(score)
        
        DIAGNOSTIC('The node '+str(node)+' has score ='+str(score))
    
    total=[]
    for elements in lista.values():
        #print(elements)
        # Summing the participation in all sizes, so that we can compute the 
        # curvature (TOTAL IS ACTUALLY THE CURVATURE - WITHOUT NORMALIZATION)
        total.append(sum(elements)) # This is good if one wants to normalize by the maximum
    DIAGNOSTIC(total)
    
    nor = sum(total) ####!!! not being used //REMOVE ?
    nor2 = max(total) ####!!! not being used //REMOVE ?
    # nt is normalized by the sum
    #nt2 is normalized by the max"
    nt = []
    nt2 = []
    
    # I just removed where one could find division by zero
    #for i in range(0,len(total)):
    #    nt.append(total[i]/nor)
    #    nt2.append(total[i]/nor2)
    most = np.argsort(-np.array(total))#
    
    #def showrank():
    for i in most:
            DIAGNOSTIC('the node ' + str(i)+ ' is in '+ str(total[i])+ ' cliques')
    #    return 
    #DIAGNOSTIC(showrank())
    
    DIAGNOSTIC('These are the most important nodes ranked according to the total clique score')
    DIAGNOSTIC(most)
    DIAGNOSTIC('These is the array nt')

    DIAGNOSTIC(nt)
    DIAGNOSTIC('These is the array nt2')

    DIAGNOSTIC(nt2)
    DIAGNOSTIC('These is the array lista')

    DIAGNOSTIC(lista)
    DIAGNOSTIC('The output is one vector normalizing the value from the maximum')
    #vector=10000*np.array(nt)
    # nor2 is the maximum- The output nt2 is in percentage - 
    # That means the max get 100 and the rest bet 0-100
    
    # curv gives the curvature  - put Sc instead of curv to get that 
    # the particiaption rank - notice that you can normalize in many ways"
    curv = []
    for i in lista.values():
        # Summing up for a fixed node all the curvature scores gives the 
        # curvature of the nodes
        curv.append(sum(i))
        
    curv = np.array(curv)
    # Now, the curvature is not normalized!!!
    return curv#fden, curv

def betti(G, C=None, verbose=False):
    # G is a networkx graph
    # C is networkx.find_cliques(G)

    # RA, 2017-11-03, CC-BY-4.0

    # Ref: 
    # A. Zomorodian, Computational topology (Notes), 2009
    # http://www.ams.org/meetings/short-courses/zomorodian-notes.pdf



    def DIAGNOSTIC(*params):
        if verbose: print(*params)

    #
    # 1. Prepare maximal cliques
    #

    # If not provided, compute maximal cliques
    if (C is None): C = nx.find_cliques(G)

    # Sort each clique, make sure it's a tuple
    C = [tuple(sorted(c)) for c in C]
    
    if len(C) == 0:
        print('No neighbourhood!')
        return [0,0,0]
    
    DIAGNOSTIC("Number of maximal cliques: {} ({}M)".format(len(C), round(len(C) / 1e6)))

    #
    # 2. Enumerate all simplices
    #

    # S[k] will hold all k-simplices - if you are interested in the simplices, change the codes!!!
    # S[k][s] is the ID of simplex s
    S = []
    
    for k in range(0, max(len(s) for s in C)):
        # Get all (k+1)-cliques, i.e. k-simplices, from max cliques mc
        Sk = set(c for mc in C for c in itertools.combinations(mc, k + 1))
        # Check that each simplex is in increasing order
        assert (all((list(s) == sorted(s)) for s in Sk))
        # Assign an ID to each simplex, in lexicographic order
        S.append(dict(zip(sorted(Sk), range(0, len(Sk)))))

    for (k, Sk) in enumerate(S):
        DIAGNOSTIC("Number of {}-simplices: {}".format(k, len(Sk)))

    # Euler characteristic
    ec = sum(((-1) ** k * len(S[k])) for k in range(0, len(S)))

    DIAGNOSTIC("Euler characteristic:", ec)

    #
    # 3. Construct the boundary operator
    #

    # D[k] is the boundary operator 
    #      from the k complex 
    #      to the k-1 complex
    D = [None for _ in S]

    # D[0] is the zero matrix
    D[0] = lil_matrix((1, G.number_of_nodes()))

    # Construct D[1], D[2], ...
    for k in range(1, len(S)):
        D[k] = lil_matrix((len(S[k - 1]), len(S[k])))
        SIGN = np.asmatrix([(-1) ** i for i in range(0, k + 1)]).transpose()

        for (ks, j) in S[k].items():
            # Indices of all (k-1)-subsimplices s of the k-simplex ks
            I = [S[k - 1][s] for s in sorted(itertools.combinations(ks, k))]
            D[k][I, j] = SIGN.squeeze()

    for (k, d) in enumerate(D):
        DIAGNOSTIC("D[{}] has shape {}".format(k, d.shape))

    # Check that D[k-1] * D[k] is zero
    assert (all((0 == np.dot(D[k - 1], D[k]).count_nonzero()) for k in range(1, len(D))))

    #
    # 4. Compute rank and dimker of the boundary operators
    #

    # Rank and dimker
    rk = [np.linalg.matrix_rank(d.todense()) for d in D[:4]]
    ns = [(d.shape[1] - rk[n]) for (n, d) in enumerate(D[:4])]

    DIAGNOSTIC("rk:", rk)
    DIAGNOSTIC("ns:", ns)

    #
    # 5. Infer the Betti numbers
    #

    # Betti numbers
    # B[0] is the number of connected components
    B = [(n - r) for (n, r) in zip(ns[:-1], rk[1:])]

    DIAGNOSTIC("B:", B)

    ec_alt = sum(((-1) ** k * B[k]) for k in range(0, len(B)))
    DIAGNOSTIC("Euler characteristic (from Betti numbers):", ec_alt)

    # Check: Euler-Poincare formula
    #assert (ec == ec_alt)

    return B


DEBUG = False # True


######################################################################
## disparity filter for extracting the multiscale backbone of
## complex weighted networks

def get_nes (graph, label):
    """
    find the neighborhood attention set (NES) for the given label
    """
    for node_id in graph.nodes():
        node = graph.node[node_id]

        if node["label"].lower() == label:
            return set([node_id]).union(set([id for id in graph.neighbors(node_id)]))


def disparity_integral (x, k):
    """
    calculate the definite integral for the PDF in the disparity filter
    """
    assert x != 1.0, "x == 1.0"
    assert k != 1.0, "k == 1.0"
    return ((1.0 - x)**k) / ((k - 1.0) * (x - 1.0))


def get_disparity_significance (norm_weight, degree):
    """
    calculate the significance (alpha) for the disparity filter
    """
    return 1.0 - ((degree - 1.0) * (disparity_integral(norm_weight, degree) - disparity_integral(0.0, degree)))


def disparity_filter (graph):
    """
    implements a disparity filter, based on multiscale backbone networks
    https://arxiv.org/pdf/0904.2389.pdf
    """
    alpha_measures = []
    
    for node_id in graph.nodes():
        node = graph.nodes[node_id]
        degree = graph.degree(node_id)
        strength = 0.0

        for id0, id1 in graph.edges(nbunch=[node_id]):
            edge = graph[id0][id1]
            strength += edge["weight"]

        node["strength"] = strength

        for id0, id1 in graph.edges(nbunch=[node_id]):
            edge = graph[id0][id1]

            norm_weight = edge["weight"] / strength
            edge["norm_weight"] = norm_weight

            if degree > 1:
                try:
                    if norm_weight == 1.0:
                        norm_weight -= 0.0001

                    alpha = get_disparity_significance(norm_weight, degree)
                except AssertionError:
                    report_error("disparity {}".format(repr(node)), fatal=True)

                edge["alpha"] = alpha
                alpha_measures.append(alpha)
            else:
                edge["alpha"] = 0.0

    for id0, id1 in graph.edges():
        edge = graph[id0][id1]
        edge["alpha_ptile"] = percentileofscore(alpha_measures, edge["alpha"]) / 100.0

    return alpha_measures


######################################################################
## related metrics

def calc_centrality (graph, min_degree=1):
    """
    to conserve compute costs, ignore centrality for nodes below `min_degree`
    """
    sub_graph = graph.copy()
    sub_graph.remove_nodes_from([ n for n, d in list(graph.degree) if d < min_degree ])

    centrality = nx.betweenness_centrality(sub_graph, weight="weight")
    #centrality = nx.closeness_centrality(sub_graph, distance="distance")

    return centrality


def calc_quantiles (metrics, num):
    """
    calculate `num` quantiles for the given list
    """
    global DEBUG

    bins = np.linspace(0, 1, num=num, endpoint=True)
    s = pd.Series(metrics)
    q = s.quantile(bins, interpolation="nearest")

    try:
        dig = np.digitize(metrics, q) - 1
    except ValueError as e:
        print("ValueError:", str(e), metrics, s, q, bins)
        sys.exit(-1)

    quantiles = []

    for idx, q_hi in q.items():
        quantiles.append(q_hi)

        if DEBUG:
            print(idx, q_hi)

    return quantiles


def calc_alpha_ptile (alpha_measures, show=False):
    """
    calculate the quantiles used to define a threshold alpha cutoff
    """
    quantiles = calc_quantiles(alpha_measures, num=100)
    num_quant = len(quantiles)

    if show:
        print("\tptile\talpha")

        for i in range(num_quant):
            percentile = i / float(num_quant)
            print("\t{:0.2f}\t{:0.4f}".format(percentile, quantiles[i]))

    return quantiles, num_quant


def cut_graph (graph, min_alpha_ptile=0.5, min_degree=2):
    """
    apply the disparity filter to cut the given graph
    """
    filtered_set = set([])

    for id0, id1 in graph.edges():
        edge = graph[id0][id1]

        if edge["alpha_ptile"] < min_alpha_ptile:
            filtered_set.add((id0, id1))

    for id0, id1 in filtered_set:
        graph.remove_edge(id0, id1)

    filtered_set = set([])

#     for node_id in graph.nodes():
#         node = graph.nodes[node_id]

#         if graph.degree(node_id) < min_degree:
#             filtered_set.add(node_id)

#     for node_id in filtered_set:
#         graph.remove_node(node_id)

    return graph

######################################################################
## profiling utilities

def start_profiling ():
    """start profiling"""
    pr = cProfile.Profile()
    pr.enable()

    return pr


def stop_profiling (pr):
    """stop profiling and report"""
    pr.disable()

    s = io.StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)

    ps.print_stats()
    print(s.getvalue())


def report_error (cause_string, logger=None, fatal=False):
    """
    TODO: errors should go to logger, and not be fatal
    """
    etype, value, tb = sys.exc_info()
    error_str = "{} {}".format(cause_string, str(format_exception(etype, value, tb, 3)))

    if logger:
        logger.info(error_str)
    else:
        print(error_str)

    if fatal:
        sys.exit(-1)


######################################################################
## graph serialization

def load_graph (graph_path):
    """
    load a graph from JSON
    """
    with open(graph_path) as f:
        data = json.load(f)
        graph = json_graph.node_link_graph(data, directed=True)
        return graph


def save_graph (graph, graph_path):
    """
    save a graph as JSON
    """
    with open(graph_path, "w") as f:
        data = json_graph.node_link_data(graph)
        json.dump(data, f)


######################################################################
## testing

def random_graph (n, k, seed=0):
    """
    populate a random graph (with an optional seed) with `n` nodes and
    up to `k` edges for each node
    """
    graph = nx.DiGraph()
    random.seed(seed)

    for node_id in range(n):
        graph.add_node(node_id, label=str(node_id))

    for node_id in range(n):
        population = set(range(n)) - set([node_id])

        for neighbor in random.sample(population, random.randint(0, k)):
            weight = random.random()
            graph.add_edge(node_id, neighbor, weight=weight)

    return graph


def describe_graph (graph, min_degree=1, show_centrality=False):
    """
    describe a graph
    """
    print("\ngraph: {} nodes {} edges\n".format(len(graph.nodes()), len(graph.edges())))

    if show_centrality:
        print(calc_centrality(graph, min_degree))


def main (n=100, k=10, min_alpha_ptile=0.5, min_degree=2):
    # generate a random graph (from seed, always the same)
    graph = random_graph(n, k)

    save_graph(graph, "g.json")
    describe_graph(graph, min_degree)

    # calculate the multiscale backbone metrics
    alpha_measures = disparity_filter(graph)
    quantiles, num_quant = calc_alpha_ptile(alpha_measures)
    alpha_cutoff = quantiles[round(num_quant * min_alpha_ptile)]

    print("\nfilter: percentile {:0.2f}, min alpha {:0.4f}, min degree {}".format(
            min_alpha_ptile, alpha_cutoff, min_degree
            ))

    # apply the filter to cut the graph
    cut_graph(graph, min_alpha_ptile, min_degree)

    save_graph(graph, "h.json")
    describe_graph(graph, min_degree)
    
    
def to_distance_matrix(A):
    A = 1 - A
    #print(A)
    for r in range(len(A)):
        for c in range(len(A)):
            if A[c][r] == 1:
                A[c][r] = 0
                
    #print(A)
    return A

def pruned_matrix(A, density, binarize = True): #this can be certainly optimized
    
    B = np.zeros(np.shape(A))
    v = A[np.where(A > 0)] # >= if you want to inclu
    v = np.sort(v)
    if density <= 1:
        threshold = v[round(len(v)*density)]
        for r in range(len(A)):
            for c in range(len(A[0])):
                if A[c,r] < threshold and c!= r:
                    if binarize == True:
                        B[c,r] = 1
                    else:
                        B[c,r] = A[c,r]
                        
        return B


    
def get_ego_adj_matrix(A, rad, ego_node):
    G = nx.Graph(A)
    EG = nx.ego_graph(G, ego_node, radius=rad, center = False, undirected=True)

    return EG

def betti_analysis_single_case_local(args):
    band, meg, density, rad  = args[0], args[1], args[2], args[3]
    min_density = .2
    max_density = .2
    precision = 1e-1
    #print(case)


    matrix_dict = pd.read_pickle(f'Data/matrix_dict')
        
    A = matrix_dict[band][meg]
    A = to_distance_matrix(A)
    G = nx.from_numpy_matrix(A, parallel_edges=False, create_using=None)
    
    res = {d : {ego_node : None for ego_node in range(90)} for d in [density] }
    
    min_alpha_ptile= 1- density
    min_degree=2
    # calculate the multiscale backbone metrics
    alpha_measures = disparity_filter(G)
    quantiles, num_quant = calc_alpha_ptile(alpha_measures, show = False)
    alpha_cutoff = quantiles[round(num_quant * min_alpha_ptile)]

#     print("\nfilter: percentile {:0.2f}, min alpha {:0.4f}, min degree {}".format(
#             min_alpha_ptile, alpha_cutoff, min_degree
#             ))

    # apply the filter to cut the graph
    G = cut_graph(G, min_alpha_ptile, min_degree)
    
    A = nx.adjacency_matrix(G).todense()
    
    B =  (A > 0).astype(np.int_) #this is for binarize
    
    
    #G = nx.Graph(B)
    for ego_node in tqdm.tqdm(range(len(B))):
        
            G = get_ego_adj_matrix(B, rad, ego_node)
            aux = []
            
            #try:
            x = betti(G, C=None, verbose=False)   
                
#             except:
#                 x = []
            
            #print(x )
            #aux.append(x)
            betti_aux = np.zeros(3)
            if len(x) >0:
                for b in range(len(x)):
                    betti_aux[b] = x[b]
                    print(betti_aux)
            for b in betti_aux:
                
                aux.append(b)
            
            try:
                x = Curv_density(G)
            except:
                x = None
            aux.append(x)
    
            
            try:
                x = nx.number_of_nodes(G)
            except:
                x = None
            aux.append(x)
            
            try:
                x = nx.number_of_edges(G)
            except:
                x = None
            aux.append(x)
            

            

            res[density][ego_node] = aux.copy()
        #res[density] = betti(G, C=None,  verbose=False)
        
    with open(f'Betti curves/betti_curves_dict_case_{meg}_disparity_filter_density_{density}_rad_{rad}_band_{band}_local', 'wb') as f:

        pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
        
        
        
def complete_analysis_single_case_global(args):
    band, meg, density  = args[0], args[1], args[2]
    min_density = .2
    max_density = .2
    precision = 1e-1
    
    matrix_dict = pd.read_pickle(f'Data/matrix_dict')

    
    A = matrix_dict[band][meg]
    A = to_distance_matrix(A)
    G = nx.from_numpy_matrix(A, parallel_edges=False, create_using=None)
    
    res = {d : {ego_node : None for ego_node in range(90)} for d in [density] }
    
    min_alpha_ptile= 1- density
    min_degree=2
    # calculate the multiscale backbone metrics
    alpha_measures = disparity_filter(G)
    quantiles, num_quant = calc_alpha_ptile(alpha_measures, show = False)
    alpha_cutoff = quantiles[round(num_quant * min_alpha_ptile)]

#     print("\nfilter: percentile {:0.2f}, min alpha {:0.4f}, min degree {}".format(
#             min_alpha_ptile, alpha_cutoff, min_degree
#             ))

    # apply the filter to cut the graph
    G = cut_graph(G, min_alpha_ptile, min_degree)
    
    A = nx.adjacency_matrix(G).todense()
    
    B =  (A > 0).astype(np.int_) #this is for binarize


    res = {density : {} for density in [density] }


    #B = pruned_matrix(A, density)
    G = nx.Graph(B)
    aux = []

    try:
        x = betti(G, C=None, verbose=False)   
    except:
        x = None
    aux.append(x)


    try:
        x = Curv_density(G)
    except:
        x = None
    aux.append(x)

    try:
        x = nx.number_of_nodes(G)
    except:
        x = None
    aux.append(x)

    try:
        x = nx.number_of_edges(G)
    except:
        x = None
    aux.append(x)

    try:
        x = nx.average_clustering(G)
    except:
        x = None
    aux.append(x)

    try:
        x = nx.betweenness_centrality(G)
    except:
        x = None
    aux.append(x)

    res[density] = aux.copy()

           
    with open(f'Betti curves/global_analysis_case_{meg}_disparity_filter_density_{density}_band_{band}_global', 'wb') as f:

        pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)

def local_statistical_analysis(res, density,patient_type1, patient_type2, node_type1, node_type2, metric, band):
    matrix_dict = pd.read_pickle(f'Data/matrix_dict')
    RA_dict = pd.read_pickle(f'Data/RA_dict')
    sf_dict = pd.read_pickle(f'Data/sf_dict')
    meg_list = pd.read_pickle(f'Data/meg_list')
    
        
    neighs_of_RA_dict = pd.read_pickle(f'Data/neighs_of_RA_dict')
    
    metric_list = {'Betti 0':0, 'Betti 1':1, 'Betti 2':2 ,'Average curvature':3, 'Nodes':4, 'Edges':5, 'Local curvature':6, 'Betweeness':7}
    if patient_type1 in ['sf','nsf','all'] and node_type1 in ['ra', 'nra', 'neigh', 'other', 'all'] and metric in metric_list.keys():
        if patient_type1 == 'sf':
            p_list1 = [p for p in meg_list if sf_dict[p] ==1]
        elif patient_type1 == 'nsf':
            p_list1 = [p for p in meg_list if sf_dict[p] ==0]
        elif patient_type1 == 'all':
            p_list1 = [p for p in meg_list ]
        
        if patient_type2 == 'sf':
            
            p_list2 = [p for p in meg_list if sf_dict[p] ==1]
        elif patient_type2 == 'nsf':
            
            p_list2 = [p for p in meg_list if sf_dict[p] ==0]
        elif patient_type2 == 'all':
            
            p_list2 = [p for p in meg_list ]
        
            
        if node_type1 =='ra':            
            results_1 = [ np.mean([res[p][density][n][metric_list[metric]] for n in [n for n in range(90) if n in RA_dict[p] and len(RA_dict[p]) >0 ]]) for p in meg_list if p in p_list1 ]
            #print(results_1)
        elif node_type1 == 'nra':   
            
            results_1 = [ np.mean([res[p][density][n][metric_list[metric]] for n in [n for n in range(90) if n  not in RA_dict[p] and len(RA_dict[p]) >0 ]]) for p in meg_list if p in p_list1 ]
            #print(results_1)
        elif node_type1 == 'all':
            
            results_1 = [ np.mean([res[p][density][n][metric_list[metric]] for n in range(90)]) for p in meg_list if p in p_list1 ]
        elif node_type1 == 'neigh':
            
            results_1 = [ np.mean([res[p][density][n][metric_list[metric]] for n in [n for n in range(90) if n in neighs_of_RA_dict[band][density][p] and n not in RA_dict[p] ]]) for p in meg_list if p in p_list1 ]
            
        elif node_type1 == 'other':
            
            results_1 = [ np.mean([ res[p][density][n][metric_list[metric]] for n in [n for n in range(90) if n not in neighs_of_RA_dict[band][density][p] and n not in RA_dict[p]  ] ]) for p in meg_list if p in p_list1 ]
            #print(results_1)
        #print('-----------------------')
        #print(p_list2)
        if node_type2 =='ra':    
            #print('test 2')
            results_2 = [ np.mean([res[p][density][n][metric_list[metric]] for n in [n for n in range(90) if n in RA_dict[p] and len(RA_dict[p]) >0 ]]) for p in meg_list if p in p_list2 ]
            #print(results_2)
        elif node_type2 == 'nra':            
            results_2 = [ np.mean([res[p][density][n][metric_list[metric]] for n in [n for n in range(90) if n  not in RA_dict[p] and len(RA_dict[p]) >0 ]]) for p in meg_list if p in p_list2 ]
            
        elif node_type2 == 'all':
            results_2 = [ np.mean([res[p][density][n][metric_list[metric]] for n in range(90)]) for p in meg_list if p in p_list2 ]
        elif node_type2 == 'neigh':
            
            results_2 = [ np.mean([res[p][density][n][metric_list[metric]] for n in [n for n in range(90) if n in neighs_of_RA_dict[band][density][p]  and n not in RA_dict[p]  ]]) for p in meg_list if p in p_list2 ]
            
        elif node_type2 == 'other':
            
            results_2 = [ np.mean([res[p][density][n][metric_list[metric]] for n in [n for n in range(90) if n not in neighs_of_RA_dict[band][density][p]  and n not in RA_dict[p]  ]]) for p in meg_list if p in p_list2 ]
    
#         print(f'{metric} distribution for all nodes. Density = 5%.'+ 
#               f'\n ks-test p-value = {np.round(stats.kstest(results_1, results_2)[1],2)} \n'+ 
#               f't-test  p-value = {np.round(stats.ttest_ind(results_1, results_2)[1],2)} \n '+ 
#              f'Mean values: nsf= {np.round(np.mean(results_1),2)}, sf={np.round(np.mean(results_2),2)} ')
        #print(results_1)
    
        results_1 = [x for x in results_1 if str(x) != 'nan' and str(np.shape(x)) != '(0,)' ]
        #print(results_1)
        results_2 = [x for x in results_2 if str(x) != 'nan'  and str(np.shape(x)) != '(0,)' ]
        
        #return results_1
        return np.round(np.mean(results_1),4),np.round(np.mean(results_2),4), np.round(stats.kstest(results_1, results_2)[1],4), np.round(stats.ttest_ind(results_1, results_2)[1],4)
    

    else:
        print('Wrong input!')
        return [0,0,0,0]  
    
    
def z_score_and_pvalue(res, density,patient_type1, patient_type2, node_type1, node_type2, metric, band):
    matrix_dict = pd.read_pickle(f'Data/matrix_dict')
    RA_dict = pd.read_pickle(f'Data/RA_dict')
    sf_dict = pd.read_pickle(f'Data/sf_dict')
    meg_list = pd.read_pickle(f'Data/meg_list')
    
        
    neighs_of_RA_dict = pd.read_pickle(f'Data/neighs_of_RA_dict')
    
    metric_list = {'Betti 0':0, 'Betti 1':1, 'Betti 2':2 ,'Average curvature':3, 'Nodes':4, 'Edges':5, 'Local curvature':6, 'Betweeness':7}
    if patient_type1 in ['sf','nsf', 'all'] and node_type1 in ['ra', 'nra', 'neigh','other', 'all'] and metric in metric_list.keys():
        if patient_type1 == 'sf':
            p_list1 = [p for p in meg_list if sf_dict[p] ==1]
        elif patient_type1 == 'nsf':
            p_list1 = [p for p in meg_list if sf_dict[p] ==0]
        elif patient_type1 == 'all':
            p_list1 = [p for p in meg_list ]
        
        if patient_type2 == 'sf':
            
            p_list2 = [p for p in meg_list if sf_dict[p] ==1]
        elif patient_type2 == 'nsf':
            
            p_list2 = [p for p in meg_list if sf_dict[p] ==0]
        elif patient_type2 == 'all':
            
            p_list2 = [p for p in meg_list ]
        
            
        if node_type1 =='ra':            
            results_1 = [ np.mean([res[p][density][n][metric_list[metric]] for n in [n for n in range(90) if n in RA_dict[p] and len(RA_dict[p]) >0 ]]) for p in meg_list if p in p_list1 ]
            #print(results_1)
        elif node_type1 == 'nra':   
            
            results_1 = [ np.mean([res[p][density][n][metric_list[metric]] for n in [n for n in range(90) if n  not in RA_dict[p] and len(RA_dict[p]) >0 ]]) for p in meg_list if p in p_list1 ]
            #print(results_1)
        elif node_type1 == 'all':
            
            results_1 = [ np.mean([res[p][density][n][metric_list[metric]] for n in range(90)]) for p in meg_list if p in p_list1 ]
        
        elif node_type1 == 'neigh':
            
            results_1 = [ np.mean([res[p][density][n][metric_list[metric]] for n in [n for n in range(90) if n in neighs_of_RA_dict[band][density][p] and n not in RA_dict[p] ]]) for p in meg_list if p in p_list1 ]
            
        elif node_type1 == 'other':
            
            results_1 = [ np.mean([ res[p][density][n][metric_list[metric]] for n in [n for n in range(90) if n not in neighs_of_RA_dict[band][density][p] and n not in RA_dict[p]  ] ]) for p in meg_list if p in p_list1 ]
            #print(results_1)
        #print('-----------------------')
        #print(p_list2)
        if node_type2 =='ra':    
            #print('test 2')
            results_2 = [ np.mean([res[p][density][n][metric_list[metric]] for n in [n for n in range(90) if n in RA_dict[p] and len(RA_dict[p]) >0 ]]) for p in meg_list if p in p_list2 ]
            #print(results_2)
        elif node_type2 == 'nra':            
            results_2 = [ np.mean([res[p][density][n][metric_list[metric]] for n in [n for n in range(90) if n  not in RA_dict[p] and len(RA_dict[p]) >0 ]]) for p in meg_list if p in p_list2 ]
            
        elif node_type2 == 'all':
            results_2 = [ np.mean([res[p][density][n][metric_list[metric]] for n in range(90)]) for p in meg_list if p in p_list2 ]
        
        elif node_type2 == 'neigh':
            
            results_2 = [ np.mean([res[p][density][n][metric_list[metric]] for n in [n for n in range(90) if n in neighs_of_RA_dict[band][density][p]  and n not in RA_dict[p]  ]]) for p in meg_list if p in p_list2 ]
            
        elif node_type2 == 'other':
            
            results_2 = [ np.mean([res[p][density][n][metric_list[metric]] for n in [n for n in range(90) if n not in neighs_of_RA_dict[band][density][p]  and n not in RA_dict[p]  ]]) for p in meg_list if p in p_list2 ]
#         print(f'{metric} distribution for all nodes. Density = 5%.'+ 
#               f'\n ks-test p-value = {np.round(stats.kstest(results_1, results_2)[1],2)} \n'+ 
#               f't-test  p-value = {np.round(stats.ttest_ind(results_1, results_2)[1],2)} \n '+ 
#              f'Mean values: nsf= {np.round(np.mean(results_1),2)}, sf={np.round(np.mean(results_2),2)} ')
        #print(results_1)
        results_1 = [x for x in results_1 if str(x) != 'nan' and str(np.shape(x)) != '(0,)' ]
        #print(results_1)
        results_2 = [x for x in results_2 if str(x) != 'nan'  and str(np.shape(x)) != '(0,)' ]
        #print(len(results_1), len(results_1) )
        #return results_1
        
        results_0 = [np.mean([res[p][density][n][metric_list[metric]] for n in range(90)]) for p in meg_list if p in [p for p in meg_list]]
        return (np.mean(results_1) - np.mean(results_2))/np.std(results_0), np.round(stats.ttest_ind(results_1, results_2)[1],4), np.mean(results_1) , np.mean(results_2)
#     np.round(np.mean(results_1),4),np.round(np.mean(results_2),4), np.round(stats.kstest(results_1, results_2)[1],4), np.round(stats.ttest_ind(results_1, results_2)[1],4)
    

    else:
        print('Wrong input!')
        return [0,0,0,0]  
    
        
    