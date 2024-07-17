import sys
from igraph import *
import pandas as pd
if len(sys.argv) != 5:
    print('Usage:python NetInfo.py edge.csv nodeinfo.txt sd.txt transitivity.txt')
else:
    # read in edge list from first command-line argument
    edges = []
    with open(sys.argv[1], 'r') as f:
        header = f.readline().strip().split(',')
        for line in f:
            edge = tuple(line.strip().split(','))
            edges.append(edge)
    
    # create graph from edge list
    g = Graph.TupleList(edges, directed=False)
    
    # calculate network measures for each node
    eccentricity = g.eccentricity(mode="all")
    closeness = g.closeness(normalized=True)
    degree = g.degree()
    eigen_centrality = g.eigenvector_centrality()
    
    # combine measures into a single data frame
    genes = g.vs['name']
    total = {'Gene': genes, 'eccentricity': eccentricity,
             'closeness': closeness, 'degree': degree,
             'eigen_centrality': eigen_centrality}
    total = pd.DataFrame(total)
    
    # write out data frames to files specified by command-line arguments
    total.to_csv(sys.argv[2], sep='\t', index=False)
    distances = g.shortest_paths()
    distances = pd.DataFrame(distances, index=genes, columns=genes)
    distances.index.name = 'Gene'
    distances.to_csv(sys.argv[3], sep='\t', index=True)
    transitivity = g.transitivity_undirected()
    avg_path_length = g.average_path_length()
    a = pd.DataFrame({'transitivity': [transitivity], 'average_path_length': [avg_path_length]})
    a.to_csv(sys.argv[4], sep='\t', index=False)
    

