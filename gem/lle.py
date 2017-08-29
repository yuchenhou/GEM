from gem.embedding.lle import LocallyLinearEmbedding
from gem.utils import graph_util
import pandas


def main(data_set_name):
    dimensions = 4
    input_file = './graph/' + data_set_name + '.tsv'
    output_file = './emb/' + data_set_name + '.emb'
    # Instatiate the embedding method with hyperparameters
    lle = LocallyLinearEmbedding(dimensions)

    # Load graph
    graph = graph_util.loadGraphFromEdgeListTxt(input_file)

    # Learn embedding - accepts a networkx graph or file with edge list
    embeddings_array, t = lle.learn_embedding(graph, edge_f=None, is_weighted=True, no_python=True)
    embeddings = pandas.DataFrame(embeddings_array)
    embeddings.to_csv(output_file, sep=' ', na_rep=0.1)


if __name__ == '__main__':
    for data_set_name in ['airport', 'collaboration', 'congress', 'forum']:
        main(data_set_name)
