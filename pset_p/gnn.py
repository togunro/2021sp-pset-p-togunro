"""

Detailed comparison of these three libraries (https://blog.paperspace.com/geometric-deep-learning-framework-comparison/)
Here are a couple of suggestions for common use-cases:
Scientific Research: PyTorch Geometric is developed by a Ph.D. student that is working on SOTA algorithms in the field.
This library is probably your best bet. From the ease of integration of common benchmark datasets to implementations of
other papers, it allows for a seamless integration if you want to quickly test your new findings against the SOTA.
Production-ready development: Here, it's not as clear cut. With the newest version and full TensorFlow support on its way,
the decision between using DGL and Graph Nets is hard to make. With AWS deeply involved, DGL will very likely offer superior
support for large-scale applications soon. Otherwise, if you're keen on being able to apply Deepmind's newest research to
your applications, then Graph Nets will be the only option.


More about Deep Graph Library:
** Choice between TensorFlow, PyTorch and MXNet back-end
** Has additional versions for domain applications (e.g. life sciences and knowledge graphs)
** AWS integration

"""

import dgl
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from ogb.linkproppred import DglLinkPropPredDataset
from dgl.nn import SAGEConv

"""
Train-validation-test split
We usually need positive and negative examples - positive are existing edges in the graph, negative are edges that are 
not in the graph (this example is for link prediction task, for other tasks the samples would be nodes or graphs).
"""

# Code from DGL tutorial
data = dgl.data.KarateClub()
graph = data[0]

# Split edge set for training and testing
u, v = graph.edges()
u, v = u.numpy(), v.numpy()
eids = np.arange(graph.number_of_edges())
eids = np.random.permutation(eids)
test_pos_u, test_pos_v = u[eids[:50]], v[eids[:50]]
train_pos_u, train_pos_v = u[eids[50:]], v[eids[50:]]

# Find all negative edges and split them for training and testing
adj = sp.coo_matrix((np.ones(len(u)), (u, v)))
adj_neg = 1 - adj.todense() - np.eye(34)
neg_u, neg_v = np.where(adj_neg != 0)
neg_eids = np.random.choice(len(neg_u), 200)
test_neg_u, test_neg_v = neg_u[neg_eids[:50]], neg_v[neg_eids[:50]]
train_neg_u, train_neg_v = neg_u[neg_eids[50:]], neg_v[neg_eids[50:]]

# Code from OGB instructions
# You may need to use DGL backend, otherwise PyTorch will be used.
dataset = DglLinkPropPredDataset(name='ogbl-collab')
graph = dataset[0]

split_edge = dataset.get_edge_split()
train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], \
                                    split_edge["test"]

"""
Building models
I stack different layers together, here is the full list of supported layers in Tensorflow https://docs.dgl.ai/api/python/nn.tensorflow.html, 
PyTorch https://docs.dgl.ai/api/python/nn.pytorch.html, and MXNet https://docs.dgl.ai/api/python/nn.mxnet.html backends

Below I'm using GraphSage (SAmple and aggreGatE) -- this aggregate feature information from a node's local neighborhood 
(e.g., the degrees or text attributes of nearby nodes)

Paper: https://arxiv.org/pdf/1706.02216.pdf
Blog: https://towardsdatascience.com/an-intuitive-explanation-of-graphsage-6df9437ee64f
Github: https://github.com/dglai/WWW20-Hands-on-Tutorial/tree/master/basic_tasks_tf
"""


# build a two-layer GraphSAGE model
class GraphSAGE(tf.keras.layers.Layer):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def call(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = tf.nn.relu(h)
        h = self.conv2(g, h)
        return h


# Create the model with given dimensions
# input layer dimension: 5, node embeddings
# hidden layer dimension: 16
net = GraphSAGE(5, 16)
