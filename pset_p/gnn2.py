"""
GNN implementation of Cora dataset

"""
import dgl
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from dgl.nn import SAGEConv
from pset_p.plot import plot_results

data = dgl.data.CitationGraphDataset('cora')
g = data[0]
# NumEdges: 10556

node_embed = tf.keras.layers.Embedding(g.number_of_nodes(), 5,
                                       embeddings_initializer='glorot_uniform')  # Every node has an embedding of size 5.
node_embed(1)  # intialize embedding layer
inputs = node_embed.embeddings  # Use the embedding weight as the node features.

# Split edge set for training, validation and testing
u, v = g.edges()
u, v = u.numpy(), v.numpy()
eids = np.arange(g.number_of_edges())
eids = np.random.permutation(eids)
test_pos_u, test_pos_v = u[eids[:3000]], v[eids[:3000]]
val_pos_u, val_pos_v = u[eids[3000:6000]], v[eids[3000:6000]]
train_pos_u, train_pos_v = u[eids[6000:]], v[eids[6000:]]

# Find all negative edges and split them for training, validation and testing
adj = sp.coo_matrix((np.ones(len(u)), (u, v)))
adj_neg = 1 - adj.todense() - np.eye(2708)
neg_u, neg_v = np.where(adj_neg != 0)
neg_eids = np.random.choice(len(neg_u), 200)
test_neg_u, test_neg_v = neg_u[neg_eids[:3000]], neg_v[neg_eids[:3000]]
val_neg_u, val_neg_v = neg_u[neg_eids[3000:6000]], neg_v[neg_eids[3000:6000]]
train_neg_u, train_neg_v = neg_u[neg_eids[6000:]], neg_v[neg_eids[6000:]]

# Create training set.
train_u = tf.concat([train_pos_u, train_neg_u], axis=0)
train_v = tf.concat([train_pos_v, train_neg_v], axis=0)
train_label = tf.concat([tf.zeros(len(train_pos_u)), tf.ones(len(train_neg_u))], axis=0)

# Create validation set.
val_u = tf.concat([val_pos_u, val_neg_u], axis=0)
val_v = tf.concat([val_pos_v, val_neg_v], axis=0)
val_label = tf.concat([tf.zeros(len(val_pos_u)), tf.ones(len(val_neg_u))], axis=0)

# Create testing set.
test_u = tf.concat([test_pos_u, test_neg_u], axis=0)
test_v = tf.concat([test_pos_v, test_neg_v], axis=0)
test_label = tf.concat([tf.zeros(len(test_pos_u)), tf.ones(len(test_neg_u))], axis=0)


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


def train_model(model, epochs=100):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_fcn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    all_logits = []
    accs = []
    val_accs = []
    losses = []
    val_losses = []
    for e in range(epochs):

        with tf.GradientTape() as tape:
            tape.watch(inputs)  # optimize embedding layer also
            # forward
            logits = model(g, inputs)
            pred = tf.sigmoid(tf.reduce_sum(tf.gather(logits, train_u) *
                                            tf.gather(logits, train_v), axis=1))

            # Compute training accuracy and loss
            acc = ((pred.numpy() >= 0.5) == train_label.numpy()).sum().item() / len(pred)
            loss = loss_fcn(train_label, pred)
            accs.append(acc)
            losses.append(loss)

            # Compute validation accuracy and loss
            val_pred = tf.sigmoid(tf.reduce_sum(tf.gather(logits, val_u) *
                                                tf.gather(logits, val_v), axis=1))
            val_acc = ((val_pred.numpy() >= 0.5) == val_label.numpy()).sum().item() / len(val_pred)
            val_loss = loss_fcn(val_label, val_pred)
            val_accs.append(val_acc)
            val_losses.append(val_loss)

            # backward
            grads = tape.gradient(loss, model.trainable_weights + node_embed.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights + node_embed.trainable_weights))
            all_logits.append(logits.numpy())

        if e % 5 == 0:
            print('In epoch {}, loss: {}, acc: {}, val loss: {}, val acc: {}'.format(
                e, loss, acc, val_loss, val_acc))
    return losses, val_losses, accs, val_accs, logits


net = GraphSAGE(5, 16)
losses, val_losses, accs, val_accs, logits = train_model(net, epochs=20)

# Plot results
plot_results(losses, val_losses, accs, val_accs)
