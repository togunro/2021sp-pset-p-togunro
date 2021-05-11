"""
Graph Convolutional Networks (GCN) implementation using streamlit app maker
"""

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import GCNConv
from spektral.transforms import AdjToSpTensor, LayerPreprocess
import os

import pandas as pd
import streamlit as st

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

st.title('Graph Convolutional Network Models')

st.markdown("""
This app uses **Spektral** library to retrieve Planetoid **Citation** graphs data from Thomas Kipf repository. The output is a Semi-Supervised edge Classification with **Graph Convolutional Network**
* ** Python libraries:** pandas, tensorflow, keras, streamlit, and spektral
* **Data source:** [Planetoid](https://github.com/kimiyoung/planetoid). [Cora](https://relational.fit.cvut.cz/dataset/CORA). 

 """)

st.sidebar.header('User Input Features')


def user_input_features():
    dataset_type = st.sidebar.selectbox('Select a Planetoid Dataset', ('pubmed', 'cora', 'citeseer'))
    epoch_sidebar = st.sidebar.slider('Epoch', 20, 1000, 20)
    learn_sidebar = st.sidebar.slider('Learning Rate', 0.1, 2.0, 0.2)
    pat_sidebar = st.sidebar.slider('Patience', 10, 400, 20)
    data_fi = {'dataset': dataset_type,
               'epochs': epoch_sidebar,
               'learning': learn_sidebar,
               'patience': pat_sidebar}
    feat_fi = pd.DataFrame(data_fi, index=[0])
    return feat_fi


df = user_input_features()
st.subheader('User Input parameters')
st.write(df)
st.write('-----')


def param_data():
    df_type = df.iloc[0, 0]
    return df_type


def param_learn():
    df_type = df.iloc[0, 2]
    return df_type


def param_epoch():
    df_type = df.iloc[0, 1]
    return df_type


def param_pat():
    df_type = df.iloc[0, 3]
    return df_type


class SGCN:
    def __init__(self, K):
        self.K = K

    def __call__(self, graph):
        out = graph.a
        for _ in range(self.K - 1):
            out = out.dot(out)
        out.sort_indices()
        graph.a = out
        return graph


# Load data
K = 2  # Propagation steps for SGCN
dataset = Citation(
    param_data(), transforms=[LayerPreprocess(GCNConv), SGCN(K), AdjToSpTensor()]
)
mask_tr, mask_va, mask_te = dataset.mask_tr, dataset.mask_va, dataset.mask_te

# Parameters
l2_reg = 5e-6  # L2 regularization rate
learning_rate = param_learn()  # Learning rate
epochs = param_epoch()  # Number of training epochs
patience = param_pat()  # Patience for early stopping

a_dtype = dataset[0].a.dtype  # Only needed for TF 2.1
N = dataset.n_nodes  # Number of nodes in the graph
F = dataset.n_node_features  # Original size of node features
n_out = dataset.n_labels  # Number of classes


# Model definition
def model1():
    x_in = Input(shape=(F,))
    a_in = Input((N,), sparse=True, dtype=a_dtype)
    output = GCNConv(
        n_out, activation="softmax", kernel_regularizer=l2(l2_reg), use_bias=False)([x_in, a_in])
    return x_in, a_in, output


x_in, a_in, output = model1()


class EvalModel:
    # Build model
    def build_model(self):
        model = Model(inputs=[x_in, a_in], outputs=output)
        optimizer = Adam(lr=learning_rate)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", weighted_metrics=["acc"]
        )
        return model

    # model.summary()

    # Train model
    def train_model(self):
        loader_tr = SingleLoader(dataset, sample_weights=mask_tr)
        loader_va = SingleLoader(dataset, sample_weights=mask_va)
        result = self.build_model().fit(
            loader_tr.load(),
            steps_per_epoch=loader_tr.steps_per_epoch,
            validation_data=loader_va.load(),
            validation_steps=loader_va.steps_per_epoch,
            epochs=epochs,
            callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)],
        )

        loader_te = SingleLoader(dataset, sample_weights=mask_te)
        eval_results = self.build_model().evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
        print("Done.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))

        return result, eval_results


eval = EvalModel()
result, eval_results = eval.train_model()

test_l, test_a = eval_results
data_result = {"Test loss": test_l,
               "Test Accuracy": test_a}

df_result = pd.DataFrame(data_result, index=[0])
st.subheader('Test Model Performance')
st.write(df_result)

loss = pd.DataFrame(result.history['loss'], columns=['Loss'])
val_loss = pd.DataFrame(result.history['val_loss'], columns=['Validation_Loss'])
acc = pd.DataFrame(result.history['acc'], columns=['Accuracy'])
val_acc = pd.DataFrame(result.history['val_acc'], columns=['Validation Accuracy'])

loss['Validation Loss'] = val_loss
acc['Validation Accuracy'] = val_acc

st.write("""
### Model Training Accuracy
""")
st.line_chart(acc)

st.write("""
### Model Training Loss
""")
st.line_chart(loss)
