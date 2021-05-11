# 2021sp CSCI E-29 Final Project (Pset P)


<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*




<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Problem Statement

Many learning tasks require dealing with graph data, which contains rich relationship information 
among elements. For example, modeling physics system, learning molecular fingerprints, predicting protein 
interface, and classifying diseases require a model to learn from graph inputs. A typical graph data consists 
of vertices and edges, while vertices are nodes and edges are relationship between nodes. This project aims to 
implement graph neural network models that captures the dependence of graphs through message passing between nodes 
and their edges.

## Goal
I plan to provide insight into open source packages and functional programming methods for working with graphs and 
graph neural networks (gnn). Using real world application, this project will implement advanced python concepts to 
showcase the importance of functional programming in solving graph data. Three open sources for gnn will be considered:

1. PyTorch Geometric
2. Deep Graph Library
3. GraphNet

## How To
1. Fork or download this repo
2. Install required packages
3. Use the code below in your terminal to launch the web app

```python
$ streamlit run directory_name/pset_p/gcn.py
```
A locahost port will open in your browser to view the Graph Convolutional Network implementation of 3 Planetiod graph datasets.

## Additional Resources:
This repo aggregates the [Must-read Papers on GNN](https://github.com/thunlp/GNNPapers)