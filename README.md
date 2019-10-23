# Deep Precision Medicine

## Context Setting

Predicting clinical labels from gene expression data is hard.
This is because the problem exists in a high dimensionality, low sample size (HDLSS) regime: the sizes of data sets
are small compared with each data points' number of features.  For example, the median size of a data set added to the
NIH Gene Expression Omnibus (GEO) hovers around 25.  On the other hand, the number of genes measured per data
point can be 15,000+.  Models trained on such datasets end up suffering from overfitting and high variance.

To work around this problem, specific genes are often selected by a domain expert (e.g. a trained bioinformatician)
to use as features when training a prediction model.  While this process can lead to accurate, low variance models,
it doesn't help us to discover new genetic mechanisms behind diseases (e.g. different cancer subtypes).  As such,
there is much value to be gained from new methods that can both produce stable models and work without input from a
domain expert.

One line of research that has gained more attention in this domain has been to use gene interaction graphs to impose bias
on deep models.  To evaluate the performance of this biasing procedure, Dutil et. al proposed the "Single Gene Inference" (SGI)
task, which they believe to be a proxy for more relevant prediction tasks.  This task compares the performance of a nonlinear
model's prediction of a given gene's expression level using only the gene's neighbors in a graph, versus using the whole gene
set.  If accuracy is similar across the two scenarios, then we can conclude that the gene interaction graph captures much of 
the signal present in the whole gene set.  Results on this task have been mixed.  The expression levels of a small number of 
genes have been shown to be better predicted by only using neighbors in an interaction graph.  However, most genes experience
a degradation in performance when only using neighbor data.

Previous work has demonstrated the ability to reliably predict gene expression markers for drug sensitivity by including
multi-omic prior information relevant to each gene's ability to drive cancer (i.e., the MERGE paper).  In particular, each
gene was assigned a driver score made up of a weighted sum of five "driver features".

## Room for Future Work

While MERGE incorporated a gene's hubness as one of its driver features for drug sensitivity, MERGE scores did not incorporate
information about a given gene's neighbors.  One could imagine that information about a gene's neighbors would have a
nontrivial impact on that gene's impact on drug sensitivity.  For example, if a gene A has close neighbors B and C in a 
gene-gene interaction graph, we would expect that e.g. methylation in the DNA regions corresponding to B and C, which will 
silence B and C, would have an impact on A's ability to influence drug response. 

Recent advances in machine models that learn from graph data, such as graph convolutional networks, might be able to capture
such information and use it to make more accurate predictions.  Furthermore, incorporating these driver features into the nodes
on gene interaction graphs might be able to improve the underwhelming results previously observed when attempting to learn
from these graphs with gene expresion level as the only feature.

Finally, assuming that we can obtain better results by incorporating a stronger biological prior, new advances have been made
in the realm of graph neural network interpretability.  Such advances would allow us to attribute drug sensitivity predictions
not only to specific genes, but also to _neighborhoods_ of genes with differing driver feature values.

### Potential Example Datasets

* [Functional genomic landscape of acute myeloid leukaemia](https://www.nature.com/articles/s41586-018-0623-z/)
  * 537(!) samples of gene expression and drug response data for AML patients
* MERGE AML dataset https://www.nature.com/articles/s41467-017-02465-5#MOESM4

## Relevant Papers

### On Precision Medicine

* [A machine learning approach to integrate big data for precision medicine in acute myeloid leukemia](https://www.nature.com/articles/s41467-017-02465-5#MOESM4)
  * Introduces a method for using a strong biological prior to predict how much individual genes influence drug sensitivity

### On General Methods For Dealing High Dimensional Low Sample Size (HDLSS) Data

* [Deep Neural Networks for High Dimension, Low Sample Size Data](https://www.ijcai.org/proceedings/2017/0318.pdf)
  * Proposes a new algorithm to train neural networks in the HDLSS setting by greedily selecting features to use

### On Predicting with Gene Expression Graphs

* [Towards Gene Expression Convolutions using Gene Interaction Graphs](https://arxiv.org/pdf/1806.06975.pdf)
   * Proposes biasing models trained on gene expression data by using gene interaction graphs.
   * Introduces the single gene inference (SGI) task as a proxy for understanding signal vs. noise in gene interaction graphs
* [Analysis of Gene Interaction Graphs for Biasing Machine Learning Models](https://arxiv.org/pdf/1905.02295.pdf)
   * Tests the performance of neural nets biased with different publicly available gene interaction graphs on the SGI task
   
### On Graph Neural Networks

This list focuses on graph convolutional networks, which (AFAIK), along with their variants, are the state of the art in deep learning based graph processing algorithms

#### 
* [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1609.02907.pdf)
  * Introduces the concept of graph convolutional networks.  Very briefly, these operate by filtering
  (performing convolutions) in the spectral domain
* [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907.pdf)
  * Introduces a more scalable version of GCNs
  * Blogpost giving a good explanation at https://tkipf.github.io/graph-convolutional-networks/
* [Inductive Representation Learning on Large Graphs](https://arxiv.org/pdf/1706.02216.pdf)
  * Original formulation of graph convolutional nets was transductive, meaning that models did not generalize to new graph structures.  This paper proposes an _inductive_ graph CNN, which can generalize
