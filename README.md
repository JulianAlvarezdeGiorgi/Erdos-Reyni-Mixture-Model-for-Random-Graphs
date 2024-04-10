# Erdos-Reyni Mixture Model for Random Graphs.

Project developed for the "Introduction to Probabilistic Graphical Models" of the MVA Master program (ENS Paris-Saclay) by:

- Borachhun You
- Nicolás Schecchi
- Julián Alvarez de Giorgi

We present 2 different mixture models with 3 different inference methods:
- SBM with a Variational Expectation Maximisation inference approach based on *Daudin, J. J., Picard, F., & Robin, S. (2008). A mixture model for random graphs. Statistics and Computing, 18(2), 173-183.*
- SBM with greedy optimization approach based on the exactly Integrated Complete Likelihood (ICL), based on *Côme, E., & Latouche, P. (2015). Model selection and clustering in stochastic block models based on the exact integrated complete data likelihood. Statistical Modelling, 15(6), 564-589.*
- A Mixture model with an expectation maximization inference method following *Newman, M. E., & Leicht, E. A. (2007). Mixture models and exploratory analysis in networks. Proceedings of the National Academy of Sciences, 104(23), 9564-9569.*

This repository contains:

- *src/SBM*: A definition of a class to manipulate a stochastic block model with 3 different inference methods.
- *src/keystoneGraph*: A class to define a Keystones graph following the topology described in *Newman, M. E., & Leicht, E. A. (2007). Mixture models and exploratory analysis in networks. Proceedings of the National Academy of Sciences, 104(23), 9564-9569.*
- *src/Dataset*: Dataset class definition
- *src/CommunitiesGraph*: Synthetic communities graph
- *src/NewmanMixtureModel*: The class modeling the Mixture model proposed in *Newman, M. E., & Leicht, E. A. (2007). Mixture models and exploratory analysis in networks. Proceedings of the National Academy of Sciences, 104(23), 9564-9569.*.

**Remark: In the SBM class, there's only 2 methods working, the 'Variational_EM' proposed by Daudin et Al. and the 'spectral clustering' approach, the 'EM'method which refers to the optimisation algorithm proposed by Côme, E. and Latouche, P. Also, the Variational_EM has mistakes, it only retrieve good results for simples datasets**
