# Reproduction:
Install the necessary packages with :
```
$ pip install -r requirements.txt
```
Run Jupyter Notebook:
```
$ jupyter notebook
```
# Repository Organisation:
- **Different seeds**:
  - different_seeds.ipynb
  - different_seeds_CV.ipynb
- **Parallelized training**:
  - training_parallelization.ipynb
- **Different datasets (sentiment analysis)**:
  - training_downstream.ipynb
- **Multilingual fusion**:
  - https://colab.research.google.com/drive/12chbVk1d7XBkFPstvsdUJRiwklwPFep_?usp=sharing
  - https://colab.research.google.com/drive/1lO_4JsqtUHCpjHLu5TR7QJ1CDYjYbC1y?usp=sharing
  - https://colab.research.google.com/drive/1M4OgV5F_JBmjhD8Jhm_U4O7EiGR5CVv9?usp=sharing
- **Different support mass functions**:
  - ProbMassDist_TransformerNeuronImportance.ipynb



# Project Proposal:
## Exploring Model Fusion with Optimal Transport on Transformers

### 1 Motivation
As the AI field is constantly developing and broadening its application range, a
natural question that arises is whether we can utilize the knowledge of individual
models in which we put in considerable amounts of time and resources for train-
ing. There are AI models that are specifically trained to effectively detect objects,
segment images, generate art from a text, and translate text between languages at
high levels of performance [2]. Could we improve the performance by combining
the individual models’ knowledge so that the child network performs better than
its parent networks?

We choose the model fusion technique based on weight averaging to explore this
question. Other strategies to combine parent networks and their difficulties, such
as the computational overhead, are discussed in [3]. In contrast to vanilla av-
eraging [4], we consider the permutation invariance of neural networks and use
optimal transport (OT), intending to find the least costly transport [1], to align
neurons before averaging weights.

Aside from the performance improvement, we are also concerned with covering
other aspects of today’s AI world, such as the handling of data. Whether it is a
company or an individual, sharing valuable/sensitive data to train a model is not
risk-free and sometimes also regulated by international authorities such as The Eu-
ropean Union’s General Data Protection Regulation (GDPR) [6]. Our approach
eliminates the need for sharing private data by aligning the neurons and averaging
the weights, abstracting the training data from the model-building phase.

### 2 Task Description
Before we discuss the milestones of the project, we present the datasets we want
to utilize for different steps.

**Datasets** The first dataset indicates the dataset we will use for our first task
(parallelizing training), and the second and third datasets will be used for the
second task. The last dataset contains English and Finnish sentences which we
will use for the third task.
1. IMDB review
2. Stanford Sentiment Treebank
3. Sentiment 140
4. XED dataset

As mentioned in Section 1, one of the incentives that lead us to use model
fusion instead of other techniques is resources. We are going to explore whether
splitting a big dataset will give a comparable performance aside from the training
duration improvement by:

1. Parallelizing training, i.e., splitting a big corpus -or any big dataset- into
   smaller, separate segments, and training different simplistic transformer
   models[5] on those segments. We will then compare the performances of
   the model trained with whole data, the fused model, and the model trained
   on partial data.
   Another main focus we have among our motivations is federated learning. To
   this end, we are keen on exploring a decentralized approach by:
2. Doing sentiment analysis on different/abstracted datasets with transformers,
   and finally fusing the models to increase downstream performance/better-
   generalized model.
3. Experimenting with fusing models trained on bilingual data to see whether
   the fused model can perform well in both languages.

After completing the previous steps, we want to try out:

4. Trying different support mass distributions (default is the uniform distribution).
5. Machine translation to see how fusing decoders perform.

### 3 Evaluation
To better understand how model fusion impacts performance and whether it results in a reliable performance boost compared to its parent models, we run the
experiments multiple times. Different parent models are trained by e.g. different
initializations or different dataset splits and then fused.

For sentiment analysis tasks we will use accuracy, and (weighted) F1-score as
evaluation metrics, as well as probability calibration. For machine translation
tasks we want to use the BLEU score for performance analysis.

### 4 Contact
This master project will be advised by Sidak Pal Singh (sidak.singh@inf.
ethz.ch).

### References
[1] Gabriel Peyr ́e, Marco Cuturi, et al. “Computational optimal transport”. In:
Center for Research in Economics and Statistics Working Papers 2017-86
(2017).

[2] Terrence J Sejnowski. “The unreasonable effectiveness of deep learning in
artificial intelligence”. In: Proceedings of the National Academy of Sciences
117.48 (2020), pp. 30033–30038.

[3] Sidak Pal Singh and Martin Jaggi. “Model fusion via optimal transport”. In:
Advances in Neural Information Processing Systems 33 (2020), pp. 22045–22055.

[4] Joshua Smith and Michael Gashler. “An investigation of how neural net-
works learn from the experiences of peers through periodic weight averag-
ing”. In: 2017 16th IEEE International Conference on Machine Learning
and Applications (ICMLA). IEEE. 2017, pp. 731–736.

[5] Ashish Vaswani et al. “Attention is all you need”. In: Advances in neural
information processing systems 30 (2017).

[6] Qiang Yang et al. “Federated learning”. In: Synthesis Lectures on Artificial
Intelligence and Machine Learning 13.3 (2019), pp. 1–207.