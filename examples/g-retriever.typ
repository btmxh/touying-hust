// paper: https://github.com/XiaoxinHe/G-Retriever
#import "@preview/touying:0.6.1": *
#import "../lib.typ": *

#show: hust-theme.with(
  theme: "red",
  aspect-ratio: "16-9",
  config-info(
    title: [G-Retriever: Retrieval-Augmented Generation for Textual Graph
      Understanding and Question Answering],
  ),
)

#title-slide()

#outline-slide()

= Introduction

== Introduction

*Chat With Your Graph*: An LLM is given a textual description of a _textual_
graph $G$, and is tasked to answer questions about the content of $G$.

#figure(image("assets/g-retriever/chat-graph.png"))

== Problems

- The lack of a comprehensive benchmark of question answering tasks on graphs.
#pause

- Graph LLMs are prone to hallucinations, due to the limitations of current
  graph embedding techniques.
#pause

- Textual description of complex graphs result in too many tokens.

= Problem Formulation

== Textual Graph

A graph $G(V, E)$ equipped with a *textual* mapping $x: V union E
-> T$, where $T$ is the set of labels, i.e. finite-dimensional vectors of
tokens.

#pause

*Paper notation*: For every node $n in V$ and edge $e in E$:
- $D$ is the vocabulary (the set of all tokens).
- $x_n = x(n)$, $x_e = x(e)$.
- $L_n = dim x_n$ and $L_e = dim x_e$. Therefore $x_n in D^(L_n)$ and $x_e in
  D^(L_e)$.

== Text Encoding via Language Models

Given a piece of text $x_n$, a language model $"LM"$ can be used to encode $x$
into a (numeric) vector representation $z_n$:

$ z_n = "LM"(x_n) in RR^d, $

where $d$ is the embedding dimension.

== Large Language Models

A LLM with parameters $theta$ is a stochastic process that takes a sequence of
tokens $I$ as input and generate a sequence of tokens $Y$ as output such that:

$ p_theta (Y | I) = product_(i = 1)^r p_theta (y_i | y_0, y_1, ..., y_(i-1), I) $

#pause

$I$ is oftenly the concatenation of a prompt $P$ and the a sequence of tokens
$X$ describing the problem.

== Soft-prompting

When a LLM is given input $I$, it encodes $I$ into a vector representation $I_e$
before continuing. This process is done independently for each token before the
result is concatenated into a matrix.

#pause

Hence, instead of manual prompt engineering for $P$, *soft-prompting*
directly pass the prompt encoding $P_e$ to the LLM instead of $P$.

= G-Retriever

== Overview

G-Retriever is a retrieval-aumentation generation approach for general textual
graph.

#pause

It follows the standard retrieval-augmented generation (RAG) pipeline:
#pause

- *Indexing*: G-Retriever indexes the textual graph embedding into a nearest
  neighbor data-structure for efficient querying.
#pause

- *Retrieval*: Given a query $x_q$, G-Retriever find the most relevant nodes and
  edges to $x_q$.
#pause

- *Subgraph Construction* (Augmentation): A subgraph is constructed _based on_ the
  received nodes and edges.
#pause

- *Generation*: The subgraph is fed into a language model to generate an answer.

== Indexing

A pre-trained LM is used to encode the textual description of each node $n$ and
edge $e$ into vector representations $z_n = "LM"(x_n)$ and $z_e = "LM"(x_e)$.

#pause

Entries $(n, z_n)$ and $(e, z_e)$ are stored into a nearest neighbor data-structure
for efficient querying.

== Retrieval

Given a query $x_q$, we first transform it into embedding space:

$ z_q = "LM"(x_q), $

#pause

$k$-nearest neighbors is used to retrieve the $k$ most relevant nodes and $k$
most relevant edges to a query $x_q$:

#let argtopk = "argtopk"
$
  V_k &= argtopk_(n in V) cos(z_q, z_n)\
  E_k &= argtopk_(e in E) cos(z_q, z_e)
$

== Subgraph Construction

If we only use $V_k$ and $E_k$ for construction, we might end up with _node-less
edges_ (an edge in $E_k$ but its endpoints are not in $V_k$) and _nodes with
insufficient edges_ (a node in $V_k$ but its edges are not in $E_k$).

#pause

Instead, we find a connected subgraph $S(V_S, E_S)$ of $G$ that maximizes a certain
objective.

== Subgraph Construction

If we define some $"prize"(x)$ as the reward of choosing node/edge
$x$ in $S$, for example:

$
  "prize"(n) = cases(0 "if" x in.not V_k union E_k, k - i "if" x in V_k union
E_k "and" x "is the top" i "node/edge"),
$

#pause

$"cost"(S)$ as a measure of $S$'s _size_, for example:
$ "cost"(S) = C_e |E_S|, $

for some constant $C_e$ (cost per edge), we have an optimizing problem:

$
  max &sum_(x in V_S union E_S) "prize"(x) - "cost"(S),\
  "s.t." &S subset.eq G, S "is connected"
$

== Subgraph Construction

$
  max sum_(x in V_S union E_S) "prize"(x) - "cost"(S), "s.t." S subset.eq G, S "is connected"
$

This is a minor variation of Prize-Collecting Steiner Tree problem. However, the
original problem does not assign prize values to edges (edge cost must be
positive).

#pause

Given an edge $e$ with prize $p = "prize"(e)$, either:
- If $C_e > p$, then the cost of $e$ is set to $C_e - p$.
- If $C_e <= p$, $e$ is replaced by a virtual node $v(e)$, connected to two of
  its endpoints with 2 new edges of cost $0$. The prize of $v(e)$ is $p - C_e$.

This problem can now be solved in near-linear time complexity.

== Generation

The graph $S^*$ is first encoded by a standard Graph Attention Network $"GNN"$:
$ h_g = "POOL"("GNN"_phi_1 (S^*)) in RR^(d_g), $

#pause

Then, $h_g$ is projected to $RR^(d_l)$ with a MLP to yield the soft prompt $P_e
= hat(h)_g$:

$ hat(h)_g = "MLP"_phi_2 (h_g) in RR^(d_l), $

#pause

Finally, the sequence of tokens $X$ constructed from the textualization of $S^*$
and the query $x_q$:

$ X = ["textualize"(S^*), x_q]. $

= Experimental results

== Setup

Pre-trained models:
- Encoding LM: SentenceBert
- Generation LLM: Llama2-7b
- Graph encoder: Graph Transformer#footnote[https://arxiv.org/abs/2009.03509]

#pause

Model configurations:
- Inference-only: a frozen LLM is used for direct QA.
- Frozen LLM w/ PT: only the prompt is adapted.
- Tuned LLM: the entire LLM is fine-tuned with LoRA.

== GraphQA dataset

GraphQA dataset contains a collection of graph questions and answers on three
types of graphs: explanation graphs, scene graphs and knowledge graphs.

#figure(image("assets/g-retriever/graphqa.png"))

== Experimental results

#figure(image("assets/g-retriever/results.png"))

== Ablation study

G-Retriever significantly reduced the number of tokens.

#figure(image("assets/g-retriever/tokens.png"))

== Ablation study

G-Retriever also reduced LLM hallucinations by a large margin.

#figure(image("assets/g-retriever/ablation.png"))

== Further discussions

- PCST retrieval is justified by comparison with other retrieval methods (top-k,
  top-k + neighbors, shortest path)
- The parameter $k$ in kNN.
- Choice of similarity function other than $cos$.
- Complexity (GNN + LLM + GraphRAG) and performance.

#ending-slide(title: [THANK YOU!])

