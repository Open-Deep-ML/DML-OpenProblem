
# Learn Section

**Mixture of Experts** is a sparse architecture (commonly applied in transformers), where dense FullyConnected layers that are applied to the entire batch of tokens are replaced with N networks (experts) that are only applied to a subset of tokens.
In-detail explanation with beautiful vizualizaiotns is available in this HuggingFace [post](https://huggingface.co/blog/moe).


## Implementation

MoE consists of two main elements:

- Sparse MoE layers are used instead of dense feed-forward network (FFN) layers. MoE layers have a certain number of “experts” (e.g. 8), where each expert is a neural network

- A gate network or router, that determines which tokens are sent to which expert. We can send a token to more than one expert. How to route a token to an expert is one of the big decisions when working with MoEs - the router is composed of learned parameters and is pretrained at the same time as the rest of the network.

**All equeations below are applied per token.**
In the most traditional setup, we just use a simple network with a softmax function as a gating function. The network will learn which expert to send the input.
$$
G(x) = Softmax(xW_g)
$$

Each token is usually only routed to top K experts.
$$
w = getTopK(G(x))
$$

The weights of the top K experts are renormalized after selection.
$$
w_i = \frac{w_i}{\sum_j w_j}
$$

For each token embedding the output of expert $i$ is computed as a simple FNN.
$$
e = xW^i_e
$$

To get the final output per token we combine the outputs of all experts to which this token was routed as weighted sum using renormalized expert weights:
$$
y = \sum_i e_i * w_i
$$
    