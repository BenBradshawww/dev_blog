+++
date = '2025-01-19T17:34:44Z'
draft = false
title = 'My True First Post'
image = "/images/test.jpg"
categories = ["Programming", "AI", "Machine Learning"]
math = true
+++

+++
date = '2025-01-19T17:34:44Z'
draft = false
title = 'My True First Post'
image = "/images/test.jpg"
categories = ["Programming", "AI", "Machine Learning"]
math = true
+++

+++
date = '2025-01-19T17:34:44Z'
draft = false
title = 'My True First Post'
image = "/images/test.jpg"
categories = ["Programming", "AI", "Machine Learning"]
math = true
+++

+++
date = '2025-01-19T17:34:44Z'
draft = false
title = 'My True First Post'
image = "/images/test.jpg"
categories = ["Programming", "AI", "Machine Learning"]
math = true
+++


Over the last 10 years, transformer based architectures have surpassed the prior state of the art models in natural language processing tasks, computer vision tasks, and in foundation model creation. However, many researchers has questioned their performance in time series forecasting.

In time series forecasting, transformer based architectures typically embed multiple variates of the same timestamp into indistinguishable channels (i.e. the variables at one timestamp are encoded into a single embedding). We'll call these as temporal tokens. Attention based mechanism will then applied to these temporal tokens to capture temporal dependencies.

Using an attention mechanism enables the model to learn which timestamps are most relevant to predict the target value at a specific point in time. There are benefits to this approach such as:
1. Long-Range dependency capture. Self-attention is able to capture complex dependencies over long horizons, which is typically not possible in traditional RNNs.
2. Unified Handling of Multivariate Data. All the variables at a timestep are combined into one, simplifying the representation of multivariate relationships.
3. Parallelisation. Many of these temporal tokens can be processed in parallel, providing a more effective training process over RNNs.

However, even with these token embedding methods, simple linear layers (such as ARIMA based models) tend to outperform the complicated transformers. One of the reasons behind this is because these models can ensure that the multivariate correlations in time series forecasting will not impact the model's accuracy.

By embedding the multivariate features, the variables independence cannot be preserved which may limit the model's ability to capture the complex multivariate dependencies. This becomes an issue because transformers are meant to model temporal dependencies but they are not meant to incorporate inter-variable correlations.

As a result of the potential risks involved in embedding multivariate points of a timestamp to a temporal token. The iTransformer [1] takes an "inverted" approach and embeds the the whole time series of each variate independently into variable tokens to enlarge the receptive field. This inverted approach allows for embedding representations which are more variable-centric and they can then be leveraged more powerfully by attention based mechanisms for multivariate correlating. When this is used in conjunction with feed-forward networks, the transformer will be able to capture the generalised representations within the encoded token and can then be used to create forecasts for future values. This is the basic idea behind the iTransformer.

![Image](/images/itransformer_1.png)

### The Background To The Problem

In general, the authors of the iTransformer categorised time-series transformer based architectures into 4 different categories depending on whether the architecture or the component is modified.

The first of these categories (and the most common category) modifies the component, specifically focusing on modifying the attention mechanism. However, the success of these methods have been put under question because they struggle to surpass the success of simpler linear methods.

The second group of these methods focus on leveraging the strength of the transformer in time series forecasting by modifying the time series data. Some of these methods include:
1. Stationarisation: Making the data stationary to improve forecasts.
2. Channel Independence: Processing multiple channels of data independently to reduce noise.
3. Pathching: Grouping time series data into patches to help capture local and global patterns.
These methods have shown to achieve a consistent performance.

The third group of these methods modify both the component and architecture to improve the model's ability to capture complex relationships in multivariate time series forecasting. These model address complex interactions such as independence and mutual interactions.

The final of these methods (and the new method by Hu et al.) do not modify any of the native components of the transformer. Instead, the component is modified to work on the inverted dimension, by using an altered architecture, specifically, this architecture adopts the encoder-only architecture (appendix 1) of the Transformer. The reason for this choice of architecture stems from the authors struggle to reason with the necessity of heavy encoder-decoder architectures when compared to the triumph of linear forecasts.

## How The iTransformer Works

Given a multivariate time series forecasting observations of $X = \{ x_1, x_2, ..., x_T \} \in \mathbb{R}^{T \times N}$ with $T$ time steps and $N$ variates. The transformer will predict the future $S$ time steps $Y = \{x_{T+1}, x_{T+2}, ..., x_{T+S} \} \in \mathbb{R}^{R\times N}$. For simplicity, we will denote $X_t$ as the recorded time points at step $t$ and $X_{:, n}$ will denote the whole time series of each variate indexed by $n$.

The value of the future variate $\hat{Y}_{:, N}$ given a lookback series $X_{:, n}$ is formulated using:

$$\begin{aligned}h_n^0 &= \text{Embedding}(X_{:,n}), \\ H^{l+1} &= \text{TrmBlock}(H^l), \quad l = 0, ..., L-1, \\\hat{Y}_{:, N} &= \text{Projection}(h_n^l),  \end{aligned}$$

where $H = \{h_1, ..., h_N\} \in \mathbb{R}^{N\times D}$ contains the $N$ embedded tokens of dimension $D$. and the superscript denotes the layer index. The Embedding: $\mathbb{R}^T \rightarrow \mathbb{R}^D$ and projection $\mathbb{R}^D \rightarrow \mathbb{R}^S$ are implemented with a multi-layer perceptron (MLP). Finally, the TrmBlock denotes the Transformer block. The image below breaks down the components of the TrmBlock.
![[trm_block.png]]

Each of the 3 components in the TrmBlock were carefully chosen to be kept in this model based on the following reasonings:
1. Layer Normalisation: Layer Normalisation was initially propose to improve the convergence of deep neural networks by increasing their stability. Prior to the iTransformer, the Layer Normalisation was applied to normalise variates along the same timestamp, gradually fusing the variates together. However, this method would lead to discrepancies caused by inconsistent measurement scales. For example, one feature may be recorded in cm and another feature may be recorded in celsius; normalising over these two measurement scales will lead to discrepancies. As a result, normalising over the temporal dimension allows each feature to retain it's temporal structure while improving the model's ability to generalise as the impact of individual outliers are diminished.
2. Feed Forward Network (FFN): In traditional time-series Transformer based models, the FFN is used to applied identically to each time point, generating a token representation. Prior models embedded multiple variates to create a token for each timestamp. As a result, the embeddings suffered from being too localised and couldn't capture the complex relationships between variates. Secondly, the token would often be malpositioned as the embedding mechanism is unable to distinguish the individual importance of each feature. Taking the iTransformer's approach to encode by each variate rather than by each timestamp allowing the model to treat each variate as a sequence over time. The FFN will then create a representation of the full time series of that variate allowing it to capture complex patterns.
3. Self-Attention: Rather than applying attention across different timestamps the iTransformer applies attention across different variates. Rather than focusing on how past values of a single variate predict future values, the model focuses on the interactions between multiple variates. This is done by computing the correlations between each variate allowing the model to learn which variate influences each other more.

## Results

The iTransformer performance was evaluated on 7 of the standard time series benchmarks against 10 state of the art models including Transformer-based models, Linear-based models, and TCN-based models. The results of the models performance are displayed below. The red number indicate the best performing model and the blue indicate the second best result for the corresponding benchmark.

![](/images/itransformer_performance.png)

The results indicate the iTransformer is competitive among the prior state of the art models and even surpasses prior benchmarks. Where the iTransformer excels is in high-dimensional time series and surpasses the prior state of art model for multivariate time series forecasting (PatchTST).


### References:
1. Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L. and Long, M. (2023). _iTransformer: Inverted Transformers Are Effective for Time Series Forecasting_. [online] arXiv.org. doi:https://doi.org/10.48550/arXiv.2310.06625.

‌

### Appendix:
1. Encoder-only transformer architectures refer to the group of transformers that only utilize the encoder portion of the Transformer (as introduced by the original Attention is All You Need) paper. These models are optimised for tasks where the input sequence is the primary focus, rather than generating new sequences. Some of the common tasks these models perform well for include text classification, named entity recognition, question answering, semantic search, and language understanding tasks. Some of the well known models that fall into this category include BERT, RoBERTa, DistilBERT, and ALBERT.
