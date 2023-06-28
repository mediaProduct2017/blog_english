Theoretical understanding of NBCE

NBCE, Naive Bayes-based Context Extension, is a method[1] proposed by Jianlin Su. The calculation and result of NBCE will lead to the similar models designed in other papers[2, 3], which were published by scientists in AI21 Labs and microsoft research.

We will first introduce the PCW model in [2], which can be viewed as a specific case of NBCE, with $\beta$=0 (The larger $\beta$ is, the more context it considers when answering questions; The smaller $\beta$ is, the more knowledge in the LLM it considers when answering questions). To understand PCW model, we first need to know basic knowledge of attention and transformer.

In the attention paper[4], Scaled Dot-Product Attention (Or dot-product attention) is defined as:
$$Attention(Q, K, V) = [softmax(\frac{QK^T}{\sqrt{d_k}})]V$$

The results of attention are not logits. Assume the length of the sentence is L, and the hidden size of the model is d (which is usually the same as the dimension of word embedding), the results of attention have dimension of $L \times d$. As we can see below, $\frac{QK^T}{\sqrt{d_k}}$ is a matrix of $L \times L$. For this matrix, the softmax of it means softmax for each row of the matrix, the result of the softmax is still a matrix of $L \times L$. V is a matrix of $L \times d$, therefore, the attention result is a matrix of $L \times d$.

For logits, the dimension is $L \times n$, where n is the number of classes when giving predictions, such as the number of words or tokens in language model. To get logits from attention results, we need another transformation:

$$logits = Attention(Q, K, V)*W$$

Where W is a matrix of $d \times n$. If we want probabilities, then

$$ probabilities = softmax(logits) $$

Where "probabilities" is a matrix of $L \times n$.

Assume the length of a sentence is 10 (the sentence has 10 words) and the dimension of word embedding is 8, for the self attention of this sentence, Q is a matrix of $10 \times 8$, and so is K and V. Therefore, $\frac{QK^T}{\sqrt{d_k}}$ is a matrix of 10*10.

Therefore, if the length of the sentence is L, then we need a matrix calculation of $L \times L$ (for example, for softmax of the matrix of $L \times L$, we need the softmax calculation $L \times L$ times). If L becomes larger, because of the existence of self attention (self attention in transformer), the increase of calculation is quadratic, not linear. Of course, for matrix calculation, it can be paralleled in GPU, not the quadratic complex in traditional algorithms.

For attention used in language model, attention masks are needed. We have padding mask and future (blinding) mask [5]. In language model and other sequence generation situations, in order to avoid information leak, we need future mask even for only one sentence (one sample of a batch). Future mask is usually used before softmax of the attention process, making some elements of the $\frac{QK^T}{\sqrt{d_k}}$ attention matrix not participate the calculation of softmax (For specific methods to do this, refer to [6]). When the element of future attention mask matrix is 1, the corresponding element in attention matrix will participate the calculation of softmax; when the element of future attention mask matrix is 0, the corresponding element in attention matrix will not participate the calculation of softmax.

For attention used in language model, the attention matrix (future attention matrix) means the generation of the sentence for a single sentence sample (including start and end sign) in a batch. For the first row, it means the generation of the next token following the first token; for the second row, it means the generation of the next token following the second token; so on and so forth. For unidirectional attention, a token will only attend to itself and the left tokens. For bidirectional attention, a token can attend to tokens both left and right (such as the uniLM model in seq2seq. If we have question and want to predict answer, the token in question can attend to all the tokens in question, no matter the left tokens or the right tokens).

If we import a causal language model (unidirectional attention), when training and inferring, lower triangular matrix will automatically be used as the future attention matrix. We don't have to assign future attention matrix by hand (It's automatic).

For padding mask, we need to assign it according to the sentence samples in a batch. The padding mask and future mask will be combined automatically by deep learning framework to produce final future mask (final mask) before softmax of (self) attention manipulation.

With padding, the final mask (with both future mask and padding mask) matrix will become larger. Assume the final mask is a matrix of $L \times L$, with padding, L will become larger because of the paddings and there will be more 0's in the mask matrix. If we have several samples in a batch, each sample will correspond to a mask matrix, and a batch will correspond to several mask matrix, that is, a tensor which has three dimensions, one dimension is for batch size, the other two dimensions is for mask matrix. Because the paddings for different sentences in a batch are different, the mask matrixes for different sentences are also different.

In PCW figure in [1] or Figure 3 in [2], we can see that, assume each context has token length of L, then n contexts have a total token length of $n \times L$. Assume task tokens have token length of T. The total length of prompt is nL+T.
If we don't use pcw or nbce, we need to input the prompt of length nL+T into LLM directly. Because of self attention in transformer, we will have a calculation amount of $(nL+T)^2$, which will be very large if n is large. That is, we need to conduct softmax calculation $(nL+T)^2$ times.


References：

[1] 苏剑林. (May. 23, 2023). 《NBCE：使用朴素贝叶斯扩展LLM的Context处理长度 》[Blog post]. Retrieved from https://kexue.fm/archives/9617 air: /Users/arfu/Documents/tensent_cloud/2023/papers2023

[2] Ratner, N., Levine, Y., Belinkov, Y., Ram, O., Abend, O., Karpas, E., ... & Shoham, Y. (2022). Parallel Context Windows Improve In-Context Learning of Large Language Models. arXiv preprint arXiv:2212.10947. Parallel Context Windows for Large Language Models.

[3] Hao, Y., Sun, Y., Dong, L., Han, Z., Gu, Y., & Wei, F. (2022). Structured Prompting: Scaling In-Context Learning to 1,000 Examples. arXiv preprint arXiv:2212.06713.

[4] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.

[5] Attention中的Mask; query mask, key mask, future mask; Posted by ZhangWenXiang on July 27, 2019. Retrieved from https://demmon-tju.github.io/2019/07/27/attention-masks/

[6] 苏剑林. (Jul. 16, 2019). 《“让Keras更酷一些！”：层中层与mask 》[Blog post]. Retrieved from https://kexue.fm/archives/6810

[7] 苏剑林. (Sep. 18, 2019). 《从语言模型到Seq2Seq：Transformer如戏，全靠Mask 》[Blog post]. Retrieved from https://kexue.fm/archives/6933


Understanding diffusion models (1): DDPM

$$x_t = \alpha_tx_{t-1} + \beta_t\varepsilon_t\, , \ \varepsilon_t \sim Normal \ (0,I)$$

We have $\alpha_t, \beta_t > 0$ and $\alpha_t^2 + \beta_t^2 = 1$ (an extra constraint which makes the subsequent calculation easier and which is reasonable because if you add more noise, the left part of the data is less).

$\begin{eqnarray}
x_t &=& \alpha_tx_{t-1} + \beta_t\varepsilon_t\\
&=& \alpha_t(\alpha_{t-1}x_{t-2}+\beta_{t-1}\varepsilon_{t-1}) + \beta_t\varepsilon_t\\
&=& ...\\
&=& (\alpha_t...\alpha_1)x_0 + (\alpha_t...\alpha_2)\beta_1\varepsilon_1 + (\alpha_t...\alpha_3)\beta_2\varepsilon_2 + ... + \alpha_t\beta_{t-1}\varepsilon_{t-1} + \beta_t\varepsilon_t  \tag{1}\label{eq1}
\end{eqnarray}$

We assume $\varepsilon_{all} = (\alpha_t...\alpha_2)\beta_1\varepsilon_1 + (\alpha_t...\alpha_3)\beta_2\varepsilon_2 + ... + \alpha_t\beta_{t-1}\varepsilon_{t-1} + \beta_t\varepsilon_t$. &nbsp; $\varepsilon_{all}$ is the sum of multiple independent normal variables, and therefore it also follows normal distribution, with the mean of 0, and the variance of $(\alpha_t...\alpha_2)^2\beta_1^2 + (\alpha_t...\alpha_3)^2\beta_2^2 + ... + \alpha_t^2\beta_{t-1}^2 + \beta_t^2$.

With the assumption of $\alpha_t^2 + \beta_t^2 = 1$, see equation ($\ref{eq1}$), the sum of square of coefficients $(\alpha_t...\alpha_1)^2+(\alpha_t...\alpha_2)^2\beta_1^2 + (\alpha_t...\alpha_3)^2\beta_2^2 + ... + \alpha_t^2\beta_{t-1}^2 + \beta_t^2$ needs to be calculated.

$\begin{eqnarray}
&& (\alpha_t...\alpha_1)^2+(\alpha_t...\alpha_2)^2\beta_1^2\\
&=& (\alpha_t...\alpha_2)^2\alpha_1^2+(\alpha_t...\alpha_2)^2\beta_1^2\\
&=& (\alpha_t...\alpha_2)^2(\alpha_1^2+\beta_1^2)\\
&=& (\alpha_t...\alpha_2)^2
\end{eqnarray}$

$\begin{eqnarray}
&& (\alpha_t...\alpha_1)^2+(\alpha_t...\alpha_2)^2\beta_1^2+(\alpha_t...\alpha_3)^2\beta_2^2\\
&=& (\alpha_t...\alpha_2)^2+(\alpha_t...\alpha_3)^2\beta_2^2\\
&=& (\alpha_t...\alpha_3)^2\alpha_2^2+(\alpha_t...\alpha_3)^2\beta_2^2\\
&=& (\alpha_t...\alpha_3)^2(\alpha_2^2+\beta_2^2)\\
&=& (\alpha_t...\alpha_3)^2
\end{eqnarray}$

Therefore,

$\begin{eqnarray}
(\alpha_t...\alpha_1)^2+(\alpha_t...\alpha_2)^2\beta_1^2 + (\alpha_t...\alpha_3)^2\beta_2^2 + ... + \alpha_t^2\beta_{t-1}^2 = \alpha_t^2
\end{eqnarray}$

Now we have

$\begin{align*}
& (\alpha_t...\alpha_1)^2+(\alpha_t...\alpha_2)^2\beta_1^2 + (\alpha_t...\alpha_3)^2\beta_2^2 + ... + \alpha_t^2\beta_{t-1}^2 + \beta_t^2\\
&= \alpha_t^2 + \beta_t^2\\
&= 1
\end{align*}$
