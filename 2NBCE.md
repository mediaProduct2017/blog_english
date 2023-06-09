Theoretical understanding of NBCE

NBCE, Naive Bayes-based Context Extension, is a method[1] proposed by Jianlin Su. The calculation and result of NBCE will lead to the similar models designed in other papers[2, 3], which were published by scientists in AI21 Labs and microsoft research.

We will first introduce the PCW model in [2], which can be viewed as a specific case of NBCE. To understand PCW model, we first need to know basic knowledge of attention and transformer.

In the attention paper[4], Scaled Dot-Product Attention is defined as:
$$Attention(Q, K, V) = [softmax(\frac{QK^T}{\sqrt{d_k}})]V$$

Assume the length of a sentence is 10 (the sentence has 10 words) and the dimension of word embedding is 8, for the self attention of this sentence, Q is a matrix of $10 \times 8$, and so is K and V. Therefore, $\frac{QK^T}{\sqrt{d_k}}$ is a matrix of 10*10.

For attention used in language model, attention masks are needed. We have padding mask and future (blinding) mask [5]. In language model and other sequence generation situations, in order to avoid information leak, we need future mask even for only one sentence (one sample of a batch). Future mask is usually used before softmax of the attention process, making some elements of the $\frac{QK^T}{\sqrt{d_k}}$ attention matrix not participate the calculation of softmax.


References：

[1] 苏剑林. (May. 23, 2023). 《NBCE：使用朴素贝叶斯扩展LLM的Context处理长度 》[Blog post]. Retrieved from https://kexue.fm/archives/9617

[2] Ratner, N., Levine, Y., Belinkov, Y., Ram, O., Abend, O., Karpas, E., ... & Shoham, Y. (2022). Parallel Context Windows Improve In-Context Learning of Large Language Models. arXiv preprint arXiv:2212.10947.

[3] Hao, Y., Sun, Y., Dong, L., Han, Z., Gu, Y., & Wei, F. (2022). Structured Prompting: Scaling In-Context Learning to 1,000 Examples. arXiv preprint arXiv:2212.06713.

[4] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.

[5] Attention中的Mask; query mask, key mask, future mask; Posted by ZhangWenXiang on July 27, 2019. Retrieved from https://demmon-tju.github.io/2019/07/27/attention-masks/



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
