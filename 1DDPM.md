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

$\begin{eqnarray}
&& (\alpha_t...\alpha_1)^2+(\alpha_t...\alpha_2)^2\beta_1^2 + (\alpha_t...\alpha_3)^2\beta_2^2 + ... + \alpha_t^2\beta_{t-1}^2 + \beta_t^2\\
&=& \alpha_t^2 + \beta_t^2\\
&=& 1
\end{eqnarray}$
