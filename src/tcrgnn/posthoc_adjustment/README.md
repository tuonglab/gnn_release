### **Overall Idea**
The process begins with a list of T-cell receptor (TCR) sequences.  
Each sequence includes:  
- a **model-predicted probability** of being cancer-associated (`P`), and  
- a **clonal frequency** that reflects how expanded or common that TCR clone is in the sample (`F_raw`).  

The objective is to **refine** or **adjust** the model’s cancer-association probabilities by taking clonal expansion into account. In this way, expanded clones, which may indicate immune activation, can slightly increase or decrease their cancer-association score depending on the characteristics of the data distribution.

---

### **Step-by-Step Explanation**

#### 1. **Sample Skewness**
This step measures how **asymmetric** the distribution of model probabilities (`P`) is.  
- When the probabilities are evenly distributed, the skewness is close to zero.  
- When most TCRs have very low probabilities and only a few have high values, the skewness is positive.  
- This helps identify potential bias in the score distribution.

#### 2. **Skew-Aware Adjustment**
Using the computed skewness, the algorithm applies a **global correction** (`a`) to all probabilities:  
- If the distribution is excessively skewed, the scores are shifted slightly upward or downward to restore balance.  
- The magnitude of this correction depends on the skew sensitivity (`λ_s`) and the clipping limit (`λ_c`).  
- All resulting scores are then clipped to remain within the range of 0 to 1.

#### 3. **Score–Frequency Blending**
In this stage, each TCR’s model probability (`P`) is combined with its **frequency percentile** (`R`):  
- TCRs that have both **high probability and high frequency** (potentially meaningful clones) receive a **boost**.  
- TCRs that have **low probability and low frequency** (likely noise) are **penalized**.  
- Control parameters (`α`, `β`, `γ`) determine how strongly these adjustments are applied.

Mathematically:  
- `A_h` enhances high-confidence, high-frequency TCRs.  
- `A_l` reduces confidence for low-frequency, low-probability TCRs.  
- The final adjusted score (`S'`) is a smooth combination of the base model score and the adjusted values.

#### 4. **Full Pipeline**
In practical terms:  
1. The raw frequency is normalized (`F_raw` → `R`).  
2. The scores are blended and then adjusted for skew.  
3. The final cancer-association probability (`S'`) is obtained.

---

### **Intuitive Summary**
The algorithm refines the model’s cancer-association probabilities by:  
1. Detecting bias in the score distribution using skewness.  
2. Applying a global correction to mitigate that bias.  
3. Incorporating clonal frequency information to weight probabilities more meaningfully.  

In essence, **expanded clones with strong cancer-association signals are slightly enhanced, weak and rare clones are downweighted, and the overall score distribution is balanced for fairness and stability.**

## Mathematical Formulation

### 1. Sample Skewness

$$
\text{skew}(x) =
\begin{cases}
0, & n < 3 \text{ or } \sigma = 0, \\[6pt]
\dfrac{\frac{1}{n} \sum_{i=1}^n (x_i - \mu)^3}{\sigma^3}, & \text{otherwise.}
\end{cases}
\tag{1}
$$

$$
\mu = \frac{1}{n} \sum_{i=1}^n x_i, \quad
\sigma = \sqrt{\frac{1}{n} \sum_{i=1}^n (x_i - \mu)^2}.
\tag{2}
$$


### 2. Skew-Aware Score Adjustment

$$
a = -\tanh\!\big(\text{skew}(P) \cdot \lambda_s\big) \cdot \lambda_c
\tag{3}
$$

$$
P' = \mathrm{clip}(P + a, f, c)
\tag{4}
$$

where $ \lambda_s $ is the skew sensitivity, $ \lambda_c $ is the clipping strength, and $(f, c)$ are the lower and upper bounds (typically $0$ and $1$).


### 3. High-Confidence Score–Frequency Blending

$$
M_h = \mathbb{I}(P > p_h) \cdot \mathbb{I}(R > f_h), \quad
M_l = \mathbb{I}(P < 1 - p_h) \cdot \mathbb{I}(R < 1 - f_h)
\tag{5}
$$

$$
A_h = \min(P, R) \cdot M_h, \quad
A_l = \min(1 - P, 1 - R) \cdot M_l
\tag{6}
$$

$$
S_{\text{adj}} = P + \alpha A_h - \beta A_l
\tag{7}
$$

$$
S = (1 - \gamma) P + \gamma S_{\text{adj}}
\tag{8}
$$

$$
S' = \mathrm{clip}(S, 0, 1)
\tag{9}
$$


### 4. Full Pipeline

$$
\text{blend} = \text{combined\_score\_sample\_blend}(P, F_{\text{raw}})
\tag{10}
$$

$$
\text{score\_adj} = \text{combined\_score\_distribution\_aware\_simple}(\text{blend})
\tag{11}
$$


### 5. Variable Definitions

| Symbol | Meaning |
|:--------|:---------|
| $x_i$ | Individual data sample |
| $n$ | Number of samples |
| $\mu$ | Mean of samples |
| $\sigma$ | Standard deviation |
| $P$ | Base model score (probability) |
| $F_{\text{raw}}$ | Raw clone frequency |
| $R$ | Percentile-normalized frequency, $R = F_{\text{percentile}}(F_{\text{raw}})$ |
| $p_h, f_h$ | High-confidence thresholds for $P$ and $R$ |
| $M_h, M_l$ | Masks for high and low confidence regions |
| $A_h, A_l$ | Adjustment terms for high and low confidence |
| $\alpha, \beta, \gamma$ | Blend coefficients controlling balance and smoothing |
| $\lambda_s$ | Skew sensitivity (scaling factor) |
| $\lambda_c$ | Clipping limit for skew adjustment |
| $(f, c)$ | Lower and upper clipping bounds |
| $a$ | Global skew-based adjustment |
| $S, S_{\text{adj}}, S'$ | Intermediate and final adjusted scores |
| $\mathrm{clip}(\cdot)$ | Truncation to a specified range (e.g. $[0, 1]$) |
| $\mathbb{I}(\cdot)$ | Indicator function (1 if condition true, else 0) |

## Why We Average in Log-Odds (Not Probabilities)

The final step in our method is to convert the model’s output probabilities into **log-odds space**.  
We then compute the **average of all logits** in this space and transform the result **back into probability space** using the inverse logit function.

### Why this makes sense

- The final layer of the GNN model, before producing probabilities, is a **classifier layer** that combines learned features in a **linear way**.  
  This means the model operates additively and symmetrically in the log-odds domain.  
  Averaging in this space therefore **preserves the underlying geometry** of the model.

- From a **Bayesian perspective**, log-odds represent the **natural scale of evidence** for a binary event.  
  Each TCR in a sample repertoire can be viewed as an **independent piece of evidence**, and combining these observations through averaging in log-odds space reflects the **additive nature of evidence accumulation**.

- Converting the averaged log-odds back into probability space yields a **coherent, robust, and Bayesian-consistent aggregate probability**.

### Why not just average probabilities?

Because the **sigmoid layer compresses logits** into a bounded range between 0 and 1, taking a simple mean of probabilities would **bias the result toward extreme predictions**.  
In contrast, averaging in log-odds space avoids this issue because logits are **unbounded real numbers**.  
This allows the aggregated final score—such as a representative **cancer-risk probability**—to **preserve the true relative differences** in model confidence.
