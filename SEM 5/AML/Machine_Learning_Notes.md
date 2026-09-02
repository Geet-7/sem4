# Machine Learning — Complete Exam Notes

> **Target Exam**: University End-Semester Examination  
> **Course**: Applied Machine Learning / Machine Learning  
> **Syllabus Coverage**: Unit 1 & Unit 2 (100% Exam-Oriented)  
> **Special Focus**: Unit 2 High-Priority 4-Mark Answers, Solved Numericals, Formulas, Long Answers, and Revision Guides  

---

## Table of Contents

1. [UNIT 1 — Introduction to Machine Learning](#unit-1--introduction-to-machine-learning)
   - [1.1 Introduction to Machine Learning](#11-introduction-to-machine-learning)
   - [1.2 Examples of Machine Learning Applications](#12-examples-of-machine-learning-applications)
   - [1.3 Learning Types](#13-learning-types)
   - [1.4 Supervised Learning — Learning a Class from Examples](#14-supervised-learning--learning-a-class-from-examples)
   - [1.5 Vapnik-Chervonenkis (VC) Dimension](#15-vapnik-chervonenkis-vc-dimension)
   - [1.6 Probably Approximately Correct (PAC) Learning](#16-probably-approximately-correct-pac-learning)
   - [1.7 Noise](#17-noise)
   - [1.8 Learning Multiple Classes](#18-learning-multiple-classes)
   - [1.9 Regression](#19-regression)
   - [1.10 Model Selection and Generalization](#110-model-selection-and-generalization)
   - [1.11 Dimensions of a Supervised Machine Learning Algorithm](#111-dimensions-of-a-supervised-machine-learning-algorithm)
2. [UNIT 2 — Feature Selection (High Priority for 4-Mark Questions)](#unit-2--feature-selection)
   - [2.1 Concept of Feature](#21-concept-of-feature)
   - [2.2 Data Preprocessing](#22-data-preprocessing)
   - [2.3 Normalization and Scaling](#23-normalization-and-scaling)
   - [2.4 Standardization (Z-Score Normalization)](#24-standardization-z-score-normalization)
   - [2.5 Normalization vs Standardization](#25-normalization-vs-standardization)
   - [2.6 Managing Missing Values](#26-managing-missing-values)
   - [2.7 Introduction to Dimensionality Reduction](#27-introduction-to-dimensionality-reduction)
   - [2.8 Principal Component Analysis (PCA)](#28-principal-component-analysis-pca)
   - [2.9 Kernel PCA](#29-kernel-pca)
   - [2.10 Linear Discriminant Analysis (LDA)](#210-linear-discriminant-analysis-lda)
   - [2.11 PCA vs LDA](#211-pca-vs-lda)
   - [2.12 Feature Selection Techniques](#212-feature-selection-techniques)
   - [2.13 Sequential Forward Selection (SFS)](#213-sequential-forward-selection-sfs)
   - [2.14 Sequential Backward Selection (SBS)](#214-sequential-backward-selection-sbs)
   - [2.15 SFS vs SBS](#215-sfs-vs-sbs)
3. [IMPORTANT NUMERICAL QUESTION BANK](#important-numerical-question-bank)
   - [Unit 1 Solved Numericals](#unit-1-solved-numericals)
   - [Unit 2 Solved Numericals](#unit-2-solved-numericals)
4. [🔥 MOST IMPORTANT 4-MARK QUESTION BANK](#-most-important-4-mark-question-bank)
   - [Unit 2 — 4-Mark Questions (25 Solved)](#unit-2--4-mark-questions-25-solved)
   - [Unit 1 — 4-Mark Questions (20 Solved)](#unit-1--4-mark-questions-20-solved)
5. [IMPORTANT LONG-ANSWER QUESTIONS (8–10 MARKS)](#important-long-answer-questions-810-marks)
6. [IMPORTANT COMPARISON TABLES](#important-comparison-tables)
7. [IMPORTANT FORMULAS — LAST-MINUTE SHEET](#important-formulas--last-minute-sheet)
8. [COMMON MISTAKES TO AVOID](#common-mistakes-to-avoid)
9. [UNIVERSITY ANSWER-WRITING GUIDE](#university-answer-writing-guide)
10. [🔥 ONE-DAY BEFORE EXAM REVISION PLAN](#-one-day-before-exam-revision-plan)
11. [⚡ 30-MINUTE ULTRA-QUICK REVISION](#-30-minute-ultra-quick-revision)

---


# UNIT 1 — Introduction to Machine Learning

---

## 1.1 Introduction to Machine Learning

### 1. Definition
**Machine Learning (ML)** is a subfield of Artificial Intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.

> **Formal Definition (Tom M. Mitchell, 1997):**  
> *"A computer program is said to learn from experience $E$ with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$, improves with experience $E$."*

### 2. Intuition
In traditional computing, humans write explicit rules and algorithms to convert input data into output answers. In Machine Learning, the computer takes input data alongside known answers (or structure within data) and automatically discovers the underlying rules and patterns.

### 3. Machine Learning vs Traditional Programming

```
Traditional Programming:
[Data] + [Rules / Program]  --->  [Output / Answers]

Machine Learning:
[Data] + [Answers / Target] --->  [Learned Rules / Model]  ---> [Predictions on New Data]
```

| Parameter | Traditional Programming | Machine Learning |
|---|---|---|
| **Primary Input** | Data + Hand-coded Rules | Data + Historical Answers / Interactions |
| **Logic Creation** | Designed manually by software developers | Discovered automatically by learning algorithms |
| **Adaptability** | Hard to adapt to non-linear, dynamic environments | Dynamically updates as new data arrives |
| **Problem Complexity**| Suitable for deterministic, rule-based tasks | Ideal for perception, vision, NLP, complex patterns |
| **Maintenance** | Rule sets grow complex and unmaintainable | Model retrained on fresh datasets |

### 4. Components of a Machine Learning System

1. **Dataset**: A collection of historical data points used for learning and evaluation.
2. **Features (Attributes / Inputs)**: Measurable individual properties or characteristics of the observed phenomenon, denoted as $\mathbf{x} = [x_1, x_2, \dots, x_d]^T \in \mathbb{R}^d$.
3. **Labels (Targets / Outputs)**: The ground-truth variable we wish to predict, denoted as $y$. In classification, $y \in \{0, 1\}$ or discrete classes; in regression, $y \in \mathbb{R}$.
4. **Model (Hypothesis $h$)**: The mathematical representation or function $h(\mathbf{x}; \theta)$ parametrized by weights $\theta$ that maps inputs to predicted outputs.
5. **Training (Learning Phase)**: The process of optimizing model parameters $\theta$ by minimizing a loss function over the training dataset.
6. **Testing / Evaluation**: Assessing the generalization performance of the trained model on unseen test data.
7. **Inference / Prediction**: Deploying the trained model $h(\mathbf{x})$ to output predicted values for completely new, real-world input samples.
8. **Training Data vs Testing Data**:
   - **Training Data**: Historical subset (typically 70%–80%) used exclusively to fit the model parameters.
   - **Testing Data**: Independent, held-out subset (typically 20%–30%) used solely to evaluate how well the model generalizes to unseen data.

### 5. Real-World Examples of Machine Learning Systems

1. **Spam Detection**:
   - *Input*: Email text body, sender domain, frequency of suspicious keywords ("free", "claim").
   - *Output*: Binary label — Spam ($1$) or Not Spam ($0$).
2. **House-Price Prediction**:
   - *Input*: Square footage, number of bedrooms, location zip code, age of house.
   - *Output*: Continuous price in dollars ($\$350,000$).
3. **Image Classification (Medical Diagnosis)**:
   - *Input*: Pixel values of X-ray images ($256 \times 256$ pixels).
   - *Output*: Multi-class label — Pneumonia, Tuberculosis, Normal.
4. **Recommendation Systems (Netflix / E-commerce)**:
   - *Input*: User watch history, user ratings, item metadata, demographic info.
   - *Output*: Ranked list of recommended movies or products.
5. **Fraud Detection (Banking)**:
   - *Input*: Transaction amount, time stamp, merchant location, spending history.
   - *Output*: Fraudulent ($1$) vs Legitimate ($0$).

---

## 1.2 Examples of Machine Learning Applications

| Application | Input / Features ($\mathbf{x}$) | Target Output ($y$) | Learning Type | Possible ML Algorithm | Why ML is Useful? |
|---|---|---|---|---|---|
| **Spam Filtering** | Word frequencies, header info | Binary (Spam/Ham) | Supervised | Naive Bayes, Logistic Regression | Spam patterns evolve rapidly; manual rules break easily. |
| **House Valuation** | Area, bedrooms, zip code | Continuous Price ($) | Supervised | Linear Regression, Decision Trees | Handles non-linear multi-variate pricing interactions. |
| **Customer Segmentation** | Purchase history, annual income, age | Cluster ID (Group 1, 2, 3) | Unsupervised | K-Means Clustering, DBSCAN | Discovers hidden customer personas without manual labeling. |
| **Autonomous Driving** | Camera pixels, LiDAR range, speed | Steering angle, brake force | Supervised / Reinforcement | Deep Convolutional Neural Networks, PPO | Real-time visual perception in unconstrained environments. |
| **Credit Scoring** | Income, debt ratio, credit history | High Risk / Low Risk | Supervised | Support Vector Machines (SVM), Random Forest | Automates objective lending decisions with statistical rigor. |
| **Handwritten Digit Recognition**| $28 \times 28$ grayscale pixel matrix | Digit class ($0, 1, \dots, 9$) | Supervised | Convolutional Neural Network (CNN), $k$-NN | Handwritten styles vary infinitely across individuals. |
| **Game Playing (AlphaGo)**| Board state configurations | Next optimal move | Reinforcement Learning | Deep Q-Learning, Monte Carlo Tree Search | Search space exceeds total atoms in the observable universe. |

---

## 1.3 Learning Types

Machine Learning algorithms are broadly categorized into four primary paradigms based on the nature of the learning feedback.

```
                    ┌─────────────────────────────────────────┐
                    │       Machine Learning Paradigms        │
                    └────────────────────┬────────────────────┘
                                         │
        ┌───────────────────┬────────────┴───────┬────────────────────┐
        ▼                   ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌─────────────────┐   ┌────────────────────────┐
│  Supervised   │   │ Unsupervised  │   │ Semi-Supervised │   │ Reinforcement Learning │
│   Learning    │   │   Learning    │   │    Learning     │   │          (RL)          │
└───────┬───────┘   └───────┬───────┘   └─────────────────┘   └───────────┬────────────┘
        │                   │                                             │
   ┌────┴────┐         ┌────┴────┐                                   ┌────┴────┐
   ▼         ▼         ▼         ▼                                   ▼         ▼
Classifi- Regression Cluster- Dimension-                           Agent   Environment
cation               ing     ality Red.                                (State, Action, Reward)
```

### 1. Supervised Learning
- **Definition**: The algorithm is provided with a labeled training dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$, where $\mathbf{x}_i$ is the feature vector and $y_i$ is the target ground truth label.
- **Goal**: Learn a mapping function $h: \mathcal{X} \rightarrow \mathcal{Y}$ that accurately predicts target $y$ for new unseen inputs $\mathbf{x}$.
- **Sub-types**:
  1. **Classification**: Target label $y$ is discrete / categorical (e.g., Disease diagnosis: Yes/No; Digit recognition: $0-9$).
  2. **Regression**: Target label $y$ is continuous real-valued (e.g., Stock price prediction, Temperature forecasting).
- **Popular Algorithms**: Linear Regression, Logistic Regression, Decision Trees, Support Vector Machines (SVM), $k$-Nearest Neighbors ($k$-NN), Neural Networks.

### 2. Unsupervised Learning
- **Definition**: The algorithm receives an unlabeled dataset $\mathcal{D} = \{\mathbf{x}_i\}_{i=1}^N$ containing inputs without target labels.
- **Goal**: Uncover underlying structure, probability distributions, latent patterns, or groupings within the data.
- **Sub-types**:
  1. **Clustering**: Partitioning data into groups of similar instances (e.g., K-Means, Hierarchical Clustering).
  2. **Dimensionality Reduction**: Projecting high-dimensional data into a lower-dimensional subspace while preserving variance or distance relationships (e.g., PCA, LDA, t-SNE).
  3. **Association Rule Learning**: Discovering interesting relations between variables in large databases (e.g., Apriori algorithm for Market Basket Analysis).

### 3. Semi-Supervised Learning
- **Definition**: The algorithm trains on a dataset containing a small amount of labeled data $\mathcal{D}_L = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N_L}$ and a large volume of unlabeled data $\mathcal{D}_U = \{\mathbf{x}_j\}_{j=1}^{N_U}$ (where $N_U \gg N_L$).
- **Why useful?**: Data labeling is expensive, time-consuming, and requires domain expertise (e.g., medical image annotation), whereas unlabeled data is abundant and cheap.
- **Common Approaches**: Pseudo-labeling, Graph-based Semi-Supervised Learning, Generative Adversarial Networks (GANs).

### 4. Reinforcement Learning (RL)
- **Definition**: An agent learns optimal decision-making behavior by interacting dynamically with an environment through trial and error.
- **Key Concepts**:
  - **Agent**: The learner or decision-maker.
  - **Environment**: Everything the agent interacts with.
  - **State ($s \in \mathcal{S}$)**: The current situation or configuration of the environment.
  - **Action ($a \in \mathcal{A}$)**: The choice made by the agent in state $s$.
  - **Reward ($r \in \mathbb{R}$)**: Immediate scalar feedback signal returned by the environment evaluating action $a$.
  - **Policy ($\pi(a|s)$)**: The strategy mapping state $s$ to action $a$.
  - **Goal**: Maximize cumulative expected future rewards over time ($\sum_{t=0}^T \gamma^t r_t$, where $\gamma \in [0, 1)$ is the discount factor).

### 5. Comparison: Supervised vs Unsupervised vs Semi-Supervised vs Reinforcement Learning

| Feature | Supervised Learning | Unsupervised Learning | Semi-Supervised Learning | Reinforcement Learning |
|---|---|---|---|---|
| **Data Nature** | Labeled $(\mathbf{x}, y)$ | Unlabeled $(\mathbf{x})$ | Small Labeled + Large Unlabeled | Environment States & Delayed Rewards |
| **Objective** | Predict target output $y$ | Discover hidden patterns/clusters | Leverage unlabeled data to improve accuracy | Maximize cumulative long-term reward |
| **Feedback Signal** | Explicit correct target answer | No feedback signal | Partial target feedback | Delayed scalar reward/punishment |
| **Main Tasks** | Classification, Regression | Clustering, Dim. Reduction | Semi-supervised classification | Robot navigation, Game AI (Chess, Go) |
| **Human Supervision**| High (Requires full labeling) | Low / None | Medium | Environment reward specification |

---

## 1.4 Supervised Learning — Learning a Class from Examples

### 1. Fundamental Terminology
- **Training Examples**: Sample instances $(\mathbf{x}_i, y_i)$ provided to the learning algorithm.
- **Features**: Representational vector $\mathbf{x} = [x_1, x_2]^T$ spanning the feature space $\mathcal{X}$.
- **Target Concept ($C$)**: The true underlying function $C: \mathcal{X} \rightarrow \{0, 1\}$ that generates positive ($1$) and negative ($0$) labels.
- **Hypothesis ($h \in \mathcal{H}$)**: An approximation of the target concept formed by the learning algorithm. $\mathcal{H}$ denotes the hypothesis space.
- **Decision Boundaries**: The surface or hyper-plane in feature space separating different class predictions.

### 2. Conceptual Walkthrough: Learning a Family Car Concept
Consider predicting whether a vehicle is a "Family Car" based on two continuous features:
- $x_1$: Price of the car
- $x_2$: Engine Power

Let the true concept class $C$ be defined by an axis-aligned rectangle:
$$p_1 \le \text{Price} \le p_2 \quad \text{and} \quad e_1 \le \text{Engine Power} \le e_2$$

```
Engine Power (x2)
    ^
    │          + (Positive: Family Car)
    │          - (Negative: Non-Family Car)
 e2 ┼───────┬───────────────┐
    │       │   +   +   +   │
    │   -   │   +   +   +   │   -
 e1 ┼───────┴───────────────┴───
    │   -               -       -
    └───────┼───────────────┼────────> Price (x1)
           p1              p2
```

### 3. Most Specific and Most General Hypotheses
- **Most Specific Hypothesis ($S$)**: The smallest bounding box containing all positive examples and no negative examples.
- **Most General Hypothesis ($G$)**: The largest bounding box that contains all positive examples without encompassing any negative examples.
- **Version Space**: The set of all hypotheses $h \in \mathcal{H}$ consistent with the training set, bounded between $S$ and $G$.

### 4. Training Error vs Test Error vs Generalization
- **Training Error ($E_{train}$)**: Proportion of incorrect predictions made by hypothesis $h$ on the training dataset.
  $$E_{train}(h) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(h(\mathbf{x}_i) \neq y_i)$$
- **Test Error / Generalization Error ($E_{test}$ / $E(h)$)**: Expected error rate of hypothesis $h$ on new, unseen instances sampled from the underlying data distribution $P(\mathcal{X}, \mathcal{Y})$.
  $$E_{test}(h) = \mathbb{E}_{(\mathbf{x}, y) \sim P}[ \mathbb{I}(h(\mathbf{x}) \neq y) ]$$
- **Generalization**: The ability of a trained machine learning model to perform accurately on previously unseen data.

---

## 1.5 Vapnik-Chervonenkis (VC) Dimension

### 1. Definition & Technical Foundation
The **VC Dimension** is a formal measure of the capacity, complexity, and expressive power of a hypothesis space $\mathcal{H}$.

### 2. Concept of Shattering
A set of points $N$ is said to be **shattered** by a hypothesis space $\mathcal{H}$ if, for *every* possible binary labeling ($2^N$ possible label combinations) of those points, there exists a hypothesis $h \in \mathcal{H}$ that achieves zero classification error (perfect separation).

> **Formal Definition of VC Dimension:**  
> The VC Dimension of a hypothesis space $\mathcal{H}$, denoted as $VC(\mathcal{H})$, is the **maximum number of points $d$** that can be shattered by $\mathcal{H}$.  
> If $\mathcal{H}$ can shatter arbitrarily large point sets, then $VC(\mathcal{H}) = \infty$.

### 3. Geometric Intuition & Shattering Examples

#### A. Linear Classifier (Lines) in 2D ($VC = 3$)
- **Can 3 points be shattered by a 2D line?**  
  Yes! For any non-collinear 3 points in 2D space, all $2^3 = 8$ binary label combinations can be separated by a straight line.
- **Can 4 points be shattered by a 2D line?**  
  No! Consider 4 points arranged in a square where diagonally opposite points share the same label (XOR configuration). No single straight line can separate positive diagonal points from negative diagonal points.
- **Conclusion**: For 2D linear classifiers, $VC(\mathcal{H}) = 3$. In general, for hyperplanes in $d$-dimensional space $\mathbb{R}^d$, $VC(\mathcal{H}) = d + 1$.

```
Can 3 points be shattered in 2D by a line? YES (2^3 = 8 combinations separable)
  +   +        +   -        -   -        +   +
  ─── line ─── line ─── line ─── line
    +            +            -            -

Can 4 points (XOR pattern) be shattered by a line? NO!
  +    -
    ╳   <-- Requires 2 non-linear lines (XOR Problem)
  -    +
```

### 4. Growth Function & Sauer's Lemma
- **Growth Function ($\Pi_{\mathcal{H}}(N)$)**: The maximum number of distinct labelings that $\mathcal{H}$ can assign to any set of $N$ points.
  - If $N \le VC(\mathcal{H})$, $\Pi_{\mathcal{H}}(N) = 2^N$ (Polynomial growth).
  - If $N > VC(\mathcal{H})$, growth is bounded polynomially according to **Sauer's Lemma**:
    $$\Pi_{\mathcal{H}}(N) \le \sum_{i=0}^{d} \binom{N}{i} \le \left(\frac{e N}{d}\right)^d \quad (\text{where } d = VC(\mathcal{H}))$$

### 5. Why VC Dimension Matters in ML?
1. **Generalization Bound**: VC Dimension quantifies model complexity in generalization error bounds:
   $$E_{test}(h) \le E_{train}(h) + \sqrt{\frac{d \left(\ln\left(\frac{2N}{d}\right) + 1\right) - \ln\left(\frac{\delta}{4}\right)}{N}}$$
2. **Structural Risk Minimization (SRM)**: Prevents overfitting by balancing training error minimization against hypothesis capacity $d = VC(\mathcal{H})$.

---

## 1.6 Probably Approximately Correct (PAC) Learning

### 1. Motivation & Framework
Introduced by Leslie Valiant (1984), the **PAC Learning Framework** provides a mathematical framework for analyzing sample complexity and computational learnability in supervised machine learning.

### 2. Key Parameters
- **Target Concept ($c \in \mathcal{C}$)**: True function generating data labels.
- **Hypothesis ($h \in \mathcal{H}$)**: Learner's candidate function.
- **Accuracy Parameter ($\epsilon \in (0, 1)$)**: Maximum permissible error. The hypothesis $h$ is **approximately correct** if its generalization error $R(h) \le \epsilon$.
- **Confidence Parameter ($\delta \in (0, 1)$)**: Maximum permissible probability of failure. The learner succeeds **probably** if $P[R(h) \le \epsilon] \ge 1 - \delta$.

> **Formal Definition of PAC Learnability:**  
> A concept class $\mathcal{C}$ is **PAC-learnable** using hypothesis space $\mathcal{H}$ if there exists an algorithm $\mathcal{A}$ such that for every target concept $c \in \mathcal{C}$, distribution $D$ over $\mathcal{X}$, $\epsilon > 0$, and $\delta > 0$, the algorithm draws a sample $S$ of size $m(\epsilon, \delta)$ and outputs hypothesis $h \in \mathcal{H}$ satisfying:
> $$P_{S \sim D^m}\Big[ R(h) \le \epsilon \Big] \ge 1 - \delta$$
> where sample complexity $m(\epsilon, \delta)$ grows polynomially with $\frac{1}{\epsilon}$, $\frac{1}{\delta}$, and problem dimension.

### 3. Sample Complexity Derivation for Finite Hypothesis Space $|H|$
For a consistent learner choosing $h \in \mathcal{H}$ that makes zero training errors ($E_{train}(h) = 0$):
- Probability that a single bad hypothesis ($R(h) > \epsilon$) agrees with 1 random example is $\le (1 - \epsilon)$.
- Probability that a bad hypothesis agrees with all $m$ independent training examples is $\le (1 - \epsilon)^m \le e^{-\epsilon m}$.
- By Union Bound over all bad hypotheses in $\mathcal{H}$:
  $$P[\exists h \in \mathcal{H} \text{ s.t. } R(h) > \epsilon \text{ and } E_{train}(h)=0] \le |\mathcal{H}| e^{-\epsilon m}$$
- To ensure this failure probability is at most $\delta$:
  $$|\mathcal{H}| e^{-\epsilon m} \le \delta \implies e^{-\epsilon m} \le \frac{\delta}{|\mathcal{H}|}$$
  $$\implies -\epsilon m \le \ln\left(\frac{\delta}{|\mathcal{H}|}\right) \implies m \ge \frac{1}{\epsilon} \left( \ln|\mathcal{H}| + \ln\left(\frac{1}{\delta}\right) \right)$$

> **Fundamental PAC Sample Complexity Formula:**
> $$m \ge \frac{1}{\epsilon} \left( \ln |\mathcal{H}| + \ln \left(\frac{1}{\delta}\right) \right)$$

---

## 1.7 Noise

### 1. Definition
**Noise** refers to corruptions, errors, or unwanted random variations present in the training dataset that distort the underlying true relationship between features and target output.

### 2. Sources of Noise
1. **Label Noise (Teacher Error)**: Incorrect target assignment (e.g., a spam email erroneously labeled as non-spam by human annotators).
2. **Feature Noise (Sensor / Input Error)**: Corrupted feature values caused by sensor malfunction, typos, or measurement precision limits.
3. **Measurement / Environmental Noise**: Background interference affecting signal capture.

### 3. Effect of Noise on Learning & Overfitting
- Complex models attempt to fit noisy data points, resulting in complex, jagged decision boundaries (**Overfitting**).
- Destroys model generalization on clean test sets.

```
Clean Data Decision Boundary:          Noisy Data (Overfitted Boundary):
      +  +  + | -  -                         +  +  + ─┐ -  -
      +  +  + | -  -                         +  +  +  │ -  -
      +  +  + | -  -                         +  + -<-─┘ -  -  (Fits noisy minus!)
   Smooth Linear Boundary                 Complex Overfitted Boundary
```

### 4. Handling Noisy Data
- **Robust Loss Functions**: Use Huber loss or MAE instead of MSE to mitigate outlier sensitivity.
- **Regularization ($L_1, L_2$)**: Penalizes overly complex model parameters.
- **Pruning**: Trim deep branches in Decision Trees.
- **Soft-Margin Classifiers**: Use SVM slack variables ($\xi_i$) to permit controlled training misclassifications.

---

## 1.8 Learning Multiple Classes

### 1. Multi-Class Classification Problem
When target class label $y \in \{1, 2, \dots, K\}$ where $K > 2$. Standard binary classifiers (SVM, Logistic Regression) can be adapted using decomposition strategies.

### 2. Strategy 1: One-vs-Rest (OvR) / One-vs-All (OvA)
- **Concept**: Train $K$ distinct binary classifiers. The $k$-th classifier $h_k(\mathbf{x})$ is trained treating class $k$ as positive ($+1$) and all remaining $K-1$ classes as negative ($-1$).
- **Decision Rule**: Predict class index achieving highest confidence score:
  $$\hat{y} = \arg\max_{k \in \{1, \dots, K\}} h_k(\mathbf{x})$$
- **Total Classifiers Trained**: $K$

### 3. Strategy 2: One-vs-One (OvO)
- **Concept**: Train a binary classifier for every pair of classes $(i, j)$ where $i < j$.
- **Decision Rule**: Pass instance $\mathbf{x}$ through all $\frac{K(K-1)}{2}$ binary classifiers. Each classifier votes for its preferred class; assign class receiving maximum total votes (Majority Voting).
- **Total Classifiers Trained**: $\frac{K(K-1)}{2}$

```
Multiclass Strategy Diagram (K = 3 classes: A, B, C):

One-vs-Rest (3 Classifiers):              One-vs-One (3 Classifiers):
Classifier 1: A vs {B, C}                 Classifier 1: A vs B
Classifier 2: B vs {A, C}                 Classifier 2: A vs C
Classifier 3: C vs {A, B}                 Classifier 3: B vs C
```

### 4. Comparison: One-vs-One vs One-vs-Rest

| Parameter | One-vs-Rest (OvR) | One-vs-One (OvO) |
|---|---|---|
| **Number of Classifiers** | $K$ | $\frac{K(K-1)}{2}$ |
| **Training Set Size per Model**| Full dataset size ($N$) | Subset containing only two classes ($\approx \frac{2N}{K}$) |
| **Training Speed** | Faster for small $K$, slower for huge $K$ | Fast per classifier, scales quadratically $\mathcal{O}(K^2)$ |
| **Class Imbalance** | High imbalance (1 positive vs $K-1$ negative)| Balanced (Equal instances per pair) |
| **Decision Ambiguity** | Ambiguous overlap regions possible | Ambiguous tie votes possible |

---

## 1.9 Regression

### 1. Definition
**Regression** is a supervised learning task where the target output variable $y$ is continuous and real-valued ($y \in \mathbb{R}$).

### 2. Simple Linear Regression
Model assumes a linear relationship between a single independent input feature $x$ and dependent target $y$:
$$\hat{y} = w_0 + w_1 x$$
- $w_0$: $y$-intercept
- $w_1$: Slope coefficient
- Residual error for $i$-th instance: $e_i = y_i - \hat{y}_i = y_i - (w_0 + w_1 x_i)$

### 3. Derivation of Least Squares Normal Equations
We minimize the **Sum of Squared Errors (SSE)** loss function:
$$S(w_0, w_1) = \sum_{i=1}^N (y_i - w_0 - w_1 x_i)^2$$

Setting partial derivatives with respect to $w_0$ and $w_1$ to zero:
$$\frac{\partial S}{\partial w_0} = -2 \sum_{i=1}^N (y_i - w_0 - w_1 x_i) = 0 \implies \sum y_i - N w_0 - w_1 \sum x_i = 0$$
$$\implies w_0 = \bar{y} - w_1 \bar{x} \quad \left(\text{where } \bar{x} = \frac{\sum x_i}{N}, \bar{y} = \frac{\sum y_i}{N}\right)$$

Now for slope $w_1$:
$$\frac{\partial S}{\partial w_1} = -2 \sum_{i=1}^N x_i (y_i - w_0 - w_1 x_i) = 0$$
Substituting $w_0 = \bar{y} - w_1 \bar{x}$:
$$\sum x_i (y_i - (\bar{y} - w_1 \bar{x}) - w_1 x_i) = 0 \implies \sum x_i (y_i - \bar{y}) = w_1 \sum x_i (x_i - \bar{x})$$

> **Closed-Form Formula for Slope ($w_1$) and Intercept ($w_0$):**
> $$w_1 = \frac{\sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^N (x_i - \bar{x})^2} = \frac{N \sum x_i y_i - \sum x_i \sum y_i}{N \sum x_i^2 - (\sum x_i)^2}$$
> $$w_0 = \bar{y} - w_1 \bar{x}$$

### 4. Regression Loss & Evaluation Metrics
1. **Sum of Squared Errors (SSE)**: $\text{SSE} = \sum_{i=1}^N (y_i - \hat{y}_i)^2$
2. **Mean Squared Error (MSE)**: $\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$
3. **Mean Absolute Error (MAE)**: $\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$
4. **Root Mean Squared Error (RMSE)**: $\text{RMSE} = \sqrt{\text{MSE}}$
5. **Coefficient of Determination ($R^2$ Score)**:
   $$R^2 = 1 - \frac{\text{SSE}}{\text{SST}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

---

## 1.10 Model Selection and Generalization

### 1. The Overfitting vs Underfitting Dilemma

```
 Error
   ^
   │       \                         / Validation Error / Test Error
   │        \                       /  (High Variance -> Overfitting)
   │         \   Optimal Model     /
   │          \    Complexity     /
   │           \       │         /
   │            └──────┼────────┘
   │                   │          \
   │                   │           \ Training Error
   └───────────────────┴────────────\───────────────────> Model Complexity
      Underfitting              Overfitting
      (High Bias)              (High Variance)
```

- **Underfitting (High Bias)**: Model is too simple to capture underlying structure (High $E_{train}$, High $E_{val}$).
- **Overfitting (High Variance)**: Model memorizes noise in training data (Low $E_{train}$, High $E_{val}$).

### 2. Mathematical Bias-Variance Decomposition
For target $y = f(\mathbf{x}) + \epsilon$ with noise variance $\mathbb{E}[\epsilon^2] = \sigma^2$:
$$\mathbb{E}\left[(y - \hat{f}(\mathbf{x}))^2\right] = \underbrace{\Big(f(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})]\Big)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\left[\left(\hat{f}(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})]\right)^2\right]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible Noise}}$$

### 3. Model Selection Techniques
1. **Train / Validation / Test Split**: Divide dataset into 70% Training, 15% Validation (hyperparameter tuning), and 15% Testing.
2. **$k$-Fold Cross-Validation**:
   - Randomly split data into $k$ equal-sized folds.
   - Iterate $k$ times: train model on $k-1$ folds and validate on remaining $1$ fold.
   - Compute average cross-validation score: $\text{CV}_{(k)} = \frac{1}{k} \sum_{i=1}^k E_i$.
3. **Regularization**: Adding a penalty term $R(\theta)$ to the loss function to constrain model weight magnitude:
   - **$L_2$ Regularization (Ridge)**: $L_{Ridge} = \text{Loss} + \lambda \sum w_j^2$
   - **$L_1$ Regularization (Lasso)**: $L_{Lasso} = \text{Loss} + \lambda \sum |w_j|$

---

## 1.11 Dimensions of a Supervised Machine Learning Algorithm

Every supervised learning algorithm is defined across 8 core dimensions:

1. **Input Space ($\mathcal{X}$)**: Set of all possible raw input instances.
2. **Feature Space ($\mathcal{F}$)**: $d$-dimensional vector space representation of inputs $\mathbf{x} \in \mathbb{R}^d$.
3. **Output Space ($\mathcal{Y}$)**: Set of valid target predictions (Discrete for classification, continuous for regression).
4. **Hypothesis Space ($\mathcal{H}$)**: Set of all possible candidate functions $h: \mathcal{X} \rightarrow \mathcal{Y}$ considered by the model family.
5. **Training Set ($\mathcal{D}$)**: Instance-label pairs $\mathcal{D} = \{(\mathbf{x}_1, y_1), \dots, (\mathbf{x}_N, y_N)\}$.
6. **Target Function ($f$)**: True, unknown ideal mapping function $y = f(\mathbf{x}) + \epsilon$.
7. **Loss / Error Function ($\mathcal{L}(y, h(\mathbf{x}))$)**: Quantifies discrepancy between target label and model prediction.
8. **Learning Algorithm ($\mathcal{A}$)**: Optimization procedure (e.g., Gradient Descent) searching $\mathcal{H}$ to find optimal hypothesis $h^*$.

```
+─────────────────────────────────────────────────────────────────────────────+
│                       SUPERVISED LEARNING PIPELINE                          │
+─────────────────────────────────────────────────────────────────────────────+

 [ Raw Data Input ] ──> [ Feature Extraction ] ──> Feature Space X in R^d
                                                           │
                                                           ▼
 [ Ground Truth y ] ─────────────────────────────> [ Loss Function L ]
                                                           │
                                                           ▼
 [ Training Data D ] ──> [ Learning Alg A ] ──> Optimal Hypothesis h* in H
                                                           │
                                                           ▼
 [ New Instance x_test ] ─────────────────────────> [ Final Prediction y_hat ]
```


# UNIT 2 — Feature Selection

> **🔥 HIGH PRIORITY EXAM ALERT:**  
> Unit 2 is the primary source for **4-Mark short theory questions and core numericals** (PCA, LDA, Scaling, SFS/SBS). Every topic in this unit includes a dedicated **4-MARK EXAM ANSWER** formatted for maximum marks in university examinations.

---

## 2.1 Concept of Feature

### 1. Definition
A **Feature** (also referred to as an attribute, variable, or predictor) is an individual, measurable property or characteristic of an observed phenomenon used as an input to a Machine Learning model.

### 2. Intuition
Think of a feature as a single piece of evidence used to make a decision. For instance, when evaluating a patient's health, features include blood pressure, heart rate, age, and cholesterol level.

### 3. Technical Explanation
Mathematically, a dataset instance is represented as a $d$-dimensional **Feature Vector**:
$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_d \end{bmatrix} \in \mathbb{R}^d$$
where each $x_j$ represents the value of the $j$-th feature, spanning a $d$-dimensional **Feature Space** $\mathcal{X}$.

### 4. Classification of Features

#### A. By Data Type
1. **Numerical Features**: Quantitative continuous or discrete numeric values.
   - *Continuous*: Height ($175.5 \text{ cm}$), Price ($\$45.50$).
   - *Discrete*: Count of bedrooms ($3$), number of doctor visits ($5$).
2. **Categorical Features**: Qualitative discrete values belonging to fixed categories.
   - *Nominal* (Unordered): Color (Red, Green, Blue), Zip code.
   - *Ordinal* (Ordered): Education Level (Bachelors, Masters, PhD), Customer Rating (Poor, Fair, Excellent).

#### B. By Relevance to Machine Learning (Kohavi & John Definition)
1. **Relevant Features**: Features that carry direct statistical correlation or mutual information with the target label $y$. Removing them degrades predictive accuracy.
2. **Irrelevant Features**: Features that provide zero useful information regarding the target $y$ (e.g., Patient ID number when predicting heart disease risk).
3. **Redundant Features**: Features that provide duplicate information already fully captured by other features (e.g., Height in Inches and Height in Centimeters in the same dataset).

### 5. Exam Points to Remember
- A feature vector $\mathbf{x} \in \mathbb{R}^d$ maps an instance to $d$-dimensional space.
- Redundant features increase computational complexity without adding new information.
- Irrelevant features introduce noise and degrade model accuracy.

---

### 📝 4-MARK EXAM ANSWER: Concept of Feature

**Q: Define Feature, Feature Vector, and explain Relevant, Irrelevant, and Redundant features with examples.**

**Answer:**
1. **Definition**: A **Feature** is an individual measurable property or variable of an observed sample used by an ML algorithm for training and inference.
2. **Feature Vector**: An ordered $d$-dimensional column vector representing all attributes of an instance:
   $$\mathbf{x} = [x_1, x_2, \dots, x_d]^T \in \mathbb{R}^d$$
3. **Types of Features based on Relevance**:
   - **Relevant Feature**: Essential feature strongly correlated with target output.  
     *Example*: House area (sq. ft.) when predicting House Price.
   - **Irrelevant Feature**: Uncorrelated attribute providing zero predictive power.  
     *Example*: House Owner's Name or Roll Number when predicting House Price.
   - **Redundant Feature**: Attribute supplying duplicate information already present in existing features.  
     *Example*: Including both "Age in Years" and "Year of Birth" in the same dataset.

---

## 2.2 Data Preprocessing

### 1. Definition
**Data Preprocessing** is the foundational stage in Machine Learning involving cleaning, transforming, scaling, and structuring raw data to make it suitable for algorithmic model training.

### 2. Why is Preprocessing Required?
Real-world data is inherently **dirty**, characterized by:
- **Incompleteness**: Missing values due to sensor failures or unrecorded entries.
- **Noise**: Corrupted readings or incorrect measurements.
- **Inconsistency**: Discrepancies in naming conventions, formats, or scaling units.
- **Scale Variance**: Features measured on widely different scales (e.g., Salary in $\$100,000$ vs Age in $25$ years) which skew distance metrics.

```
+─────────────────────────────────────────────────────────────────────────────+
│                       DATA PREPROCESSING PIPELINE                           │
+─────────────────────────────────────────────────────────────────────────────+

 [ Raw Dirty Data ] ──> [ Data Cleaning ] (Handle missing values & outliers)
                                │
                                ▼
 [ Feature Encoding ] ──> [ Feature Scaling ] (Normalization / Standardization)
                                │
                                ▼
               [ Clean Preprocessed Data Matrix X ]
```

### 3. Core Preprocessing Steps
1. **Data Cleaning**: Removing noise, handling missing entries, and filtering outliers.
2. **Data Transformation**: Encoding categorical strings into numerical vectors (One-Hot Encoding, Label Encoding).
3. **Feature Scaling**: Rescaling numeric feature ranges via Normalization or Standardization.
4. **Data Reduction**: Reducing feature dimensionality via PCA or LDA.

---

### 📝 4-MARK EXAM ANSWER: Data Preprocessing

**Q: What is Data Preprocessing? Explain why it is essential and list its main steps.**

**Answer:**
1. **Definition**: Data Preprocessing refers to the techniques used to clean, convert, scale, and transform raw data into an accurate, structured format suitable for ML model training.
2. **Why Essential?**: Real-world raw data contains noise, missing attributes, duplicate records, and inconsistent scaling. Feeding raw data directly into algorithms causes poor convergence, biased weights, and incorrect predictions.
3. **Key Steps in Data Preprocessing Pipeline**:
   - **Data Cleaning**: Imputing missing values and removing corrupted records/outliers.
   - **Categorical Encoding**: Converting string labels into numbers (e.g., One-Hot Encoding).
   - **Feature Scaling**: Rescaling continuous numeric features to uniform ranges (Min-Max Normalization / Z-score Standardization).
   - **Dimensionality Reduction**: Eliminating redundant features to avoid the Curse of Dimensionality.

---

## 2.3 Normalization and Scaling

### 1. Definition
**Min-Max Normalization** (or Feature Scaling) is a data preprocessing technique that rescales continuous numeric feature values into a fixed, bounded range (typically $[0, 1]$ or $[-1, 1]$).

### 2. Why Feature Scaling is Required?
Distance-based algorithms (such as $k$-NN, SVM, K-Means) compute Euclidean distances between data points:
$$d(\mathbf{x}_a, \mathbf{x}_b) = \sqrt{(x_{a1} - x_{b1})^2 + (x_{a2} - x_{b2})^2}$$
If Feature 1 ($\text{Income}$) ranges from $\$10,000$ to $\$1,000,000$ while Feature 2 ($\text{Age}$) ranges from $18$ to $65$, the squared difference of Income completely dominates the distance metric, rendering Age practically invisible to the model.

### 3. Mathematical Formulation (Min-Max Normalization)

#### A. Standard $[0, 1]$ Range Formula:
$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
- $x$: Original feature value
- $x_{\min}$: Minimum value of feature in the dataset
- $x_{\max}$: Maximum value of feature in the dataset
- $x'$: Rescaled normalized feature value ($x' \in [0, 1]$)

#### B. Custom Bounded Range $[a, b]$ Formula:
$$x'' = a + \frac{(x - x_{\min})(b - a)}{x_{\max} - x_{\min}}$$

### 4. Advantages & Limitations
- **Advantages**: Preserves exact relative relationships between data points; bounds features strictly within $[0, 1]$; ideal for algorithms not assuming Gaussian distributions.
- **Limitations**: **Highly sensitive to extreme outliers**. If a single outlier $x_{\max} = 10,000$ exists while normal data lies between $1-10$, all normal instances compress into a tiny fraction near $0$.

---

### 📝 4-MARK EXAM ANSWER: Min-Max Normalization

**Q: Explain Min-Max Normalization with formula, example, and limitations.**

**Answer:**
1. **Definition**: Min-Max Normalization rescales a continuous numeric feature $x$ linearly to a fixed bounded range $[0, 1]$.
2. **Formula**:
   $$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
   Where $x_{\min}$ and $x_{\max}$ are minimum and maximum feature values in the dataset.
3. **Example**: Given feature values $\{20, 40, 60, 80, 100\}$, here $x_{\min} = 20, x_{\max} = 100$.  
   For $x = 40$:
   $$x' = \frac{40 - 20}{100 - 20} = \frac{20}{80} = 0.25$$
4. **Limitations**: Extreme outliers skew $x_{\min}$ or $x_{\max}$, compressing non-outlier data into an extremely narrow sub-range.

---

## 2.4 Standardization (Z-Score Normalization)

### 1. Definition
**Standardization** (or Z-score Normalization) transforms feature values such that the resulting distribution has a **mean of 0 ($\mu = 0$)** and a **standard deviation of 1 ($\sigma = 1$)**.

### 2. Mathematical Formulation
$$z = \frac{x - \mu}{\sigma}$$
Where:
- $x$: Original feature value
- $\mu$: Sample mean of the feature: $\mu = \frac{1}{N} \sum_{i=1}^N x_i$
- $\sigma$: Sample standard deviation: $\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2}$
- $z$: Standardized Z-score ($z \in (-\infty, +\infty)$)

### 3. Key Properties & Intuition
- Z-score indicates how many standard deviations a data point $x$ lies above or below the sample mean $\mu$.
- If $z = 0$, $x$ is exactly equal to the mean $\mu$.
- If $z = +2.0$, $x$ lies $2$ standard deviations above the mean.

### 4. When to Use Standardization?
- Algorithms assuming underlying normal (Gaussian) distributions (e.g., Linear Regression, Logistic Regression, Linear Discriminant Analysis).
- Algorithms relying on gradient descent optimization (faster convergence when features are centered around 0).
- Principal Component Analysis (PCA) where variance maximization requires mean-centered features.

---

### 📝 4-MARK EXAM ANSWER: Standardization (Z-Score)

**Q: Explain Standardization (Z-score Normalization). State its formula, step-by-step procedure, and advantages.**

**Answer:**
1. **Definition**: Standardization transforms a numeric feature distribution to have zero mean ($\mu = 0$) and unit variance ($\sigma^2 = 1$).
2. **Formula**:
   $$z = \frac{x - \mu}{\sigma} \quad \text{where } \mu = \frac{\sum x_i}{N}, \, \sigma = \sqrt{\frac{\sum (x_i - \mu)^2}{N}}$$
3. **Step-by-Step Procedure**:
   - Step 1: Calculate sample mean $\mu$ of feature column.
   - Step 2: Compute variance $\sigma^2$ and standard deviation $\sigma$.
   - Step 3: Subtract mean $\mu$ from each feature value $x_i$ and divide by $\sigma$.
4. **Advantages**: Outliers do not bound data into a artificial range; essential for PCA and gradient-descent algorithms.

---

## 2.5 Normalization vs Standardization

---

### 📝 4-MARK EXAM ANSWER: Normalization vs Standardization

**Q: Differentiate between Min-Max Normalization and Standardization in tabular form.**

**Answer:**

| Parameter | Min-Max Normalization | Standardization (Z-Score) |
|---|---|---|
| **Primary Goal** | Rescales feature values to a bounded interval $[0, 1]$ | Rescales distribution to zero mean ($\mu=0$) and unit variance ($\sigma=1$) |
| **Formula** | $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ | $z = \frac{x - \mu}{\sigma}$ |
| **Output Range** | Strictly bounded $[0, 1]$ (or $[a, b]$) | Unbounded $(-\infty, +\infty)$ (typically $[-3, +3]$) |
| **Outlier Sensitivity** | **High** (Outliers distort $x_{\min}$ and $x_{\max}$) | **Low** (Outliers do not compress non-outlier data) |
| **Distribution Assumption**| Preserves original distribution (No assumption)| Works best when data follows Gaussian distribution |
| **Best Used For** | Distance algorithms ($k$-NN, Neural Nets with bounded inputs) | PCA, LDA, Gradient Descent, Linear/Logistic Regression |

---

## 2.6 Managing Missing Values

### 1. Why Missing Values Occur?
- Sensor defects or network packet losses during data transmission.
- Optional survey questions left blank by respondents.
- Data corruption during database joining/merging.

### 2. Impact of Missing Data
ML algorithms like SVM, Matrix Inversion, and Neural Networks crash or fail when encountering `NaN` / null values because mathematical operators (addition, matrix multiplication) are undefined for missing entries.

### 3. Strategies for Managing Missing Values

#### A. Deletion Techniques
1. **Listwise Deletion (Row Removal)**: Drop any dataset row containing one or more missing values.  
   - *Pros*: Simple.  
   - *Cons*: Discards valuable training samples; leads to severe data loss if missingness is high.
2. **Feature Deletion (Column Removal)**: Drop an entire feature column if $>50\%$ of its entries are missing.

#### B. Statistical Imputation Techniques
1. **Mean Imputation**: Replace missing numerical values with the mean ($\mu$) of known values in that column.  
   - *Best for*: Normally distributed numeric data without outliers.
2. **Median Imputation**: Replace missing numerical values with the median of known column values.  
   - *Best for*: Skewed numeric data containing extreme outliers.
3. **Mode Imputation**: Replace missing categorical entries with the most frequently occurring category.  
   - *Best for*: Categorical / nominal variables.

```
Missing Value Treatment Decision Tree:

                 ┌───────────────────────────┐
                 │ Are Missing Values > 50%? │
                 └─────────────┬─────────────┘
                               │
                ┌──────────────┴──────────────┐
             YES│                             │NO
                ▼                             ▼
   ┌───────────────────────┐    ┌───────────────────────────┐
   │ Drop Column / Feature │    │ Is Data Numeric or Cat.?  │
   └───────────────────────┘    └─────────────┬─────────────┘
                                              │
                       ┌──────────────────────┴──────────────────────┐
                Numeric│                                             │Categorical
                       ▼                                             ▼
        ┌──────────────────────────────┐              ┌───────────────────────────┐
        │ Contains Extreme Outliers?   │              │ Replace with Mode (Most   │
        └──────────────┬───────────────┘              │ Frequent Category)        │
                       │                              └───────────────────────────┘
         ┌─────────────┴─────────────┐
      YES│                           │NO
         ▼                           ▼
 ┌───────────────┐           ┌───────────────┐
 │ Impute Median │           │  Impute Mean  │
 └───────────────┘           └───────────────┘
```

---

### 📝 4-MARK EXAM ANSWER: Managing Missing Values

**Q: Explain different techniques for handling missing values in a dataset.**

**Answer:**
1. **Definition**: Missing value treatment involves identifying unrecorded `NaN` entries and applying deletion or statistical imputation to preserve data integrity.
2. **Main Handling Techniques**:
   - **Row Deletion (Listwise)**: Delete rows containing missing values. Suitable only when missing data is $<5\%$.
   - **Mean Imputation**: Replace missing numeric values with column mean $\mu$. Preferred for symmetric numeric data without outliers.
   - **Median Imputation**: Replace missing values with column median. Preferred for skewed numerical data with heavy outliers.
   - **Mode Imputation**: Replace missing categorical data with the most frequent category.
3. **Summary Rule**: Deletion discards data; Mean/Median imputation retains data size without biasing model dimensions.

---

## 2.7 Introduction to Dimensionality Reduction

### 1. Definition
**Dimensionality Reduction** is the process of reducing the number of input random variables (features) in a dataset by obtaining a lower-dimensional set of principal features.

### 2. The Curse of Dimensionality
As feature dimensionality $d$ increases:
- The volume of feature space grows exponentially ($V \propto r^d$), causing data points to become extremely sparse.
- Euclidean distances between arbitrary pairs of points converge to the same value ($d_{\max} \approx d_{\min}$), making distance-based clustering and classification ineffective.
- Overfitting risk increases drastically as the number of parameters outgrows available training samples.

```
Visualizing Data Sparsity & Curse of Dimensionality:

1D Space (Line):           2D Space (Square):              3D Space (Cube):
Points dense [• • • •]     Points moderately spread        Points extremely sparse
Density = N / L            Density = N / L^2               Density = N / L^3
```

### 3. Dimensionality Reduction vs Feature Selection vs Feature Extraction

---

### 📝 4-MARK EXAM ANSWER: Dimensionality Reduction vs Feature Selection vs Feature Extraction

**Q: Explain the Curse of Dimensionality. Compare Feature Selection and Feature Extraction in tabular format.**

**Answer:**
1. **Curse of Dimensionality**: Phenomenon where increasing feature dimensions $d$ causes data sparsity, exponential volume growth, and distance metric breakdown, leading to severe overfitting.
2. **Comparison Table**:

| Parameter | Feature Selection | Feature Extraction |
|---|---|---|
| **Mechanism** | Selects a subset of $k$ original features without modifying them. | Transforms original $d$ features into $k$ completely new features. |
| **Interpretability**| High (Original feature names and units are preserved). | Low (Transformed features like principal components lack physical meaning). |
| **Information Loss**| May discard useful information in unselected features. | Retains combined variance/signal across all original features. |
| **Key Algorithms** | Sequential Forward Selection (SFS), Sequential Backward Selection (SBS). | Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA). |


## 2.8 Principal Component Analysis (PCA)

> **🔥 HIGH PRIORITY TOPIC**: Principal Component Analysis (PCA) is an extremely important numerical and theory topic in university examinations.

### 1. What is PCA?
**Principal Component Analysis (PCA)** is an unsupervised, linear feature extraction technique that transforms a set of $d$ possibly correlated features into a smaller set of $k$ ($k < d$) linearly uncorrelated variables called **Principal Components**, while preserving the maximum possible variance of the dataset.

### 2. Intuition & Geometry
Imagine a 2D cloud of data points oriented diagonally. PCA finds a new orthogonal coordinate system:
- **1st Principal Component ($PC_1$)**: The axis along which data points exhibit the maximum variance (spread).
- **2nd Principal Component ($PC_2$)**: The axis perpendicular (orthogonal) to $PC_1$ capturing the remaining variance.

```
       x2 ^             PC1 (Direction of Max Variance)
          │            /
          │          •/  •
          │        • / •
          │      •  /•
          │    •  / •
          │  •  / •       PC2 (Orthogonal to PC1)
          │ • / •        /
          └─────────────┼───────────> x1
                       /
```

### 3. Key Mathematical Definitions
1. **Variance ($\sigma_x^2$)**: Measures the spread of a single feature $x$:
   $$\sigma_x^2 = \frac{1}{N-1} \sum_{i=1}^N (x_i - \bar{x})^2$$
2. **Covariance ($\text{Cov}(x, y)$)**: Measures joint variability between two features $x$ and $y$:
   $$\text{Cov}(x, y) = \frac{1}{N-1} \sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})$$
   - If $\text{Cov}(x, y) > 0$: Features increase together.
   - If $\text{Cov}(x, y) = 0$: Features are linearly independent.
3. **Covariance Matrix ($\Sigma \in \mathbb{R}^{d \times d}$)**: For mean-centered data matrix $X_{centered}$:
   $$\Sigma = \frac{1}{N-1} X_{centered}^T X_{centered} = \begin{bmatrix} \text{Var}(x_1) & \text{Cov}(x_1, x_2) \\ \text{Cov}(x_2, x_1) & \text{Var}(x_2) \end{bmatrix}$$
4. **Eigenvalues ($\lambda$) and Eigenvectors ($\mathbf{v}$)**: Satisfy the characteristic matrix equation:
   $$\Sigma \mathbf{v} = \lambda \mathbf{v} \implies (\Sigma - \lambda I) \mathbf{v} = 0$$
   - **Eigenvector ($\mathbf{v}$)**: Gives the direction of a Principal Component.
   - **Eigenvalue ($\lambda$)**: Quantifies the amount of variance captured along that direction.
5. **Explained Variance Ratio ($EVR_k$)**: Proportion of total dataset variance explained by the $k$-th principal component:
   $$\text{EVR}_k = \frac{\lambda_k}{\sum_{j=1}^d \lambda_j}$$

### 4. Complete Step-by-Step PCA Algorithm Pipeline

```
+─────────────────────────────────────────────────────────────────────────────+
│                           PCA ALGORITHM PIPELINE                            │
+─────────────────────────────────────────────────────────────────────────────+

 [ Step 1: Input Data X (N x d) ] ──> [ Step 2: Mean Centering: X_c = X - mu ]
                                                   │
                                                   ▼
 [ Step 4: Solve det(Sigma - lambda I)=0 ] <── [ Step 3: Compute Covariance Matrix Sigma ]
                   │
                   ▼
 [ Step 5: Sort Eigenvalues & Select Top k Eigenvectors ] ──> Transformation Matrix W (d x k)
                                                                       │
                                                                       ▼
                                          [ Step 6: Project Data: Y = X_c * W (N x k) ]
```

1. **Step 1: Mean Centering**: Compute column means $\mu_j$ and subtract from data matrix: $X_{centered} = X - \mu$.
2. **Step 2: Compute Covariance Matrix**: $\Sigma = \frac{1}{N-1} X_{centered}^T X_{centered}$.
3. **Step 3: Solve Eigenvalues & Eigenvectors**: Solve characteristic determinant $\det(\Sigma - \lambda I) = 0$ to find $\lambda_1, \lambda_2, \dots, \lambda_d$ and corresponding eigenvectors $\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_d$.
4. **Step 4: Select Top $k$ Components**: Sort eigenvalues in descending order ($\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d$) and select top $k$ eigenvectors to construct projection matrix $W = [\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k] \in \mathbb{R}^{d \times k}$.
5. **Step 5: Project Data**: Transform original dataset into $k$-dimensional subspace:
   $$Y = X_{centered} W \in \mathbb{R}^{N \times k}$$

---

### 📝 4-MARK EXAM ANSWER: Principal Component Analysis (PCA) Steps

**Q: What is PCA? Explain the step-by-step algorithm of PCA.**

**Answer:**
1. **Definition**: Principal Component Analysis (PCA) is an unsupervised linear feature extraction technique that projects $d$-dimensional data into a $k$-dimensional subspace ($k < d$) along directions of maximum variance.
2. **Steps of PCA Algorithm**:
   - **Step 1 (Mean Centering)**: Calculate mean vector $\mu$ and center dataset: $X_c = X - \mu$.
   - **Step 2 (Covariance Matrix)**: Compute $d \times d$ covariance matrix $\Sigma = \frac{1}{N-1} X_c^T X_c$.
   - **Step 3 (Eigendecomposition)**: Solve $(\Sigma - \lambda I)\mathbf{v} = 0$ to compute eigenvalues $\lambda_i$ and eigenvectors $\mathbf{v}_i$.
   - **Step 4 (Projection Matrix)**: Rank eigenvalues descendingly and choose top $k$ eigenvectors to build transformation matrix $W = [\mathbf{v}_1, \dots, \mathbf{v}_k]$.
   - **Step 5 (Data Projection)**: Calculate lower-dimensional projected dataset: $Y = X_c W$.

---

## 2.9 Kernel PCA

### 1. Motivation: Failure of Linear PCA on Non-Linear Data
Linear PCA projects data onto straight planes. If data patterns lie on non-linear manifolds (e.g., concentric circles or Swiss roll), standard linear PCA cannot separate the patterns.

```
Linear PCA Failure on Concentric Circles:
  ( • • ( ┼ ┼ ) • • )  <-- Linear axis projection merges inner (+) and outer (•) points!

Kernel PCA Solution:
  Maps 2D non-linear points into 3D space Z where classes become linearly separable!
```

### 2. The Kernel Trick
Kernel PCA implicitly maps original low-dimensional input vectors $\mathbf{x} \in \mathbb{R}^d$ into a high-dimensional feature space $\Phi(\mathbf{x}) \in \mathcal{H}$ using a non-linear kernel function $K(\mathbf{x}_i, \mathbf{x}_j)$:
$$K(\mathbf{x}_i, \mathbf{x}_j) = \langle \Phi(\mathbf{x}_i), \Phi(\mathbf{x}_j) \rangle$$
This computes dot products in feature space without explicitly evaluating high-dimensional mapping $\Phi(\mathbf{x})$.

### 3. Popular Kernel Functions
1. **Polynomial Kernel**: $K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T \mathbf{x}_j + c)^p$
2. **Radial Basis Function (RBF / Gaussian) Kernel**:
   $$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$
3. **Sigmoid Kernel**: $K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\alpha \mathbf{x}_i^T \mathbf{x}_j + c)$

### 4. Basic Kernel PCA Procedure
1. Construct $N \times N$ Kernel Matrix $K$ where $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$.
2. Center the Kernel Matrix: $K' = K - \mathbf{1}_N K - K \mathbf{1}_N + \mathbf{1}_N K \mathbf{1}_N$ (where $\mathbf{1}_N$ is an $N \times N$ matrix with values $\frac{1}{N}$).
3. Solve $N \times N$ eigenvalue equation $K' \mathbf{a}_k = \lambda_k \mathbf{a}_k$.
4. Normalize eigenvectors $\mathbf{a}_k$ and compute projections.

---

### 📝 4-MARK EXAM ANSWER: Kernel PCA vs Linear PCA

**Q: Explain Kernel PCA and state how it differs from standard PCA.**

**Answer:**
1. **Definition**: Kernel PCA is a non-linear extension of PCA that uses kernel functions to project data into a higher-dimensional feature space $\Phi(\mathbf{x})$ where linear PCA is performed.
2. **How it Works**: Uses the **Kernel Trick** $K(\mathbf{x}_i, \mathbf{x}_j) = \langle \Phi(\mathbf{x}_i), \Phi(\mathbf{x}_j) \rangle$ to compute scalar inner products in high-dimensional space without explicit feature expansion.
3. **Comparison Table**:

| Parameter | Linear PCA | Kernel PCA |
|---|---|---|
| **Separability** | Finds linear hyperplanes | Finds non-linear complex decision boundaries |
| **Computation** | Eigendecomposition of $d \times d$ covariance matrix | Eigendecomposition of $N \times N$ Kernel matrix |
| **Kernel Function**| None (Uses raw linear dot products) | Employs RBF, Polynomial, or Sigmoid kernels |
| **Complexity** | $O(d^3)$ — Dependent on feature count $d$ | $O(N^3)$ — Dependent on dataset sample size $N$ |

---

## 2.10 Linear Discriminant Analysis (LDA)

> **🔥 HIGH PRIORITY TOPIC**: Linear Discriminant Analysis (LDA) is a major supervised dimensionality reduction and classification method frequently featured in exam numericals.

### 1. What is LDA?
**Linear Discriminant Analysis (LDA)** (or Fisher's Linear Discriminant) is a **supervised** linear dimensionality reduction algorithm that projects data onto a lower-dimensional space while **maximizing class separability**.

### 2. Core Intuition: Fisher's Objective
LDA finds a projection direction $\mathbf{w}$ that simultaneously achieves two goals:
1. **Maximize distance between class means** (Maximize Between-Class Variance).
2. **Minimize variance within each individual class** (Minimize Within-Class Variance).

```
Unfavorable Projection (PCA):          Optimal Fisher Projection (LDA):
  Overlapping class projections          Separated class projections
       Class 1  Class 2                       Class 1       Class 2
     ( • • • •  + + + + )                   ( • • • • )   ( + + + + )
   Means close, high spread                 Means far, low spread within
```

### 3. Mathematical Formulation

#### A. Class Means
- Mean vector of class $c$: $\mathbf{\mu}_c = \frac{1}{N_c} \sum_{\mathbf{x} \in \mathcal{D}_c} \mathbf{x}$
- Overall mean vector: $\mathbf{\mu} = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i$

#### B. Within-Class Scatter Matrix ($S_W \in \mathbb{R}^{d \times d}$)
Measures variance spread of samples within their respective classes:
$$S_W = S_1 + S_2 = \sum_{\mathbf{x} \in \mathcal{D}_1} (\mathbf{x} - \mathbf{\mu}_1)(\mathbf{x} - \mathbf{\mu}_1)^T + \sum_{\mathbf{x} \in \mathcal{D}_2} (\mathbf{x} - \mathbf{\mu}_2)(\mathbf{x} - \mathbf{\mu}_2)^T$$

#### C. Between-Class Scatter Matrix ($S_B \in \mathbb{R}^{d \times d}$)
Measures separation distance between different class means:
$$S_B = (\mathbf{\mu}_1 - \mathbf{\mu}_2)(\mathbf{\mu}_1 - \mathbf{\mu}_2)^T \quad (\text{for 2 classes})$$
For multi-class ($C$ classes): $S_B = \sum_{c=1}^C N_c (\mathbf{\mu}_c - \mathbf{\mu})(\mathbf{\mu}_c - \mathbf{\mu})^T$

#### D. Fisher's Criterion Objective Function ($J(\mathbf{w})$)
$$J(\mathbf{w}) = \frac{\mathbf{w}^T S_B \mathbf{w}}{\mathbf{w}^T S_W \mathbf{w}}$$

#### E. Optimal Projection Vector ($\mathbf{w}^*$)
Maximizing $J(\mathbf{w})$ leads to the generalized eigenvalue problem:
$$S_W^{-1} S_B \mathbf{w} = \lambda \mathbf{w}$$
For a 2-class problem, the closed-form optimal direction vector is simply:
$$\mathbf{w}^* = S_W^{-1} (\mathbf{\mu}_1 - \mathbf{\mu}_2)$$

#### F. Dimension Constraint
Maximum output dimensions reduced by LDA is bounded by:
$$k \le C - 1 \quad (\text{where } C \text{ is the number of target classes})$$
*(e.g., For a binary classification problem $C=2$, LDA can project data to at most $C-1 = 1$ dimension).*

---

### 📝 4-MARK EXAM ANSWER: Linear Discriminant Analysis (LDA)

**Q: Explain Linear Discriminant Analysis (LDA). State Fisher's Criterion and formulas for scatter matrices.**

**Answer:**
1. **Definition**: LDA is a supervised linear dimensionality reduction algorithm that projects data onto lower dimensions while maximizing separation between target classes.
2. **Fisher's Criterion**: Maximizes the ratio of between-class scatter to within-class scatter:
   $$J(\mathbf{w}) = \frac{\mathbf{w}^T S_B \mathbf{w}}{\mathbf{w}^T S_W \mathbf{w}}$$
3. **Scatter Matrices Formulas**:
   - **Within-Class Scatter Matrix ($S_W$)**: $S_W = \sum_{c=1}^C \sum_{\mathbf{x} \in \mathcal{D}_c} (\mathbf{x} - \mathbf{\mu}_c)(\mathbf{x} - \mathbf{\mu}_c)^T$
   - **Between-Class Scatter Matrix ($S_B$)**: $S_B = \sum_{c=1}^C N_c (\mathbf{\mu}_c - \mathbf{\mu})(\mathbf{\mu}_c - \mathbf{\mu})^T$
4. **Optimal Direction Vector**: For 2 classes, $\mathbf{w}^* = S_W^{-1}(\mathbf{\mu}_1 - \mathbf{\mu}_2)$. Max output dimension is $C-1$.

---

## 2.11 PCA vs LDA

---

### 📝 4-MARK EXAM ANSWER: PCA vs LDA

**Q: Compare PCA and LDA in tabular form.**

**Answer:**

| Parameter | Principal Component Analysis (PCA) | Linear Discriminant Analysis (LDA) |
|---|---|---|
| **Learning Paradigm** | **Unsupervised** (Ignores class labels) | **Supervised** (Requires target class labels $y$) |
| **Primary Objective** | Maximize total data variance / minimize reconstruction error | Maximize class separability (Fisher's Criterion) |
| **Scatter / Variance**| Computes overall Covariance Matrix $\Sigma$ | Computes Within-Class ($S_W$) and Between-Class ($S_B$) matrices |
| **Max Components** | Up to $d$ components ($k \le d$) | Bound by number of classes: $k \le C - 1$ |
| **Data Requirements**| Does not require categorical labels | Requires discrete categorical class labels |
| **Outlier Behavior** | Sensitive to extreme feature outliers | Sensitive to class-mean outliers and class imbalance |
| **Best Application** | Unsupervised compression, visualization | Supervised pre-classification feature reduction |

---

## 2.12 Feature Selection Techniques

### 1. Definition
**Feature Selection** is the process of selecting a subset of $k$ relevant features from total $d$ original features ($k < d$) **without altering or transforming original feature definitions**.

### 2. Three Main Categories of Feature Selection

#### A. Filter Methods
- Evaluate features based on intrinsic statistical properties (correlation, chi-square, ANOVA, mutual information) independent of any ML model.
- *Pros*: Extremely fast, highly scalable.
- *Cons*: Ignores feature interactions and model specific biases.

#### B. Wrapper Methods
- Use a specific ML predictive model as an evaluator to score subsets of features based on validation accuracy.
- Examples: Sequential Forward Selection (SFS), Sequential Backward Selection (SBS).
- *Pros*: High predictive accuracy; captures feature interaction.
- *Cons*: Computationally expensive ($O(d^2)$ iterations).

#### C. Embedded Methods
- Feature selection occurs naturally during model training (e.g., $L_1$ Lasso Regularization penalizing weights to 0, Decision Trees selecting root split features).

```
Feature Selection Paradigms:

Filter Method:    [ All Features ] ──> [ Statistical Test ] ──> [ Top k Features ] ──> [ ML Model ]

Wrapper Method:   [ Feature Subset ] ──> [ Train Model ] ──> [ Evaluate Acc ] ──> [ Loop Update ]

Embedded Method:  [ All Features ] ──> [ Model Training with L1 / Trees ] ──> [ Selected Model ]
```

---

## 2.13 Sequential Forward Selection (SFS)

### 1. Definition & Strategy
**Sequential Forward Selection (SFS)** is a bottom-up greedy wrapper feature selection algorithm that starts with an empty feature set $\mathcal{Y}_0 = \emptyset$ and iteratively adds one feature at a time that maximizes model performance.

### 2. Step-by-Step SFS Algorithm & Pseudocode

```python
# Pseudocode for Sequential Forward Selection (SFS)
Input: Full feature set X = {f_1, f_2, ..., f_d}, Target size k
Output: Selected feature subset Y

1. Initialize Y = {} (empty set)
2. For step = 1 to k:
     best_feature = None
     best_score = -infinity
     For each feature f in (X - Y):
         candidate_set = Y U {f}
         score = Evaluate_Model(candidate_set) # Cross-validation score
         If score > best_score:
             best_score = score
             best_feature = f
     Y = Y U {best_feature}
3. Return Y
```

### 3. Worked Iterative Example
Consider original features $X = \{F_1, F_2, F_3, F_4\}$ targeting $k=2$ features:
- **Iteration 1**: Evaluate singletons:
  - $\{F_1\} \rightarrow 72\%$, $\{F_2\} \rightarrow 78\%$, $\{F_3\} \rightarrow 65\%$, $\{F_4\} \rightarrow 70\%$
  - Best single feature: **Select $F_2$**. Current subset $\mathcal{Y}_1 = \{F_2\}$.
- **Iteration 2**: Evaluate pairs including $F_2$:
  - $\{F_2, F_1\} \rightarrow 85\%$, $\{F_2, F_3\} \rightarrow 80\%$, $\{F_2, F_4\} \rightarrow 82\%$
  - Best feature pair: **Select $F_1$**. Final subset $\mathcal{Y}_2 = \{F_2, F_1\}$.

### 4. Advantages & Limitations
- **Advantages**: Simple; computationally cheaper than exhaustive search ($O(d^2)$ vs $O(2^d)$).
- **Limitations**: **Nesting Effect**: Once a feature is added in an early iteration, it can never be removed later even if it becomes redundant.

---

### 📝 4-MARK EXAM ANSWER: Sequential Forward Selection (SFS)

**Q: Explain Sequential Forward Selection (SFS) with its algorithm, example, and limitations.**

**Answer:**
1. **Definition**: SFS is a bottom-up greedy wrapper feature selection algorithm that starts with an empty set and sequentially appends the feature providing the maximum performance increase.
2. **Algorithm Steps**:
   - Start with $\mathcal{Y}_0 = \emptyset$.
   - Evaluate model accuracy for each candidate feature $f \in X \setminus \mathcal{Y}_{k-1}$.
   - Add feature $f^+$ maximizing validation score: $\mathcal{Y}_k = \mathcal{Y}_{k-1} \cup \{f^+\}$.
   - Repeat until $k$ features are selected.
3. **Limitations**: Suffers from the **nesting effect** (cannot drop a feature once selected).

---

## 2.14 Sequential Backward Selection (SBS)

### 1. Definition & Strategy
**Sequential Backward Selection (SBS)** is a top-down greedy wrapper feature selection algorithm that starts with the complete set of features $\mathcal{Y}_0 = X$ and iteratively removes one feature at a time whose removal causes the least performance loss.

### 2. Step-by-Step SBS Algorithm & Pseudocode

```python
# Pseudocode for Sequential Backward Selection (SBS)
Input: Full feature set X = {f_1, f_2, ..., f_d}, Target size k
Output: Selected feature subset Y

1. Initialize Y = X (all d features)
2. While size(Y) > k:
     worst_feature = None
     best_score = -infinity
     For each feature f in Y:
         candidate_set = Y - {f}
         score = Evaluate_Model(candidate_set)
         If score > best_score:
             best_score = score
             worst_feature = f
     Y = Y - {worst_feature}
3. Return Y
```

### 3. Advantages & Limitations
- **Advantages**: Evaluates feature contributions in the presence of all other remaining features (captures joint interactions better than SFS).
- **Limitations**: Computationally expensive for large $d$ since initial iterations train models on near-full feature sets; cannot re-add dropped features.

---

### 📝 4-MARK EXAM ANSWER: Sequential Backward Selection (SBS)

**Q: Explain Sequential Backward Selection (SBS) algorithm and contrast it with SFS.**

**Answer:**
1. **Definition**: SBS is a top-down greedy wrapper algorithm starting with all $d$ features and sequentially eliminating the feature whose removal results in minimum accuracy drop.
2. **Workflow**: Starts at $\mathcal{Y}_0 = X$; at step $k$, tests removal of each $f \in \mathcal{Y}$, dropping $f^-$ to set $\mathcal{Y}_k = \mathcal{Y}_{k-1} \setminus \{f^-\}$ until target size is reached.
3. **Contrast with SFS**: SFS builds up from $\emptyset$ (bottom-up); SBS prunes down from full set $X$ (top-down).

---

## 2.15 SFS vs SBS

---

### 📝 4-MARK EXAM ANSWER: SFS vs SBS Comparison

**Q: Compare Sequential Forward Selection (SFS) and Sequential Backward Selection (SBS) in tabular form.**

**Answer:**

| Parameter | Sequential Forward Selection (SFS) | Sequential Backward Selection (SBS) |
|---|---|---|
| **Starting State** | Empty feature set $\mathcal{Y}_0 = \emptyset$ | Full feature set $\mathcal{Y}_0 = X$ |
| **Search Direction** | Bottom-up (Adding features) | Top-down (Removing features) |
| **Operation per Step**| Appends 1 best performing feature | Drops 1 least contributing feature |
| **Initial Computational Cost**| Low (Starts with 1-feature models) | High (Starts with full $d$-feature models)|
| **Feature Interaction**| Misses interactions until features added| Captures multi-feature interactions early |
| **Nesting Flaw** | Cannot delete an added feature | Cannot re-add a deleted feature |
| **Best Suited For** | Datasets with large feature count $d$ | Datasets with moderate $d$ where $k \approx d$ |


# IMPORTANT NUMERICAL QUESTION BANK

---

## Unit 1 Solved Numericals

### 1. VC Dimension Numericals

#### 📌 Problem 1.1: VC Dimension of 1D Closed Intervals
**Question**: Determine the VC Dimension of the hypothesis space $\mathcal{H}$ consisting of closed intervals $[a, b]$ on the real line $\mathbb{R}^1$. Show step-by-step shattering proof.

**Given**:
- Instance space $\mathcal{X} = \mathbb{R}^1$
- Hypothesis space $\mathcal{H} = \{ h_{[a,b]}(x) = 1 \text{ if } a \le x \le b \text{ else } 0 \}$

**Concept / Formula**:
VC Dimension $VC(\mathcal{H})$ is the maximum number of points $d$ such that *there exists* a set of $d$ points that can be shattered (all $2^d$ labelings realized).

**Step-by-Step Solution**:
1. **Test $N=2$ points ($x_1 = 1, x_2 = 3$)**:
   - $2^2 = 4$ labelings: $(0,0), (1,0), (0,1), (1,1)$.
   - All 4 labelings are realizable using intervals $[a, b]$ (e.g., $(1,0)$ achieved by $[0.5, 1.5]$). Thus, 2 points can be shattered.
2. **Test $N=3$ points ($x_1 = 1, x_2 = 2, x_3 = 3$)**:
   - $2^3 = 8$ labelings.
   - Consider the labeling $(+1, -1, +1)$ where $x_1$ and $x_3$ are positive ($+1$) and $x_2$ is negative ($-1$).
   - To make $x_1 \in [a, b]$ and $x_3 \in [a, b]$, the interval must span at least from $a \le 1$ to $b \ge 3$.
   - But if $a \le 1$ and $b \ge 3$, then $x_2 = 2$ MUST also fall inside $[a, b]$, giving $x_2$ label $+1$.
   - It is impossible for any interval $[a, b]$ to label $x_1=+1, x_2=-1, x_3=+1$.
3. **Conclusion**: $\mathcal{H}$ cannot shatter any set of 3 points.

**Final Answer**: $VC(\mathcal{H}) = 2$.  
**Exam Tip**: Remember that if even a single labeling cannot be achieved for ANY arrangement of $N$ points, $VC(\mathcal{H}) < N$.

---

#### 📌 Problem 1.2: VC Dimension of 2D Linear Classifiers
**Question**: Prove that the VC Dimension of linear classifiers (lines $w_0 + w_1 x_1 + w_2 x_2 = 0$) in $\mathbb{R}^2$ is $3$.

**Given**: Hypothesis space $\mathcal{H} = \{ \text{sgn}(w_0 + w_1 x_1 + w_2 x_2) \}$.

**Step-by-Step Solution**:
1. **Can 3 points be shattered?**  
   Choose 3 non-collinear points arranged in a triangle. All $2^3 = 8$ binary label combinations can be separated by a straight line. Thus, $VC \ge 3$.
2. **Can 4 points be shattered?**  
   Take any 4 points in $\mathbb{R}^2$. By Radon's Theorem, 4 points can be split into two disjoint sets whose convex hulls intersect (e.g., XOR configuration). The labeling where one set is positive and the other is negative CANNOT be separated by any straight line.
3. **General Formula**: For linear hyperplanes in $\mathbb{R}^d$, $VC(\mathcal{H}) = d + 1$. Here $d=2 \implies VC = 3$.

**Final Answer**: $VC(\mathcal{H}) = 3$.

---

#### 📌 Problem 1.3: VC Dimension of 2D Axis-Aligned Rectangles
**Question**: Calculate the VC Dimension of axis-aligned rectangles in 2D space $\mathbb{R}^2$.

**Step-by-Step Solution**:
1. **Test $N=4$ points**: Place 4 points in a diamond configuration $(1,0), (-1,0), (0,1), (0,-1)$. An axis-aligned rectangle can shatter all $2^4 = 16$ labelings. Thus, $VC \ge 4$.
2. **Test $N=5$ points**: For any 5 points, identify the extreme points (leftmost, rightmost, topmost, bottommost). These define at most 4 boundary points. The 5th internal point cannot be labeled negative while all 4 extreme points are labeled positive.
3. Therefore, 5 points cannot be shattered.

**Final Answer**: $VC(\mathcal{H}) = 4$.

---

### 2. PAC Learning Solved Numericals

#### 📌 Problem 1.4: Sample Complexity for Finite Hypothesis Space
**Question**: A machine learning student wants to train a text classifier over a finite hypothesis space $|\mathcal{H}| = 100,000$. If the acceptable generalization error is $\epsilon = 0.05$ (5%) and the confidence requirement is $99\%$ ($\delta = 0.01$), calculate the minimum number of training examples $m$ required under the PAC framework.

**Given**:
- Hypothesis space size $|\mathcal{H}| = 100,000 = 10^5$
- Accuracy parameter $\epsilon = 0.05$
- Failure parameter $\delta = 1 - 0.99 = 0.01$

**Formula**:
$$m \ge \frac{1}{\epsilon} \left( \ln |\mathcal{H}| + \ln \left(\frac{1}{\delta}\right) \right)$$

**Step-by-Step Solution**:
1. Compute $\ln |\mathcal{H}|$:
   $$\ln(100,000) = 11.5129$$
2. Compute $\ln(1/\delta)$:
   $$\ln(1 / 0.01) = \ln(100) = 4.6052$$
3. Sum the logarithmic terms:
   $$11.5129 + 4.6052 = 16.1181$$
4. Divide by $\epsilon = 0.05$:
   $$m \ge \frac{16.1181}{0.05} = 322.362$$

**Final Answer**: The minimum training sample size required is $m = 323$ examples.  
**Exam Tip**: Always round UP to the nearest integer for sample complexity $m$.

---

#### 📌 Problem 1.5: PAC Bound for Boolean Conjunctions
**Question**: Consider learning Boolean conjunctions over $n = 10$ Boolean features. Find the PAC sample complexity $m$ for $\epsilon = 0.1$ and $\delta = 0.05$.

**Given**:
- Number of Boolean variables $n = 10$
- Accuracy $\epsilon = 0.1$, Confidence parameter $\delta = 0.05$

**Step-by-Step Solution**:
1. Each variable $x_i$ can appear in a conjunction as $x_i$, $\neg x_i$, or be absent.
2. Total hypotheses $|\mathcal{H}| = 3^n = 3^{10} = 59,049$.
3. Compute $\ln |\mathcal{H}| = 10 \times \ln(3) = 10 \times 1.0986 = 10.9861$.
4. Compute $\ln(1/\delta) = \ln(1/0.05) = \ln(20) = 2.9957$.
5. Calculate $m$:
   $$m \ge \frac{1}{0.1} \Big( 10.9861 + 2.9957 \Big) = 10 \times 13.9818 = 139.818$$

**Final Answer**: $m = 140$ samples.

---

### 3. Regression Solved Numericals

#### 📌 Problem 1.6: Complete Simple Linear Regression Fit
**Question**: Given the following dataset of Study Hours ($x$) and Exam Marks ($y$):

| Student | Study Hours ($x$) | Exam Marks ($y$) |
|---|---|---|
| 1 | 1 | 2 |
| 2 | 2 | 3 |
| 3 | 3 | 5 |
| 4 | 4 | 4 |
| 5 | 5 | 6 |

Calculate:
1. Mean $\bar{x}$ and $\bar{y}$
2. Slope $w_1$ and Intercept $w_0$
3. Regression line equation $\hat{y} = w_0 + w_1 x$
4. Predicted marks for a student studying $x = 6$ hours
5. Residuals, Sum of Squared Errors (SSE), and Mean Squared Error (MSE)

**Step-by-Step Solution**:

**Step 1: Compute Means**:
- $N = 5$
- $\sum x = 1 + 2 + 3 + 4 + 5 = 15 \implies \bar{x} = \frac{15}{5} = 3.0$
- $\sum y = 2 + 3 + 5 + 4 + 6 = 20 \implies \bar{y} = \frac{20}{5} = 4.0$

**Step 2: Construct Calculation Table**:

| $i$ | $x_i$ | $y_i$ | $(x_i - \bar{x})$ | $(y_i - \bar{y})$ | $(x_i - \bar{x})(y_i - \bar{y})$ | $(x_i - \bar{x})^2$ |
|---|---|---|---|---|---|---|
| 1 | 1 | 2 | $-2$ | $-2$ | $+4$ | 4 |
| 2 | 2 | 3 | $-1$ | $-1$ | $+1$ | 1 |
| 3 | 3 | 5 | $0$ | $+1$ | $0$ | 0 |
| 4 | 4 | 4 | $+1$ | $0$ | $0$ | 1 |
| 5 | 5 | 6 | $+2$ | $+2$ | $+4$ | 4 |
| **Sum**| **15**| **20**| **0** | **0** | **9** | **10** |

**Step 3: Compute Slope ($w_1$) and Intercept ($w_0$)**:
$$w_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2} = \frac{9}{10} = 0.9$$
$$w_0 = \bar{y} - w_1 \bar{x} = 4.0 - (0.9 \times 3.0) = 4.0 - 2.7 = 1.3$$

**Step 4: Regression Equation & Prediction**:
- Regression Line: $\hat{y} = 1.3 + 0.9 x$
- For $x = 6$: $\hat{y}(6) = 1.3 + 0.9(6) = 1.3 + 5.4 = 6.7$ marks.

**Step 5: Error Metrics Calculation**:

| $i$ | $y_i$ | $\hat{y}_i = 1.3 + 0.9 x_i$ | Residual $e_i = y_i - \hat{y}_i$ | Squared Error $e_i^2$ | Absolute Error $|e_i|$ |
|---|---|---|---|---|---|
| 1 | 2 | $1.3 + 0.9(1) = 2.2$ | $2 - 2.2 = -0.2$ | 0.04 | 0.2 |
| 2 | 3 | $1.3 + 0.9(2) = 3.1$ | $3 - 3.1 = -0.1$ | 0.01 | 0.1 |
| 3 | 5 | $1.3 + 0.9(3) = 4.0$ | $5 - 4.0 = +1.0$ | 1.00 | 1.0 |
| 4 | 4 | $1.3 + 0.9(4) = 4.9$ | $4 - 4.9 = -0.9$ | 0.81 | 0.9 |
| 5 | 6 | $1.3 + 0.9(5) = 5.8$ | $6 - 5.8 = +0.2$ | 0.04 | 0.2 |
| **Total**| | | **0.0** | **SSE = 1.90** | **SAE = 2.40** |

- $\text{SSE} = 1.90$
- $\text{MSE} = \frac{\text{SSE}}{N} = \frac{1.90}{5} = 0.38$
- $\text{MAE} = \frac{\text{SAE}}{N} = \frac{2.40}{5} = 0.48$

**Final Answer**:
- Regression Equation: $\hat{y} = 1.3 + 0.9 x$
- Prediction for $x=6$: $\hat{y} = 6.7$
- $\text{SSE} = 1.90$, $\text{MSE} = 0.38$, $\text{MAE} = 0.48$

---

## Unit 2 Solved Numericals

### 1. Normalization & Scaling Numericals

#### 📌 Problem 2.1: Min-Max Normalization to $[0, 1]$
**Question**: Normalize the dataset values $X = \{10, 25, 40, 55, 70\}$ to range $[0, 1]$ using Min-Max Normalization.

**Step-by-Step Solution**:
1. Identify $x_{\min} = 10$ and $x_{\max} = 70$.
2. Range $x_{\max} - x_{\min} = 70 - 10 = 60$.
3. Apply formula $x' = \frac{x - 10}{60}$:
   - For $x = 10 \implies x' = \frac{10-10}{60} = 0.00$
   - For $x = 25 \implies x' = \frac{25-10}{60} = \frac{15}{60} = 0.25$
   - For $x = 40 \implies x' = \frac{40-10}{60} = \frac{30}{60} = 0.50$
   - For $x = 55 \implies x' = \frac{55-10}{60} = \frac{45}{60} = 0.75$
   - For $x = 70 \implies x' = \frac{70-10}{60} = \frac{60}{60} = 1.00$

**Final Answer**: Normalized Dataset $X' = \{0.00, 0.25, 0.50, 0.75, 1.00\}$.

---

#### 📌 Problem 2.2: Min-Max Scaling to Custom Bounded Range $[-1, +1]$
**Question**: Transform feature value $x = 45$ from dataset with $x_{\min} = 20$ and $x_{\max} = 100$ into the custom range $[a, b] = [-1, +1]$.

**Formula**: $x'' = a + \frac{(x - x_{\min})(b - a)}{x_{\max} - x_{\min}}$

**Step-by-Step Solution**:
$$x'' = -1 + \frac{(45 - 20)(1 - (-1))}{100 - 20} = -1 + \frac{25 \times 2}{80} = -1 + \frac{50}{80} = -1 + 0.625 = -0.375$$

**Final Answer**: $x'' = -0.375$.

---

### 2. Standardization (Z-Score) Numericals

#### 📌 Problem 2.3: Z-Score Standardization of 1D Dataset
**Question**: Standardize the feature values $X = \{2, 4, 6, 8, 10\}$. Show all intermediate statistics ($\mu, \sigma^2, \sigma$).

**Step-by-Step Solution**:

**Step 1: Compute Sample Mean ($\mu$)**:
$$\mu = \frac{2 + 4 + 6 + 8 + 10}{5} = \frac{30}{5} = 6.0$$

**Step 2: Compute Variance ($\sigma^2$) and Standard Deviation ($\sigma$)**:

| $x_i$ | $(x_i - \mu)$ | $(x_i - \mu)^2$ |
|---|---|---|
| 2 | $2 - 6 = -4$ | 16 |
| 4 | $4 - 6 = -2$ | 4 |
| 6 | $6 - 6 = 0$ | 0 |
| 8 | $8 - 6 = +2$ | 4 |
| 10 | $10 - 6 = +4$ | 16 |
| **Sum**| **0** | **40** |

$$\sigma^2 = \frac{\sum (x_i - \mu)^2}{N} = \frac{40}{5} = 8.0 \implies \sigma = \sqrt{8.0} \approx 2.8284$$

**Step 3: Calculate Z-Scores ($z = \frac{x - 6.0}{2.8284}$)**:
- $z_1 = \frac{2 - 6}{2.8284} = \frac{-4}{2.8284} = -1.4142$
- $z_2 = \frac{4 - 6}{2.8284} = \frac{-2}{2.8284} = -0.7071$
- $z_3 = \frac{6 - 6}{2.8284} = 0.0000$
- $z_4 = \frac{8 - 6}{2.8284} = +0.7071$
- $z_5 = \frac{10 - 6}{2.8284} = +1.4142$

**Final Answer**:  
Standardized values $Z = \{-1.4142, -0.7071, 0.0000, +0.7071, +1.4142\}$.

---

### 3. Principal Component Analysis (PCA) Solved Numericals

#### 📌 Problem 2.4 (PCA Full 2D Worked Problem — VERY IMPORTANT)
**Question**: Given the 2D dataset with $N=4$ samples:
$$X = \begin{bmatrix} 2 & 4 \\ 3 & 5 \\ 5 & 7 \\ 6 & 8 \end{bmatrix}$$
Perform complete Principal Component Analysis step-by-step:
1. Calculate mean centered matrix $X_c$.
2. Compute Covariance Matrix $\Sigma$.
3. Compute Eigenvalues $\lambda_1, \lambda_2$.
4. Compute normalized Eigenvectors $\mathbf{v}_1, \mathbf{v}_2$.
5. Calculate Explained Variance Ratio for each component.
6. Project data onto the 1st Principal Component.

**Step-by-Step Solution**:

**Step 1: Compute Feature Means and Centered Matrix $X_c$**:
- Mean of $x_1$: $\mu_1 = \frac{2 + 3 + 5 + 6}{4} = \frac{16}{4} = 4.0$
- Mean of $x_2$: $\mu_2 = \frac{4 + 5 + 7 + 8}{4} = \frac{24}{4} = 6.0$

Subtract mean vector $\mathbf{\mu} = [4.0, 6.0]$ from each row of $X$:
$$X_c = \begin{bmatrix} 2-4 & 4-6 \\ 3-4 & 5-6 \\ 5-4 & 7-6 \\ 6-4 & 8-6 \end{bmatrix} = \begin{bmatrix} -2 & -2 \\ -1 & -1 \\ 1 & 1 \\ 2 & 2 \end{bmatrix}$$

**Step 2: Compute Covariance Matrix $\Sigma$**:
$$\Sigma = \frac{1}{N-1} X_c^T X_c = \frac{1}{3} \begin{bmatrix} -2 & -1 & 1 & 2 \\ -2 & -1 & 1 & 2 \end{bmatrix} \begin{bmatrix} -2 & -2 \\ -1 & -1 \\ 1 & 1 \\ 2 & 2 \end{bmatrix}$$

Calculate elements:
- $\text{Var}(x_1) = \frac{(-2)^2 + (-1)^2 + 1^2 + 2^2}{3} = \frac{4 + 1 + 1 + 4}{3} = \frac{10}{3}$
- $\text{Cov}(x_1, x_2) = \frac{(-2)(-2) + (-1)(-1) + (1)(1) + (2)(2)}{3} = \frac{4 + 1 + 1 + 4}{3} = \frac{10}{3}$
- $\text{Var}(x_2) = \frac{10}{3}$

$$\Sigma = \begin{bmatrix} 3.333 & 3.333 \\ 3.333 & 3.333 \end{bmatrix}$$

**Step 3: Calculate Eigenvalues ($\lambda$)**:
Solve characteristic equation $\det(\Sigma - \lambda I) = 0$:
$$\det \begin{bmatrix} \frac{10}{3} - \lambda & \frac{10}{3} \\ \frac{10}{3} & \frac{10}{3} - \lambda \end{bmatrix} = 0 \implies \left(\frac{10}{3} - \lambda\right)^2 - \left(\frac{10}{3}\right)^2 = 0$$
$$\implies \left(\frac{10}{3} - \lambda - \frac{10}{3}\right)\left(\frac{10}{3} - \lambda + \frac{10}{3}\right) = 0 \implies (-\lambda)\left(\frac{20}{3} - \lambda\right) = 0$$

- $\lambda_1 = \frac{20}{3} \approx 6.667$
- $\lambda_2 = 0$

**Step 4: Compute Eigenvectors**:
For $\lambda_1 = \frac{20}{3}$:
$$(\Sigma - \lambda_1 I)\mathbf{v}_1 = 0 \implies \begin{bmatrix} -\frac{10}{3} & \frac{10}{3} \\ \frac{10}{3} & -\frac{10}{3} \end{bmatrix} \begin{bmatrix} v_{11} \\ v_{12} \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$
$$\implies -\frac{10}{3} v_{11} + \frac{10}{3} v_{12} = 0 \implies v_{11} = v_{12}$$

Normalizing eigenvector $\mathbf{v}_1$:
$$\|\mathbf{v}_1\| = \sqrt{v_{11}^2 + v_{12}^2} = 1 \implies \sqrt{2 v_{11}^2} = 1 \implies v_{11} = \frac{1}{\sqrt{2}}$$
$$\mathbf{v}_1 = \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix} \approx \begin{bmatrix} 0.7071 \\ 0.7071 \end{bmatrix}$$

**Step 5: Explained Variance Ratio**:
$$\text{EVR}_1 = \frac{\lambda_1}{\lambda_1 + \lambda_2} = \frac{6.667}{6.667 + 0} = 1.0 \quad (100\% \text{ of variance captured by } PC_1)$$

**Step 6: Project Data onto 1st Principal Component ($Y = X_c \mathbf{v}_1$)**:
$$Y = \begin{bmatrix} -2 & -2 \\ -1 & -1 \\ 1 & 1 \\ 2 & 2 \end{bmatrix} \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix} = \begin{bmatrix} \frac{-4}{\sqrt{2}} \\ \frac{-2}{\sqrt{2}} \\ \frac{2}{\sqrt{2}} \\ \frac{4}{\sqrt{2}} \end{bmatrix} = \begin{bmatrix} -2.8284 \\ -1.4142 \\ +1.4142 \\ +2.8284 \end{bmatrix}$$

**Final Answer**:
- Eigenvalues: $\lambda_1 = 6.667, \lambda_2 = 0$
- 1st Principal Component Eigenvector: $\mathbf{v}_1 = [0.7071, 0.7071]^T$
- Transformed 1D Projected Data $Y = [-2.8284, -1.4142, +1.4142, +2.8284]^T$

---

#### 📌 Problem 2.5: PCA Component Selection & Variance Threshold
**Question**: A PCA algorithm computed the following eigenvalues for a 5-dimensional dataset:
$$\lambda_1 = 4.0, \quad \lambda_2 = 2.5, \quad \lambda_3 = 1.5, \quad \lambda_4 = 1.0, \quad \lambda_5 = 1.0$$
1. Calculate total dataset variance.
2. Compute Explained Variance Ratio (EVR) for each principal component.
3. Determine the minimum number of components required to retain at least $80\%$ of total variance.

**Step-by-Step Solution**:
1. **Total Variance**:
   $$\lambda_{\text{total}} = 4.0 + 2.5 + 1.5 + 1.0 + 1.0 = 10.0$$
2. **Individual EVR**:
   - $\text{EVR}_1 = \frac{4.0}{10.0} = 40.0\%$
   - $\text{EVR}_2 = \frac{2.5}{10.0} = 25.0\%$
   - $\text{EVR}_3 = \frac{1.5}{10.0} = 15.0\%$
   - $\text{EVR}_4 = \frac{1.0}{10.0} = 10.0\%$
   - $\text{EVR}_5 = \frac{1.0}{10.0} = 10.0\%$
3. **Cumulative EVR**:
   - $k=1 \implies 40.0\%$
   - $k=2 \implies 40.0\% + 25.0\% = 65.0\%$
   - $k=3 \implies 65.0\% + 15.0\% = 80.0\%$

**Final Answer**: Minimum $k = 3$ principal components are required to retain $80\%$ total variance.

---

### 4. Linear Discriminant Analysis (LDA) Solved Numericals

#### 📌 Problem 2.6 (LDA Full 2D Worked Numerical — VERY IMPORTANT)
**Question**: Consider a 2-class binary classification problem with 2D data points:
- **Class 1 ($\mathcal{D}_1$)**: $\mathbf{x}_1 = [1, 2]^T, \mathbf{x}_2 = [2, 3]^T$
- **Class 2 ($\mathcal{D}_2$)**: $\mathbf{x}_3 = [4, 5]^T, \mathbf{x}_4 = [5, 6]^T$

Compute:
1. Class mean vectors $\mathbf{\mu}_1, \mathbf{\mu}_2$.
2. Within-class scatter matrices $S_1, S_2$ and total $S_W$.
3. Between-class scatter matrix $S_B$.
4. Optimal Fisher discriminant direction vector $\mathbf{w}^* = S_W^{-1}(\mathbf{\mu}_1 - \mathbf{\mu}_2)$.

**Step-by-Step Solution**:

**Step 1: Compute Class Means**:
$$\mathbf{\mu}_1 = \frac{1}{2} \left( \begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 2 \\ 3 \end{bmatrix} \right) = \begin{bmatrix} 1.5 \\ 2.5 \end{bmatrix}$$
$$\mathbf{\mu}_2 = \frac{1}{2} \left( \begin{bmatrix} 4 \\ 5 \end{bmatrix} + \begin{bmatrix} 5 \\ 6 \end{bmatrix} \right) = \begin{bmatrix} 4.5 \\ 5.5 \end{bmatrix}$$

**Step 2: Compute Within-Class Scatter Matrices ($S_1, S_2, S_W$)**:
- For Class 1:
  $$\mathbf{x}_1 - \mathbf{\mu}_1 = \begin{bmatrix} 1 - 1.5 \\ 2 - 2.5 \end{bmatrix} = \begin{bmatrix} -0.5 \\ -0.5 \end{bmatrix}$$
  $$\mathbf{x}_2 - \mathbf{\mu}_1 = \begin{bmatrix} 2 - 1.5 \\ 3 - 2.5 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}$$
  $$S_1 = \begin{bmatrix} -0.5 \\ -0.5 \end{bmatrix} [-0.5, -0.5] + \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix} [0.5, 0.5] = \begin{bmatrix} 0.25 & 0.25 \\ 0.25 & 0.25 \end{bmatrix} + \begin{bmatrix} 0.25 & 0.25 \\ 0.25 & 0.25 \end{bmatrix} = \begin{bmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{bmatrix}$$

- For Class 2:
  $$\mathbf{x}_3 - \mathbf{\mu}_2 = \begin{bmatrix} -0.5 \\ -0.5 \end{bmatrix}, \quad \mathbf{x}_4 - \mathbf{\mu}_2 = \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix} \implies S_2 = \begin{bmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{bmatrix}$$

- Total Within-Class Scatter Matrix $S_W$:
  $$S_W = S_1 + S_2 = \begin{bmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{bmatrix} + \begin{bmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{bmatrix} = \begin{bmatrix} 1.0 & 1.0 \\ 1.0 & 1.0 \end{bmatrix}$$

*(Note: In practice, a small regularization term $\epsilon I$ is added if $S_W$ is singular. Adding $0.1 I$ gives $S_W = \begin{bmatrix} 1.1 & 1.0 \\ 1.0 & 1.1 \end{bmatrix}$).*

**Step 3: Compute Between-Class Scatter Matrix $S_B$**:
$$\mathbf{\mu}_1 - \mathbf{\mu}_2 = \begin{bmatrix} 1.5 - 4.5 \\ 2.5 - 5.5 \end{bmatrix} = \begin{bmatrix} -3.0 \\ -3.0 \end{bmatrix}$$
$$S_B = (\mathbf{\mu}_1 - \mathbf{\mu}_2)(\mathbf{\mu}_1 - \mathbf{\mu}_2)^T = \begin{bmatrix} -3 \\ -3 \end{bmatrix} [-3, -3] = \begin{bmatrix} 9.0 & 9.0 \\ 9.0 & 9.0 \end{bmatrix}$$

**Step 4: Compute Optimal Linear Discriminant Direction $\mathbf{w}^*$**:
Using regularized inverse $S_W^{-1}$:
$$\det(S_W) = (1.1)(1.1) - (1.0)(1.0) = 1.21 - 1.0 = 0.21$$
$$S_W^{-1} = \frac{1}{0.21} \begin{bmatrix} 1.1 & -1.0 \\ -1.0 & 1.1 \end{bmatrix}$$
$$\mathbf{w}^* = S_W^{-1}(\mathbf{\mu}_1 - \mathbf{\mu}_2) = \frac{1}{0.21} \begin{bmatrix} 1.1 & -1.0 \\ -1.0 & 1.1 \end{bmatrix} \begin{bmatrix} -3 \\ -3 \end{bmatrix} = \frac{1}{0.21} \begin{bmatrix} -3.3 + 3.0 \\ 3.0 - 3.3 \end{bmatrix} = \begin{bmatrix} -1.428 \\ -1.428 \end{bmatrix}$$

Normalizing direction vector: $\mathbf{w}^* = [-0.7071, -0.7071]^T$.

**Final Answer**:
- Class Means: $\mathbf{\mu}_1 = [1.5, 2.5]^T, \mathbf{\mu}_2 = [4.5, 5.5]^T$
- Within-Class Scatter Matrix $S_W = \begin{bmatrix} 1.0 & 1.0 \\ 1.0 & 1.0 \end{bmatrix}$
- Between-Class Scatter Matrix $S_B = \begin{bmatrix} 9.0 & 9.0 \\ 9.0 & 9.0 \end{bmatrix}$
- Optimal Fisher Projection Vector $\mathbf{w}^* = [-0.7071, -0.7071]^T$


# 🔥 MOST IMPORTANT 4-MARK QUESTION BANK

---

## Unit 2 — 4-Mark Questions (25 Solved)

> **HIGH EXAM FREQUENCY**: Unit 2 contributes heavily to 4-mark short questions. Memorize these answers for exam writing.

---

### Q1: What is a Feature Vector? Explain with a 2D medical dataset example.
**Answer**:
1. **Definition**: A **Feature Vector** $\mathbf{x}$ is an ordered $d$-dimensional column vector containing numerical or categorical attribute values representing a single instance:
   $$\mathbf{x} = [x_1, x_2, \dots, x_d]^T \in \mathbb{R}^d$$
2. **Medical Example**: For a patient undergoing diabetes screening:
   $$\mathbf{x} = \begin{bmatrix} \text{Glucose Level} \\ \text{BMI} \end{bmatrix} = \begin{bmatrix} 140 \text{ mg/dL} \\ 28.5 \text{ kg/m}^2 \end{bmatrix}$$
3. **Key Points for Full Marks**:
   - Mention $d$-dimensional vector space.
   - Give vector equation $\mathbf{x} = [x_1, \dots, x_d]^T$.
   - Provide concrete domain example with units.

---

### Q2: What is Data Preprocessing? Why is it essential in Machine Learning?
**Answer**:
1. **Definition**: Data Preprocessing encompasses techniques applied to transform messy, unrefined raw data into a clean, structured format suitable for ML algorithms.
2. **Why Essential?**:
   - Handles missing values and corrupted entries (`NaN`).
   - Rescales attributes to prevent high-magnitude features from dominating distance functions.
   - Encodes non-numeric categorical attributes into vectors.
3. **Key Points for Full Marks**:
   - Define data cleaning and transformation.
   - Mention impact on model convergence and distance metrics.

---

### Q3: Explain Min-Max Normalization with formula and range.
**Answer**:
1. **Definition**: Min-Max Normalization rescales numeric feature values linearly into a fixed bounded range $[0, 1]$.
2. **Formula**:
   $$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
3. **Properties**:
   - Bounds all features to $[0, 1]$.
   - Preserves relative distance ratios between points.
4. **Key Points for Full Marks**: State formula, symbol definitions ($x, x_{\min}, x_{\max}$), and bounded output range.

---

### Q4: Explain Z-Score Standardization with formula and mean/variance values.
**Answer**:
1. **Definition**: Standardization transforms a continuous feature distribution to have a mean of zero ($\mu = 0$) and a standard deviation of one ($\sigma = 1$).
2. **Formula**:
   $$z = \frac{x - \mu}{\sigma} \quad \text{where } \mu = \frac{1}{N}\sum x_i, \quad \sigma = \sqrt{\frac{1}{N}\sum (x_i - \mu)^2}$$
3. **Key Points for Full Marks**: Include Z-score equation, specify resulting $\mu=0$ and $\sigma=1$, and mention robustness to outliers.

---

### Q5: Differentiate between Normalization and Standardization.
**Answer**:

| Parameter | Min-Max Normalization | Z-Score Standardization |
|---|---|---|
| **Formula** | $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ | $z = \frac{x - \mu}{\sigma}$ |
| **Output Range** | Strictly bounded $[0, 1]$ | Unbounded $(-\infty, +\infty)$ |
| **Outlier Sensitivity** | High (Outliers compress normal data) | Low (Retains variance structure) |
| **Mean / Std Dev**| Arbitrary mean and variance | Fixed Mean $\mu=0$, Std Dev $\sigma=1$ |

---

### Q6: How do Mean and Median Imputation differ for managing missing values?
**Answer**:
1. **Mean Imputation**: Replaces missing numeric values with the sample mean $\mu$.  
   *Best used when*: Feature data is normally distributed without extreme outliers.
2. **Median Imputation**: Replaces missing numeric values with the sample median (middle rank).  
   *Best used when*: Feature data contains heavy skewness or extreme outliers.
3. **Key Points for Full Marks**: State formulas/concepts for mean vs median and outlier sensitivity.

---

### Q7: What is the Curse of Dimensionality? State two negative effects.
**Answer**:
1. **Definition**: The phenomenon where exponentially increasing the feature dimension $d$ causes data points to become extremely sparse in high-dimensional space.
2. **Two Negative Effects**:
   - **Distance Breakdown**: Euclidean distances between points converge to the same value ($d_{\max} \approx d_{\min}$), crippling $k$-NN and clustering.
   - **Overfitting**: Model parameters outnumber training instances, causing severe memorization of noise.
3. **Key Points for Full Marks**: Mention data sparsity, volume expansion, and model overfitting.

---

### Q8: Differentiate between Feature Selection and Feature Extraction.
**Answer**:
1. **Feature Selection**: Chooses a subset of $k$ original features without altering their original representation (e.g., SFS, SBS). High interpretability.
2. **Feature Extraction**: Combines and transforms $d$ original features into $k$ completely new features in a lower-dimensional space (e.g., PCA, LDA). Low physical interpretability.
3. **Key Points for Full Marks**: Tabulate definition, transformation nature, and example algorithms.

---

### Q9: Define Principal Component Analysis (PCA) and state its primary objective.
**Answer**:
1. **Definition**: PCA is an unsupervised linear feature extraction technique that projects $d$-dimensional data into a $k$-dimensional subspace along orthogonal axes called Principal Components.
2. **Primary Objective**: Maximize the captured dataset variance while minimizing total reconstruction error.
3. **Key Points for Full Marks**: Mention unsupervised nature, variance maximization, and orthogonal projection axes.

---

### Q10: What is the Covariance Matrix in PCA? Write its equation.
**Answer**:
1. **Definition**: An symmetric $d \times d$ matrix storing variance of individual features on the diagonal and pairwise covariances on off-diagonal entries.
2. **Formula**: For mean-centered data $X_c$:
   $$\Sigma = \frac{1}{N-1} X_c^T X_c = \begin{bmatrix} \text{Var}(x_1) & \text{Cov}(x_1, x_2) \\ \text{Cov}(x_2, x_1) & \text{Var}(x_2) \end{bmatrix}$$
3. **Key Points for Full Marks**: Write matrix formula, specify diagonal (variance) vs off-diagonal (covariance).

---

### Q11: Explain the role of Eigenvalues and Eigenvectors in PCA.
**Answer**:
1. **Eigenvector ($\mathbf{v}$)**: Defines the spatial direction of a Principal Component line in feature space ($\Sigma \mathbf{v} = \lambda \mathbf{v}$).
2. **Eigenvalue ($\lambda$)**: Represents the quantitative scalar magnitude of variance captured along the corresponding eigenvector's direction.
3. **Component Ranking**: Eigenvalues are sorted descendingly ($\lambda_1 \ge \lambda_2 \ge \dots$); top eigenvectors form the projection matrix $W$.
4. **Key Points for Full Marks**: State matrix equation $\Sigma \mathbf{v} = \lambda \mathbf{v}$ and link eigenvalue magnitude to captured variance.

---

### Q12: Define Explained Variance Ratio in PCA. Write its formula.
**Answer**:
1. **Definition**: The percentage of total dataset variance accounted for by a specific $k$-th principal component.
2. **Formula**:
   $$\text{EVR}_k = \frac{\lambda_k}{\sum_{j=1}^d \lambda_j}$$
3. **Application**: Used to determine how many components $k$ to retain to reach a cumulative variance threshold (e.g., $90\%$).

---

### Q13: What is Kernel PCA? Why is it needed?
**Answer**:
1. **Definition**: A non-linear extension of PCA that uses kernel functions $K(\mathbf{x}_i, \mathbf{x}_j)$ to map input data into a high-dimensional feature space $\Phi(\mathbf{x})$ where linear PCA is applied.
2. **Why Needed?**: Standard linear PCA fails when dataset decision boundaries lie on complex non-linear manifolds (e.g., concentric circles).
3. **Key Points for Full Marks**: Mention kernel trick $K(x,y)=\langle\Phi(x),\Phi(y)\rangle$ and non-linear separability.

---

### Q14: List three common Kernel Functions used in Kernel PCA.
**Answer**:
1. **Polynomial Kernel**: $K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T \mathbf{x}_j + c)^d$
2. **RBF / Gaussian Kernel**: $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$
3. **Sigmoid Kernel**: $K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\alpha \mathbf{x}_i^T \mathbf{x}_j + c)$

---

### Q15: What is Linear Discriminant Analysis (LDA)? State its main goal.
**Answer**:
1. **Definition**: LDA is a supervised linear dimensionality reduction algorithm used for classification.
2. **Main Goal**: Finds a projection direction $\mathbf{w}$ that maximizes the distance between class means while minimizing variance within each class (Fisher's Criterion).
3. **Key Points for Full Marks**: Contrast supervised nature against PCA and state class separability goal.

---

### Q16: Explain Within-Class Scatter ($S_W$) and Between-Class Scatter ($S_B$) in LDA.
**Answer**:
1. **Within-Class Scatter ($S_W$)**: Measures feature variance spread within individual classes:
   $$S_W = \sum_{c=1}^C \sum_{\mathbf{x} \in \mathcal{D}_c} (\mathbf{x} - \mathbf{\mu}_c)(\mathbf{x} - \mathbf{\mu}_c)^T$$
2. **Between-Class Scatter ($S_B$)**: Measures separation distance between class mean vectors:
   $$S_B = \sum_{c=1}^C N_c (\mathbf{\mu}_c - \mathbf{\mu})(\mathbf{\mu}_c - \mathbf{\mu})^T$$

---

### Q17: Write and explain Fisher's Criterion in LDA.
**Answer**:
1. **Objective Function**:
   $$J(\mathbf{w}) = \frac{\mathbf{w}^T S_B \mathbf{w}}{\mathbf{w}^T S_W \mathbf{w}}$$
2. **Explanation**: $J(\mathbf{w})$ is maximized when between-class variance ($\mathbf{w}^T S_B \mathbf{w}$) is large and within-class variance ($\mathbf{w}^T S_W \mathbf{w}$) is small.
3. **Optimal Vector**: $\mathbf{w}^* = S_W^{-1}(\mathbf{\mu}_1 - \mathbf{\mu}_2)$.

---

### Q18: What is the maximum number of dimensions LDA can project data into?
**Answer**:
1. **Constraint**: For a classification problem with $C$ target classes and $d$ original features, LDA can reduce dimensionality to at most:
   $$k \le C - 1$$
2. **Reason**: The between-class scatter matrix $S_B$ has a maximum rank of $C - 1$.
3. **Example**: For a 3-class problem ($C=3$), LDA can project data to at most $3-1 = 2$ dimensions.

---

### Q19: Compare PCA and LDA based on supervision, objective, and output dimensions.
**Answer**:

| Parameter | PCA | LDA |
|---|---|---|
| **Supervision** | Unsupervised (No labels) | Supervised (Uses class labels $y$) |
| **Objective** | Maximize total variance | Maximize class separability |
| **Max Dimensions**| Up to $d$ components | Up to $C - 1$ components |

---

### Q20: Explain Filter Methods of Feature Selection with pros and cons.
**Answer**:
1. **Definition**: Filter methods select feature subsets based on statistical tests (Pearson correlation, Chi-Square, Mutual Info) independent of ML training algorithms.
2. **Pros**: Computationally fast; scalable to massive feature counts.
3. **Cons**: Ignores feature interactions and model-specific learning biases.

---

### Q21: Explain Wrapper Methods of Feature Selection.
**Answer**:
1. **Definition**: Wrapper methods score feature subsets by training a specific predictive ML model and evaluating validation accuracy (e.g., SFS, SBS).
2. **Pros**: Captures feature dependencies; achieves high predictive accuracy.
3. **Cons**: Computationally expensive ($O(d^2)$ complexity).

---

### Q22: Explain Sequential Forward Selection (SFS) algorithm in 4 points.
**Answer**:
1. **Strategy**: Bottom-up greedy wrapper feature selection.
2. **Initial State**: Starts with an empty set $\mathcal{Y}_0 = \emptyset$.
3. **Iteration**: At step $k$, appends single feature $f^+$ maximizing cross-validation performance: $\mathcal{Y}_k = \mathcal{Y}_{k-1} \cup \{f^+\}$.
4. **Flaw**: Suffers from nesting effect (cannot remove a feature once added).

---

### Q23: Explain Sequential Backward Selection (SBS) algorithm in 4 points.
**Answer**:
1. **Strategy**: Top-down greedy wrapper feature selection.
2. **Initial State**: Starts with full feature set $\mathcal{Y}_0 = X$.
3. **Iteration**: At step $k$, removes feature $f^-$ whose deletion causes minimal accuracy drop: $\mathcal{Y}_k = \mathcal{Y}_{k-1} \setminus \{f^-\}$.
4. **Flaw**: High initial computational cost for large $d$; cannot re-add dropped features.

---

### Q24: Differentiate between SFS and SBS.
**Answer**:

| Feature | SFS | SBS |
|---|---|---|
| **Start State** | Empty set $\emptyset$ | Full set $X$ |
| **Operation** | Adds 1 feature per step | Removes 1 feature per step |
| **Initial Cost** | Low | High |
| **Best For** | Large feature count $d$ | Moderate feature count $d$ |

---

### Q25: What is the Nesting Effect in Sequential Feature Selection?
**Answer**:
1. **Definition**: The structural limitation of greedy selection algorithms where past decisions cannot be undone.
2. **In SFS**: Once a feature is selected, it can never be removed in later iterations even if it becomes redundant.
3. **In SBS**: Once a feature is removed, it can never be re-added in subsequent steps.

---

## Unit 1 — 4-Mark Questions (20 Solved)

---

### Q1: Define Machine Learning. Differentiate it from Traditional Programming.
**Answer**:
1. **Definition**: Machine Learning is a field of AI that provides systems the ability to automatically learn and improve from experience without explicit programming (Tom Mitchell's $E, T, P$ definition).
2. **Comparison**:
   - *Traditional*: Input Data + Hand-coded Rules $\rightarrow$ Outputs.
   - *Machine Learning*: Input Data + Historical Outputs $\rightarrow$ Learned Model/Rules.

---

### Q2: Explain Supervised Learning with its two main sub-types.
**Answer**:
1. **Definition**: Algorithms trained on labeled datasets $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}$.
2. **Sub-types**:
   - **Classification**: Target label $y$ is discrete/categorical (e.g., Spam vs Ham).
   - **Regression**: Target label $y$ is continuous real-valued (e.g., Price prediction).

---

### Q3: Explain Unsupervised Learning with two key applications.
**Answer**:
1. **Definition**: Algorithms trained on unlabeled data $\mathcal{D} = \{\mathbf{x}_i\}$ to discover hidden patterns or groupings.
2. **Applications**:
   - **Clustering**: Grouping similar customers (K-Means).
   - **Dimensionality Reduction**: Visualizing high-dimensional data (PCA).

---

### Q4: Explain Reinforcement Learning using its 5 core components.
**Answer**:
1. **Definition**: An agent learns optimal policy by taking actions in an environment to maximize cumulative scalar rewards.
2. **5 Components**: Agent, Environment, State ($s$), Action ($a$), Reward ($r$).

---

### Q5: What is Concept Learning? Define Hypothesis Space and Target Concept.
**Answer**:
1. **Concept Learning**: Inferring a boolean-valued function from training examples.
2. **Target Concept ($C$)**: The true function $C: \mathcal{X} \rightarrow \{0, 1\}$ being learned.
3. **Hypothesis Space ($\mathcal{H}$)**: Set of all candidate functions considered by the model.

---

### Q6: Define VC Dimension. What does $VC(\mathcal{H}) = d$ signify?
**Answer**:
1. **Definition**: The maximum number of points $d$ that can be shattered by hypothesis space $\mathcal{H}$.
2. **Significance**: Quantifies model capacity and expressive power. Higher VC dimension implies greater capacity but higher risk of overfitting.

---

### Q7: What does "Shattering" mean in VC theory? Give an example.
**Answer**:
1. **Definition**: A set of $N$ points is shattered by $\mathcal{H}$ if $\mathcal{H}$ can realize all $2^N$ possible binary labelings of those points.
2. **Example**: 3 non-collinear points in 2D space can be shattered by straight lines ($2^3 = 8$ labelings).

---

### Q8: Explain the PAC Learning framework parameters $\epsilon$ and $\delta$.
**Answer**:
1. **Accuracy Parameter ($\epsilon$)**: Upper bound on generalization error ($R(h) \le \epsilon$).
2. **Confidence Parameter ($\delta$)**: Probability of algorithm failure ($P[R(h) > \epsilon] \le \delta$).
3. **PAC Goal**: Ensures hypothesis is "Approximately Correct" ($1-\epsilon$) with "High Probability" ($1-\delta$).

---

### Q9: Write the PAC sample complexity formula for finite $\mathcal{H}$ and define every symbol.
**Answer**:
1. **Formula**:
   $$m \ge \frac{1}{\epsilon} \left( \ln |\mathcal{H}| + \ln \left(\frac{1}{\delta}\right) \right)$$
2. **Symbols**: $m$ = sample size, $\epsilon$ = error bound, $\delta$ = confidence failure probability, $|\mathcal{H}|$ = hypothesis space size.

---

### Q10: What is Noise in Machine Learning? Name three sources.
**Answer**:
1. **Definition**: Corruption or unwanted random variation present in feature values or target labels.
2. **3 Sources**: Label Noise (human annotator error), Feature Noise (sensor error), Measurement Noise.

---

### Q11: Explain One-vs-Rest (OvR) multi-class classification strategy.
**Answer**:
1. **Strategy**: Trains $K$ binary classifiers for $K$ classes. $k$-th model treats class $k$ as positive and all others as negative.
2. **Decision Rule**: $\hat{y} = \arg\max_{k} h_k(\mathbf{x})$.

---

### Q12: Explain One-vs-One (OvO) multi-class classification strategy.
**Answer**:
1. **Strategy**: Trains $\frac{K(K-1)}{2}$ binary classifiers for every pair of classes $(i, j)$.
2. **Decision Rule**: Majority voting across all binary classifiers.

---

### Q13: Differentiate between Classification and Regression.
**Answer**:

| Parameter | Classification | Regression |
|---|---|---|
| **Target $y$** | Discrete categorical values | Continuous numerical values |
| **Output Goal** | Assign class label / decision boundary | Fit continuous mapping function |
| **Metrics** | Accuracy, Precision, Recall, F1-Score | MSE, MAE, RMSE, $R^2$ Score |

---

### Q14: Write Least Squares formulas for slope $w_1$ and intercept $w_0$ in Simple Linear Regression.
**Answer**:
$$\text{Slope } w_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, \qquad \text{Intercept } w_0 = \bar{y} - w_1 \bar{x}$$

---

### Q15: Define MSE, MAE, and SSE in Regression.
**Answer**:
1. **Sum of Squared Errors**: $\text{SSE} = \sum (y_i - \hat{y}_i)^2$
2. **Mean Squared Error**: $\text{MSE} = \frac{1}{N} \sum (y_i - \hat{y}_i)^2$
3. **Mean Absolute Error**: $\text{MAE} = \frac{1}{N} \sum |y_i - \hat{y}_i|$

---

### Q16: What is Overfitting? State two causes and two remedies.
**Answer**:
1. **Definition**: When a model fits training data noise perfectly, yielding low $E_{train}$ but high $E_{test}$.
2. **Causes**: Excessive model complexity; noisy or insufficient training data.
3. **Remedies**: Apply $L_1/L_2$ regularization; simplify model / reduce features.

---

### Q17: What is Underfitting? State two remedies.
**Answer**:
1. **Definition**: When a model is too simple to capture underlying data relationships, yielding high $E_{train}$ and high $E_{test}$.
2. **Remedies**: Increase model complexity; add more relevant features.

---

### Q18: Explain the Bias-Variance Tradeoff.
**Answer**:
1. **Bias**: Error introduced by approximating a complex real-world problem with an overly simple model (Underfitting).
2. **Variance**: Sensitivity of model predictions to small fluctuations in training data (Overfitting).
3. **Tradeoff**: Total Error = $\text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$. Optimal model minimizes the sum of bias and variance.

---

### Q19: Explain $k$-Fold Cross-Validation in 3 points.
**Answer**:
1. Partition dataset into $k$ equal-sized folds.
2. Iterate $k$ times: train on $k-1$ folds and validate on the remaining fold.
3. Average validation performance across $k$ runs: $\text{CV} = \frac{1}{k} \sum E_i$.

---

### Q20: List the 8 Dimensions of a Supervised Machine Learning Algorithm.
**Answer**:
1. Input Space ($\mathcal{X}$)
2. Feature Space ($\mathcal{F}$)
3. Output Space ($\mathcal{Y}$)
4. Hypothesis Space ($\mathcal{H}$)
5. Training Dataset ($\mathcal{D}$)
6. Target Function ($f$)
7. Loss Function ($\mathcal{L}$)
8. Learning Algorithm ($\mathcal{A}$)


# IMPORTANT LONG-ANSWER QUESTIONS (8–10 MARKS)

---

### 📝 Long-Answer Model 1: Principal Component Analysis (PCA) End-to-End Derivation and Algorithm Pipeline

**Question**: Explain Principal Component Analysis (PCA) in detail. Derive the covariance matrix and eigenvalue problem. Present the step-by-step algorithm and illustrate with a neat flowchart.

#### 1. Definition & Core Philosophy
Principal Component Analysis (PCA) is an unsupervised linear feature extraction technique that transforms a dataset of $d$ correlated continuous features into $k$ ($k < d$) uncorrelated orthogonal variables called **Principal Components**. PCA aims to maximize the variance of projected data while minimizing squared reconstruction errors.

#### 2. Mathematical Derivation of 1st Principal Component
Let $X \in \mathbb{R}^{N \times d}$ be a mean-centered data matrix where column means $\bar{x}_j = 0$.
We seek a unit direction vector $\mathbf{w}_1 \in \mathbb{R}^d$ ($\|\mathbf{w}_1\|^2 = \mathbf{w}_1^T \mathbf{w}_1 = 1$) such that the variance of the projected data points $y_i = \mathbf{x}_i^T \mathbf{w}_1$ is maximized.

1. **Variance of Projected Data**:
   $$\text{Var}(y) = \frac{1}{N-1} \sum_{i=1}^N (\mathbf{x}_i^T \mathbf{w}_1)^2 = \frac{1}{N-1} \mathbf{w}_1^T X^T X \mathbf{w}_1 = \mathbf{w}_1^T \Sigma \mathbf{w}_1$$
   where $\Sigma = \frac{1}{N-1} X^T X$ is the $d \times d$ Sample Covariance Matrix.

2. **Optimization Problem Formulation (Lagrange Multipliers)**:
   $$\max_{\mathbf{w}_1} \mathbf{w}_1^T \Sigma \mathbf{w}_1 \quad \text{subject to } \mathbf{w}_1^T \mathbf{w}_1 = 1$$
   Construct Lagrangian:
   $$\mathcal{L}(\mathbf{w}_1, \lambda_1) = \mathbf{w}_1^T \Sigma \mathbf{w}_1 - \lambda_1 (\mathbf{w}_1^T \mathbf{w}_1 - 1)$$

3. **Setting Partial Derivative to Zero**:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{w}_1} = 2 \Sigma \mathbf{w}_1 - 2 \lambda_1 \mathbf{w}_1 = 0 \implies \Sigma \mathbf{w}_1 = \lambda_1 \mathbf{w}_1$$

4. **Conclusion**:
   The optimal projection direction $\mathbf{w}_1$ is an **eigenvector** of the covariance matrix $\Sigma$, and the projected variance $\text{Var}(y) = \mathbf{w}_1^T (\Sigma \mathbf{w}_1) = \mathbf{w}_1^T (\lambda_1 \mathbf{w}_1) = \lambda_1$ equals the corresponding **eigenvalue** $\lambda_1$. To maximize variance, $\mathbf{w}_1$ must be the eigenvector corresponding to the largest eigenvalue $\lambda_1$.

#### 3. Step-by-Step PCA Algorithm
1. **Center Data**: $X_c = X - \mu$.
2. **Compute Covariance Matrix**: $\Sigma = \frac{1}{N-1} X_c^T X_c$.
3. **Eigendecomposition**: Solve $\det(\Sigma - \lambda I) = 0$ to get eigenvalues $\lambda_i$ and eigenvectors $\mathbf{v}_i$.
4. **Sort and Select**: Rank eigenvalues descendingly ($\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d$). Form projection matrix $W = [\mathbf{v}_1, \dots, \mathbf{v}_k] \in \mathbb{R}^{d \times k}$.
5. **Project**: $Y = X_c W \in \mathbb{R}^{N \times k}$.

#### 4. Advantages & Limitations
- **Advantages**: Removes multi-collinearity; compresses storage; improves model speed.
- **Limitations**: Linear transformation only; sensitive to unscaled features and outliers; components lack physical interpretability.

---

### 📝 Long-Answer Model 2: Linear Discriminant Analysis (LDA) Derivation and Fisher's Criterion

**Question**: Derive Linear Discriminant Analysis (LDA) for a two-class problem using Fisher's Linear Discriminant. Define Within-Class and Between-Class scatter matrices.

#### 1. Definition & Supervised Objective
LDA is a supervised linear dimensionality reduction algorithm that projects $d$-dimensional data onto a 1D line (or $k$-dimensional subspace) such that target classes are maximally separated.

#### 2. Derivation of Fisher's Criterion
Let dataset samples belong to Class 1 ($\mathcal{D}_1$, size $N_1$) and Class 2 ($\mathcal{D}_2$, size $N_2$).
Let $\mathbf{w}$ be the projection vector. Projected point $y_i = \mathbf{w}^T \mathbf{x}_i$.

1. **Projected Class Means**:
   $$\tilde{\mu}_1 = \frac{1}{N_1} \sum_{\mathbf{x} \in \mathcal{D}_1} \mathbf{w}^T \mathbf{x} = \mathbf{w}^T \mathbf{\mu}_1, \qquad \tilde{\mu}_2 = \mathbf{w}^T \mathbf{\mu}_2$$

2. **Projected Between-Class Distance**:
   $$(\tilde{\mu}_1 - \tilde{\mu}_2)^2 = (\mathbf{w}^T \mathbf{\mu}_1 - \mathbf{w}^T \mathbf{\mu}_2)^2 = \mathbf{w}^T (\mathbf{\mu}_1 - \mathbf{\mu}_2)(\mathbf{\mu}_1 - \mathbf{\mu}_2)^T \mathbf{w} = \mathbf{w}^T S_B \mathbf{w}$$
   where $S_B = (\mathbf{\mu}_1 - \mathbf{\mu}_2)(\mathbf{\mu}_1 - \mathbf{\mu}_2)^T$ is the Between-Class Scatter Matrix.

3. **Projected Within-Class Variance**:
   $$\tilde{s}_1^2 = \sum_{\mathbf{x} \in \mathcal{D}_1} (\mathbf{w}^T \mathbf{x} - \mathbf{w}^T \mathbf{\mu}_1)^2 = \mathbf{w}^T S_1 \mathbf{w}$$
   $$\tilde{s}_2^2 = \sum_{\mathbf{x} \in \mathcal{D}_2} (\mathbf{w}^T \mathbf{x} - \mathbf{w}^T \mathbf{\mu}_2)^2 = \mathbf{w}^T S_2 \mathbf{w}$$
   Total Within-Class Variance $\tilde{s}_1^2 + \tilde{s}_2^2 = \mathbf{w}^T (S_1 + S_2) \mathbf{w} = \mathbf{w}^T S_W \mathbf{w}$, where $S_W = S_1 + S_2$ is the Within-Class Scatter Matrix.

4. **Fisher's Criterion Objective Function**:
   $$J(\mathbf{w}) = \frac{(\tilde{\mu}_1 - \tilde{\mu}_2)^2}{\tilde{s}_1^2 + \tilde{s}_2^2} = \frac{\mathbf{w}^T S_B \mathbf{w}}{\mathbf{w}^T S_W \mathbf{w}}$$

5. **Solving for Optimal Projection $\mathbf{w}^*$**:
   Differentiating $J(\mathbf{w})$ with respect to $\mathbf{w}$ and setting to zero yields:
   $$S_B \mathbf{w} = \lambda S_W \mathbf{w} \implies S_W^{-1} S_B \mathbf{w} = \lambda \mathbf{w}$$
   Since $S_B \mathbf{w} = (\mathbf{\mu}_1 - \mathbf{\mu}_2)(\mathbf{\mu}_1 - \mathbf{\mu}_2)^T \mathbf{w}$ is always in the direction of $(\mathbf{\mu}_1 - \mathbf{\mu}_2)$, the optimal solution vector is:
   $$\mathbf{w}^* = S_W^{-1} (\mathbf{\mu}_1 - \mathbf{\mu}_2)$$

---

# IMPORTANT COMPARISON TABLES

---

### Table 1: Supervised vs Unsupervised vs Reinforcement Learning

| Parameter | Supervised Learning | Unsupervised Learning | Reinforcement Learning |
|---|---|---|---|
| **Input Data** | Labeled pairs $(\mathbf{x}, y)$ | Unlabeled features $\mathbf{x}$ | Environment states $s$ |
| **Output Goal** | Predict discrete/continuous target $y$ | Discover latent patterns/clusters | Maximize cumulative reward |
| **Feedback** | Direct ground truth label | Zero feedback signal | Delayed scalar reward $r_t$ |
| **Key Algorithms** | SVM, Linear Reg, Decision Trees | K-Means, PCA, DBSCAN | Q-Learning, PPO, Deep SARSA |

---

### Table 2: Normalization vs Standardization

| Parameter | Min-Max Normalization | Z-Score Standardization |
|---|---|---|
| **Formula** | $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ | $z = \frac{x - \mu}{\sigma}$ |
| **Range** | Fixed $[0, 1]$ | Unbounded $(-\infty, +\infty)$ |
| **Outlier Effect** | Severe compression of normal values | Robust to outliers |
| **Best Used In** | Distance algorithms ($k$-NN, Neural Nets) | PCA, LDA, Logistic Regression |

---

### Table 3: Feature Selection vs Feature Extraction

| Parameter | Feature Selection | Feature Extraction |
|---|---|---|
| **Output Type** | Subset of original features | Transformed linear/non-linear space |
| **Interpretability**| High (Original names retained) | Low (Principal components lack names) |
| **Information Loss**| High if unselected features matter | Low (Retains maximum combined variance) |
| **Algorithms** | SFS, SBS, Lasso Regularization | PCA, LDA, Kernel PCA, t-SNE |

---

### Table 4: PCA vs LDA

| Parameter | PCA | LDA |
|---|---|---|
| **Type** | Unsupervised | Supervised |
| **Goal** | Maximize total data variance | Maximize class separability |
| **Labels Required**| No | Yes |
| **Max Dimensions**| $k \le d$ | $k \le C - 1$ |

---

### Table 5: SFS vs SBS

| Parameter | Sequential Forward Selection (SFS) | Sequential Backward Selection (SBS) |
|---|---|---|
| **Start State** | Empty set $\emptyset$ | Full feature set $X$ |
| **Direction** | Bottom-up (Adding features) | Top-down (Removing features) |
| **Complexity** | Low initially | High initially |
| **Nesting Risk** | Cannot remove added feature | Cannot re-add deleted feature |

---

### Table 6: One-vs-Rest (OvR) vs One-vs-One (OvO)

| Parameter | One-vs-Rest (OvR) | One-vs-One (OvO) |
|---|---|---|
| **Model Count** | $K$ classifiers | $\frac{K(K-1)}{2}$ classifiers |
| **Training Size** | Full dataset $N$ | Subset per pair $\approx \frac{2N}{K}$ |
| **Class Imbalance**| High imbalance | Balanced pair distributions |

---

# IMPORTANT FORMULAS — LAST-MINUTE SHEET

---

### 1. Preprocessing Formulas
- **Min-Max Normalization**: $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$
- **Custom Bounded Scaling**: $x'' = a + \frac{(x - x_{\min})(b - a)}{x_{\max} - x_{\min}}$
- **Z-Score Standardization**: $z = \frac{x - \mu}{\sigma}$
- **Sample Mean**: $\mu = \frac{1}{N} \sum_{i=1}^N x_i$
- **Sample Standard Deviation**: $\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2}$

### 2. Regression Formulas
- **Simple Linear Model**: $\hat{y} = w_0 + w_1 x$
- **Slope ($w_1$)**: $w_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2} = \frac{N \sum x_i y_i - \sum x_i \sum y_i}{N \sum x_i^2 - (\sum x_i)^2}$
- **Intercept ($w_0$)**: $w_0 = \bar{y} - w_1 \bar{x}$
- **Sum of Squared Errors (SSE)**: $\text{SSE} = \sum (y_i - \hat{y}_i)^2$
- **Mean Squared Error (MSE)**: $\text{MSE} = \frac{1}{N} \text{SSE}$
- **Mean Absolute Error (MAE)**: $\text{MAE} = \frac{1}{N} \sum |y_i - \hat{y}_i|$

### 3. PCA & LDA Formulas
- **Covariance**: $\text{Cov}(x, y) = \frac{1}{N-1} \sum (x_i - \bar{x})(y_i - \bar{y})$
- **Covariance Matrix**: $\Sigma = \frac{1}{N-1} X_c^T X_c$
- **PCA Characteristic Equation**: $\det(\Sigma - \lambda I) = 0$
- **PCA Explained Variance Ratio**: $\text{EVR}_k = \frac{\lambda_k}{\sum \lambda_j}$
- **LDA Within-Class Scatter Matrix**: $S_W = \sum_{c=1}^C \sum_{\mathbf{x} \in \mathcal{D}_c} (\mathbf{x} - \mathbf{\mu}_c)(\mathbf{x} - \mathbf{\mu}_c)^T$
- **LDA Between-Class Scatter Matrix**: $S_B = \sum_{c=1}^C N_c (\mathbf{\mu}_c - \mathbf{\mu})(\mathbf{\mu}_c - \mathbf{\mu})^T$
- **Fisher Criterion**: $J(\mathbf{w}) = \frac{\mathbf{w}^T S_B \mathbf{w}}{\mathbf{w}^T S_W \mathbf{w}}$
- **LDA Optimal Projection Vector**: $\mathbf{w}^* = S_W^{-1}(\mathbf{\mu}_1 - \mathbf{\mu}_2)$

### 4. Theoretical ML Bounds
- **PAC Sample Complexity (Finite $\mathcal{H}$)**: $m \ge \frac{1}{\epsilon} \left( \ln |\mathcal{H}| + \ln \frac{1}{\delta} \right)$
- **Sauer's Lemma Bound**: $\Pi_{\mathcal{H}}(N) \le \sum_{i=0}^d \binom{N}{i} \le \left(\frac{e N}{d}\right)^d$
- **Generalization VC Bound**: $E_{test} \le E_{train} + \sqrt{\frac{d (\ln(2N/d) + 1) - \ln(\delta/4)}{N}}$

---

# COMMON MISTAKES TO AVOID

1. **Confusing PCA and LDA Supervision**: Never write that PCA uses class labels! PCA is strictly unsupervised.
2. **Dividing by $N$ vs $N-1$ in Covariance**: Use $N-1$ for unbiased sample covariance matrices.
3. **Rounding Down Sample Complexity $m$ in PAC**: PAC sample complexity $m$ MUST be rounded UP to ensure accuracy guarantees.
4. **Forgetting to Subtract Mean in PCA**: Always mean-center data ($X_c = X - \mu$) before calculating covariance matrix!
5. **Ignoring Outlier Sensitivity in Min-Max Normalization**: Remember that a single outlier completely skews Min-Max scaling.
6. **Writing $C$ Dimensions for LDA**: Remember that the maximum dimension for LDA output is $C - 1$, NOT $C$.

---

# UNIVERSITY ANSWER-WRITING GUIDE

### How to Structure Answers for Full Marks:

#### 1. 2-Mark Questions (2-3 lines)
- Direct definition + Formula or key point. No lengthy intro.

#### 2. 4-Mark Questions (1 page max)
- **Structure**: Definition $\rightarrow$ Formula / Equations $\rightarrow$ Short Example / Diagram $\rightarrow$ Comparison or Advantage.
- Use bullet points and bold key terms.

#### 3. 8–10 Mark Questions (2–3 pages)
- **Structure**:
  1. Introduction & Formal Definition.
  2. Intuition & ASCII / Flow Diagrams.
  3. Complete Step-by-Step Mathematical Derivation.
  4. Algorithm Pseudocode / Pipeline Steps.
  5. Solved Example / Numerical demonstration.
  6. Advantages, Limitations, and Applications.

---

# 🔥 ONE-DAY BEFORE EXAM REVISION PLAN

| Time Block | Focus Area | High-Yield Topics |
|---|---|---|
| **08:00 AM – 10:30 AM** | **Unit 2: Preprocessing & Scaling** | Min-Max Normalization, Z-Score Standardization, Missing Value Imputation, Solved Scaling Numericals. |
| **10:30 AM – 01:30 PM** | **Unit 2: PCA & LDA (High Priority)** | PCA derivation, Covariance matrix, Eigenvalues/vectors, PCA Numerical, LDA derivation, Fisher Criterion, LDA Numerical. |
| **02:30 PM – 04:30 PM** | **Unit 2: Feature Selection & Methods** | SFS vs SBS algorithms, Kernel PCA, Feature Selection vs Extraction, 4-Mark Question Bank. |
| **04:30 PM – 07:00 PM** | **Unit 1: Foundations & Regression** | Supervised vs Unsupervised, Regression Derivation & Solved Numerical, OvR vs OvO. |
| **07:30 PM – 10:00 PM** | **Unit 1: Theory & Learning Bounds** | VC Dimension (Shattering, 5 Numericals), PAC Learning Framework & Sample Complexity Numericals. |
| **10:00 PM – 11:00 PM** | **Formula & Comparison Sheet** | Rapid review of Formula Sheet, 12 Comparison Tables, and Common Mistakes. |

---

# ⚡ 30-MINUTE ULTRA-QUICK REVISION

- **Min-Max Normalization**: $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ (Range $[0, 1]$, Outlier sensitive).
- **Z-Score Standardization**: $z = \frac{x - \mu}{\sigma}$ (Zero mean $\mu=0$, unit variance $\sigma=1$).
- **PCA Goal**: Unsupervised linear variance maximization ($\Sigma \mathbf{v} = \lambda \mathbf{v}$).
- **LDA Goal**: Supervised linear class separability ($\mathbf{w}^* = S_W^{-1}(\mathbf{\mu}_1 - \mathbf{\mu}_2)$). Max output dimension $C-1$.
- **SFS**: Starts empty $\emptyset$, appends 1 feature at a time (Bottom-up).
- **SBS**: Starts full $X$, drops 1 feature at a time (Top-down).
- **PAC Sample Complexity**: $m \ge \frac{1}{\epsilon} \left(\ln|\mathcal{H}| + \ln\frac{1}{\delta}\right)$.
- **VC Dimension**: Maximum number of points shattered by $\mathcal{H}$ ($VC = d+1$ for lines in $\mathbb{R}^d$).
- **Simple Linear Regression**: $w_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, w_0 = \bar{y} - w_1 \bar{x}$.


