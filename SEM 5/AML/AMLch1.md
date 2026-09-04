# AML Chapter  1
## Types of ML
### Supervised ML : 
- It is a type of ml where labled data is given that si the imput and the output is given and the model learns from finddign patterns / mapping them.
- Algorithms : 
  - Regression 
  - SVM 
  - Decision Trees
  - Random forest


### Unsupervised Learning : 
- It is a type of ml where unlabeled data is given and the model learns from finding patterns / mapping them 
- Needs to discover hidden patterns

- Applications :
    - Clustering 
    - Market Basket Analysis
    - Customer segmentation 

- Alogrithms :
    - K means cluturing
    - hierarchial clustring
    - DBSCAN

### Semi - supervised Learning :
- Small amount of labeled data
- large amount of unlabled data 

### Reinforcement Learning :
- Learns from  trial and error 
- If correct == reward,  worng == penalty 

- Applications :
  - NPC in games
  - Robots 
  - self driving cars

### Supervised learning : Learning a class from examples 
- basically just means that we have to do classifaction using supervised leanign so something like logistic regression.

#### Classification  Process :
- Training data
- feature extraxtion
- Learning model 
- classifier model
- Prediction

### Hypothesis Class :
- A hypthesis is one possible rule / model that can be used to make predictions.
- A hypothesis class is the set of all the possible hypothesis that the ML algorithm is allowed to use / choose from.

### Shattering  :
- It means that a hypothesis class can correctly classify a given set of data points in every possible way.  
  - Here every possible way refers to all combination of the data points.
### Vapnik Chervonenkis (VC) Dimension
- It is the max number of points that an ML model can classify in any order 
- It is basically the learning capacity / complexity of the model 


- The VC dimension of a hypothesis class is the highest number of points that can be shattered by the hypothesis class.
