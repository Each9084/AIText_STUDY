# Introduction

This worksheet focuses on Neural Networks. You will:

本工作表重点关注神经网络。你会：

- Implement your own version of a Single Layer Perceptron (SLP) to ensure you understand the details of how it works and compare it with the implementation available in `scikit-learn` to test and validate your solution.

  实现您自己的单层感知器 (SLP) 版本，以确保您了解其工作原理的详细信息，并将其与 `scikit-learn` 中提供的实现进行比较，以测试和验证您的解决方案。

- Use `scikit-learn`'s implementation of Multi-Layer Perceptrons (MLP) for both classification and regression tasks, exploring how to configure and optimise these models.

​		使用 `scikit-learn` 的多层感知器 (MLP) 实现来执行分类和回归任务，探索如何配置和优化这些模型。

This is a reasonably long and difficult worksheet, but, hopefully, an interesting one. Try your best at it and don't worry if you don't get it all done. We will be posting the solutions and you can always ask about it in a different lab in future weeks.

这是一份相当长且困难的工作表，但希望它是一份有趣的工作表。尽力而为，如果没有完成也不要担心。我们将发布解决方案，您可以在未来几周内随时在不同的实验室询问。



**Note**: This is a challenging worksheet, and you might not finish all tasks during the lab. However, it is designed to be engaging, so do as much as you can. Remember that the solutions will be made available.
注意：这是一个具有挑战性的工作表，您可能无法在实验期间完成所有任务。然而，它的设计是为了吸引人，所以尽可能多地去做。请记住，我们将提供解决方案。



# 0. Preliminaries

We firstly import NumPy and matplotlib as we will be using these throughout the worksheet. We use a function %matplotlib inline to display plots in the worksheet.
我们首先导入 NumPy 和 matplotlib，因为我们将在整个工作表中使用它们。我们使用函数 %matplotlib inline 在工作表中显示绘图。



```python
#TODO: import NumPy and matplotlib here
import numpy as np
import matplotlib as plt

%matplotlib inline
```



# 1. Single Layer Perceptron

In this question, we will use a single layer perceptron from `sklearn` to make predictions on the **breast cancer dataset**. This is a classification problem where the aim is to classify instances as either malignant or benign based on 30 features, each representing various characteristics present in the images.
在本问题中，我们将使用 `sklearn` 中的单层感知器对乳腺癌数据集进行预测。这是一个分类问题，其目的是根据 30 个特征将实例分类为恶性或良性，每个特征代表图像中存在的各种特征。

In this question, you will:
在这个问题中，你将：
(a) Download the dataset from `sklearn` and store the data and targets in suitable variables.
	 从 `sklearn` 下载数据集并将数据和目标存储在合适的变量中。

(b) Separate your data into a training and test split.
	 将数据分为训练数据和测试数据。

(c) (Optional) Write your own function to implement Single Layer Perceptron.
	（可选）编写您自己的函数来实现单层感知器。

(d) Train a neural network classifier on the training data using the implementation from `sklearn` (`Perceptron`).
	 使用 `sklearn` ( `Perceptron` ) 的实现在训练数据上训练神经网络分类器。

(e) Evaluate the performance of both models on the test data using appropriate metrics (e.g., accuracy, precision).
	  使用适当的指标（例如准确度、精度）评估两个模型在测试数据上的性能。

(f) Plot the confusion matrix to visualise the performance of your model.
	 绘制混淆矩阵以可视化模型的性能。

## Part (a) 

Import the package `datasets` from `sklearn` and then load the load_breast_cancer dataset (function is `load_breast_cancer()`). Save the data into a variable `X` and the targets into a variable `Y`.
从 `sklearn` 导入包 `datasets` ，然后加载 load_breast_cancer 数据集（函数为 `load_breast_cancer()` ）。将数据保存到变量 `X` 中，将目标保存到变量 `Y` 中。

Take a look at the data in `X`. How many datapoints are there? How many features does each datapoint have? (Hint: use `np.shape`).
查看 `X` 中的数据。有多少个数据点？每个数据点有多少个特征？ （提示：使用 `np.shape` ）。

Take a look at the targets. Is this suitable for a classification algorithm or a regression algorithm?
看看目标。这适合分类算法还是回归算法？

```python
#TODO: import suitable packages, load the dataset, and save data and targets into variables X and Y
from sklearn import datasets

breast_cancer = datasets.load_breast_cancer()

X = breast_cancer.data
Y = breast_cancer.target

print("X的内容:")

#print(X)
print(X.shape)

print("Y的内容:")
#print(Y)
print(Y.shape)
```

> ```
> X的内容:
> (569, 30)
> Y的内容:
> (569,)
> ```



## Part (b) 

Use the function `train_test_split` from `sklearn.model_selection` to split your data into a training set and a held-out test set. Use a test set that is 0.2 of the original dataset. Set the parameter `random_state` to 10 to help with replication.
使用 `sklearn.model_selection` 中的函数 `train_test_split` 将数据拆分为训练集和保留测试集。使用原始数据集 0.2 的测试集。将参数 `random_state` 设置为 10 以帮助复制。

```python
# TODO: import the package train_test_split from sklearn.model_selection.
from sklearn.model_selection import train_test_split

# Split the dataset into Xtr, Xtest, Ytr, Ytest
Xtr, Xtest, Ytr, Ytest=train_test_split(X,Y,test_size=0.2,random_state=10)
```

## (Optional) Part (c) 

Recall from the lecture that a single-layer perceptron runs as follows:
回想一下讲座中的单层感知器的运行方式如下：

**Training step**:  

- For each training datapoint $(\vec{x}_i)$:  
  
  对于每个训练样本 $(\vec{x}_i)$，感知器的训练步骤如下： 
  
  
  
  - Compute the linear combination $(z = \vec{w} \cdot \vec{x}_i + b)$.  
    - **计算线性组合**:  $$  z = \vec{w} \cdot \vec{x}_i + b  $$  其中，$\vec{w}$ 是权重向量，$\vec{x}_i$ 是输入特征向量，$b$ 是偏置。
      
      这个公式表示输入特征和权重的加权和，再加上偏置项，得到一个标量值 $z$。
    
  - Pass \(z\) through the activation function (step function in this case) to get the predicted class $(y_{\text{pred}})$.
  
    - **通过激活函数获得预测值**:  $$  y_{\text{pred}} = \text{activation}(z)  $$  
  
      在这个任务中，使用的是 **阶跃函数**（step function）作为激活函数。阶跃函数的行为是：  
  
      如果 $z \geq 0$，输出 1。  
  
      如果 $z < 0$，输出 0。   
  
      这个激活函数会将线性组合的结果转换为类别预测值。  
  
  - Compute the error as $(e = y_i - y_{\text{pred}})$, where $(y_i)$ is the true label.  
  
    - **计算误差**:  $$  e = y_i - y_{\text{pred}}  $$  其中 $y_i$ 是实际的标签，$y_{\text{pred}}$ 是通过感知器得到的预测值。
  
      误差 $e$ 是实际标签与预测标签之间的差异。对于二分类问题，误差可能是 1 或 -1。
  
  - Update the weights and bias using the perceptron learning rule:  
    $[
    \vec{w} \gets \vec{w} + \eta \cdot e \cdot \vec{x}_i  
    ]  
    [
    b \gets b + \eta \cdot e
    ]$  
    Here, $(\eta)$ is the learning rate.  
    
    - **更新权重和偏置**:  根据感知器学习规则，更新权重和偏置：  
    
      $$  \vec{w} \gets \vec{w} + \eta \cdot e \cdot \vec{x}_i  $$  $$  b \gets b + \eta \cdot e  $$  其中，$\eta$ 是学习率，控制每次更新的幅度。
    
      更新过程是根据误差 $e$ 来调整权重和偏置，使得感知器的预测值更加接近实际标签。   
    
      -权重更新的过程是将误差与输入特征相乘并加到现有的权重值上。  
    
      -偏置更新的过程是将误差乘以学习率并加到现有的偏置上。

**Prediction step**:  

预测步骤 给定一个新的数据点 $\vec{x}$，预测步骤如下：

- For a given datapoint $(\vec{x})$:  
  - Compute the linear combination $(z = \vec{w} \cdot \vec{x} + b)$.  
    - **计算线性组合**:  $$  z = \vec{w} \cdot \vec{x} + b  $$  计算该数据点与当前权重和偏置的线性组合。
  - Pass $(z)$ through the step function to obtain the class prediction.
    - **通过激活函数进行预测**:  $$  y_{\text{pred}} = \text{activation}(z)  $$  将 $z$ 输入到阶跃函数中，输出最终的类别预测值。  

Write function(s) to implement the training and prediction steps. Y

```python
class SingleLayerPerceptron:
    def __init__(self, input_size, learning_rate, iterat):
        #TO DO# initialise the weights to random values and set the bias to 0
        
        # 初始化权重为随机值
        self.weights = np.random.rand(input_size)  # Random initial weights 
        self.bias = 0  # 初始化偏置为0
        self.learning_rate = learning_rate
        self.iterat = iterat

    def activation(self, z):
        #TO DO # Write a function to implement the **step activation function**. This activation function should output return 1 if z >= 0, else 0
        return np.where(z >= 0, 1, 0) 
   
    def train(self, X, y):
        for epoch in range(self.iterat):
            for i in range(X.shape[0]):
                # Calculate the linear combination
                z = np.dot(X[i], self.weights) + self.bias
                y_pred = self.activation(z)

                #TODO# Calculate error between target and predicted values
                error = y[i] - y_pred
                
                #TODO# update the weights and bias according to the above equations
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)
#Train the perceptron
input_size = Xtr.shape[1] # To pass the number of features
perceptron1 = SingleLayerPerceptron(input_size=input_size, learning_rate=0.01, iterat=20)
perceptron1.train(Xtr, Ytr)

#Test the perceptron
my_Ypred = perceptron1.predict(Xtest)

```



一些问题

#### 为什么要**初始化权重为随机值**?

在神经网络中，**初始化权重为随机值**是一个重要的步骤，特别是在训练开始时。以下是几个原因，解释了为什么通常会将权重初始化为随机值：

1.**避免对称性问题**

如果所有权重初始化为相同的值（例如全为零），在训练过程中，所有的神经元将做出相同的计算，并且会得到相同的梯度更新。因此，所有的神经元会以相同的方式学习，导致它们不会有所区别，这样就无法充分利用神经网络的多样性。

通过将权重初始化为随机值，每个神经元会在训练过程中以不同的方式调整其权重，从而保证它们学习到不同的特征，从而提高网络的表达能力和准确性。

2.**帮助网络快速收敛**

随机初始化权重能够打破网络的对称性，使得神经网络能够从不同的起点开始学习。这有助于避免陷入局部最小值或鞍点，从而加快网络的收敛速度。

3.**避免梯度消失或爆炸**

如果权重初始化得过大，计算得到的梯度可能会很大，从而导致“梯度爆炸”（梯度值过大，使得权重更新过度）。相反，如果初始化得过小，梯度可能变得非常小，导致“梯度消失”（梯度几乎为零，使得权重几乎不更新）。通过随机初始化，可以使得权重的大小处于一个适当的范围，避免这些问题。

4.**探索更广泛的解空间**

随机初始化的权重可以使神经网络探索更广泛的解空间，而不仅仅是一个固定的路径。这对于神经网络的训练至关重要，因为不同的初始权重可能导致网络在不同的区域收敛，从而提高找到最佳解的概率。

5.**增强模型的泛化能力**

权重初始化为随机值可以减少训练数据的过拟合问题，增强模型在未见过的数据上的泛化能力。通过从不同的初始点开始训练，网络能够更好地适应不同类型的数据模式。

总结

初始化权重为随机值是一种防止网络在训练过程中出现对称性问题、加速收敛、防止梯度消失和爆炸、以及增强模型泛化能力的技术。正确的权重初始化是神经网络成功训练的重要因素之一。



#### 为什么初始化偏置为0,偏置值有什么用

偏置在模型中起着“平移”作用，它允许模型在没有任何输入特征（即输入全为零）时，依然能够进行预测。简单来说，偏置提供了一个额外的自由度，使得模型不完全依赖于输入特征的加权和。

如果没有偏置（即 b=0），那么当输入特征 \(x_i\)  为零时，模型的输出就是零。这在很多情况下是不可取的，因为模型的输出无法满足不同类别的区分。因此，偏置 bbb 是为了确保模型的灵活性和表达能力。

偏置在训练过程中帮助模型做出更合理的预测。没有偏置，感知器的决策边界将仅依赖于输入特征的加权和，但很多问题的最佳决策边界是倾斜的，不经过原点。通过调整偏置，感知器可以选择一个合适的决策边界，从而提高分类性能。

虽然偏置的值不影响模型的表达能力，但初始化偏置为0是一个简单且有效的策略。常见的做法是将权重初始化为随机值，而将偏置初始化为零。这样做有以下几个原因：

- **简化初始设置**：初始偏置值为0时，模型训练时的偏置调整会直接由梯度信息来决定，而不需要额外的假设。
- **避免对初始阶段的影响**：如果初始偏置不为0，可能会对训练过程产生额外的影响，而初始化为0能够保持训练的简洁性。

在训练过程中，偏置会逐步更新，直到模型能够更好地拟合数据。

有时在某些任务中，可能会使用偏置的非零初始值，这取决于任务的性质、网络的设计以及实验结果。某些特定的任务中，初始化偏置为一个较大的正值或负值可能会有助于加快学习过程，特别是在深度学习网络中。