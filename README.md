# Neural_Networks

Neural networks are not a new method, the first artificial neural network was devised in 1943, but advances in computational power and speed have made them a much more viable strategy for solving complex problems over the last 5-10 years. Originally devised by mathmaticians and neuroscientists to illustrate the fundamental principles of how brains might work they lost favor in the second half of the 20th century only to surge in popularity in the 20-teens as software engineers used them to resolve mathmatically intractable problems. The application of neural networks to learning problems has been ongoing for 20 years, often to predict student behvior or to parse unstructured data such as student writing samples and provide natural sounding feedback through AI avatars.

![nn](https://github.com/ab4499/Neural_Networks/blob/master/graphs/NN.png "github")

In this simple project, we use neural network to predict whether a student is attending.

### Build a Neural Network

In the attached data sets attention1.csv and attention2.csv, there is data that describe features assocaited with webcam images of 100 students' faces as they particpate in an online discussion. The variables are:

eyes - student has their eyes open (1 = yes, 0 = no)
face.forward - student is facing the camera (1 = yes, 0 = no)
chin.up - student's chin is raised above 45 degrees (1 = yes, 0 = no)
attention - whether the student was paying attention when asked (1 = yes, 0 = no)

We will use the webcam data to build a neural net to predict whether or not a student is attending.

After loading the package and data, I can build a neural net that predicts attention based on webcam images. The command "neuralnet" sets up the model. It is composed of four basic arguments:

- A formula that describes the inputs and outputs of the neural net (attention is our output)
- The data frame that the model will use
- How many hidden layers are in our neural net
- A threshold that tells the model when to stop adjusting weights to find a better fit. If error does not change more than the threshold from one iteration to the next, the algorithm will stop (We will use 0.01, so if prediction error does not change by more than 1% from one iteration to the next the algorithm will halt)

```{r}
net <- neuralnet(attention ~ eyes + face.forward + chin.up, D1, hidden = 1, threshold = 0.01)

plot(net)
```

![net1](https://github.com/ab4499/Neural_Networks/blob/master/graphs/net1.png "github")


I have now trained a neural network! The plot shows the layers of the newtork as black nodes and edges with the calculated weights on each edge. The blue nodes and edges are called bias terms. The bias term anchors the activation function, the weights change the shape of the activation function while the bias term changes the overall position of the activation function - if I have used linear regression the bias term is like the intercept of the regression equation, it shifts the trend line up and down the y axis, while the other parameters change the angle of the line. The plot also reports the final error rate and the number of iterations ("steps") that it took to reach these weights.

What happens if I increase the number of hidden layers in the neural net? I will build a second neural net with more layers in it and determine if this improves my predictions or not. 

```{r}
net2 <- neuralnet(attention ~ eyes + face.forward + chin.up, D1, hidden=3, threshold=0.01)
plot(net2)
```

![net2](https://github.com/ab4499/Neural_Networks/blob/master/graphs/net2.png "github")


Use my preferred neural net to predict the second data set.

```{r}
D3 <- D2[,-4]
```

Now I can create predictions using your neural net
```{r}
# predict with the first neural network (1 hidden layer)
net.prediction <- neuralnet::compute(net, D3)
net.prediction$net.result
prediction1<-ifelse(net.prediction$net.result<0.5, 0, 1)
table(prediction1, D2$attention)
```

(41+53)/(41+3+3+53) -> the accuracy is 0.94

```{r}
# predict with the second neural network (3 hidden layers)
net.prediction2 <- neuralnet::compute(net2, D3)
prediction2<-ifelse(net.prediction2$net.result<0.5, 0, 1)
table(prediction2, D2$attention)
```
(41+53)/(41+3+3+53) -> the accuracy is 0.94

![predict_table](https://github.com/ab4499/Neural_Networks/blob/master/graphs/predict%20table.png "github")

### Some related questions:

1. How accurate is the neural net? 

The accurate for both of the networks are 94%. I create an accuracy check table and found out the accuracy by dividing correctly predicted number by the total number. 

2. How would I explain the model to the students whose behavior I am predicting? 

This model predicts whether they are paying attention in class (output) based on three kinds of behaviors (input - whether he/she has eyes open, whether he/she is facing the camera, and whether he/she raised above 45 degrees). 

3. This is a very simple example of a neural network. Real facial recognition is very complex though. Would a neural network be a good solution for predicting real facial movements? Why, why not? 

Neural network would be a good solution for facial recognition. Because it can recognize different edges and patterns in different layers and this is very useful/important in image recognition. When it comes to predicting facial movement, it could be harder. But if we constantly train the algorithm, and help it to correct its bad prediction, the final results could be good. 

