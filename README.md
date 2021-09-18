# AOS551_Curve-fitting

The goal is to fit a x = sin(5t) curve with a training data set randomly selected from the t=[-1, 1] domain.
1. Find the suitable learning rate for the problem so that the loss converges with iterations.
2. Try different ratio of amount of training data/amount of testing data. Note that in this particular problem you only need ~10 data points to get a good x prediction. In general you should try a few different training/test data ratio, and make sure you are in the regime where NN prediction doesn't vary much with increasing amount of training data. 
3. Test the effect of other activation function (In this case, using tanh activation function will greatly increase the # of iterations required to make the loss converge)
4. We tested the effects of fitting the NN to noisy data.
5. When we have a small amount of noisy data it is easy to overfit the data, see the following plot. As the model weights and biases update throughout different iterations, the model fits the training better and better but the model prediction of testing data gets worse and worse. 
