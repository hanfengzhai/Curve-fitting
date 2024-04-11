# AOS551_Curve-fitting

The goal is to fit a x = sin(5t) curve with a training data set randomly selected from the t=[-1, 1] domain.
1. Find the suitable learning rate for the problem so that the loss converges with iterations.
2. Try different ratio of amount of training data/amount of testing data. In this particular problem how many data points do you need to get a good NN prediction? In general you should try a few different training/test data ratio, and make sure you are in the regime where NN prediction doesn't vary much with increasing amount of training data. 
3. Test the effect of other activation function (In this case, using tanh activation function will greatly increase the # of iterations required to make the loss converge)
4. Test the effects of fitting the NN to noisy data.
5. When we have a small amount of noisy data it is easy to overfit the data, see the following plot. As the model weights and biases update throughout different iterations, the model fits the training better and better but the model prediction of testing data gets worse and worse. Find the parameters that can cause overfitting (test loss increases over iterations). The maximum iterations you can use is 20k.

![image](https://github.com/GP248CME215/Curve-fitting/assets/44380166/0cc65d7e-1932-4ff4-b5ce-ea4dd97fcc6a)

