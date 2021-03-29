class LogisticRegression:
    
    '''
    datax = Independent Variable
    datay = Dependent Variable
    alpha = Learning rate
    
    train(iterations=10)
    ----------------------------------------------------
    Train method takes one argument as input i.e itertions 
    default = 10
    
    predict(x)
    -----------------------------------------------------
    predict method takes data as argument, the data has to 
    be enclosed either as a 2D array or 2D list
    
    example:
    
    obj.predict([[10]])
    
    output: [some_value]
    '''
    
    def __init__(self,datax,datay,alpha):
        self.x = np.hstack((np.array([[1 for i in range(len(datax))]]).transpose(),datax))
        self.y = datay
        self.alpha = alpha
        self.cols = datax.shape[1]
        self.theta = np.array([0 for i in range(self.cols+1)])
        
    def update_train(self,x):
        y = 0
        for i in range(len(x)):
            y+=(1/(1+np.exp(-np.dot(self.theta,x))))
        return y
    
    def predict(self,x):
        y =[]
        x = np.hstack((np.array([[1 for i in range(len(x))]]).transpose(),x))
        for i in range(len(x)):
            y.append(1/(1+np.exp(-np.dot(self.theta,x[i]))))
        return y
        
        
    def update_sgd(self,k):
        sum_error =0
        for i in range(len(self.x)):
            error = self.update_train(self.x[i])-self.y[i]
            self.theta = self.theta - self.alpha*error*self.x[i]
            sum_error+=error
        print(f'{k}: {sum_error**2} {error}')
    
    def  train(self, iterations=10):
        for i in range(iterations):
            self.update_sgd(i)
    
    def __repr__(self):
        return f'{self.theta}'
        
        
