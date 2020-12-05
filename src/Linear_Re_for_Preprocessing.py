import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML # for animation in colab

class LinearRegression:

  def __init__(self, lr, step,opt):
    self.lr = lr
    self.epoch_num = step
    self.opt=opt
  def mean_squared_error(self,y_hat,y):
    loss=1/2*1/y.shape[0] * np.sum ((y-y_hat)**2)
    return loss

  def optimizer(self,X,y):
    if self.opt=="batch":
        delta = np.dot(X.T, ( np.dot(X, self.weight) - y))/len(X)/2
        self.weight = self.weight - self.lr * delta
    elif self.opt=="SGD":
        for i in range(len(X)):
            delta = np.dot(X[i].reshape(9,1), ( np.dot(X[i], self.weight) - y[i]).reshape(1,1))/len(X)/2
            self.weight = self.weight - self.lr * delta
    elif self.opt=='mini-batch':
        BATCH_SIZE=5
        X_=X.reshape(-1,5,9)
        for i in range(len(X_)):
            delta = np.dot(X_[i].reshape(9,5), ( np.dot(X_[i], self.weight) - y[i]).reshape(5,1))/len(X)/2
            self.weight = self.weight - self.lr * delta
    elif self.opt=="AdaGrad":
        epsi=np.random.rand()
        di=X.shape[1]
        G_matrix=np.zeros((di,di))
        delta = np.dot(X.T, ( np.dot(X, self.weight) - y))/len(X)/2
        for i in range(X.shape[1]):
            G_matrix[i][i]+=np.sum(delta**2)
        
            self.weight = self.weight - self.lr * delta*1/np.sqrt(epsi+ G_matrix[i][i])
     

  def fit(self, X, y):
    self.n_features = X.shape[1] 
    train_size = len(X) 
    y = y.reshape([-1, 1])

    one = np.ones([train_size, 1])
    X = np.concatenate([X, one], 1)
    self.weight = np.ones([self.n_features + 1, 1]) # init
    self.train_loss = []
    # train
    for i in range(self.epoch_num):        
      loss = self.mean_squared_error(y , np.dot(X, self.weight))
      self.optimizer(X,y)   
      #if(i%5==0):       
      self.train_loss.append(loss)


  def loss_visualize(self):
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    draw, = plt.plot([], [], 'ro')
    def update(frame):
        xdata.append(frame)
        ydata.append(self.train_loss[frame])
        draw.set_data(xdata,ydata)
        return draw


    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Running Loss!')	  
    ani = animation.FuncAnimation(fig,update,frames=np.arange(0,len(self.train_loss))) 
    HTML(ani.to_html5_video())


  def predict(self, X):
    X = X.reshape(-1, self.n_features)
    one = np.ones([len(X), 1])
    X = np.concatenate([X, one], 1)
    y_hat = np.dot(X, self.weight)    
    return y_hat
  
  def print_weight(self):
    print("Weights using {} : \n{}".format(self.opt,self.weight))
