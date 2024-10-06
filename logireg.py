class Logistic_regression:

  def __init__(self, alpha,lam):
    self.alpha=alpha
    self.lam=lam
    self.W=None

  def h(self,X):
    return np.dot(X,self.W.T)

  def S(self,X):
    return 1/(1+np.exp(-self.h(X)))

  def loss(self,y,y_aprox):#y_trainig and y de S(X)
    n=len(y)
    ep=1e-15
    y_aporx=np.clip(y_aprox,ep,1-ep)
    loss=-1/n*np.sum(y*np.log(y_aprox)+(1-y)*np.log(1-y_aprox)) + self.lam* np.sum(self.W**2)
    return loss

  def derivada(self,X,Y):
    s=self.S(X)
    return (np.dot((s - Y).T, X) / len(Y)) +2 * self.lam * self.W

  def actualizar_parametros(self,dW):
    self.W-=(self.alpha*dW)

  def fit(self,X,Y,epochs):#x_train and y_train
    n=X.shape[1]
    self.W=np.zeros(n)
    for epoch in range(epochs): 
      y_aprox=self.S(X)
      dW=self.derivada(X,Y)
      self.actualizar_parametros(dW)

  def pred(self,X):#x_test apra obtener y_pred 
    prob=self.S(X)
    return (prob >= np.median(prob)).astype(int)  #ponemos como umbral la mediana del dato de probabilidades para tener un mejor equilibrio de valores
