class LinearRegressionGDBatch:
    def __init__(self, learning_rate=0.001, num_iterations=20,initial_m=0, initial_b=0):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.initial_m = initial_m
        self.initial_b = initial_b

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.b, self.m = self.gradient_descent()

    def gradient_descent(self):
        X = self.X
        y = self.y
        initial_m = self.initial_m
        initial_b = self.initial_b
        N = float(len(X))
        iteration = 0 
        diff = 1
        convg = 10 ** -10
        while iteration <= self.num_iterations and diff > convg:
            b_gradient = 0
            m_gradient = 0
            for i in range(0, len(X)):
                x = X[i]
                b_gradient += -2/N * (y - ((initial_m * x) + initial_b))
                m_gradient += -2/N  * x *  (y - ((initial_m * x) + initial_b))
            diff = min(np.abs(b_gradient - initial_b).min(), np.abs(m_gradient - initial_m).min())
            initial_b -= self.learning_rate * b_gradient
            initial_m -= self.learning_rate * m_gradient
            iteration += 1
        return initial_b, initial_m

    def predict(self, X):
        return self.m*X + self.b
    
    def coefients(self):
        self.b, self.m = self.gradient_descent()
        return self.b, self.m
