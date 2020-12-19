'''
    File name: ML.py
    Author: Kunal Khade
    Date created: 9/20/2020
    Date last modified: 9/24/2020 - Perceptron Model Done
                       10/30/2020 - Linear Regression 2D
    Python Version: 3.7

'''

import math
import numpy as np
from numpy.random import seed
from numpy.random import rand
from numpy.random import randint
from numpy import mean
from numpy import median
from numpy import percentile
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
seed(1)

############################### Section-F Support Vector Machine #################################################################################
class SVM:
    def __init__(self):
        #self.visualization = visualization
        self.colors = {1:'r',-1:'b'}        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        self.SVM_fit = self.SVM_fit
        self.data_handle = self.data_handle
        self.predict = self.predict
        self.visualize = self.visualize

    def data_handle(self, X, Y):
    	self.data_dict = {-1:np.array(X),1:np.array(Y)}

    def SVM_fit(self):
        #train with data
        opt_dict = {}
        
        transforms = [[1,1],[-1,1],[-1,-1],[1,-1]]
        
        all_data = np.array([])
        for yi in self.data_dict:
            all_data = np.append(all_data,self.data_dict[yi])
                    
        self.max_feature_value = max(all_data)         
        self.min_feature_value = min(all_data)
        all_data = None
        
        #with smaller steps our margins and db will be more precise
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      #point of expense
                      self.max_feature_value * 0.001,]
        
        #extremly expensise
        b_range_multiple = 5
        #we dont need to take as small step as w
        b_multiple = 5
        
        latest_optimum = self.max_feature_value*10
        
        """
        objective is to satisfy yi(x.w)+b>=1 for all training dataset such that ||w|| is minimum
        for this we will start with random w, and try to satisfy it with making b bigger and bigger
        """
        #making step smaller and smaller to get precise value
        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            
            #we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1*self.max_feature_value*b_range_multiple,
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        
                        #weakest link in SVM fundamentally
                        #SMO attempts to fix this a bit
                        for i in self.data_dict:
                            for xi in self.data_dict[i]:
                                yi=i
                                if not yi*(np.dot(w_t,xi)+b)>=1:
                                    found_option=False
                        if found_option:
                            """
                            all points in dataset satisfy y(w.x)+b>=1 for this cuurent w_t, b
                            then put w,b in dict with ||w|| as key
                            """
                            opt_dict[np.linalg.norm(w_t)]=[w_t,b]
                
                #after w[0] or w[1]<0 then values of w starts repeating itself because of transformation
                #Think about it, it is easy
                #print(w,len(opt_dict)) Try printing to understand
                if w[0]<0:
                    optimized=True
                    print("optimized a step")
                else:
                    w = w-step
                    
            # sorting ||w|| to put the smallest ||w|| at poition 0 
            norms = sorted([n for n in opt_dict])
            #optimal values of w,b
            opt_choice = opt_dict[norms[0]]

            self.w=opt_choice[0]
            self.b=opt_choice[1]
            
            #start with new latest_optimum (initial values for w)
            latest_optimum = opt_choice[0][0]+step*2
            #visualize()
    
    def predict(self,features):
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification!=0 and self.visualization:
            self.ax.scatter(features[0],features[1],s=200,marker='*',c=self.colors[classification])
        return (classification,np.dot(np.array(features),self.w)+self.b)
    
    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,c=self.colors[i]) for x in self.data_dict[i]] for i in self.data_dict]
        
        # hyperplane = x.w+b (actually its a line)
        # v = x0.w0+x1.w1+b -> x1 = (v-w[0].x[0]-b)/w1
        #psv = 1     psv line ->  x.w+b = 1a small value of b we will increase it later
        #nsv = -1    nsv line ->  x.w+b = -1
        # dec = 0    db line  ->  x.w+b = 0
        def hyperplane(x,w,b,v):
            #returns a x2 value on line when given x1
            return (-w[0]*x-b+v)/w[1]
       
        hyp_x_min= self.min_feature_value*0.9
        hyp_x_max = self.max_feature_value*1.1
        # (w.x+b)=1
        # positive support vector hyperplane
        pav1 = hyperplane(hyp_x_min,self.w,self.b,1)
        pav2 = hyperplane(hyp_x_max,self.w,self.b,1)
        self.ax.plot([hyp_x_min,hyp_x_max],[pav1,pav2],'k')
        
        # (w.x+b)=-1
        # negative support vector hyperplane
        nav1 = hyperplane(hyp_x_min,self.w,self.b,-1)
        nav2 = hyperplane(hyp_x_max,self.w,self.b,-1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nav1,nav2],'k')
        
        # (w.x+b)=0
        # db support vector hyperplane
        db1 = hyperplane(hyp_x_min,self.w,self.b,0)
        db2 = hyperplane(hyp_x_max,self.w,self.b,0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2],'y--')

        plt.xlabel('Sepal_Length')
        plt.ylabel('Sepal_Width')
        plt.title('Support Vector Machine')
        #print(hyp_x_min, hyp_x_max, db1, db2)
        plt.show()

############################### Section-A Perceptron Model Basics #####################################################

class Perceptron:
	error = 0
	def __init__(self, learning_rate, iterations):
		#Initialize Instance variables and function
		self.lr = learning_rate
		self.iterations = iterations
		self.active = self.step_input
		self.weights = None
		self.bias = None
		self.error_Val = None

	def fit(self, X, y):
		#Fit method for training data (X) and respected output(y)
		#Training module for perceptron
		n_samples, n_features = X.shape
		self.weights = np.zeros(n_features)
		self.bias = 0
		y_ = np.array([1 if i > 0 else 0 for i in y])
		for _ in range(self.iterations):
			for idx, x_i in enumerate(X):
				lin_out = np.dot(x_i, self.weights)+self.bias
				y_predicted = self.active(lin_out)
				update = self.lr * (y_[idx] - y_predicted)
				self.error_Val =+ abs(lin_out*lin_out)
				self.weights += update * x_i
				self.bias += update
			self.error = np.append(self.error, int(self.error_Val)/n_samples)


	def predict(self, X):
		#Predict the resultant data with respect to training dataset
		lin_out = np.dot(X, self.weights) + self.bias
		y_predicted = self.active(lin_out)
		return y_predicted

	def step_input(self, x):
		#Convert data (x) into -1 and 1
		return np.where(x>=0, 1, -1)

	def net_input(self, X):
		#Display function for net_input
		lin_out = np.dot(X, self.weights) + self.bias
		print(lin_out)

	def plot_decision_regions(self, X, y, classifier, resolution=0.02):
		#Convert complete training data into points
		#plot points on 2d plane
		#Setup marker generator and color map
		markers = ('s', 'x', 'o', '^', 'v')
		colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
		cmap = ListedColormap(colors[:len(np.unique(y))])
		# plot the decision surface
		x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
		x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
		xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
		Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
		Z = Z.reshape(xx1.shape)
		plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
		plt.xlim(xx1.min(), xx1.max())
		plt.ylim(xx2.min(), xx2.max())
		
		# plot class samples
		for idx, cl in enumerate(np.unique(y)):
			plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
		#Display Result
		plt.xlabel('sepal length')
		plt.ylabel('petal length')
		plt.title('Perceptron Model Classifier')
		plt.legend(loc='upper left')
		plt.show()
############################### Section-B Linear Regression #####################################################

class Linear_Regression:
	#Initialize Instance variables and function
	def __init__(self, LR_learning_rate, LR_Iteration):
		self.LR_learning_rate = LR_learning_rate
		self.LR_Iteration = LR_Iteration
		self.init_b = 0
		self.init_m = 0
		self.compute_error = self.compute_error
		self.gradient_descent_runner = self.gradient_descent_runner
		self.step_gradient_calculate = self.step_gradient_calculate
		self.plot_graph = self.plot_graph 
		self.plot_gradient_Curve = self.plot_gradient_Curve
		self.X = 0
		self.Y = 0

	def plot_graph(self, points_xy, m, b):
	    #Display Final Output with Points and Regression Line
	    m_trim = np.round(m, 2)
	    b_trim = np.round(b, 2)
	   #print(points_xy[0][0], points_xy[0][1])
	    for i in range(len(points_xy)):
	    	self.X = np.append(self.X, points_xy[i][0])
	    	self.Y = np.append(self.Y, points_xy[i][1])
	    slope = m*self.X[1:] + b
	    plt.plot(self.X[1:], slope, '-r', label='Y='+str(m_trim)+'X+'+str(b_trim))#Y = mx+b
	    plt.scatter(self.X[1:], self.Y[1:], alpha=0.8)
	    plt.xlabel('sepal_width')
	    plt.ylabel('sepal_length')
	    plt.title('Plot of Regression Line')
	    plt.legend()
	    plt.show()

	def plot_gradient_Curve(self, grad_points, iteration, m_val):
	    #Display Gradient Curve with points
	    updated_m = np.round(m_val, 2)
	    update_grad_val = np.round(grad_points[1:], 1)
	    argmin = min(float(sub) for sub in update_grad_val)
	    max_step = np.where(update_grad_val == argmin)
	    max_step_value = max_step[0][0]
	    Random_Points = np.linspace(0,0.1,max_step_value)
	    #print(len(grad_points[1:max_step_value+1]),len(Random_Points))
	    plt.scatter(Random_Points, grad_points[1:max_step_value+1], alpha=0.8, label='Cost')
	    #print(Random_Points, update_grad_val, argmin, max_step, max_step_value, len(grad_points[1:max_step_value]),len(Random_Points)) #res_min = min(float(sub) for sub in test_list)
	    plt.title('Plot of Gradient Curve, Step = '+ str(max_step_value) ) #string_value = str(float_value)
	    plt.xlabel('Points')
	    plt.ylabel('Cost Function')
	    plt.legend()
	    plt.show()


	def compute_error(self, b, m, points):
	    #Calculate Errors in the start and at the end
	    #return average of all error points
	    totalError = 0
	    for i in range(0, len(points)):
	        x = points[i, 0]
	        y = points[i, 1]
	        totalError += (y - (m * x + b)) ** 2
	    return totalError / float(len(points))

	def step_gradient_calculate(self, b_current, m_current, points, learningRate):
	    #Calculate Gradient 
	    b_gradient = 0
	    m_gradient = 0
	    N = float(len(points))
	    for i in range(0, len(points)):
	        x = points[i, 0]
	        y = points[i, 1]
	        b_gradient += -(1/N) * (y - ((m_current * x) + b_current))
	        m_gradient += -(1/N) * x * (y - ((m_current * x) + b_current))
	    new_b = b_current - (learningRate * b_gradient)
	    new_m = m_current - (learningRate * m_gradient)
	    return [new_b, new_m]

	def gradient_descent_runner(self, points, starting_b, starting_m, LR_learning_rate, num_iterations):
	    #Calculate Steps for gradient 
	    b = starting_b
	    m = starting_m
	    gradient_plot = 0
	    for i in range(num_iterations):
	        b, m = self.step_gradient_calculate(b, m, points, LR_learning_rate)
	        gradient_parameter = self.compute_error(b, m, points)
	        gradient_plot = np.append(gradient_plot, gradient_parameter)
	    return [b, m, gradient_plot]

	def final_run(self, points):
	    #Initialize starting parameters, Display all parameters, Calculate Gradient for Line 
	    self.points = points
	    print("Initial Parameters: b = ", self.init_b,", m = ", self.init_m ,", error = ",self.compute_error(self.init_b, self.init_m, self.points), ", iterations = ", self.LR_Iteration)
	    print("After Calculation")
	    [b, m, self.gradient_plot_update] = self.gradient_descent_runner(self.points, self.init_b, self.init_m, self.LR_learning_rate, self.LR_Iteration)
	    print("b = ", b,", m = ", m, ", error = ",self.compute_error(b, m, points))
	    self.plot_graph(points, m, b)
	    self.plot_gradient_Curve(self.gradient_plot_update, self.LR_Iteration, m)



############################### Section-C Log-Regression #############################################################

class Logistic_Regression:
	
	def __init__(self, alpha, iterations):
		#Initialize Instance variables and function
		self.alpha = alpha
		self.num_iters = iterations
		self.theta = [0,0]
		self.Sigmoid = self.Sigmoid
		self.Gradient_Descent = self.Gradient_Descent
		self.Hypothesis = self.Hypothesis
		self.Cost_Function = self.Cost_Function
		self.Cost_Function_Derivative = self.Cost_Function_Derivative  


	def Sigmoid(self, z):
		G_of_Z = float(1.0 / float((1.0 + math.exp(-1.0*z))))
		return G_of_Z 

	##The hypothesis is the linear combination of all the known factors x[i] and their current estimated coefficients theta[i] 
	##This hypothesis will be used to calculate each instance of the Cost Function
	def Hypothesis(self, theta, x):
		z = 0
		for i in range(len(theta)):
			z += x[i]*theta[i]
		return self.Sigmoid(z)

	##For each member of the dataset, the result (Y) determines which variation of the cost function is used
	##The Y = 0 cost function punishes high probability estimations, and the Y = 1 it punishes low scores
	##The "punishment" makes the change in the gradient of ThetaCurrent - Average(CostFunction(Dataset)) greater
	def Cost_Function(self, X,Y,theta,m):
		sumOfErrors = 0
		for i in range(m):
			xi = X[i]
			hi = self.Hypothesis(theta,xi)
			if Y[i] == 1:
				error = Y[i] * math.log(hi)
			elif Y[i] == 0:
				error = (1-Y[i]) * math.log(1-hi)
			sumOfErrors += error
		const = -1/m
		J = const * sumOfErrors
		#print ('cost is ', J )
		return J

	##This function creates the gradient component for each Theta value 
	##The gradient is the partial derivative by Theta of the current value of theta minus 
	##a "learning speed factor aplha" times the average of all the cost functions for that theta
	##For each Theta there is a cost function calculated for each member of the dataset
	def Cost_Function_Derivative(self, X,Y,theta,j,m,alpha):
		sumErrors = 0
		for i in range(m):
			xi = X[i]
			xij = xi[j]
			hi = self.Hypothesis(self.theta,X[i])
			error = (hi - Y[i])*xij
			sumErrors += error
		m = len(Y)
		constant = float(self.alpha)/float(m)
		J = constant * sumErrors
		return J

	##For each theta, the partial differential 
	##The gradient, or vector from the current point in Theta-space (each theta value is its own dimension) to the more accurate point, 
	##is the vector with each dimensional component being the partial differential for each theta value
	def Gradient_Descent(self, X,Y,theta,m,alpha):
		new_theta = []
		constant = alpha/m
		for j in range(len(theta)):
			CFDerivative = self.Cost_Function_Derivative(X,Y,self.theta,j,m,self.alpha)
			new_theta_value = theta[j] - CFDerivative
			new_theta.append(new_theta_value)
		return new_theta

	##The high level function for the LR algorithm which, for a number of steps (num_iters) finds gradients which take 
	##the Theta values (coefficients of known factors) from an estimation closer (new_theta) to their "optimum estimation" which is the
	##set of values best representing the system in a linear combination model
	def Logistic_Regression(self, X,Y,X_t,Y_t,X_tt,Y_tt):
		self.X_t = X_t
		self.Y_t = Y_t 
		self.X_tt = X_tt
		self.Y_tt = Y_tt 
		m = len(Y)
		for x in range(self.num_iters):
			new_theta = self.Gradient_Descent(X,Y,self.theta,m,self.alpha)
			theta = new_theta
			if x % 100 == 0:
				#here the cost function is used to present the final hypothesis of the model in the same form for each gradient-step iteration
				self.Cost_Function(X,Y,theta,m)
				print ('theta ', theta)	
				print ('cost is ', self.Cost_Function(X,Y,theta,m))
		self.Score_Update(theta, X_t, Y_t, X_tt, Y_tt)

	##This method compares the accuracy of the model generated by the scikit library with the model generated by this implementation
	def Score_Update(self, theta, X_test, Y_test, X_total, Y_total):
	    score = 0
	    #first scikit LR is tested for each independent var in the dataset and its prediction is compared against the dependent var
	    #if the prediction is the same as the dataset measured value it counts as a point for thie scikit version of LR

	    length = len(X_test)
	    for i in range(length):
	        prediction = round(self.Hypothesis(X_test[i],theta))
	        answer = Y_test[i]
	        if prediction == answer:
	            score += 1
	    #the same process is repeated for the implementation from this module and the scores compared to find the higher match-rate
	    my_score = float(score) / float(length)
	    pos = np.where(Y_total == 1)
	    neg = np.where(Y_total == 0)
	    plt.scatter(X_total[pos, 0], X_total[pos, 1], marker='o', c='b')
	    plt.scatter(X_total[neg, 0], X_total[neg, 1], marker='x', c='r')
	    plt.xlabel('Sepal_Length')
	    plt.ylabel('Sepal_Width')
	    plt.legend(['Virginica', 'Setosa'])
	    plt.title('Logistic Regression')
	    plt.show()
############################### Section-D Decision Stump ################################################################
class Decision_Stump:
	def __init__(self):
		self.D = np.mat(np.ones((100,1))/5)
		self.stumpClassify = self.stumpClassify
		self.buildStump = self.buildStump

	def stumpClassify(self,dataMatrix,dimen,threshVal,threshIneq):
	    retArray = np.ones((np.shape(dataMatrix)[0],1))
	    if threshIneq == 'lt':
	        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
	    else:
	        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
	    return retArray

	def buildStump(self, dataArr,classLabels):
	    dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).T
	    m,n = np.shape(dataMatrix)
	    numSteps = 10.0; bestStump = {}; bestClasEst = np.mat(np.zeros((m,1)))
	    minError = np.inf     
	    for i in range(n):
	        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
	        stepSize = (rangeMax-rangeMin)/numSteps
	        for j in range(-1,int(numSteps)+1):
	            for inequal in ['lt', 'gt']: 
	                threshVal = (rangeMin + float(j) * stepSize)
	                predictedVals = self.stumpClassify(dataMatrix,i,threshVal,inequal)
	                errArr = np.mat(np.ones((m,1)))
	                errArr[predictedVals == labelMat] = 0
	                weightedError = self.D.T*errArr  
	                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
	                if weightedError < minError:
	                    minError = weightedError
	                    bestClasEst = predictedVals.copy()
	                    bestStump['dim'] = i
	                    bestStump['thresh'] = threshVal
	                    bestStump['ineq'] = inequal
	    return bestStump,minError,bestClasEst
############################### Section-E Intervals #################################################################################

class Intervals:
	def __init__(self):
		self.scores = list()

	def intervals(self, X):
		for _ in range(100):
			# bootstrap sample
			indices = randint(0, 50, 100)
			sample = X[indices]
			# calculate and store statistic
			statistic = mean(sample)
			self.scores.append(statistic)
		print('50th percentile (median) = %.3f' % median(self.scores))
		# calculate 95% confidence intervals (100 - alpha)
		alpha = 5.0
		# calculate lower percentile (e.g. 2.5)
		lower_p = alpha / 2.0
		# retrieve observation at lower percentile
		lower = max(0.0, percentile(self.scores, lower_p))
		print('%.1fth percentile = %.3f' % (lower_p, lower))
		# calculate upper percentile (e.g. 97.5)
		upper_p = (100 - alpha) + (alpha / 2.0)
		# retrieve observation at upper percentile
		upper = min(1.0, percentile(self.scores, upper_p))
		print('%.1fth percentile = %.3f' % (upper_p, upper))


############################### Section-F Knn ################################################################

class knn:
	def __init__(self):
		self.test = 2
		self.scores = list()
		self.Complte_Train_Date = self.Complte_Train_Date
		self.EU_Distance = self.EU_Distance
		self.Get_Neighbors = self.Get_Neighbors
		self.knn_process = self.knn_process

	def Complte_Train_Date(self, X_setosa_train, X_Versicolor_train, Y_setosa_train, Y_Versicolor_train):
	    X_Update = np.concatenate((X_setosa_train, X_Versicolor_train),axis=0)
	    Y_Update = np.concatenate((Y_setosa_train, Y_Versicolor_train),axis=0)
	    return X_Update, Y_Update

	def EU_Distance(self, X_Train, Y_Train, X_Test, Y_Test):
	    Current_distance = 0.0;
	    for i in range(len(X_Train-1)):
	        distance = np.sqrt( (X_Test-X_Train[i])**2 + (Y_Test-Y_Train[i])**2 )
	        Current_distance = np.append(Current_distance, [distance])

	    return Current_distance
	def Get_Neighbors(self, EU_Distance_Train, Num_Neighbors):
	    for i in range(Num_Neighbors):
	        Smallest_Values = np.min(EU_Distance_Train[1:])
	        index = np.where(EU_Distance_Train == Smallest_Values)
	    return index 

	def knn_process(self, X_setosa_train, X_Versicolor_train, Y_setosa_train, Y_Versicolor_train, X_setosa_test, Y_setosa_test):
		self.X_Train, self.Y_Train = self.Complte_Train_Date(X_setosa_train, X_Versicolor_train, Y_setosa_train, Y_Versicolor_train)
		self.EU_Distance_Train = self.EU_Distance(self.X_Train, self.Y_Train, X_setosa_test[self.test], Y_setosa_test[self.test])
		self.Index_Of_Neighbors = self.Get_Neighbors(self.EU_Distance_Train ,1)
		self.Clean_Index =  self.Index_Of_Neighbors[0][0] - 1
		if self.Index_Of_Neighbors[0][0] == 0:
		    self.Clean_Index = self.Index_Of_Neighbors[0][1] - 1
		    plt.plot([X_setosa_test[self.test], self.X_Train[self.Clean_Index]], [Y_setosa_test[self.test], self.Y_Train[self.Clean_Index]], marker = 'o',  color='g') 
		else:
		    plt.plot([X_setosa_test[self.test], self.X_Train[self.Clean_Index]], [Y_setosa_test[self.test], self.Y_Train[self.Clean_Index]], marker = 'o')

		print("Coordinates of Neighbour - ",self.Index_Of_Neighbors[0])
		print("Index - ",self.Clean_Index)
		#print(self.X_Train[self.Clean_Index], self.Y_Train[self.Clean_Index], X_setosa_test[self.test], Y_setosa_test[self.test])
		#print(self.EU_Distance_Train[1:])

		plt.scatter(X_setosa_train, Y_setosa_train, color='b', alpha=0.8)
		plt.scatter(X_Versicolor_train, Y_Versicolor_train, color='r', alpha=0.8)
		plt.scatter(X_setosa_test[self.test], Y_setosa_test[self.test], color='y', alpha=0.8) 
		plt.title('1-Nearest Neighbor')
		plt.xlabel('sepal_width')
		plt.ylabel('sepal_length')
		plt.legend()
		plt.show()
	

