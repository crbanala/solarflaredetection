import numpy as np





#function declarations
def sigmoid(s):
	return 1/(1+np.exp(-float(s)));


#function to print the accuracy
def print_accuracy(weights,X,Y,candidate_weights):
	#for val in weights:
		#print val;
	for o in range(0,len(weights)):
		sum = 0.00;
		for i in range(0,len(X)):
			arr = [];
			for j in range(0,len(candidate_weights)):
				arr.append(candidate_value(j,X[i],candidate_weights));
			arr = X[i]+arr;
			calc_y = np.dot(weights[o],arr);
			calc_y = sigmoid(calc_y);
			if(calc_y>0.5):
				calc_y = 1;
			else:
				calc_y = 0;
			sum = sum+ abs(Y[i][o] - calc_y)*1.0;
		error = (sum*1.0)/(num_inp*1.0);
		accuracy = (1-error)*100.0;
		print(str(o)+"  "+str(accuracy));

	return;




#logistic error function
def error(p,o,weights,X,Y):
	f_w = sigmoid(np.dot(weights[o],X[p]));
	err = Y[i][o]*(log(f_w))+(1-Y[i][o])*(log(1-f_w));
	return err;

#intermediate function to calculate the correlation function
def E_x_y(o,new_candidate_weights,weights,X,Y):
	E_x_y= 0;

	for p in range(0,len(X)):
		Vp = np.dot(new_candidate_weights,X[p]);
		Vp = sigmoid(Vp);
		E_x_y = E_x_y+Vp*error(p,o,weights,X,Y);

	
	return E_x_y;

#intermediate function to calculate the correlation function
def E_x(o,new_candidate_weights,weights,X,Y):
	E_x= 0;

	for p in range(0,len(X)):
		Vp = np.dot(new_candidate_weights,X[p]);
		Vp = sigmoid(Vp);
		E_x = E_x+Vp;

	
	return E_x;

#intermediate function to calculate the correlation function
def E_y(o,new_candidate_weights,weights,X,Y):
	E_y= 0;

	for p in range(0,len(X)):
		E_y = E_y+error(p,o,weights,X,Y);

	
	return E_y;


#Correlation function between the error and the value of the candidate unit
def correlation(o,new_candidate_weights,weights,X,Y):


	covariance = E_x_y(o,new_candidate_weights,weights,X,Y) - E_x(o,new_candidate_weights,weights,X,Y)*E_y(o,new_candidate_weights,weights,X,Y);
	return covariance;


#sign function
def signum(s):
	if(s>0):
		s=1;
	else:
		if(s==0):
			s=0;
		else:
			s=-1;
	return s;

#recursive function to calculate the candidate value using candidate weights
def candidate_value(i,x,candidate_weights):
	value = 0;
	for j in range(0,len(candidate_weights[i])):
		if(j<num_inp):
			value = value+ candidate_weights[i][j]*x[j];
		else:
			value = value+candidate_weights[i][j]*candidate_value((j-num_inp),x,candidate_weights);
	return value;





#Read Data from Train.csv file
train_file = open('balance.dat');

#Initialising matrices to store Training data set
X=[];
Y=[];

#Data set properties
num_feat = 0;
num_inp = 0;
num_out = 3;
num_hid_layer = 0;
b = 1.0;
learning_rate = 0.1;
threshold = 0.5;


#Reading the input file and Initialising the Training Data 
for line in train_file:
	arr = line.split(',');
	num_feat = 0;
	for val in arr:
		arr[num_feat] = float(arr[num_feat]);
		num_feat = num_feat+1;
	num_feat = num_feat - num_out;
	i=0;
	for val in arr[num_feat:num_feat+3]:
		arr[num_feat+i] = int(val);
		i=i+1;

	Y.append(arr[num_feat:num_feat+3]);
	arr[0:num_feat].append(float(b));
	num_feat = num_feat+1;
	X.append(arr[0:num_feat]);
	num_inp=num_inp+1;

#print(X);
#print(Y);


#initialising all weights to zeros
weights = [];
for o in range(0,num_out):
	weights.append([0]*num_feat);



#building the Neural Network cascade
candidate_weights =[];
candidate_learning_rate = 0.1;
num_candidate_units = 0;


while(True):





	#The learning algorithm for the currently built neural network
	#single layer perceptron update with Gradient Descent using sigmoid activation function
	while(True):
		for o in range(0,num_out):
			for i in range(0,num_inp):
				arr = [];
				for j in range(0,len(candidate_weights)):
					arr.append(candidate_value(j,X[i],candidate_weights));
				arr = X[i]+arr;
				wt_x = np.dot(weights[o],arr);
				f_w = sigmoid(wt_x);
				fac = Y[i][o]-f_w;
				fac = (fac*learning_rate)/(num_inp*1.0);
				delta = [x*fac for x in arr];
				weights[o]=np.add(weights[o],delta);
		#print the error
		print_accuracy(weights,X,Y,candidate_weights);	

#Building the network by adding a new hidden layer


	#create a new candidate unit connecting all the existing inputs and hidden units
	new_candidate_weights = [];
	for i in range(0,(num_feat+num_candidate_units)):
		new_candidate_weights.append([0]*(num_feat+num_candidate_units));





	#finding the weights of the candidate unit using Gradient ascent to maximise.....
	#.....correlation between Candidate value and error
	while(True):
		for i in range(0,len(new_candidate_weights)):

			ds_by_dwi = 0;

			for o in range(0,len(weights)):
				E_o = E_x(o,new_candidate_weights,weights,X,Y);
				sigma = signum(covariance(o,new_candidate_weights,weights,X,Y));

				for p in range(0,len(X)):

					E_p_o = error(p,o,weights,X,Y);
					I_i_p = new_candidate_weights[i]*X[p][i];
					wt_v = 	np.dot(new_candidate_weights,X[p]);
					fw_v = sigmoid(wt_v);
					df_by_dp = fw_v*(1-fw_v);
					ds_by_dwi = ds_by_dwi + (sigma*(E_p_o-E_o)*(df_by_dp)*I_i_p);
			new_candidate_weights[i] = new_candidate_weights[i] + candidate_learning_rate*ds_by_dwi;

	#fixing the weights of the candidate unit and dynamically adding it to the network
	candidate_weights.append(new_candidate_weights);
	num_candidate_units = num_candidate_units+1;

	#convert the candidate unit into hidden unit
	for o in range(0,len(weights)):
		weights[o].append([0]);



















