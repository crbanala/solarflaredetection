import numpy as np



weights = [];
new_candidate_weights = [];
candidate_weights = [];
X=[];
Y=[];

#function declarations
def sigmoid(s):
	return 1/(1+np.exp(-float(s)));


#function to print the accuracy
def print_accuracy():
	#for val in weights:
		#print val;
	#print("runnin print_accuracy");

	for o in range(0,len(weights)):
		sum = 0.00;
		for i in range(0,len(X)):
			arr = [];
			for j in range(0,len(candidate_weights)):
				arr.append(candidate_value(j,X[i]));
			arr = X[i]+arr;
			calc_y = dot_product(weights[o],arr);
			calc_y = sigmoid(calc_y);
			if(calc_y>0.5):
				calc_y = 1;
			else:
				calc_y = 0;
			sum = sum+ abs(Y[i][o] - calc_y)*1.0;
		error = (sum*1.0)/(num_inp*1.0);
		accuracy = (1-error)*100.0;
		print(str(o)+"  "+str(accuracy));
	#print("done print_accuracy");

	return;

def accuracy():
	#for val in weights:
		#print val;
	#print("runnin accuracy");
	accuracy = 0;
	for o in range(0,len(weights)):
		sum = 0.00;
		for i in range(0,len(X)):
			arr = [];
			for j in range(0,len(candidate_weights)):
				arr.append(candidate_value(j,X[i]));
			arr = X[i]+arr;
			calc_y = dot_product(weights[o],arr);
			calc_y = sigmoid(calc_y);
			if(calc_y>0.5):
				calc_y = 1;
			else:
				calc_y = 0;
			sum = sum+ abs(Y[i][o] - calc_y)*1.0;
		error = (sum*1.0)/(num_inp*1.0);
		accuracy += (1-error)*100.0;
	#print("done accuracy");
	accuracy = accuracy/num_out;

	return accuracy;




#logistic error function
def error(p,o):
	arr = [];
	for j in range(0,len(candidate_weights)):
		arr.append(candidate_value(j,X[i]));
	arr = X[p]+arr;
	f_w = sigmoid(dot_product(weights[o],arr));
	err = Y[i][o]*(np.log(f_w))+(1-Y[i][o])*(np.log(1-f_w));
	return err;

#intermediate function to calculate the correlation function
def E_x_y(o):
	E_x_y= 0;

	for p in range(0,len(X)):
		arr = [];
		for j in range(0,len(candidate_weights)):
			arr.append(candidate_value(j,X[p]));
		arr = X[p]+arr;
		#print(len(arr));
		#print(len(new_candidate_weights));
		Vp = dot_product(new_candidate_weights,arr);
		Vp = sigmoid(Vp);
		E_x_y = E_x_y+Vp*error(p,o);

	
	return E_x_y;

#intermediate function to calculate the correlation function
def E_x(o):
	E_x= 0;

	for p in range(0,len(X)):
		arr = [];
		for j in range(0,len(candidate_weights)):
			arr.append(candidate_value(j,X[p]));
		arr = X[p]+arr;
		Vp = dot_product(new_candidate_weights,arr);		
		Vp = sigmoid(Vp);
		E_x = E_x+Vp;

	
	return E_x;

#intermediate function to calculate the correlation function
def E_y(o):
	E_y= 0;

	for p in range(0,len(X)):
		
		E_y = E_y+error(p,o);

	
	return E_y;


#Correlation function between the error and the value of the candidate unit
def correlation(o):


	covariance = E_x_y(o) - E_x(o)*E_y(o);
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
def candidate_value(i,x):
	value = 0;
	for j in range(0,len(candidate_weights[i])):
		if(j<num_feat):
			value = value+ candidate_weights[i][j]*x[j];
		else:
			value = value+candidate_weights[i][j]*candidate_value((j-num_feat),x);
	return value;


def zeros(n):
	arr = [];
	for i in range(0,n):
		arr.append(0);
	return arr;
def add_arrays(a,b):
	arr=[];
	for i in range(0,len(a)):
		arr.append(a[i]+b[i]);
	return arr;

def dot_product(a,b):
	sum=0;
	for i in range(0,len(a)):
		sum = sum+a[i]*b[i];
	return sum;


#Read Data from Train.csv file
train_file = open('flaremodified.data2');


test_file = open('flaremodified.data1');

#Initialising matrices to store Training data set
X=[];
Y=[];

#Data set properties
num_feat = 0;
num_inp = 0;
num_out = 3;
b = 1.0;
learning_rate = 0.5;
threshold = 0.5;


#Reading the input file and Initialising the Training Data 
for line in train_file:
	arr = line.split(' ');
	arr = arr[0:len(arr)-1];
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

max_iter = 5000;
max_iter1 =5000;
for o in range(0,num_out):
	weights.append(zeros(num_feat));



#building the Neural Network cascade
candidate_weights =[];
candidate_learning_rate = 0.5;
num_candidate_units = 0;
curr_accuracy =0;
accuracy_required = 97;

while(True):





	#The learning algorithm for the currently built neural network
	#single layer perceptron update with Gradient Descent using sigmoid activation function
	var = 0;
	count = 0;
	yo = 0;
	while(True):

		for o in range(0,num_out):
			for i in range(0,num_inp):
				arr = [];
				for j in range(0,len(candidate_weights)):
					arr.append(candidate_value(j,X[i]));
				arr = X[i]+arr;
				#print len(weights[o]);
				#print len(arr);
				wt_x = dot_product(weights[o],arr);

				f_w = sigmoid(wt_x);
				fac = Y[i][o]-f_w;
				fac = (fac*learning_rate)/(num_inp*1.0);
				delta = [x*fac for x in arr];
				weights[o]=add_arrays(weights[o],delta);
				#print("jokes");
		#print the error
		print_accuracy();	
		#print(weights);
		var_temp = accuracy();
		if(var == var_temp and yo != 10):
			yo+=1;
			#print("jokes1");

		elif(yo == 10):
			yo = 0;
			#print("jokes2");
			curr_accuracy = var_temp;
			break;
		elif(count == max_iter):
			#print("jokes3");
			break;
		else:
			#print("jokes4");
			var = var_temp;
			count+= 1;
#Building the network by adding a new hidden layer
	if(curr_accuracy > accuracy_required):
		break;

	#create a new candidate unit connecting all the existing inputs and hidden units
	new_candidate_weights = zeros(num_feat+len(candidate_weights));






	#finding the weights of the candidate unit using Gradient ascent to maximise.....
	#.....correlation between Candidate value and error
	cov = 0;
	count1 = 0;
	while(True):
		temp1 = [];
		for o in range(0,len(weights)):
			E_o = E_y(o);
			sigma = signum(correlation(o));
			temp2 = [];
			for p in range(0,len(X)):

				E_p_o = error(p,o);
				arr = [];
				for j in range(0,len(candidate_weights)):
					arr.append(candidate_value(j,X[p]));
				arr = X[p]+arr;
				
				wt_v = 	dot_product(new_candidate_weights,arr);
				fw_v = sigmoid(wt_v);
				df_by_dp = fw_v*(1-fw_v);
				temp2.append(sigma*(E_p_o-E_o)*(df_by_dp));
			temp1.append(temp2);
		for i in range(0,len(new_candidate_weights)):
			ds_by_dwi=0;
			for o in range(0,len(weights)):
				for p in range(0,len(X)):
					if(i<num_feat):
						ds_by_dwi = ds_by_dwi + temp1[o][p]*new_candidate_weights[i]*X[p][i];
					else:
						ds_by_dwi = ds_by_dwi + temp1[o][p]*new_candidate_weights[i]*candidate_value((i-num_feat),X[p]);
			new_candidate_weights[i] = new_candidate_weights[i] + candidate_learning_rate*ds_by_dwi;
		covtemp= 0;
		for o in range(0,num_out):
			covtemp = covtemp+abs(correlation(o));
		if(cov == covtemp):
			break;
		elif(count1 < max_iter1):
			count1 = 0;
			break;
		else:
			cov = covtemp;
			count1+=1;

	#fixing the weights of the candidate unit and dynamically adding it to the network
	candidate_weights.append(new_candidate_weights);
	num_candidate_units = num_candidate_units+1;

	#convert the candidate unit into hidden unit
	for o in range(0,len(weights)):
		weights[o].append(0);





#Test data accuracy
for line in test_file:
	arr = line.split(' ');
	arr = arr[0:len(arr)-1];
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


print("final Test data accuracy using the ANN is : ")
print_accuracy();



