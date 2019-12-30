from tkinter import *
from tkinter import ttk
import numpy
import xlrd 
import os
from tkinter import filedialog
import random
from pandas import DataFrame
import matplotlib.pyplot as plt
from openn import model
import xlsxwriter
import math

def main():						
	training_iter = 0
	learning_rate = 0.001
	uni_data = []
	output_Data=[]
	output_data = []
	
	Weights = []
	Bias = []
	main.values = ("Sigmoid","Tanh")
	def select_Hidden():
		select_Hidden.hidden_Stat = []
		file = filedialog.askopenfilename()
		directory = os.path.split(file)[0]
		file_name = os.path.split(file)[1]
		dir_file = directory+"/"+file_name
		path_hidden['text']=dir_file
		loc = (dir_file)
		wb = xlrd.open_workbook(loc) 
		sheet = wb.sheet_by_index(0)
		sheet.cell_value(0, 0)
		select_Hidden.hidden_Stat=(sheet.row_values(0))
		print(hidden_Stat)
	    	


	
	def select_Excel():
	    file = filedialog.askopenfilename()
	    directory = os.path.split(file)[0]
	    file_name = os.path.split(file)[1]
	    dir_file = directory+"/"+file_name
	    path_excel['text']=dir_file
	    loc = (dir_file) 
	    wb = xlrd.open_workbook(loc) 
	    sheet = wb.sheet_by_index(0) 
	    sheet.cell_value(0, 0)
	    output_Data.append(sheet.col_values(sheet.ncols-1))
	    output_data = output_Data[0]
	    for i in range(sheet.nrows):
	    	uni_data.append(sheet.row_values(i))
	    	val = uni_data[i][sheet.ncols-1]
	    	uni_data[i].remove(val)


	    
	def new_model():
	      main()
	def open_model():
	    file = filedialog.askopenfilename()
	    directory = os.path.split(file)[0]
	    file_name = os.path.split(file)[1]
	    dir_file = directory+"/"+file_name
	    root.destroy()
	    model(dir_file)


	def save_model():
	      location = filedialog.askdirectory()
	      directory_2 = os.path.split(location)[0]
	      folder_name = os.path.split(location)[1]
	      dir_folder = directory_2+"/"+folder_name
	      excel__file = directory_2+"/"+folder_name+"/"+"data.xlsx"


	def saveas_model():
	    f = filedialog.asksaveasfile(mode='a')
	    if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
	        return
	    if(select_Hidden.hidden_Stat[0]=="NONE"):
	    	for i in range(len(start_Training.weights)-1):
	    		start_Training.bias.append(0)
	    	Cars = {'Weights': start_Training.weights,
	    		'Bias': start_Training.bias}
	    	df = DataFrame(Cars, columns= ['Weights', 'Bias'])
	    	export_excel = df.to_excel (f.name, index = None, header=True)
	    else:
	    	print("M")
	    	workbook = xlsxwriter.Workbook(f.name) 
	    	worksheet = workbook.add_worksheet() 
	    	for i in range(len(start_Training.weights)): 
	    		for j in range(len(start_Training.weights[i])):
	    			worksheet.write(j, i, start_Training.weights[i][j])
	    	for i in range(len(start_Training.bias)):
	    		worksheet.write(i, len(start_Training.weights), start_Training.bias[i])
	    	workbook.close() 
	def spl(k): 
		v = ""
		for i in range(len(k)):
			if(i==len(k)-1):
				v = v + str(k[i])
			else:
				v = v+ str(k[i]) + ","
		return v 

	def start_Training():
		start_Training.weights = []
		start_Training.bias = []
		if(w.get()=="Generative Model"):
			if(activation.get()=="Linear Activation"):
				start_Training_linear()
			elif(activation.get()=="SoftPlus"):
				start_Training_linear_softplus()

			
		elif(w.get()=="Binary Classification"):
			if(activation.get()=="Sigmoid"):
				start_Training_logistic()
			elif(activation.get()=="Tanh"):
				start_Training_logistic_tanh()
		elif(w.get()=="Multiple Classification"):
			if(activation.get()=="Sigmoid"):
				start_Training_multiple()
			elif(activation.get()=="Tanh"):
				start_Training_multiple_tanh()
	def start_Training_multiple_tanh():
		mylist.delete(0,END)
		start_Training.stat = "M"
		start_Training.output = []
		start_Training.output=output_multiple(output_Data[0])
		print(start_Training.output)
		start_Training.iter_graph = []
		start_Training.error_graph = []
		start_Training.weights,start_Training.bias = generate_weights_multiple(len(uni_data[0]),len(start_Training.output[0]))
		training_iter = iter_tbox.get()
		learning_rate = float(learn_tbox.get())
		
		for i in range(int(training_iter)):
			sum_error = 0
			start_Training.iter_graph.append(i)
			for j in range(len(uni_data)):
				prediction = forward_propagation_multiple_tanh(start_Training.weights,uni_data[j],start_Training.bias)
				start_Training.weights,start_Training.bias,curr_error = backprop_multiple_tanh(prediction,start_Training.output[j],uni_data[j],start_Training.weights,start_Training.bias,learning_rate)
				sum_error+=sum(curr_error)
			average_Error = sum_error/len(uni_data)
			start_Training.error_graph.append(average_Error)
			mylist.insert(END, "Training Iteration "+str(i)+" :- Error = "+str(average_Error))
		print(start_Training.weights)
		print(start_Training.bias)
	def start_Training_multiple():
		mylist.delete(0,END)
		start_Training.stat = "M"
		start_Training.output = []
		start_Training.output=output_multiple(output_Data[0])
		print(start_Training.output)
		start_Training.iter_graph = []
		start_Training.error_graph = []
		start_Training.weights,start_Training.bias = generate_weights_multiple(len(uni_data[0]),len(start_Training.output[0]))
		training_iter = iter_tbox.get()
		learning_rate = float(learn_tbox.get())
		
		for i in range(int(training_iter)):
			sum_error = 0
			start_Training.iter_graph.append(i)
			for j in range(len(uni_data)):
				prediction = forward_propagation_multiple(start_Training.weights,uni_data[j],start_Training.bias)
				start_Training.weights,start_Training.bias,curr_error = backprop_multiple(prediction,start_Training.output[j],uni_data[j],start_Training.weights,start_Training.bias,learning_rate)
				sum_error+=sum(curr_error)
			average_Error = sum_error/len(uni_data)
			start_Training.error_graph.append(average_Error)
			mylist.insert(END, "Training Iteration "+str(i)+" :- Error = "+str(average_Error))
		print(start_Training.weights)
		print(start_Training.bias)
		
	def start_Training_logistic():
		mylist.delete(0,END)
		start_value = 0
		progressBar['maximum'] = 0
		start_Training.stat = "L"
		start_Training.coefficient = []
		start_Training.iter_graph = []
		start_Training.error_graph = []
		start_Training.weights,start_Training.bias = generate_weights(len(uni_data[0]),select_Hidden.hidden_Stat)
		training_iter = iter_tbox.get()
		learning_rate = float(learn_tbox.get())
		progressBar['maximum'] = training_iter
		for i in range(int(training_iter)):
			sum_error = 0
			start_Training.iter_graph.append(i)
			for j in range(len(uni_data)):
				prediction = forward_propagation_logistic(start_Training.weights,uni_data[j],start_Training.bias,select_Hidden.hidden_Stat)
				start_Training.weights,start_Training.bias,curr_error = backprop_logist(prediction,output_Data[0][j],uni_data[j],start_Training.weights,start_Training.bias,learning_rate,select_Hidden.hidden_Stat)
				sum_error+=curr_error
			progressBar['value']+=i+1
			average_Error = sum_error/len(uni_data)
			start_Training.error_graph.append(average_Error)
			mylist.insert(END, "Training Iteration "+str(i)+" :- Error = "+str(average_Error))
		print(start_Training.weights)
		print(start_Training.bias)
	def start_Training_logistic_tanh():
		mylist.delete(0,END)
		start_Training.stat = "LT"
		start_Training.iter_graph = []
		start_Training.error_graph = []
		start_Training.weights,start_Training.bias = generate_weights(len(uni_data[0]))
		training_iter = iter_tbox.get()
		learning_rate = float(learn_tbox.get())
		for i in range(int(training_iter)):
			sum_error = 0
			start_Training.iter_graph.append(i)
			for j in range(len(uni_data)):
				prediction = forward_propagation_tanh(start_Training.weights,uni_data[j],start_Training.bias,select_Hidden.hidden_Stat)
				mylist1.insert(END, "Training Iteration "+str(prediction))
				start_Training.weights,start_Training.bias,curr_error = backprop_tanh(prediction,output_Data[0][j],uni_data[j],start_Training.weights,start_Training.bias,learning_rate,select_Hidden.hidden_Stat)
				sum_error+=curr_error
			average_Error = sum_error/len(uni_data)
			start_Training.error_graph.append(average_Error)
			mylist.insert(END, "Training Iteration "+str(i)+" :- Error = "+str(average_Error))
		print(start_Training.weights)
		print(start_Training.bias)
	def start_Training_linear():
		mylist.delete(0,END)
		start_value = 0
		progressBar['maximum'] = 0
		start_Training.stat = "Li"
		start_Training.coefficient = []
		
		start_Training.iter_graph = []
		start_Training.error_graph = []
		start_Training.weights,start_Training.bias = generate_weights(len(uni_data[0]),select_Hidden.hidden_Stat)
		training_iter = iter_tbox.get()
		progressBar['maximum'] = training_iter
		learning_rate = float(learn_tbox.get())
		for i in range(int(training_iter)):
			sum_error = 0
			start_Training.iter_graph.append(i)
			for j in range(len(uni_data)):
				prediction = forward_propagation(start_Training.weights,uni_data[j],start_Training.bias,select_Hidden.hidden_Stat)

				start_Training.weights,start_Training.bias,curr_error = backprop(prediction,output_Data[0][j],uni_data[j],start_Training.weights,start_Training.bias,select_Hidden.hidden_Stat,int(training_iter))
				sum_error+=curr_error
			progressBar['value']+=i+1
			average_Error = sum_error/len(uni_data)
			start_Training.error_graph.append(average_Error)
			mylist.insert(END, "Training Iteration "+str(i)+" :- Error = "+str(average_Error))
		print(start_Training.weights)
		print(start_Training.bias)
	def start_Training_linear_softplus():
		mylist.delete(0,END)
		start_value = 0
		progressBar['maximum'] = 0
		start_Training.stat = "LiS"
		print("helllll,yeah")
		start_Training.coefficient = []
		start_Training.iter_graph = []
		start_Training.error_graph = []
		start_Training.weights,start_Training.bias = generate_weights(len(uni_data[0]),select_Hidden.hidden_Stat)
		training_iter = iter_tbox.get()
		progressBar['maximum'] = training_iter
		learning_rate = float(learn_tbox.get())
		for i in range(int(training_iter)):
			sum_error = 0
			start_Training.iter_graph.append(i)
			for j in range(len(uni_data)):
				prediction = forward_propagation_softplus(start_Training.weights,uni_data[j],start_Training.bias,select_Hidden.hidden_Stat)
				start_Training.weights,start_Training.bias,curr_error = backprop_softplus(prediction,output_Data[0][j],uni_data[j],start_Training.weights,start_Training.bias,select_Hidden.hidden_Stat,int(training_iter))
				sum_error+=curr_error
			progressBar['value']+=i+1
			average_Error = sum_error/len(uni_data)
			start_Training.error_graph.append(average_Error)
			mylist.insert(END, "Training Iteration "+str(i)+" :- Error = "+str(average_Error))
		print(start_Training.weights)
		print(start_Training.bias)
	def forward_propagation(mass,inval,steep,b):
		if(b[0]=="NONE"):
			value = 0
			for i in range(len(mass)):
				value+=mass[i]*inval[i]
			value+=steep[0]

			return value
		else:
			value = 0
			for i in range(len(mass[0])):
				value+=mass[0][i]*inval[i]
			value+=steep[0]
			
			cff = value
			for i in range(len(mass)-1):
				value = 0
				for j in range(len(mass[i+1])):
					value += cff*mass[i+1][j]
				start_Training.coefficient.append(cff)
				value+=steep[i]
				cff = value
			

			return value

		
			
		
		
		
	def forward_propagation_softplus(mass,inval,steep,b):
		if(b[0]=="NONE"):
			value = 0
			for i in range(len(mass)):
				value+=mass[i]*inval[i]
			value+=steep[0]
			value = numpy.log(1+numpy.exp(value))
			return value
		else:
			value = 0
			for i in range(len(mass[0])):
				value+=mass[0][i]*inval[i]
			value+=steep[0]
			cff = value
			for i in range(len(mass)-1):
				value = 0
				for j in range(len(mass[i+1])):
					value += cff*mass[i+1][j]
				start_Training.coefficient.append(cff)
				value+=steep[i]
				cff = value
			value = numpy.log(1+numpy.exp(value))


			return value

	
	def forward_propagation_logistic(mass,inval,steep,b):
		if(b[0]=="NONE"):
			value = 0
			for i in range(len(mass)):
				value+=mass[i]*inval[i]
			value+=steep[0]
			value=sigmoid(value)
			return value
		else:
			value = 0
			print(mass)
			print(inval)
			for i in range(len(mass[0])):
				print(mass[0][i])
				print(inval[i])
				value+=mass[0][i]*inval[i]
			value+=steep[0]
			cff = value
			for i in range(len(mass)-1):
				value = 0
				for j in range(len(mass[i+1])):
					value += cff*mass[i+1][j]
				start_Training.coefficient.append(cff)
				value+=steep[i]
				cff = value
			value=sigmoid(value)

			return value

		
	def forward_propagation_tanh(mass,inval,steep):
		value = 0
		for i in range(len(mass)):
			value+=mass[i]*inval[i]
		value+=steep[0]
		value=tanh(value)
		return value
	def forward_propagation_multiple(mass,inval,steep):
		value = empty_Arr(len(mass[0]))
		for i in range(len(inval)):
			for j in range(len(mass[0])):
				value[j]+=inval[i]*mass[0][j]
		for i in range(len(steep)):
			value[i]+=steep[i]
		for i in range(len(value)):
			value[i]=sigmoid(value[i])
		return value
	def forward_propagation_multiple_tanh(mass,inval,steep):
		value = empty_Arr(len(mass[0]))
		for i in range(len(inval)):
			for j in range(len(mass[0])):
				value[j]+=inval[i]*mass[0][j]
		for i in range(len(steep)):
			value[i]+=steep[i]
		for i in range(len(value)):
			value[i]=tanh(value[i])
		return value
	def sigmoid(x):
		return 1/(1+numpy.exp(-1*x))
	def tanh(x):
		return numpy.abs((numpy.exp(2*(x))-1)/(numpy.exp(2*(x))+1))
	def backprop(predicted,outputs,inputs,weight,biase,hidden_stat,tr):
		if(hidden_stat[0]=="NONE" or hidden_stat[0]=='NONE' ):
			error = (outputs-predicted)
			
			for i in range(len(weight)):
				weight[i]+=(2/tr)*error*learning_rate*inputs[i]
			biase[0]+=(2/tr)*error*learning_rate
			return weight,biase,error
		else:
			error = outputs-predicted
			
			for i in range(len(weight[0])):
				weight[0][i]+=(2/tr)*inputs[i]*error*learning_rate*sum(weight[1])
		
			for i in range(len(weight)-1):
				for j in range(len(weight[i+1])):
					if(i+1==len(weight)-1):
						weight[i+1][j]+=(2/tr)*start_Training.coefficient[i]*error*learning_rate
					else:
						weight[i+1][j]+=(2/tr)*sum(weight[i+1])*start_Training.coefficient[i]*error*learning_rate
			for i in range(len(biase)):
				if(i==len(biase)-1):
					biase[i]+=(2/tr)*error*learning_rate
						
				else:
					
					biase[i]+=(2/tr)*error*learning_rate*sum(weight[i+1])
					
			return weight,biase,error



		
			
		
		

		
	def backprop_softplus(predicted,outputs,inputs,weight,biase,hidden_stat,tr):
		if(hidden_stat[0]=="NONE" or hidden_stat[0]=='NONE'):
			error = (outputs-predicted)
			for i in range(len(weight)):
				weight[i]+=(2/tr)*error*learning_rate*inputs[i]*(numpy.exp(predicted)/(1+numpy.exp(predicted)))
			biase[0]+=(2/tr)*error*learning_rate*(numpy.exp(predicted)/(1+numpy.exp(predicted)))
			return weight,biase,error**2
		else:
			error = outputs-predicted
			for i in range(len(weight[0])):
				weight[0][i]+=(2/tr)*inputs[i]*error*learning_rate*(numpy.exp(predicted)/(1+numpy.exp(predicted)))*sum(weight[1])
		
			for i in range(len(weight)-1):
				for j in range(len(weight[i+1])):
					if(i+1==len(weight)-1):
						weight[i+1][j]+=(2/tr)*start_Training.coefficient[i]*error*learning_rate*(numpy.exp(predicted)/(1+numpy.exp(predicted)))
					else:
						weight[i+1][j]+=(2/tr)*sum(weight[i+1])*start_Training.coefficient[i]*error*learning_rate*(numpy.exp(predicted)/(1+numpy.exp(predicted)))
			for i in range(len(biase)):
				if(i==len(biase)-1):
					biase[i]+=(2/tr)*error*learning_rate*(numpy.exp(predicted)/(1+numpy.exp(predicted)))
						
				else:
					
					biase[i]+=(2/tr)*error*learning_rate*sum(weight[i+1])*(numpy.exp(predicted)/(1+numpy.exp(predicted)))
			return weight,biase,error



	def backprop_logist(predicted,outputs,inputs,weight,biase,lr,hidden_stat):
		if(hidden_stat[0]=="NONE" or hidden_stat[0]=='NONE'):
			error = cross_entropy(predicted,outputs)

			for i in range(len(weight)):
				weight[i]-=inputs[i]*(predicted-outputs)*lr
			biase[0]-=(predicted-outputs)*lr
			return weight,biase,error**2
		else:
			error = cross_entropy(predicted,outputs)

			for i in range(len(weight[0])):
				weight[0][i]-=inputs[i]*(predicted-outputs)*lr*sum(weight[1])
		
			for i in range(len(weight)-1):
				for j in range(len(weight[i+1])):
					if(i+1==len(weight)-1):
						weight[i+1][j]-=start_Training.coefficient[i]*(predicted-outputs)*lr
					else:
						weight[i+1][j]-=sum(weight[i+1])*start_Training.coefficient[i]*(predicted-outputs)*lr
					
			for i in range(len(biase)):
				if(i==len(biase)-1):
					biase[i]-=(predicted-outputs)*lr
						
				else:
					
					biase[i]-=(predicted-outputs)*lr*sum(weight[i+1])
					
						
				
			return weight,biase,error
	def backprop_tanh(predicted,outputs,inputs,weight,biase,lr):
		error = cross_entropy(predicted,outputs)
		for i in range(len(weight)):
			c = predicted/numpy.abs(predicted)

			b = ((2*inputs[i]*numpy.exp(2*predicted))/(numpy.exp(2*predicted)+1))
			
			weight[i]-=c*b*(1-outputs*(1+(2/(numpy.exp(2*predicted)+1))))*lr
		c = predicted/numpy.abs(predicted)
		b = ((2*numpy.exp(2*predicted))/(numpy.exp(2*predicted)+1))
		biase[0]-=c*b*(1-outputs*(1+(2/(numpy.exp(2*predicted)+1))))*lr
		return weight,biase,error
	def backprop_multiple(predicted,outputs,inputs,weight,biase,lr):
		error = []
		for i in range(len(predicted)):
			error.append(cross_entropy(predicted[i],outputs[i]))
		for i in range(len(weight)):
			for j in range(len(weight[0])):
				weight[i][j]-=inputs[i]*(predicted[j]-outputs[j])*lr
		for i in range(len(biase)):
			biase[i]-=(predicted[i]-outputs[i])*lr


		
		return weight,biase,error
	def backprop_multiple_tanh(predicted,outputs,inputs,weight,biase,lr):
		error = []
		for i in range(len(predicted)):
			error.append(cross_entropy(predicted[i],outputs[i]))
		for i in range(len(weight)):
			for j in range(len(weight[0])):
				weight[i][j]-=2*inputs[i]*numpy.exp(2*predicted[j])*(((numpy.exp(2*predicted[j])-1)*(1-outputs[j])-(2*outputs[j]))/(numpy.exp(4*predicted[j])-1))*lr
		for i in range(len(biase)):
			biase[i]-=2*numpy.exp(2*predicted[i])*(((numpy.exp(2*predicted[i])-1)*(1-outputs[i])-(2*outputs[i]))/(numpy.exp(4*predicted[i])-1))*lr


		
		return weight,biase,error
	def cross_entropy(p,o):
		return -1*(o*numpy.log(p)+((1-o)*numpy.log(1-p)))

	def generate_weights_multiple(a,b):
		cur_Weights = []
		cur_bias=[]
		for i in range(a):
			temp_weight = []
			for j in range(b):
				temp_weight.append(random.random())
			cur_Weights.append(temp_weight)
		for i in range(b):
			cur_bias.append(random.random())
		return cur_Weights,cur_bias


	def generate_weights(a,b):
		
		if(b[0]=="NONE"):
			cur_Weights = []
			cur_bias = []
			for i in range(a):
				cur_Weights.append(random.random())
			cur_bias.append(random.random())

			return cur_Weights,cur_bias
		else:
			cur_Weights = []
			cur_bias = []
			for i in range(len(b)+1):
				if(i==0):
					temp = []
					for j in range(a):
						temp.append(random.random())
					cur_Weights.append(temp)
				else:
					temp = []
					
					
					for j in range(int(b[i-1])):
						temp.append(random.random())
					cur_Weights.append(temp)
			for k in range(len(b)+1):
				cur_bias.append(random.random())
		

			return cur_Weights,cur_bias

				
			



		
		
			
		
		
		
	def plot_Graph():
		plt.plot(start_Training.error_graph, start_Training.iter_graph)
		plt.xlabel('Error')
		plt.ylabel('Iteration')
		plt.title('Error Vs Iteration')
		plt.show()
	def output_multiple(a):
		output = []
		print(a)
		for i in range(len(a)):
			temp_split = a[i].split(",")
			print(temp_split)
			ref_split = [int(y) for y in temp_split]
			output.append(ref_split)
		return output

	def empty_Arr(b):
		arr = []
		for i in range(b):
			arr.append(0)
		return arr

	def scroll():
		y_coord =hbar.get()
		root.geometry(str(root.winfo_screenwidth())+"x"+str(root.winfo_screenheight())+"+ 0x"+str(100*y_coord))
	    
		
		








		

	   
	
	root=Tk()
	sizex = root.winfo_screenwidth()
	sizey = root.winfo_screenheight()
	posx  = 100
	posy  = 100
	root.wm_geometry("%dx%d" % (sizex, sizey))
	
	menubar = Menu(root)
	filemenu = Menu(menubar, tearoff=0)
	filemenu.add_command(label="New", command=new_model)
	filemenu.add_command(label="Open", command=open_model)
	filemenu.add_command(label="Save", command=save_model)
	filemenu.add_command(label="Save as...", command=saveas_model)

	filemenu.add_separator()

	filemenu.add_command(label="Exit", command=root.quit)
	menubar.add_cascade(label="File", menu=filemenu)


	helpmenu = Menu(menubar, tearoff=0)
	helpmenu.add_command(label="Help Index", command=root.quit)
	helpmenu.add_command(label="About...", command=root.quit)
	menubar.add_cascade(label="Help", menu=helpmenu)
	select_exc=PhotoImage(file="select.png")
	excel = Button(root,image=select_exc,relief=FLAT, command=select_Excel, height=30, width=195)
	excel.pack()
	excel.place(bordermode=OUTSIDE, height=30, width=195,x=20,y=150)
	select_hidden=PhotoImage(file="hidden_layer.png")
	excel_hidden = Button(root,image=select_hidden,relief=FLAT, command=select_Hidden, height=30, width=195)
	excel_hidden.pack()
	excel_hidden.place(bordermode=OUTSIDE, height=30, width=195,x=20,y=200)
	iter_png=PhotoImage(file="iter.png")
	iter_but = Label(root,image=iter_png,relief=FLAT)
	iter_but.pack()
	iter_but.place(bordermode=OUTSIDE, height=30, width=195,x=20,y=390)
	learning_png=PhotoImage(file="rate.png")
	learning_but = Label(root,image=learning_png,relief=FLAT)
	learning_but.pack()
	learning_but.place(bordermode=OUTSIDE, height=30, width=195,x=20,y=450)
	path_excel = Label(root,anchor="w",background="white",font=("Open Sans",15))
	path_excel.pack()
	path_excel.place(bordermode=OUTSIDE, height=30, width=405,x=280,y=150)
	path_hidden = Label(root,anchor="w",background="white",font=("Open Sans",15))
	path_hidden.pack()
	path_hidden.place(bordermode=OUTSIDE, height=30, width=405,x=280,y=200)
	iter_tbox = Entry(root,relief=FLAT,font=("Open Sans",10))
	iter_tbox.config(highlightbackground="black",highlightthickness="2")
	iter_tbox.pack()
	iter_tbox.place(bordermode=OUTSIDE, height=30, width=195,x=280,y=390)
	learn_tbox = Entry(root,relief=FLAT,font=("Open Sans",10))
	learn_tbox.config(highlightbackground="black",highlightthickness="2")
	learn_tbox.pack()
	learn_tbox.place(bordermode=OUTSIDE, height=30, width=195,x=280,y=440)
	start_img=PhotoImage(file="start.png")
	start_but = Button(root,image=start_img,relief=FLAT, command=start_Training)
	start_but.pack()
	start_but.place(bordermode=OUTSIDE, height=30, width=85,x=600,y=510)
	plot_img=PhotoImage(file="plot.png")
	plot_but = Button(root,image=plot_img,relief=FLAT, command=plot_Graph)
	plot_but.pack()
	plot_but.place(bordermode=OUTSIDE, height=30, width=165,x=950,y=610)
	mylist = Listbox(root,font=("Open Sans",10) )
	mylist.pack()
	mylist.place(bordermode=OUTSIDE, height=100, width=505,x=400,y=570)
	
	scrollbar = Scrollbar(root)
	scrollbar.config(command=mylist.yview)
	scrollbar.pack( side = RIGHT, fill = Y )
	scrollbar.place(bordermode=OUTSIDE,height=100,width=20,x=900,y=570)
	mylist.config(yscrollcommand=scrollbar.set)
	variable = StringVar(root)
	variable.set("one")
	def idk():
		linear_Activation = ("Linear Activation","SoftPlus")
		Probability_Activation = ("Sigmoid")
		if(w.get()=="Generative Model"):
			main.values = linear_Activation
			activation["values"] = main.values
			print(main.values)
			
	
	
		else:
			main.values=Probability_Activation
			activation["values"] = main.values
			print(main.values)
			


		

	w = Spinbox(values=("Binary Classification","Generative Model"),font=("Open Sans",10),command=idk)
	w.pack()
	w.place(bordermode=OUTSIDE, height=30, width=195,x=280,y=260)
	w.config(background="white",relief=FLAT,width=195,highlightbackground="black",highlightthickness="2",state = "readonly",buttonbackground="white",)
	progressBar = ttk.Progressbar(root)
	progressBar.place(bordermode=OUTSIDE, height=30, width=305,x=400,y=30)

	# progressBar.config(background="white",relief=FLAT,width=305,highlightbackground="black",highlightthickness="2",state = "readonly",buttonbackground="white",)
	activation = Spinbox(values=main.values,font=("Open Sans",10))
	activation.pack()
	activation.place(bordermode=OUTSIDE, height=30, width=195,x=280,y=330)
	activation.config(background="white",relief=FLAT,width=195,highlightbackground="black",highlightthickness="2",state = "readonly",buttonbackground="white",)
	
	mod_png=PhotoImage(file="selecte_moel.png")
	mod_but = Label(root,image=mod_png,relief=FLAT)
	mod_but.pack()
	mod_but.place(bordermode=OUTSIDE, height=30, width=195,x=20,y=260)
	activation_png=PhotoImage(file="sel_activation.png")
	activation_but = Label(root,image=activation_png,relief=FLAT)
	activation_but.pack()
	activation_but.place(bordermode=OUTSIDE, height=30, width=195,x=20,y=330)

	
	
	
	
	root.config(menu=menubar)
	root.configure(background='white')
	# root.iconbitmap("c4.ico")
	root.title("Alpha")
	
	
	root.mainloop()
	


main()