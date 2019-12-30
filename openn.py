from tkinter import *
import numpy
import xlrd 
import os
from tkinter import filedialog
import random
from pandas import *
import matplotlib.pyplot as plt
import xlwt
import keyboard


def model(path_file):
	master = Tk()
	sizex = master.winfo_screenwidth()
	sizey = master.winfo_screenheight()
	posx  = 100
	posy  = 100
	master.wm_geometry("%dx%d" % (sizex, sizey))
	model.values = ("Sigmoid","Tanh")
	model.path = path_file
	def newModel():
		master.destroy()
		main.main()
	def openModel():
		master.destroy()
		model("")

	
	def start():
		if(w.get()=="Generative Model"):
			if(activation.get()=="Linear Activation"):
				start_linear()
			elif(activation.get()=="SoftPlus"):
				start_softplus()
		elif(w.get()=="Binary Classification"):
			if(activation.get()=="Sigmoid"):
				start_binary()
			elif(activation.get()=="Tanh"):
				start_tanh()
		elif(w.get()=="Multiple Classification"):
			start_multiple()
			
	
	
	

		

		

	def start_tanh():
		start.uni_data= []
		start.col_Data = []
		start.weights = []
		start.bias = []
		start.output=[]
		
		loc = (get_Excel.dir_file)
		wb = xlrd.open_workbook(loc) 
		sheet = wb.sheet_by_index(0) 
		sheet.cell_value(0, 0)
		workbook = xlwt.Workbook(loc) 
		sheet3 = workbook.add_sheet("Data")
		loc2 = (model.path)
		wb2 = xlrd.open_workbook(loc2) 
		sheet2 = wb2.sheet_by_index(0) 
		sheet2.cell_value(0, 0)
		for i in range(sheet.nrows):
			start.uni_data.append(sheet.row_values(i))
		for i in range(sheet.nrows):
			start.output.append(0)

		for i in  range(sheet.ncols):
			start.col_Data.append(sheet.col_values(i))
		start.weights = sheet2.col_values(0)
		start.weights.pop(0)
		start.bias = sheet2.col_values(1)
		start.bias.pop(0)

		for i in range(len(start.uni_data)):
			curr_fact=start.uni_data[i]
			for j in range(len(start.weights)):
				print(curr_fact[j])
				print(start.weights)
				start.output[i]+=curr_fact[j]*start.weights[j]
				
				
			start.output[i]+=start.bias[0]
			start.output[i]=tanh(start.output[i])
		for i in range(len(start.col_Data)):
			curr_coldata = start.col_Data[i]
			for j in range(len(curr_coldata)):
				sheet3.write(j,i,curr_coldata[j])
		for i in range(len(start.output)):
			sheet3.write(i,sheet.ncols,start.output[i])
		for i in range(len(start.output)):
			sheet3.write(i,sheet.ncols+1,tanh_round(start.output[i]))
		os.remove(get_Excel.dir_file)
		workbook.save(get_Excel.dir_file)
	def start_binary():
		start.uni_data= []
		start.col_Data = []
		start.weights = []
		start.bias = []
		start.output=[]
		
		loc = (get_Excel.dir_file)
		wb = xlrd.open_workbook(loc) 
		sheet = wb.sheet_by_index(0) 
		sheet.cell_value(0, 0)
		workbook = xlwt.Workbook(loc) 
		sheet3 = workbook.add_sheet("Data")
		loc2 = (model.path)
		wb2 = xlrd.open_workbook(loc2) 
		sheet2 = wb2.sheet_by_index(0) 
		sheet2.cell_value(0, 0)
		for i in range(sheet.nrows):
			start.uni_data.append(sheet.row_values(i))
		for i in range(sheet.nrows):
			start.output.append(0)
		for i in  range(sheet.ncols):
			start.col_Data.append(sheet.col_values(i))
		if(sheet.nrows==2):
			start.weights = sheet2.col_values(0)
			start.weights.pop(0)
			start.bias = sheet2.col_values(1)
			start.bias.pop(0)
			for i in range(len(start.uni_data)):
				curr_fact=start.uni_data[i]
			for j in range(len(start.weights)):
				start.output[i]+=(curr_fact[j]*start.weights[j])
			start.output[i]+=start.bias[0]
			start.output[i]=sigmoid(start.output[i])
			for i in range(len(start.col_Data)):
				curr_coldata = start.col_Data[i]
				for j in range(len(curr_coldata)):
					sheet3.write(j,i,curr_coldata[j])
			for i in range(len(start.output)):
				sheet3.write(i,sheet.ncols,start.output[i])
			for i in range(len(start.output)):
				sheet3.write(i,sheet.ncols+1,numpy.round(start.output[i]))
			os.remove(get_Excel.dir_file)
			workbook.save(get_Excel.dir_file)
		else:
			for i in range(sheet2.ncols):
				temp = sheet2.col_values(i)
				for j in range(len(sheet2.col_values(i))):
					if(temp[j]==""):
						temp.pop(j)



				start.weights.append(temp)
			start.bias.append(start.weights[sheet2.ncols-1])
			start.weights.pop(sheet2.ncols-1)

			
			for k in range(len(start.uni_data)):
				print(start.uni_data)
				for i in range(len(start.weights[0])):
					print(start.uni_data[k][i])

					start.output[k]+=start.weights[0][i]*start.uni_data[k][i]
				start.output[k]+=start.bias[0][0]
				cff = start.output[k]
				for i in range(len(start.weights)-1):
					start.output[k] = 0
					for j in range(len(start.weights[i+1])):
						start.output[k] += cff*start.weights[i+1][j]
					start.output[k]+=start.bias[0][i+1]
					cff = start.output[k]
			for i in range(len(start.output)):
				start.output[i] = sigmoid(start.output[i])
			for i in range(len(start.col_Data)):
				curr_coldata = start.col_Data[i]
				for j in range(len(curr_coldata)):
					sheet3.write(j,i,curr_coldata[j])
			for i in range(len(start.output)):
				sheet3.write(i,sheet.ncols,start.output[i])
			for i in range(len(start.output)):
				sheet3.write(i,sheet.ncols+1,round(start.output[i]))
			os.remove(get_Excel.dir_file)
			workbook.save(get_Excel.dir_file)
				
				
					
				
				
				

				
			
		
	def start_multiple():
		start.uni_data= []
		start.col_Data = []
		start.weights = []
		start.bias = []
		start.output=[]
		idk = []
		
		loc = (get_Excel.dir_file)
		wb = xlrd.open_workbook(loc) 
		sheet = wb.sheet_by_index(0) 
		sheet.cell_value(0, 0)
		workbook = xlwt.Workbook(loc) 
		sheet3 = workbook.add_sheet("Data")
		loc2 = (model.path)
		wb2 = xlrd.open_workbook(loc2) 
		sheet2 = wb2.sheet_by_index(0) 
		sheet2.cell_value(0, 0)
		for i in range(sheet.nrows):
			start.uni_data.append(sheet.row_values(i))

		for i in  range(sheet.ncols):
			start.col_Data.append(sheet.col_values(i))

		for i in range(sheet2.ncols):
			start.weights.append([])
		

		for i in range(len(start.weights)):
			start.weights[i].append(sheet2.col_values(i))
		start.bias = sheet2.col_values(sheet2.ncols-1)
		start.weights.pop(len(start.weights)-1)
		for i in range(len(start.uni_data)):
			start.output.append(weight_zero(len(start.weights[0][0])))
		 
				


		for i in range(len(start.uni_data)):
			curr_fact=start.uni_data[i]
			for j in range(len(curr_fact)):
				curr_weight = start.weights[j]
				
				for k in range(len(start.weights[j][0])):
					print(len(start.weights[j][0]))
					start.output[i][k]+=(curr_fact[j]*curr_weight[0][k])
					
					

				
				
		print(start.output)
		for i in range(len(start.output)):
			for j in range(len(start.output[i])):
				start.output[i][j]+=start.bias[j]
				start.output[i][j]=sigmoid(start.output[i][j])
		print(start.output)

			
			
		
		for i in range(len(start.col_Data)):
			curr_coldata = start.col_Data[i]
			for j in range(len(curr_coldata)):
				sheet3.write(j,i,curr_coldata[j])
		for i in range(len(start.output)):
			sheet3.write(i,sheet.ncols,stringify(start.output[i]))
		for i in range(len(start.output)):
			sheet3.write(i,sheet.ncols+1,stringify_round(start.output[i]))
		for i in range(len(start.output)):
			sheet3.write(i,sheet.ncols+2,stringify_softmax(start.output[i]))
		os.remove(get_Excel.dir_file)
		workbook.save(get_Excel.dir_file)
	def start_linear():
		start.uni_data= []
		start.col_Data = []
		start.weights = []
		start.bias = []
		start.output=[]
		
		loc = (get_Excel.dir_file)
		wb = xlrd.open_workbook(loc) 
		sheet = wb.sheet_by_index(0) 
		sheet.cell_value(0, 0)
		workbook = xlwt.Workbook(loc) 
		sheet3 = workbook.add_sheet("Data")
		loc2 = (model.path)
		wb2 = xlrd.open_workbook(loc2) 
		sheet2 = wb2.sheet_by_index(0) 
		sheet2.cell_value(0, 0)
		for i in range(sheet.nrows):
			start.uni_data.append(sheet.row_values(i))
		for i in range(sheet.nrows):
			start.output.append(0)

		for i in  range(sheet.ncols):
			start.col_Data.append(sheet.col_values(i))
		if(sheet.nrows==2):
			start.weights = sheet2.col_values(0)
			start.weights.pop(0)
			start.bias = sheet2.col_values(1)
			start.bias.pop(0)
			for i in range(len(start.uni_data)):
				curr_fact=start.uni_data[i]
			for j in range(len(start.weights)):
				start.output[i]+=(curr_fact[j]*start.weights[j])
			start.output[i]+=start.bias[0]
			start.output[i]=sigmoid(start.output[i])
			for i in range(len(start.col_Data)):
				curr_coldata = start.col_Data[i]
				for j in range(len(curr_coldata)):
					sheet3.write(j,i,curr_coldata[j])
			for i in range(len(start.output)):
				sheet3.write(i,sheet.ncols,start.output[i])
			for i in range(len(start.output)):
				sheet3.write(i,sheet.ncols+1,numpy.round(start.output[i]))
			os.remove(get_Excel.dir_file)
			workbook.save(get_Excel.dir_file)
		else:
			for i in range(sheet2.ncols):
				temp = sheet2.col_values(i)
				for j in range(len(sheet2.col_values(i))):
					if(temp[j]==""):
						temp.pop(j)



				start.weights.append(temp)
			start.bias.append(start.weights[sheet2.ncols-1])
			start.weights.pop(sheet2.ncols-1)

			
			for k in range(len(start.uni_data)):
				print(start.uni_data)
				for i in range(len(start.weights[0])):
					print(start.uni_data[k][i])
					print(start.weights[0][i],start.weights)

					start.output[k]+=start.weights[0][i]*start.uni_data[k][i]
				start.output[k]+=start.bias[0][0]
				cff = start.output[k]
				for i in range(len(start.weights)-1):
					start.output[k] = 0
					for j in range(len(start.weights[i+1])):
						start.output[k] += cff*start.weights[i+1][j]
					start.output[k]+=start.bias[0][i+1]
					cff = start.output[k]
			
			for i in range(len(start.col_Data)):
				curr_coldata = start.col_Data[i]
				for j in range(len(curr_coldata)):
					sheet3.write(j,i,curr_coldata[j])
			for i in range(len(start.output)):
				sheet3.write(i,sheet.ncols,start.output[i])
			for i in range(len(start.output)):
				sheet3.write(i,sheet.ncols+1,round(start.output[i]))
			os.remove(get_Excel.dir_file)
			workbook.save(get_Excel.dir_file)
	def start_softplus():
		start.uni_data= []
		start.col_Data = []
		start.weights = []
		start.bias = []
		start.output=[]
		
		loc = (get_Excel.dir_file)
		wb = xlrd.open_workbook(loc) 
		sheet = wb.sheet_by_index(0) 
		sheet.cell_value(0, 0)
		workbook = xlwt.Workbook(loc) 
		sheet3 = workbook.add_sheet("Data")
		loc2 = (model.path)
		wb2 = xlrd.open_workbook(loc2) 
		sheet2 = wb2.sheet_by_index(0) 
		sheet2.cell_value(0, 0)
		for i in range(sheet.nrows):
			start.uni_data.append(sheet.row_values(i))
		for i in range(sheet.nrows):
			start.output.append(0)

		for i in  range(sheet.ncols):
			start.col_Data.append(sheet.col_values(i))
		if(sheet.nrows==2):
			start.weights = sheet2.col_values(0)
			start.weights.pop(0)
			start.bias = sheet2.col_values(1)
			start.bias.pop(0)
			for i in range(len(start.uni_data)):
				curr_fact=start.uni_data[i]
			for j in range(len(start.weights)):
				start.output[i]+=(curr_fact[j]*start.weights[j])
			start.output[i]+=start.bias[0]
			start.output[i]=sigmoid(start.output[i])
			for i in range(len(start.col_Data)):
				curr_coldata = start.col_Data[i]
				for j in range(len(curr_coldata)):
					sheet3.write(j,i,curr_coldata[j])
			for i in range(len(start.output)):
				sheet3.write(i,sheet.ncols,numpy.log(1 + numpy.exp(start.output[i])))
			for i in range(len(start.output)):
				sheet3.write(i,sheet.ncols+1,numpy.round(numpy.log(1 + numpy.exp(start.output[i]))))
			os.remove(get_Excel.dir_file)
			workbook.save(get_Excel.dir_file)
		else:
			for i in range(sheet2.ncols):
				temp = sheet2.col_values(i)
				for j in range(len(sheet2.col_values(i))):
					if(temp[j]==""):
						temp.pop(j)



				start.weights.append(temp)
			start.bias.append(start.weights[sheet2.ncols-1])
			start.weights.pop(sheet2.ncols-1)

			
			for k in range(len(start.uni_data)):
				print(start.uni_data)
				for i in range(len(start.weights[0])):
					print(start.uni_data[k][i])

					start.output[k]+=start.weights[0][i]*start.uni_data[k][i]
				start.output[k]+=start.bias[0][0]
				cff = start.output[k]
				for i in range(len(start.weights)-1):
					start.output[k] = 0
					for j in range(len(start.weights[i+1])):
						start.output[k] += cff*start.weights[i+1][j]
					start.output[k]+=start.bias[0][i+1]
					cff = start.output[k]
			for i in range(len(start.output)):
				start.output = SoftPlus(start.output)
			for i in range(len(start.col_Data)):
				curr_coldata = start.col_Data[i]
				for j in range(len(curr_coldata)):
					sheet3.write(j,i,curr_coldata[j])
			for i in range(len(start.output)):
				sheet3.write(i,sheet.ncols,start.output[i])
			for i in range(len(start.output)):
				sheet3.write(i,sheet.ncols+1,round(start.output[i]))
			os.remove(get_Excel.dir_file)
			workbook.save(get_Excel.dir_file)
			
	
	def sigmoid(x):
		return 1/(1+numpy.exp(-1*x))
	def tanh(x):
		return (numpy.exp(2*x)-1)/(numpy.exp(2*x)+1)
	def SoftPlus(b):
		return numpy.log(1+numpy.exp(b))
	def tanh_round(a):
		if(a<0):
			return -1*numpy.round(numpy.abs(a))
		else:
			return numpy.round(a)
	def weight_zero(j):
		arr = []
		for i in range(j):
			arr.append(0)
		return arr

	def stringify(h):
		v = ""
		for i in range(len(h)):
			if(i==(len(h)-1)):
				v = v+str(h[i])
			else:
				v = v+ str(h[i])+","
		return v
	def stringify_round(h):
		v = ""
		for i in range(len(h)):
			if(i==(len(h)-1)):
				v = v+str(numpy.round(h[i]))
			else:
				v = v+ str(numpy.round(h[i]))+","
		return v
	def stringify_softmax(h):
		exps = [numpy.exp(y) for y in h]
		sum_exps = sum(exps)
		g = []
		for i in range(len(exps)):
			g.append(exps[i]/sum_exps)
		v = ""
		for i in range(len(h)):
			if(i==(len(h)-1)):
				v = v+str(numpy.round(g[i]))
			else:
				v = v+ str(numpy.round(g[i]))+","
		return v

	def get_Excel():
		file = filedialog.askopenfilename()
		directory = os.path.split(file)[0]
		file_name = os.path.split(file)[1]
		get_Excel.dir_file = directory+"/"+file_name
		path_Excel['text']=get_Excel.dir_file
	def idk():
		linear_Activation = ("Linear Activation","SoftPlus")
		Probability_Activation = ("Sigmoid","Tanh")
		if(w.get()=="Generative Model"):
			model.values = linear_Activation
			activation["values"] = model.values
			
	
	
		else:
			model.values=Probability_Activation
			activation["values"] = model.values
	def model_path():
		file = filedialog.askopenfilename()
		directory = os.path.split(file)[0]
		file_name = os.path.split(file)[1]
		dir_file = directory+"/"+file_name
		model.path = dir_file
		

	menubar = Menu(master)
	filemenu = Menu(menubar, tearoff=0)
	filemenu.add_command(label="New", command=newModel)
	filemenu.add_command(label="Open", command=openModel)

	filemenu.add_separator()

	filemenu.add_command(label="Exit", command=master.destroy)
	menubar.add_cascade(label="File", menu=filemenu)


	helpmenu = Menu(menubar, tearoff=0)
	helpmenu.add_command(label="Help Index", command=master.destroy)
	helpmenu.add_command(label="About...", command=master.destroy)
	menubar.add_cascade(label="Help", menu=helpmenu)
	
	
	path_Excel = Label(master,text=" ",anchor="w",background="white",font=("Open Sans",13))
	path_Excel.pack()
	path_Excel.place(bordermode=OUTSIDE, height=30, width=605,x=280,y=440)
	w = Spinbox(values=("Multiple Classification","Binary Classification","Generative Model"),font=("Open Sans",10),command=idk)
	w.pack()
	w.place(bordermode=OUTSIDE, height=30, width=195,x=280,y=330)
	w.config(background="white",relief=FLAT,width=195,highlightbackground="black",highlightthickness="2",state = "readonly",buttonbackground="white",)
	activation = Spinbox(values=model.values,font=("Open Sans",10))
	activation.pack()
	activation.place(bordermode=OUTSIDE, height=30, width=195,x=280,y=390)
	activation.config(background="white",relief=FLAT,width=195,highlightbackground="black",highlightthickness="2",state = "readonly",buttonbackground="white",)
	start_img=PhotoImage(file="start.png")
	start_but = Button(master,image=start_img,relief=FLAT, command=start)
	start_but.pack()
	start_but.place(bordermode=OUTSIDE, height=30, width=85,x=600,y=490)
	mdpath_png=PhotoImage(file="modelpath.png")
	mdpath_but = Button(master,image=mdpath_png,relief=FLAT,command =model_path)
	mdpath_but.pack()
	mdpath_but.place(bordermode=OUTSIDE, height=30, width=165,x=20,y=260)
	selmode_png=PhotoImage(file="select.png")
	selmode_but = Button(master,image=selmode_png,relief=FLAT,command=get_Excel)
	selmode_but.pack()
	selmode_but.place(bordermode=OUTSIDE, height=30, width=180,x=20,y=440)
	
	path_excel = Label(master,text=model.path,anchor="w",background="white",font=("Open Sans",15))
	path_excel.pack()
	path_excel.place(bordermode=OUTSIDE, height=30, width=405,x=280,y=260)
	
	mod_png=PhotoImage(file="selecte_moel.png")
	mod_but = Label(master,image=mod_png,relief=FLAT)
	mod_but.pack()
	mod_but.place(bordermode=OUTSIDE, height=30, width=195,x=20,y=330)
	activation_png=PhotoImage(file="sel_activation.png")
	activation_but = Label(master,image=activation_png,relief=FLAT)
	activation_but.pack()
	activation_but.place(bordermode=OUTSIDE, height=30, width=195,x=20,y=390)
	

	
	master.config(menu=menubar)
	master.configure(background='white')
	master.iconbitmap("c4.ico")
	master.title("Alpha")
	master.mainloop()
