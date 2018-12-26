#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random as rd
from os import listdir
import numpy as np


MaxIteraciones=100	#número de iteraciones
Particulas=100		#numero de partículas
a=1			#parámetro a
b=1					#parámetro b
c=1					#parámetro c
theta=0				#parámetro theta
seed=52562				#parámetro semilla
k=1
solucionOptima=0


if seed>=0:  	#si semilla es positivo, tomara este valor como semilla
	np.random.seed(seed)



class Instancia(object):
	"""docstring for Instancia"""
	def __init__(self, arg):
		super(Instancia, self).__init__()
		self.arg = "BoctorProblem_90_instancias/"+arg
		self.Matrix,self.Machines,self.Parts,self.Cells,self.Mmax,self.Bsol = self.cargar_matriz(self.arg)
		print ("Máquinas: ",self.Machines ,"\t Partes: ",self.Parts,"\t Celdas: ", self.Cells ,"\t Máximo Máquinas: ",self.Mmax,"\t Mejor solucion: ", self.Bsol,"\t") #mostrar informacion de la instancia

	def cargar_matriz(self,PATH_SOURCE):
		TXT_SEP='='
		archivo = open(PATH_SOURCE, "r") 				
		Matrix=[]
		matriz = False
		for linea in archivo.readlines():					#Lee archivo de instancia
			if linea.split(TXT_SEP)[0]=="Machines":
				Machines = int(linea.split(TXT_SEP)[1])
			if linea.split(TXT_SEP)[0]=="Parts":
				Parts = int(linea.split(TXT_SEP)[1])
			if linea.split(TXT_SEP)[0]=="Cells":
				Cells = int(linea.split(TXT_SEP)[1])
			if linea.split(TXT_SEP)[0]=="Mmax":
				Mmax = int(linea.split(TXT_SEP)[1])
			if linea.split(TXT_SEP)[0]=="Best Solution":
				Bsol = int(linea.split(TXT_SEP)[1])
			if matriz == True:
				Matrix.append((linea.strip(' \\\n\r').split(' ')))	
			if linea.split(TXT_SEP)[0]=="Matrix":
				matriz = True
		return np.array(Matrix, dtype='int8'),Machines,Parts,Cells,Mmax,Bsol


class soluciones(object):	#clase donde guarda la solucion
	def __init__(self, arg):
		super(soluciones, self).__init__()
		self.instancia = arg
		self.Y=[] 				#Matriz maquina x celdas 
		self.Z=[] 				#Matriz parte x celdas
		self.S=[] 				#solucion/fitness
		for p in range (Particulas): #llenar las 3 matrices
			y,z,s = self.trabajo() #ejecuta las tareas de generacion y guardado de matrices
			self.Y.append(y)
			self.Z.append(z)
			self.S.append(s)

	def trabajo(self): # método donde genera las matrices MxC y PxC aleatorias para generar solucion
		A=self.instancia.Matrix
		restriccion=False
		while restriccion==False:	
			Yuniforme=np.random.uniform(size=self.instancia.Machines) #inicializa las variables de manera uniforme
			Zuniforme=np.random.uniform(size=self.instancia.Parts) #inicializa las variables de manera uniforme
			Y=self.transformar(Yuniforme) 	# Trasnforma las matrices de continua a discreta segun las celdas
			Z=self.crearZ(A,Y)	# Trasnforma las matrices de continua a discreta segun las celdas
			restriccion=self.probar_restriccion(Y) # llama a verificar factibilidad de la funcion
		Solucion=self.solucion(A,Y,Z) #resultado FO , fitness
		return Y,Z,Solucion	#devuelve las 3 matrices 

	def transformar(self,matriz):  #funcion que transforma de numeros uniforme a binarios
		Cells=self.instancia.Cells
		arreglo_discreta=np.zeros((len(matriz),Cells),dtype='int8') #llenar matriz de MXP
		# j = (Cells*matriz).astype(int)
		for i in range(len(matriz)): #multiplicar fila x columna para ver que tenga un 1
			j=int(matriz[i]*Cells)
			arreglo_discreta[i][j]=1
		return arreglo_discreta

	def probar_restriccion(self,Y):  #verificar factibilidad de problema
		valido=True
		for k in range(self.instancia.Cells): #solo comprueba la restriccion 3 ya que las primeras dos automaticamente está resuelta
			if sum(Y[:,k])>self.instancia.Mmax:
				valido=False
		return valido	

	def solucion(self,A,Y,Z):     #mostrar solucion
		suma_total=0
		for k in range(self.instancia.Cells):
			for i in range (self.instancia.Machines):
				for j in range(self.instancia.Parts):
					suma_total=suma_total+int(A[i][j])*Z[j][k]*(1-Y[i][k])
		return suma_total	

	def crearZ(self,A,Y):
		Z=np.zeros((self.instancia.Parts,self.instancia.Cells),dtype='int8')
		for k in range(self.instancia.Cells):
			for j in range(self.instancia.Parts):
				Z[j][k]=np.sum(A[np.where(Y[:,k]==1),j])
		for j in range(self.instancia.Parts):  #establecer seleccion aleatoria cuando existe misma cantidad de elementos
			aux=np.zeros(self.instancia.Cells)  
			rep,indice = self.repetido(Z[j])
			if rep==True:
				aux[  indice[np.random.randint(0,len(indice))]  ] = 1
				Z[j]=aux
			else:
				aux[np.argmax(Z[j])]=1
				Z[j]=aux
		return Z

	def repetido(self,tupla): #genera en caso de haber maximos repetidos, obtener los indices de las cuales se repite
		maximo=np.max(tupla)
		index=np.array((),dtype='int16')
		count=0
		if maximo>0: #si el numero maximo es superior a 0, entonces cuenta la cantidad
			for i in range(len(tupla)):
				if tupla[i]==maximo:
					count+=1
					index=np.hstack((index,i))
		if count >= 2 :
			return True,index 
		return False,index

	def  mostrarMatriz(self,A,Y,Z):
		matriz = np.array((np.copy(A)))
		x=0
		y=0
		for k in range(self.instancia.Cells):
			for i in range (self.instancia.Machines):
				if Y[i][k]==1:
					matriz[x][y]=A[:,j]
					matriz[i][y]=A[:,x]

			for j in range(self.instancia.Parts):
				print("no implementado")
					



class metaehuristia(object):   #clase donde realiza las tareas de la metaehuristica
	def __init__(self, instancia, solucion):
		super(metaehuristia, self).__init__()
		self.phi=((1+np.sqrt(5))/2) #funcion phi que indica en el texto
		self.instancia = instancia 	#instancia obtenida del txt
		self.solucion = solucion 	#solucion/es inicales para particulas
		self.v = self.generarV()	#generar aleatoriamente v
		self.x = self.generarX()
		p=solucion.S.index(min(self.solucion.S)) # conseguir indice , puntero sol
		self.Xbest=np.array((np.copy(self.x[p]),np.copy(self.solucion.S[p])))				#mejor solucion del grupo actual hay q copiar
		self.Xglobal=np.array((np.copy(self.x[p]),np.copy(self.solucion.S[p])))			#mejor solucion global 															
		self.mejorAleatoria=self.Xbest[1]
		self.algoritmo()


	def generarV(self): #inicializar velocidad una matriz del mismo tamaño q la mxp
		obj=[]
		for i in range (Particulas):
			obj.append(np.array((np.random.random(self.instancia.Machines))))
		return np.array(obj)
	
	def generarX(self):
		X=[]
		for p in range(Particulas):
			X.append(np.where(self.solucion.Y[p]==1)[1])
		return np.array(X,dtype='int8')

	def algoritmo(self): #funcion que realiza las iteraciones de la metaehuristica
		rangoTheta=100
		sinTheta=0
		for p in range (Particulas):  #inicializa partículas 
			paso=False
			intentos=0
			while paso==False:
				intentos+=1
				aux1=self.velocidad(self.Xbest[0],self.Xglobal[0],self.v[p],self.x[p]) #guarda en variable temporal (velocidad)
				aux2=self.poscicion(self.x[p],aux1,1)									   #guarda en variable temporal (pocicion) en discreto
				Y=self.binarizar(aux2)
				if intentos>10 and self.solucion.probar_restriccion(Y)==False:
					theta=aux2 + np.sin((np.random.random(len(aux2))*2*np.pi))
					Y=self.binarizar(theta.astype('int8'))
				if self.solucion.probar_restriccion(Y) == True:    # si es factible la solucion generada, entonces continua asignando las demás variables para generar la solucion
					paso=True
					self.v[p]=aux1  #guarda en la pocicion, la variable auxiliar 
					self.x[p]=theta.astype('int8')
					self.solucion.Y[p]=Y #guarda en la pocicion, la variable auxiliar
					self.solucion.Z[p]=self.solucion.crearZ(self.instancia.Matrix,self.solucion.Y[p])
					self.solucion.S[p]=self.solucion.solucion(self.instancia.Matrix,self.solucion.Y[p],self.solucion.Z[p])
		desv1 = self.desviacionStandar(self.x) #almacena las primeras desviaciones estandar
		
		p=self.solucion.S.index(min(self.solucion.S)) #obtiene la partícula con mejor fitness
		self.Xbest = (np.copy(self.x[p]),np.copy(self.solucion.S[p]))
		if self.Xbest[1] < self.Xglobal[1]: #si la mejor solucion es mejor que la anterior, ésta la actualiza
			self.Xglobal = (np.copy(self.x[p]),np.copy(self.solucion.S[p]))
		primeraIteracion=self.Xglobal[1]
		
		for it in range(1,MaxIteraciones):
			if self.Xglobal[1]<=self.instancia.Bsol:                 #utiliza tecnica fordward checking
				break
			for p in range (Particulas):
				paso=False
				intentos=0
				while paso==False:
					intentos+=1
					aux1=self.velocidad(self.Xbest[0],self.Xglobal[0],self.v[p],self.x[p]) #guarda en variable temporal (velocidad)
					aux2=self.poscicion(self.x[p],aux1,1)									   #guarda en variable temporal (pocicion)
					Y=self.binarizar(aux2)
					if intentos>10 and self.solucion.probar_restriccion(Y)==False:
						theta=aux2 + np.sin((np.random.random(len(aux2))*2*np.pi))
						Y=self.binarizar(theta.astype('int8'))
						if it==11 and p==82:
							print(Y)
					if self.solucion.probar_restriccion(Y) == True:    # si es factible la solucion generada, entonces continua asignando las demás variables para generar la solucion
						paso=True
						self.v[p]=aux1  #guarda en la pocicion, la variable auxiliar 
						self.x[p]=theta
						self.solucion.Y[p]=Y #guarda en la pocicion, la variable auxiliar
						self.solucion.Z[p]=self.solucion.crearZ(self.instancia.Matrix,self.solucion.Y[p])
						self.solucion.S[p]=self.solucion.solucion(self.instancia.Matrix,self.solucion.Y[p],self.solucion.Z[p])
			p=self.solucion.S.index(min(self.solucion.S)) #obtiene la partícula con mejor fitness
			self.Xbest = (np.copy(self.x[p]),np.copy(self.solucion.S[p]))
			if self.Xbest[1] < self.Xglobal[1]: #si la mejor solucion es mejor que la anterior, ésta la actualiza
				self.Xglobal = (np.copy(self.x[p]),np.copy(self.solucion.S[p]))
	
			

			if it%10==0:  #por cada 10 iteraciones realiza un mantenimiento para realizar nuevas exploraciones
				print("mantenimiento")
				desv2= self.desviacionStandar(self.x)  #calcula nueva desviacion estandar
				listaYt = np.where(desv2<desv1)          #obtiene indices de las comparaciones de desviacion estandar que son menores que el anterior
				listaYf = np.where(desv2>desv1)  		#obtiene indices de las comparaciones de desviacion estandar que no cumple con la condicion anterior

				for p in range (Particulas):
					estado=False
					intentos=0
					auxX=np.zeros(self.instancia.Machines,dtype='int8')
					while estado==False:
						intentos+=1 
						aux1=self.velocidad(self.Xbest[0][listaYt],self.Xglobal[0][listaYt],self.v[p][listaYt],self.x[p][listaYt])  #mismo trabajo que en la iteraciones, pero con diferente trato segun la condicion de las desviaciones estandar
						aux2=self.poscicion(self.x[p][listaYt],aux1,np.random.randint(2,6,len(listaYt)))
						aux3=self.velocidad(self.Xbest[0][listaYf],self.Xglobal[0][listaYf],self.v[p][listaYf],self.x[p][listaYf])
						aux4=self.poscicion(self.x[p][listaYf],aux3,1)
						auxX[listaYt]=aux2
						auxX[listaYf]=aux4
						Y=self.binarizar(auxX.astype('int8')) #arreglar tomar auxiliares y unirla en su equivalente		
						if intentos>10 and self.solucion.probar_restriccion(Y) == False:
							theta=auxX + np.sin((np.random.random(len(auxX))*2*np.pi))
							Y=self.binarizar(theta.astype('int8'))
						if self.solucion.probar_restriccion(Y) == True:
							estado=True
							self.solucion.Y[p]=Y #guarda en la pocicion, la variable auxiliar
							self.solucion.Z[p]=self.solucion.crearZ(self.instancia.Matrix,self.solucion.Y[p])
							self.solucion.S[p]=self.solucion.solucion(self.instancia.Matrix,self.solucion.Y[p],self.solucion.Z[p])
							self.v[p][listaYt]=aux1
							self.v[p][listaYf]=aux3
							self.x[p]=theta
				desv1=desv2	
			print("mejor local: ",self.Xbest[1],"\t mejor global: ",self.Xglobal[1])
		print("mejor solucion aleatoria: ",self.mejorAleatoria,"\t primera iteracion: ",primeraIteracion ,"\t mejor solucion: ",self.Xglobal[1],"\n")
		exit()

	#def movimiento(self):

	def velocidad(self,Xbest,Xglobal,v,x): #ecuacion velocidad
		return a*v+b*np.random.random(len(v))*(Xbest-x) + c*np.random.random(len(v))*(Xglobal-x) #en caso de emergencia, colocar un v max y v min

	def poscicion(self,x,v1,mult):      # ecuacion pocicion
		r3=np.random.randint(-1,2,len(x)) #entrega -1 0 1
		temp1=self.sigmoide(x + mult*r3*self.phi + v1)
		aux=((temp1*(self.instancia.Cells)).astype('int8')) #discretiza la pocicion como entero
		return aux  		


	def desviacionStandar(self,Y):  #genera desviacion estandar para cada dimension	
		desvY=np.zeros(self.instancia.Machines) #inicializa listas de desviacion estandar	
		for i in range(self.instancia.Machines): #guarda desviacion estandar para máquinas x celdas
			desvY[i]=np.std(np.array(Y)[:,i],ddof=1)
		return desvY


	def sigmoide(self,x):    #se aplica la regla sigmoidal para transformar cualquier número a numeros entre 0 y 111
		return (1/(1+np.exp(-x/k)))

	def binarizar(self,x):
		Y=np.zeros((self.instancia.Machines,self.instancia.Cells),dtype='int8')
		for i in range(self.instancia.Machines):
			Y[i][x[i]]=1
		return Y

def main():
	instancias=listdir("BoctorProblem_90_instancias/")
	for i in range(len(instancias)):
		instancia = Instancia("MCDP_Boctor_Problem01_C3_M6.txt")	#crear instancia a partir del archivo
		objetos=soluciones(instancia)			#generar soluciones en base de la instancia
		metaehuristia(instancia,objetos) 	#en metaehuristica pasar cuadro y las soluciones
		
		

if __name__ == '__main__':
	main()


#preguntas al profe, cuando se realiza la funcion de busqueda de mejor solucion cada 10 iteraciones, esta se debe hacer el mantenimiento o hacerlo junto con la busqueda de la iteracion 10
