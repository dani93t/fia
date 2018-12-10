#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random as rd
from os import listdir
import numpy as np


MaxIteraciones=100	#número de iteraciones
Particulas=10		#numero de partículas
a=1					#parámetro a
b=1					#parámetro b
c=1					#parámetro c
theta=0				#parámetro theta
seed=-1			#parámetro semilla


if seed>=0:  	#si semilla es positivo, tomara este valor como semilla
	np.random.seed(seed)



class problema(object):
	"""docstring for problema"""
	def __init__(self, arg):
		super(problema, self).__init__()
		self.arg = "BoctorProblem_90_instancias/"+arg
		self.Matrix,self.Machines,self.Parts,self.Cells,self.Mmax,self.Bsol = self.cargar_matriz(self.arg)
		print ("Máquinas: ",self.Machines ,"\t Partes: ",self.Parts,"\t Celdas: ", self.Cells ,"\t Máximo Máquinas: ",self.Mmax,"\t\n")

	def cargar_matriz(self,PATH_SOURCE):
		TXT_SEP='='
		archivo = open(PATH_SOURCE, "r")
		Matrix=[]
		matriz = False
		for linea in archivo.readlines():
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
		return Matrix,Machines,Parts,Cells,Mmax,Bsol


class soluciones(object):
	"""docstring for soluciones"""
	def __init__(self, arg):
		super(soluciones, self).__init__()
		self.instancia = arg
		self.Y=[] #maquina 
		self.Z=[] #parte
		self.S=[] #solucion/fitness
		for p in range (Particulas): #llenar las 3 matrices
			y,z,s = self.trabajo()
			self.Y.append(y)
			self.Z.append(z)
			self.S.append(s)
		#print ("Y:",self.Y,"\nZ:",self.Z,"\nS:",self.S)


	def trabajo(self):
		A=self.instancia.Matrix
		restriccion=False
		while restriccion==False:	
			Yuniforme=np.random.uniform(size=self.instancia.Machines)
			Zuniforme=np.random.uniform(size=self.instancia.Parts)
			Y=self.transformar(Yuniforme)
			Z=self.transformar(Zuniforme)
			restriccion=self.probar_restriccion(Y)
		Solucion=self.solucion(A,Y,Z) #resultado FO , fitness
		return Yuniforme,Zuniforme,Solucion	#devuelve las 3 matrices 

	def transformar(self,matriz):  #funcion que transforma de numeros uniforme a binarios

		Cells=self.instancia.Cells
		#print(Cells)
		arreglo_discreta=np.zeros((len(matriz),Cells)) #llenar matriz de MXP

		# j = (Cells*matriz).astype(int)

		for i in range(len(matriz)): #multiplicar fila x columna para ver que tenga un 1
			j=int(matriz[i]*Cells)
			arreglo_discreta[i][j]=1
		return arreglo_discreta

	def probar_restriccion(self,Y):  #probar que no exeda la suma de máquinas permitidas
		valido=True
		suma=0			
		for k in range(self.instancia.Cells): 
			suma=0
			for i in range(self.instancia.Machines):	
				suma=suma + Y[i][k]
			if suma>self.instancia.Mmax:
				valido=False
		return valido	

	def solucion(self,A,Y,Z):     #mostrar solucion
		suma_total=0
		for k in range(self.instancia.Cells):
			for i in range (self.instancia.Machines):
				for j in range(self.instancia.Parts):
					suma_total=suma_total+int(A[i][j])*Z[j][k]*(1-Y[i][k])
		return suma_total	





class metaehuristia(object):
	"""docstring for fibonacci"""
	def __init__(self, instancia, solucion):
		super(metaehuristia, self).__init__()
		self.phi=((1+np.sqrt(5))/2)
		self.instancia = instancia 	#instancia obtenida del txt
		self.solucion = solucion 	#solucion/es inicales para particulas
		self.v = self.generarV()	#generar aleatoriamente v
		p=solucion.S.index(min(self.solucion.S)) # conseguir indice , puntero sol
		self.Xbest=np.array((np.copy(self.solucion.Y[p]),np.copy(self.solucion.Z[p]),np.copy(self.solucion.S[p])))	#mejor solucion del grupo actual hay q copiar
		self.Xglobal=np.copy(self.Xbest)				#mejor solucion global 															
		self.algoritmo()
		
	def generarV(self): #generar velocidad una matriz del mismo tamaño q la mxp
		obj=[]
		for i in range (Particulas):
			obj.append(np.array((np.random.random(self.instancia.Machines),np.random.random(self.instancia.Parts))))
		return np.array(obj)
		


	def algoritmo(self):
		print("mejor solucion",self.Xbest[2])
		for p in range (Particulas):
			paso=False
			while paso==False:	
				for i in range(self.instancia.Machines):
					self.v[p][0][i]=self.velocidad(self.Xbest[0][i],self.Xglobal[0][i],self.v[p][0][i],self.solucion.Y[p][i]) #xbest cambia cuando pasa por y[i]
					self.solucion.Y[p][i]=self.poscicion(self.solucion.Y[p][i],self.v[p][0][i],1)
				for j in range(self.instancia.Parts):
					self.v[p][1][j]=self.velocidad(self.Xbest[1][j],self.Xglobal[1][j],self.v[p][1][j],self.solucion.Z[p][j])
					self.solucion.Z[p][j]=self.poscicion(self.solucion.Z[p][j],self.v[p][1][j],1)
				SigmY=self.sigmoide(self.solucion.Y[p])
				Y=self.solucion.transformar(SigmY)
				if self.solucion.probar_restriccion(Y) == True:
					paso=True
					SigmZ=self.sigmoide(self.solucion.Z[p])
					Z=self.solucion.transformar(SigmZ)
					self.solucion.S[p]=self.solucion.solucion(self.instancia.Matrix,Y,Z)
		
		self.desviacionStandar(self.solucion.Y,self.solucion.Z)
		
		iteracion=1


		while iteracion < MaxIteraciones and self.instancia.Bsol<self.Xglobal[2]:
			print("iteracion",iteracion)
			for p in range (Particulas):
				paso=False
				while paso==False:	
					for i in range(self.instancia.Machines):
						self.v[p][0][i]=self.velocidad(self.Xbest[0][i],self.Xglobal[0][i],self.v[p][0][i],self.solucion.Y[p][i]) #xbest cambia cuando pasa por y[i]
						self.solucion.Y[p][i]=self.poscicion(self.solucion.Y[p][i],self.v[p][0][i],1)
					for j in range(self.instancia.Parts):
						self.v[p][1][j]=self.velocidad(self.Xbest[1][j],self.Xglobal[1][j],self.v[p][1][j],self.solucion.Z[p][j])
						self.solucion.Z[p][j]=self.poscicion(self.solucion.Z[p][j],self.v[p][1][j],1)
					SigmY=self.sigmoide(self.solucion.Y[p])
					Y=self.solucion.transformar(SigmY)
					if self.solucion.probar_restriccion(Y) == True:
						paso=True
						SigmZ=self.sigmoide(self.solucion.Z[p])
						Z=self.solucion.transformar(SigmZ)
						self.solucion.S[p]=self.solucion.solucion(self.instancia.Matrix,Y,Z)			


		#	if i%10==0:
		#		print("mantencion")
		#		self.desviacion(self.solucion.Y,self.solucion.Z)
				#hacer mantencion
		#	Csol+=1
			print(min(self.solucion.S))
			iteracion+=1
		exit()



	def velocidad(self,Xbest,Xglobal,v,x):
		return a*v+b*np.random.random()*(Xbest-x) + c*np.random.random()*(Xglobal-x)

	def poscicion(self,x,v1,mult):
		r3=np.random.randint(-1,2) #entrega -1 0 1
		return x + mult*r3*self.phi + v1


	def desviacionStandar(self,Y,Z):
		
		#print(np.array(Y)[:,1])

		auxY=np.flipud(np.rot90(Y))
		auxZ=np.flipud(np.rot90(Z))

		desvY=np.zeros(self.instancia.Machines)
		desvZ=np.zeros(self.instancia.Parts)
		
		for i in range(self.instancia.Machines):
			desvY[i]=np.std(auxY[i],ddof=1)

		for j in range(self.instancia.Machines):
			desvZ[j]=np.std(auxZ[j],ddof=1)		

		return desvY, desvZ #obrener desviacion estandar


	def sigmoide(self,x):
		return (1/(1+np.exp(-x/1)))




def main():
	instancias=listdir("BoctorProblem_90_instancias/")
	for i in range(len(instancias)):
		instancia = problema(instancias[i])	#crear instancia a partir del archivo
		objetos=soluciones(instancia)			#generar soluciones en base de la instancia
		metaehuristia(instancia,objetos) 	#en metaehuristica pasar cuadro y las soluciones
		
		


if __name__ == '__main__':
	main()