#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random as rd
from os import listdir
import numpy as np


MaxIteraciones=1000	#número de iteraciones
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
		matriz = 'F'
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
			if matriz == 'T':
				Matrix.append((linea.strip(' \\\n\r').split(' ')))	
			if linea.split(TXT_SEP)[0]=="Matrix":
				matriz = 'T'
		return Matrix,Machines,Parts,Cells,Mmax,Bsol


class soluciones(object):
	"""docstring for soluciones"""
	def __init__(self, arg):
		super(soluciones, self).__init__()
		self.instancia = arg
		self.Y=[]
		self.Z=[]
		self.S=[]
		for p in range (Particulas):
			y,z,s = self.trabajo()
			self.Y.append(y)
			self.Z.append(z)
			self.S.append(s)
		#print ("Y:",self.Y,"\nZ:",self.Z,"\nS:",self.S)


	def trabajo(self):
		A=self.instancia.Matrix
		restriccion='F'
		while restriccion=='F':	
			Yuniforme=np.random.uniform(size=self.instancia.Machines)
			Zuniforme=np.random.uniform(size=self.instancia.Parts)
			Y=self.transformar(Yuniforme)
			Z=self.transformar(Zuniforme)
			restriccion=self.probar_restriccion(Y)
		Solucion=self.solucion(A,Y,Z)
		return Yuniforme,Zuniforme,Solucion	

	def transformar(self,matriz):  #funcion que transforma de numeros uniforme a binarios
		Cells=self.instancia.Cells
		arreglo_discreta=np.zeros((len(matriz),Cells))
		for i in range(len(matriz)):
			j=int(matriz[i]*Cells)
			arreglo_discreta[i][j]=1
		return arreglo_discreta

	def probar_restriccion(self,Y):  #probar que no exeda la suma de máquinas permitidas
		valido='T'
		suma=0			
		for k in range(self.instancia.Cells): 
			suma=0
			for i in range(self.instancia.Machines):	
				suma=suma + Y[i][k]
			if suma>self.instancia.Mmax:
				valido='F'
		return valido	

	def solucion(self,A,Y,Z):     #mostrar solucion
		suma_total=0
		for k in range(self.instancia.Cells):
			for i in range (self.instancia.Machines):
				for j in range(self.instancia.Parts):
					suma_total=suma_total+int(A[i][j])*Z[j][k]*(1-Y[i][k])
		return suma_total	

	def sigmoide(x):
		return (1/(1+np.exp(x/1)))



class metaehuristia(object):
	"""docstring for fibonacci"""
	def __init__(self, instancia, solucion):
		super(metaehuristia, self).__init__()
		self.phi=((1+np.sqrt(5))/2)
		self.instancia = instancia 	#instancia obtenida del txt
		self.solucion = solucion 	#solucion/es inicales para particulas
		self.v = self.generarV()	#generar aleatoriamente v
		p=solucion.S.index(min(self.solucion.S))
		self.Xbest=np.array((np.copy(self.solucion.Y[p]),np.copy(self.solucion.Z[p]),np.copy(self.solucion.S[p])))	#mejor solucion del grupo actual hay q copiar
		self.Xglobal=self.Xbest
		print(self.Xbest[0][0:2])
		print(self.solucion.Y[p][0:2])
		self.Xbest[0][1]=111
		print(self.Xbest[0][0:2])
		print(self.solucion.Y[p][0:2])
		exit() 																#mejor solucion global 
		self.algoritmo()
		
	def generarV(self):
		obj=[]
		for i in range (Particulas):
			obj.append(np.array((np.random.random(self.instancia.Machines),np.random.random(self.instancia.Parts))))
		return np.array(obj)
		


	def algoritmo(self):
		it=0
		print("mejor",self.Xbest)
		for p in range (Particulas):
			for i in range(self.instancia.Machines):
				self.v[p][0][i]=self.velocidad(self.Xbest[0][i],self.Xglobal[0][i],self.v[p][0][i],self.solucion.Y[p][i]) #xbest cambia cuando pasa por y[i]
				self.solucion.Y[p][i]=self.poscicion(self.solucion.Y[p][i],self.v[p][0][i],1)
			for j in range(self.instancia.Parts):
				self.v[p][1][j]=self.velocidad(self.Xbest[1][j],self.Xglobal[1][j],self.v[p][1][j],self.solucion.Z[p][j])
				self.solucion.Z[p][j]=self.poscicion(self.solucion.Z[p][j],self.v[p][1][j],1)
		print("mejor despues de lo otro",self.Xbest)
		self.desviacionStandar(self.solucion.Y,self.solucion.Z)
		exit()
		#print(self.v)
		#quiebre

		#while it < MaxIteraciones and self.instancia.Bsol<Csol:
			#for n in range (Particulas):
				# pocicion(1,n)
				# velocidad(n)


			#self.solucion.Y=self.poscicion(1)
			#hacer algoritmo con x e v, luego la desviacion std y la wea
		#	if i%10==0:
		#		print("mantencion")
		#		self.desviacion(self.solucion.Y,self.solucion.Z)
				#hacer mantencion
		#	Csol+=1



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

		promedioMaq=np.zeros(self.instancia.Machines)
		promedioPar=np.zeros(self.instancia.Parts)
		

		for i in range(self.instancia.Machines):
			desvY[i]=np.std(auxY[i],ddof=1)

		for j in range(self.instancia.Machines):
			desvZ[j]=np.std(auxZ[j],ddof=1)		

		return desvY, desvZ
		#obrener desviacion estandar






def main():
	instancias=listdir("BoctorProblem_90_instancias/")
	for i in range(len(instancias)):
		instancia = problema(instancias[i])	#crear instancia a partir del archivo
		objetos=soluciones(instancia)			#generar soluciones en base de la instancia
		metaehuristia(instancia,objetos) 	#en metaehuristica pasar cuadro y las soluciones
		
		


if __name__ == '__main__':
	main()