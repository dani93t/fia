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
seed=0				#parámetro semilla
k=1


if seed>=0:  	#si semilla es positivo, tomara este valor como semilla
	np.random.seed(seed)


log=open("log.txt","w")
log.write(str(("\ta: ",a,"\tb: ",b,"\tc: ",c)))
log.close()
log=open("log.txt","a")

class problema(object):
	"""docstring for problema"""
	def __init__(self, arg):
		super(problema, self).__init__()
		self.arg = "BoctorProblem_90_instancias/"+arg
		self.Matrix,self.Machines,self.Parts,self.Cells,self.Mmax,self.Bsol = self.cargar_matriz(self.arg)
		log.write(str(("Máquinas: ",self.Machines ,"\t Partes: ",self.Parts,"\t Celdas: ", self.Cells ,"\t Máximo Máquinas: ",self.Mmax,"\t Mejor solucion: ", self.Bsol,"\t\n")))
		print ("Máquinas: ",self.Machines ,"\t Partes: ",self.Parts,"\t Celdas: ", self.Cells ,"\t Máximo Máquinas: ",self.Mmax,"\t Mejor solucion: ", self.Bsol,"\t\n") #mostrar informacion de la instancia

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
		return Matrix,Machines,Parts,Cells,Mmax,Bsol


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
			Z=self.transformar(Zuniforme)	# Trasnforma las matrices de continua a discreta segun las celdas
			restriccion=self.probar_restriccion(Y) # llama a verificar factibilidad de la funcion
		Solucion=self.solucion(A,Y,Z) #resultado FO , fitness
		return Yuniforme,Zuniforme,Solucion	#devuelve las 3 matrices 

	def transformar(self,matriz):  #funcion que transforma de numeros uniforme a binarios
		Cells=self.instancia.Cells
		arreglo_discreta=np.zeros((len(matriz),Cells)) #llenar matriz de MXP

		# j = (Cells*matriz).astype(int)

		for i in range(len(matriz)): #multiplicar fila x columna para ver que tenga un 1
			j=int(matriz[i]*Cells)
			arreglo_discreta[i][j]=1
		return arreglo_discreta

	def probar_restriccion(self,Y):  #verificar factibilidad de problema
		valido=True
		suma=0			
		for k in range(self.instancia.Cells): #solo comprueba la restriccion 3 ya que las primeras dos automaticamente está resuelta
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
		p=solucion.S.index(min(self.solucion.S)) # conseguir indice , puntero sol
		self.Xbest=np.array((np.copy(self.solucion.Y[p]),np.copy(self.solucion.Z[p]),np.copy(self.solucion.S[p])))				#mejor solucion del grupo actual hay q copiar
		self.Xglobal=np.array((np.copy(self.solucion.Y[p]),np.copy(self.solucion.Z[p]),np.copy(self.solucion.S[p])))			#mejor solucion global 															
		self.algoritmo()
		
	def generarV(self): #inicializar velocidad una matriz del mismo tamaño q la mxp
		obj=[]
		for i in range (Particulas):
			obj.append(np.array((np.random.random(self.instancia.Machines),np.random.random(self.instancia.Parts))))
		return np.array(obj)
		
	def algoritmo(self): #funcion que realiza las iteraciones de la metaehuristica
		for p in range (Particulas):  #inicializa partículas 
			paso=False
			while paso==False:
				aux1=self.velocidad(self.Xbest[0],self.Xglobal[0],self.v[p][0],self.solucion.Y[p]) #guarda en variable temporal (velocidad)
				aux2=self.poscicion(self.solucion.Y[p],aux1,1)									   #guarda en variable temporal (pocicion)
				Y=self.solucion.transformar(aux2)
				if self.solucion.probar_restriccion(Y) == True:    # si es factible la solucion generada, entonces continua asignando las demás variables para generar la solucion
					paso=True
					self.v[p][0]=aux1  #guarda en la pocicion, la variable auxiliar 
					self.solucion.Y[p]=aux2 #guarda en la pocicion, la variable auxiliar
					self.v[p][1]=self.velocidad(self.Xbest[1],self.Xglobal[1],self.v[p][1],self.solucion.Z[p])
					self.solucion.Z[p]=self.poscicion(self.solucion.Z[p],self.v[p][1],1)  #sigZ
					Z=self.solucion.transformar(self.solucion.Z[p])
					self.solucion.S[p]=self.solucion.solucion(self.instancia.Matrix,Y,Z)
		stdY1,stdZ1 = self.desviacionStandar(self.solucion.Y,self.solucion.Z) #almacena las primeras desviaciones estandar
		p=self.solucion.S.index(min(self.solucion.S)) #obtiene la partícula con mejor fitness
		self.Xbest = (np.copy(self.solucion.Y[p]),np.copy(self.solucion.Z[p]),np.copy(self.solucion.S[p]))
		if self.Xbest[2] < self.Xglobal[2]: #si la mejor solucion es mejor que la anterior, ésta la actualiza
			self.Xglobal = (np.copy(self.solucion.Y[p]),np.copy(self.solucion.Z[p]),np.copy(self.solucion.S[p]))

		iteracion=1
		



		while iteracion < MaxIteraciones and self.instancia.Bsol<self.Xglobal[2]:  #realiza las iteraciones y se detiene hasta realizar todas o llegar a la solución óptima
			for p in range (Particulas):
				paso=False
				intentos=0                       
				while paso==False:
					intentos+=1
					aux1=self.velocidad(self.Xbest[0],self.Xglobal[0],self.v[p][0],self.solucion.Y[p]) #a
					aux2=self.poscicion(self.solucion.Y[p],aux1,1)
					Y=self.solucion.transformar(aux2)
					if intentos>10 and self.solucion.probar_restriccion(Y)==False:
						lineas=np.linspace(0,2*np.pi,50)
						#print("\n\n original:\n ",Y)
						for i in range(50):
							if self.solucion.probar_restriccion(self.solucion.transformar(self.sigmoide(aux2+np.sin(lineas[i])))) == True:
								Y=self.solucion.transformar(self.sigmoide(aux2+np.sin(lineas[i])))
								break
					if self.solucion.probar_restriccion(Y) == True:
						paso=True
						self.v[p][0]=aux1
						self.solucion.Y[p]=aux2
						self.v[p][1]=self.velocidad(self.Xbest[1],self.Xglobal[1],self.v[p][1],self.solucion.Z[p])
						self.solucion.Z[p]=self.poscicion(self.solucion.Z[p],self.v[p][1],1)
						Z=self.solucion.transformar(self.solucion.Z[p])
						self.solucion.S[p]=self.solucion.solucion(self.instancia.Matrix,Y,Z)
			p=self.solucion.S.index(min(self.solucion.S))
			self.Xbest = (np.copy(self.solucion.Y[p]),np.copy(self.solucion.Z[p]),np.copy(self.solucion.S[p]))
			if self.Xbest[2] < self.Xglobal[2]:
				self.Xglobal = (np.copy(self.solucion.Y[p]),np.copy(self.solucion.Z[p]),np.copy(self.solucion.S[p]))
		

			if iteracion%10==0:  #por cada 10 iteraciones realiza un mantenimiento para realizar nuevas exploraciones
				print("Exploracion nuevo espacio")
				stdY2 ,stdZ2= self.desviacionStandar(self.solucion.Y,self.solucion.Z)  #calcula nueva desviacion estandar
				listaYt=[] #lista de pociciones de dimencion sobre la condicion dada
				listaYf=[]
				listaZt=[]
				listaZf=[] 
				for i in range(self.instancia.Machines): #compara las desviaciones estandar de cada dimencion y almacena en una variable la lista de elementos que cumple con la condicion en indicies
					if stdY2[i]<stdY1[i]:
						listaYt.append(i) #guarda el indice los elementos que acierta con la condicion
					else:
						listaYf.append(i) #guarda el indice de los elementos que no acierta con la condicion
				for j in range(self.instancia.Parts):
					if stdZ2[j]<stdZ1[j]:
						listaZt.append(j)
					else:
						listaZf.append(j)		
				for p in range (Particulas):
					estado=False
					while estado==False: 
						aux1=self.velocidad(self.Xbest[0][listaYt],self.Xglobal[0][listaYt],self.v[p][0][listaYt],self.solucion.Y[p][listaYt])  #mismo trabajo que en la iteraciones, pero con diferente trato segun la condicion de las desviaciones estandar
						aux2=self.solucion.Y[p][listaYt]=self.poscicion(self.solucion.Y[p][listaYt],self.v[p][0][listaYt],np.random.randint(2,6,len(listaYt)))
						aux3=self.v[p][0][listaYf]=self.velocidad(self.Xbest[0][listaYf],self.Xglobal[0][listaYf],self.v[p][0][listaYf],self.solucion.Y[p][listaYf])
						aux4=self.solucion.Y[p][listaYf]=self.poscicion(self.solucion.Y[p][listaYf],self.v[p][0][listaYf],1)
						Y=self.solucion.transformar(self.solucion.Y[p])
						if self.solucion.probar_restriccion(Y) == True:
							estado=True
							self.v[p][0][listaYt]=aux1
							self.solucion.Y[p][listaYt]=aux2
							self.v[p][0][listaYf]=aux3
							self.solucion.Y[p][listaYf]=aux4
					self.v[p][1][listaZt]=self.velocidad(self.Xbest[1][listaZt],self.Xglobal[1][listaZt],self.v[p][1][listaZt],self.solucion.Z[p][listaZt])
					self.solucion.Z[p][listaZt]=self.poscicion(self.solucion.Z[p][listaZt],self.v[p][1][listaZt],np.random.randint(2,6,len(listaZt)))
					self.v[p][1][listaZf]=self.velocidad(self.Xbest[1][listaZf],self.Xglobal[1][listaZf],self.v[p][1][listaZf],self.solucion.Z[p][listaZf])
					self.solucion.Z[p][listaZf]=self.poscicion(self.solucion.Z[p][listaZf],self.v[p][1][listaZf],1)
				stdY1=stdY2 #actualizar desviacion estandar
				stdZ1=stdZ2
			iteracion+=1 #aumenta iteracion
			print("mejor local: ",self.Xbest[2],"\t mejor global: ",self.Xglobal[2])
			log.write(str(("mejor local: ",self.Xbest[2],"\t mejor global: ",self.Xglobal[2])))
		print("iteracion final",iteracion-1)




	def velocidad(self,Xbest,Xglobal,v,x): #ecuacion velocidad
		return a*v+b*np.random.random(len(v))*(Xbest-x) + c*np.random.random(len(v))*(Xglobal-x)

	def poscicion(self,x,v1,mult):      # ecuacion pocicion
		r3=np.random.randint(-1,2,len(x)) #entrega -1 0 1
		temp1=self.sigmoide(x + mult*r3*self.phi + v1)
		return temp1  		


	def desviacionStandar(self,Y,Z):  #genera desviacion estandar para cada dimension
		
		#print(np.array(Y)[:,1])

		auxY=np.flipud(np.rot90(Y)) 
		auxZ=np.flipud(np.rot90(Z))

		desvY=np.zeros(self.instancia.Machines)
		desvZ=np.zeros(self.instancia.Parts)
		
		for i in range(self.instancia.Machines): #guarda desviacion estandar para máquinas x celdas
			desvY[i]=np.std(auxY[i],ddof=1)

		for j in range(self.instancia.Parts): #guarda desviacion estandar para partes x celdas
			desvZ[j]=np.std(auxZ[j],ddof=1)		
		return desvY, desvZ #Devuevle desviacion estandar generada


	def sigmoide(self,x):    #se aplica la regla sigmoidal para transformar cualquier número a numeros entre 0 y 111
		return (1/(1+np.exp(-x/k)))




def main():
	instancias=listdir("BoctorProblem_90_instancias/")
	for i in range(len(instancias)):
		instancia = problema(instancias[i])	#crear instancia a partir del archivo
		objetos=soluciones(instancia)			#generar soluciones en base de la instancia
		metaehuristia(instancia,objetos) 	#en metaehuristica pasar cuadro y las soluciones
		
		


if __name__ == '__main__':
	main()


#por qué se cuelga la funcion (me fije por las altos numeros de velocidad y eso afecta la asociacion  de los random) donde llega un momento que realiza pruebas de movimiento y las soluciones dadas ya no da soluciones factibles
#preguntas al profe, cuando se realiza la funcion de busqueda de mejor solucion cada 10 iteraciones, esta se debe hacer el mantenimiento o hacerlo junto con la busqueda de la iteracion 10
#a qué se refiere solucion local que aparece en el texto y como se agrega? 