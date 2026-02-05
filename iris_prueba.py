#LIBRERÍAS 

import math # librería para distancia euclidiana (raiz)
import pandas as pd #para crear dataframes y exposrlos
from collections import Counter #se utiliza para contar votos de clases en knn
import tkinter as tk #interfaz
from tkinter import filedialog, messagebox #abrir archivos y mostrar mensajes


#LECTURA DE DATOS, LÍNEA POR LÍNEA

def leer_datos(nombre_archivo, tiene_clase=True):
    
    datos = [] #Lista donde almacenamos los datos leídos
    
    with open(nombre_archivo, "r") as f: #open es una función de python, con with abrimos y cerramos terminando y guardamos en f
        
        #recorremos el archivo línea por línea
        for linea in f:
            
            #ignoramos líneas vacías
            if linea.strip() == "": #ignoramos líneas vaciás y no las guardamos
                continue
            
            #se separa la línea usando la coma como delimitador
            partes = linea.strip().split(",") #elimina espacios al inicio y al final  
#partes[0] = '5.1'
#partes[1] = '3.5'
#partes[2] = '1.4'
#partes[3] = '0.2'
#partes[4] = 'Iris-setosa'
            
            #se convierten los primeros cuatro valores a tipo float (atributos)
            x = list(map(float, partes [:4])) #Final --> x = [5.1, 3.5, 1.4, 0.2]
            
            #si el archivo contiene clase (train o test)
            if tiene_clase:
                #se guarda (atributos, clase)
                datos.append((x, partes[4]))
            else:
                #si no hay clase (newdata), solo se guardan los atributos
                datos.append(x)
    
    #se devuelve la lista de datos procesados        
    return datos
  

#1. DISTANCIA EUCLIDIANA      

#Caluclamos la ditancia euclidiana entre dos puntos p y q

def distancia(p, q):
    return math.sqrt(sum((p[i] - q[i])**2) for i in range(len(p)))

#p-->un objeto (ejemplo: un registro del test o new data)
#q--> otro objeto (ejemplo: un registro del entrenamiento)

#sum --> suma todas las diferencias al cuadrado

#FOR -->recorre todas las posiciones de p
#Como p tiene 4 valores, esto recorre:
#i = 0
#i = 1
#i = 2
#i = 3
#Así se calcula la distancia usando los 4 atributos
    
#distancia euclidiana entre dos objetos en 4 dimensiones    



#2. IMPLEMENTACIÓN DEL KNN

def knn(datos_entrenamiento, punto, k):
#datos_entrenamiento: conjunto de entrenamiento
#punto: objeto a clasificar
# k: número de vecinos más cercanos
    
        #Calcular distancias
        distancias = [
            (distancia(punto, x), clase)
            for x, clase in datos_entrenamiento
        ]
        
        #datos_entrenamiento --> x = [5.1, 3.5, 1.4, 0.2]
                            #    clase = "Iris-setosa"
                            
       # distancia(punto, x) -->
        #Calcula la distancia euclidiana entre:
        #punto (objeto que quieremos clasificar)
        #x (un objeto del entrenamiento)
        #clase (Guarda la clase real del dato de entrenamiento)
        
        #se ordenan las distancias y se seleccionan los k vecinos más cercanos
        vecinos = sorted(distancias)[:k]  #Ordena la lista distancias de menor a mayor
                                            #K-->toma solo los primeros K elementos  --> 0:k
                                            
        #se extraen las clases de los vecinos seleccionados --> [0.9, "iris"]
        clases = [clase for _, clase in vecinos] #--> Esta línea crea una lista llamada clases que contiene solo las clases de los vecinos más cercanos
                                                #_ --> Sí existe ese valor, pero no me interesa
        #después usamos esto para hacer votación y ver qué clase aparece más
        
        #Se asigna la clase por votaci´pn mayorista
        return Counter(clases).most_common(1)[0][0] #counter(clases)--> {"Iris-setosa": 2, "Iris-versicolor": 1}
                        # el más común --> [("Iris-setosa", 2)]
                        # [0] --> ("Iris-setosa", 2)
                        # [0] --> "Iris-setosa"
    
    
#VARIBALES GLOBALES PARA ALMACENAR RUTAS DE ARCHIVOS

ruta_train = ""
ruta_test = ""
ruta_new = ""

#Esto sirve para guardar las rutas cuando cargamos desde la GUI, 
#cuando damos clic en ejecutar_kkn, La función ejecutar_knn() usa esas rutas

#Son globales porque:
#Porque las rutas se cargan en funciones diferentes:
#cargar_train()
#cargar_test()
#cargar_new()
#y después se necesitan en:
#ejecutar_knn()
#Si no fueran globales, esas funciones no podrían compartir esa información.

#Se inicializan en "" porque se van actualizando las rutas y ayudamos a obligar al usuario a cargar un algo


#FUNCIONES DE GUI

def cargar_train():
    global ruta_train
    ruta_train = filedialog.askopenfile(title="seleccionar DataTrained") # filedialog.askopenfilename() --> devuelve la ruta completa del archivo seleccionado
    lbl_train.config(text=ruta_train) #Esta línea actualiza una etiqueta (lbl_train) dentro de la interfaz gráfica --> Sirve para mostrar en pantalla la ruta del archivo seleccionado, de modo que el usuario tenga confirmación visual de que el archivo fue cargado correctamente
    
#Se abre un cuadro de diálogo para seleccionar el archivo de prueba
def cargar_test():
    global ruta_test
    ruta_test = filedialog.askopenfilename(title="Selecciona TestData")
    lbl_test.config(text=ruta_test)

#Abre un cuadro de diálogo para seleccionar el archivo de nuevos datos
def cargar_new():
    global ruta_new
    ruta_new = filedialog.askopenfilename(title="Selecciona NewData")
    lbl_new.config(text=ruta_new)
                    

#FUNCIÓN PRINCIPAL QUE EJECUTA EL KNN

#Ejecuta el proceso:
# Lectura de datos
# Evaluación del modelo
# Selección del mejor K
# Clasificación de nuevos datos

def ejecutar_knn():
    
    #Verifica que los tres archivos ya hayan sido cargados
    if ruta_train == "" or ruta_test == "" or ruta_new == "":
        messagebox.showerror("Error", "Debes cargar los tres archivos")
        return
    
    #Lectura de los datos
    train_data = leer_datos(ruta_train, True)
    test_data = leer_datos(ruta_test, True)
    new_data = leer_datos(ruta_new, False)
    
    #GENERACIÓN DE SET-PRUEBA-K357
    
    filas = [] #Filas del archivo excel
    exactitudes = {3: 0, 5: 0, 7: 0}
    total = len(test_data) #--> Se obtiene el número total de registros del conjunto de prueba
    
    #Se crea un diccionario llamado exactitudes para llevar el conteo de aciertos del modelo para cada valor de K
        #exactitudes[3] contará cuántos registros fueron clasificados correctamente con K=3
        #exactitudes[5] contará los aciertos con K=5
        #exactitudes[7] contará los aciertos con K=7
        #Se inicia en 0 porque todavía no se han evaluado datos
        
    #Se evalúaa el modelo para cada objeto de prueba
    for x, clase_real in test_data: #--> Aquí se recorre el conjunto de datos de prueba (test_data) registro por registro
        k3 = knn(train_data, x, 3) #test_data--> ([atributos], clase)
        k5 = knn(train_data, x, 5)
        k7 = knn(train_data, x, 7)
        
        #Se clasifica el registro x utilizando el algoritmo KNN con:
        #conjunto de entrenamiento: train_data
        # punto a clasificar: x
        # valor de K: 3
        
        filas.append([x[0], x[1], x[2], x[3], clase_real, k3, k5, k7]) #-> append() significa: agrega un elemento al final de la lista filas.
        # En este caso, agrega una fila completa (una lista)
        
        if k3 == clase_real: exactitudes[3] += 1 
        if k5 == clase_real: exactitudes[5] += 1
        if k7 == clase_real: exactitudes[7] += 1
        #Compara si la clase predicha usando K = 3 (k3)
        # es igual a la clase real del registro (clase_real)
        #Si son iguales, significa que el modelo clasificó correctamente, entonces:
            #se suma 1 al contador de aciertos de K=3
            
        #Cálculo de exactitud
        for k in exactitudes: #Este ciclo recorre las claves del diccionario exactitudes
            exactitudes[k] /= total #--> divide el número de aciertos entre el total de registros de prueba
            
            #Creación del archivo Excel
        df_set = pd.DataFrame( #--> Aquí se crea un DataFrame, que es una estructura tipo tabla (filas y columnas) proporcionada por la librería pandas
            filas,
            columns=[
                "SepalLength", "SepalWidth",
                "PetalLength", "PetalWidth",
                "Clase real",
                "Clase K=3", "Clase K=5", "Clase K=7"
            ]
        )

        df_set.to_excel("SET-PRUEBA-K357.xlsx", index=False)
        
        #Filas --> Se le pasa la lista filas, que contiene todos los registros del conjunto de prueba con:
        # los 4 atributos
        # la clase real
        # las predicciones para K=3, K=5 y K=7
        # Cada elemento dentro de filas es una fila completa
        
        
        #MEJOR K
        mejor_k = max(exactitudes, key=exactitudes.get) # --> Esta instrucción busca dentro del diccionario exactitudes el valor de K que tenga la mayor exactitud pero usanod sus llaves y no comparando 3, 5 y 7
        
        
        #NEWDATA-BEST-K.xlsx o CLASIFICACIÓN DE NUEVOS DATOS

        filas_new = []
        
        for x in new_data:
            clase = knn(train_data, x, mejor_k)
            #Para cada objeto desconocido, se ejecuta el algoritmo KNN usando:
            # train_data como conjunto de entrenamiento,
            # x como el punto a clasificar,
            # mejor_k como el valor de K seleccionado previamente
            # El resultado se guarda en la variable clase, la cual corresponde a la clase predicha por el modelo
            
            filas_new.append([x[0], x[1], x[2], x[3], clase])
            
        
        df_new = pd.DataFrame(
            filas_new,
            columns=[
                "SepalLength", "SepalWidth",
                "PetalLength", "PetalWidth",
                "Clase asignada"
            ]
        )

        df_new.to_excel("NEWDATA-BEST-K.xlsx", index=False)

        #Mensaje final con exactitudes y mejor k
        messagebox.showinfo(
            "Proceso terminado",
            f"Exactitudes:\n"
            f"K=3 → {exactitudes[3]:.2f}\n"
            f"K=5 → {exactitudes[5]:.2f}\n"
            f"K=7 → {exactitudes[7]:.2f}\n\n"
            f"Mejor K = {mejor_k}"
        )
            
#-----------------------------------------------------------------
#DISEÑO DE LA VENTANA EMERGENTE GUI
#-----------------------------------------------------------------

#Aquí podemos ver intuitivamente el diseño de la GUI, con la ventana de carga de archivos
# así como el uso de botones e implementeción de las funciones antes implementadas

ventana = tk.Tk() #--> Tk() es una clase de Tkinter que crea la ventana raíz (la ventana principal)
ventana.title("Ventana de carga de archivos")
ventana.geometry("600x350")

tk.Label(ventana, text="K-Nearest Neighbors usando el Dataset IRIS", font=("Arial", 14)).pack(pady=10)

tk.Button(ventana, text="CARGAR --> DataTrained-iris.data", command=cargar_train).pack(pady=8) #--> Sirve para colocar (dibujar) el elemento dentro de la ventana
lbl_train = tk.Label(ventana, text="No cargado", wraplength=500)
lbl_train.pack()

tk.Button(ventana, text="CARGAR --> TestData-iris.data", command=cargar_test).pack(pady=8)
lbl_test = tk.Label(ventana, text="No cargado", wraplength=500)
lbl_test.pack()

tk.Button(ventana, text="CARGAR --> NewData-iris.data", command=cargar_new).pack(pady=8)
lbl_new = tk.Label(ventana, text="No cargado", wraplength=500)
lbl_new.pack()

tk.Button(
    ventana,
    text="EJECUTAR KNN",
    bg="#AB1679",
    fg="white",
    font=("Arial", 12),
    command=ejecutar_knn
).pack(pady=25)

tk.Button(
    ventana,
    text="Salir",
    bg="red",
    fg="black",
    font=("Arial", 11),
    command="exit"
).pack(pady=5)

ventana.mainloop() #--> Esta línea pone en ejecución la interfaz gráfica y la mantiene abierta esperando acciones del usuario

            
        
    
    