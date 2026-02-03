#--------------------------------------------------------------------
#Librerías
#-----------------------------------------------------------------------
import math # e utiliza para operaciones matemáticas (en este caso usamos raíz cuadrada)
import pandas as pd #Se utiliza para crear DataFrames y exportar archivos Excel
from collections import Counter #Se utiliza para contar votos de clases en KNN
import tkinter as tk # Librería para crear la interfaz gráfica
from tkinter import filedialog, messagebox #Para abrir archivos y mostrar mensajes

#--------------------------------------------------------------------
#LEEMOS DATOS LÍENA POR LÍNEA
#-----------------------------------------------------------------------

#Esta función lee los archivos del dataset IRIS.
def leer_datos(nombre_archivo, tiene_clase=True):

#nombre_archivo: ruta del archivo .data
#tiene_clase: indica si el archivo incluye la clase (True para train y test)

    datos = []  #Lista donde se almacenarán los datos leídos
    
    #Se abre el archivo en modo lectura
    with open(nombre_archivo, "r") as f:
        #Se recorre el archivo línea por línea
        for linea in f:
            # e ignoran líneas vacías
            if linea.strip() == "":
                continue

            #Se separa la línea usando la coma como delimitador
            partes = linea.strip().split(",")

            #Se convierten los primeros cuatro valores a tipo float (atributos)
            x = list(map(float, partes[:4]))

            #Si el archivo contiene clase (train o test)
            if tiene_clase:
                #Se guarda (atributos, clase)
                datos.append((x, partes[4]))
            else:
                #Si no hay clase (new data), solo se guardan los atributos
                datos.append(x)

    #Se devuelve la lista de datos procesados
    return datos


#----------------------------------------------------------
#CALCULAMOS LA DISTANCIA EUCLIDIANA CON LA FORMULA O MEDIDA PROPORCIONADA
#---------------------------------------------------------

#Calcula la distancia euclidiana entre dos puntos p y q
def distancia(p, q):
    return math.sqrt(sum((p[i] - q[i])**2 for i in range(len(p))))

#---------------------------------------------------------------------
#IMPLEMENTACIÓN DE KNN
#-----------------------------------------------------------------------

#Implementación del algoritmo K-Nearest Neighbors

def knn(datos_entrenamiento, punto, k):
#datos_entrenamiento: conjunto de entrenamiento
#punto: objeto a clasificar
# k: número de vecinos más cercanos

    #Calcular distancias
    # Se calcula la distancia del punto a cada dato de entrenamiento
    distancias = [
        (distancia(punto, x), clase)
        for x, clase in datos_entrenamiento
    ]

     #Se ordenan las distancias y se seleccionan los k vecinos más cercanos
    vecinos = sorted(distancias)[:k]

    # Se extraen las clases de los vecinos seleccionados
    clases = [clase for _, clase in vecinos]
    
    #Se asigna la clase por votación mayoritaria
    return Counter(clases).most_common(1)[0][0]

#-----------------------------------------------------------------
#VARIABLES GLOBALES PARA ALMACENAR RUTAS DE ARCHIVOS
#-----------------------------------------------------------------

ruta_train = "" #Ruta del archivo de entrenamiento
ruta_test = "" #Ruta del archivo de prueba
ruta_new = "" #Ruta del archivo de nuevos datos

#-----------------------------------------------------------------
#FUNCIONES GUI
#-----------------------------------------------------------------

#Abre un cuadro de diálogo para seleccionar el archivo de entrenamiento
def cargar_train():
    global ruta_train
    ruta_train = filedialog.askopenfilename(title="Selecciona DataTrained")
    lbl_train.config(text=ruta_train)

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

#--------------------------------------------------------------------
#FUNCIÓN PRINCIPAL QUE EJECUTA EL KNN
#------------------------------------------------------------------

#Ejecuta el proceso:
# Lectura de datos
# Evaluación del modelo
# Selección del mejor K
# Clasificación de nuevos datos

def ejecutar_knn():
    #Verifica que los tres archivos hayan sido cargados
    if ruta_train == "" or ruta_test == "" or ruta_new == "":
        messagebox.showerror("Error", "Debes cargar los tres archivos")
        return
    
    #Lectura de los datos
    train_data = leer_datos(ruta_train, True)
    test_data = leer_datos(ruta_test, True)
    new_data = leer_datos(ruta_new, False)

    #-----------------------------------------------------------------
    #GENERACIÓN DE SET-PRUEBA-K357.xlsx
    #-----------------------------------------------------------------

    filas = []  #Filas del archivo Excel
    exactitudes = {3: 0, 5: 0, 7: 0}
    total = len(test_data)

     #Se evalúa el modelo para cada objeto de prueba
    for x, clase_real in test_data:
        k3 = knn(train_data, x, 3)
        k5 = knn(train_data, x, 5)
        k7 = knn(train_data, x, 7)

        filas.append([x[0], x[1], x[2], x[3], clase_real, k3, k5, k7])

        if k3 == clase_real: exactitudes[3] += 1
        if k5 == clase_real: exactitudes[5] += 1
        if k7 == clase_real: exactitudes[7] += 1

    #Cálculo de exactitud
    for k in exactitudes:
        exactitudes[k] /= total

    #Creación del archivo Excel
    df_set = pd.DataFrame(
        filas,
        columns=[
            "SepalLength", "SepalWidth",
            "PetalLength", "PetalWidth",
            "Clase real",
            "Clase K=3", "Clase K=5", "Clase K=7"
        ]
    )

    df_set.to_excel("SET-PRUEBA-K357.xlsx", index=False)

    #-----------------------------------------------------------------
    #MEJOR K
    #-----------------------------------------------------------------

    mejor_k = max(exactitudes, key=exactitudes.get)

    #-----------------------------------------------------------------
    #NEWDATA-BEST-K.xlsx o CLASIFICACIÓN DE NUEVOS DATOS
    #-----------------------------------------------------------------

    filas_new = []

    for x in new_data:
        clase = knn(train_data, x, mejor_k)
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

ventana = tk.Tk()
ventana.title("Ventana de carga de archivos")
ventana.geometry("600x350")

tk.Label(ventana, text="K-Nearest Neighbors usando el Dataset IRIS", font=("Arial", 14)).pack(pady=10)

tk.Button(ventana, text="CARGAR --> DataTrained-iris.data", command=cargar_train).pack(pady=8)
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

ventana.mainloop()
