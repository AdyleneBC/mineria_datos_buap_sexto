# ================================
# PRÁCTICA 4 - MÉTODO DE WARD
# IMPLEMENTACIÓN DESDE CERO
# ================================

# Importo la librería matplotlib para poder graficar el dendrograma
import matplotlib.pyplot as plt

# -------------------------------
# 1. LEER CSV (SIN LIBRERÍAS)
# -------------------------------
# Defino una función para leer un archivo CSV
def leer_csv(ruta):
    # Inicializo lista para guardar los datos (features)
    datos = []
    # Inicializo lista para guardar las clases (última columna)
    clases = []
    # Abro el archivo en modo lectura
    with open(ruta, 'r') as f:
        # Recorro cada línea del archivo
        for linea in f:
            # Quito espacios/saltos de línea y separo por comas
            valores = linea.strip().split(',')
            # Creo una lista para la fila actual
            fila = []
            # Recorro todos los valores excepto el último (clase)
            for i in range(len(valores)-1):
                try:
                    # Intento convertir a float
                    fila.append(float(valores[i]))
                except:
                    # Si falla, guardo 0
                    fila.append(0)
            # Agrego la fila a los datos
            datos.append(fila)
            # Guardo la última columna como clase
            clases.append(valores[-1])

    # Regreso los datos y las clases
    return datos, clases


# -------------------------------
# 2. NORMALIZACIÓN (MIN-MAX)
# -------------------------------
# Defino función para normalizar los datos
def normalizar(datos):
    # Obtengo la dimensión (número de columnas)
    dim = len(datos[0])

    # Calculo el mínimo por columna
    min_vals = [min(fila[i] for fila in datos) for i in range(dim)]
    # Calculo el máximo por columna
    max_vals = [max(fila[i] for fila in datos) for i in range(dim)]

    # Lista para datos normalizados
    datos_norm = []
    # Recorro cada fila
    for fila in datos:
        nueva = []
        for i in range(dim):
            # Evito división entre cero
            if max_vals[i] - min_vals[i] == 0:
                nueva.append(0)
            else:
                # Aplico fórmula de normalización Min-Max
                nueva.append((fila[i] - min_vals[i]) / (max_vals[i] - min_vals[i]))
        # Guardo la fila normalizada
        datos_norm.append(nueva)

    # Regreso los datos normalizados
    return datos_norm


# -------------------------------
# 3. DISTANCIA EUCLIDIANA
# -------------------------------
# Calculo la distancia entre dos puntos
def distancia(a, b):
    suma = 0
    # Recorro cada dimensión
    for i in range(len(a)):
        # Sumo la diferencia al cuadrado
        suma += (a[i] - b[i])**2
    # Regreso la raíz cuadrada
    return suma**0.5


# -------------------------------
# 4. CENTROIDE
# -------------------------------
# Calculo el centroide de un cluster
def centroide(cluster):
    # Número de puntos
    n = len(cluster)
    # Dimensión
    dim = len(cluster[0])

    # Inicializo el centroide en cero
    c = [0]*dim
    # Sumo cada punto
    for punto in cluster:
        for i in range(dim):
            c[i] += punto[i]

    # Promedio
    for i in range(dim):
        c[i] /= n

    # Regreso el centroide
    return c


# -------------------------------
# 5. COSTO DE WARD
# -------------------------------
# Calculo el costo de unir dos clusters
def ward_cost(c1, c2):
    # Tamaños de los clusters
    n1 = len(c1)
    n2 = len(c2)

    # Obtengo centroides
    mu1 = centroide(c1)
    mu2 = centroide(c2)

    # Distancia entre centroides
    d = distancia(mu1, mu2)

    # Fórmula de Ward
    return (n1 * n2) / (n1 + n2) * (d**2)


# -------------------------------
# 6. CLUSTERING + HISTORIAL
# -------------------------------
# Implemento clustering jerárquico
def ward_clustering(datos):
    # Inicio cada punto como cluster individual
    clusters = [[p] for p in datos]
    # IDs de clusters
    ids = list(range(len(datos)))

    # Historial para dendrograma
    historial = []
    # ID siguiente
    next_id = len(datos)

    # Mientras haya más de un cluster
    while len(clusters) > 1:
        # Inicializo mejor par
        mejor_i, mejor_j = 0, 1
        mejor_costo = ward_cost(clusters[0], clusters[1])

        # Busco el par con menor costo
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                costo = ward_cost(clusters[i], clusters[j])
                if costo < mejor_costo:
                    mejor_costo = costo
                    mejor_i, mejor_j = i, j

        # Obtengo IDs
        id1 = ids[mejor_i]
        id2 = ids[mejor_j]
        tamaño = len(clusters[mejor_i]) + len(clusters[mejor_j])

        # Guardo en historial
        historial.append([id1, id2, mejor_costo, tamaño])

        # Uno clusters
        nuevo = clusters[mejor_i] + clusters[mejor_j]

        # Elimino los clusters usados
        clusters.pop(mejor_j)
        clusters.pop(mejor_i)

        ids.pop(mejor_j)
        ids.pop(mejor_i)

        # Agrego el nuevo cluster
        clusters.append(nuevo)
        ids.append(next_id)

        next_id += 1

    # Regreso historial
    return historial


# -------------------------------
# 7. OBTENER K CLUSTERS
# -------------------------------
# Obtengo k clusters finales
def obtener_clusters(datos, k):
    clusters = [[p] for p in datos]

    # Mientras haya más clusters que k
    while len(clusters) > k:
        mejor_i, mejor_j = 0, 1
        mejor_costo = ward_cost(clusters[0], clusters[1])

        # Busco mejor unión
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                costo = ward_cost(clusters[i], clusters[j])
                if costo < mejor_costo:
                    mejor_costo = costo
                    mejor_i, mejor_j = i, j

        # Uno clusters
        nuevo = clusters[mejor_i] + clusters[mejor_j]

        clusters.pop(mejor_j)
        clusters.pop(mejor_i)

        clusters.append(nuevo)

    return clusters


# -------------------------------
# 8. GUARDAR CSV (EXCEL)
# -------------------------------
# Guardo clusters en archivo CSV
def guardar_clusters(nombre, clusters):
    with open(nombre, 'w') as f:
        for i, cluster in enumerate(clusters):
            for punto in cluster:
                # Convierto fila a texto
                fila = ",".join(str(x) for x in punto)
                # Escribo con su cluster
                f.write(fila + ",Cluster_" + str(i+1) + "\n")


# -------------------------------
# 9. MATRIZ DE CONFUSIÓN
# -------------------------------
# Comparo clusters con clases reales
def matriz_confusion(clusters, clases_reales):
    # Mapeo punto a cluster
    mapa = {}
    for i, cluster in enumerate(clusters):
        for punto in cluster:
            mapa[tuple(punto)] = i

    # Predicciones
    y_pred = []
    for i in range(len(clases_reales)):
        punto = tuple(datos[i])
        y_pred.append(mapa[punto])

    # Clases únicas
    etiquetas = list(set(clases_reales))

    # Creo matriz clases x clusters
    matriz = [[0]*len(clusters) for _ in range(len(etiquetas))]

    # Lleno matriz
    for i in range(len(clases_reales)):
        fila = etiquetas.index(clases_reales[i])
        col = y_pred[i]
        matriz[fila][col] += 1

    return matriz, etiquetas


# -------------------------------
# 10. DENDROGRAMA
# -------------------------------
# Dibujo el dendrograma
def dibujar_dendrograma(historial, cortes=None):
    n = len(historial) + 1

    # Posiciones iniciales
    pos = {i: i for i in range(n)}
    altura = {i: 0 for i in range(n)}

    # Creo figura
    fig, ax = plt.subplots(figsize=(12, 6))

    max_dist = 0

    # Dibujo uniones
    for i, (c1, c2, dist, size) in enumerate(historial):
        x1 = pos[c1]
        x2 = pos[c2]
        y1 = altura[c1]
        y2 = altura[c2]

        ax.plot([x1, x1], [y1, dist], c='black')
        ax.plot([x2, x2], [y2, dist], c='black')
        ax.plot([x1, x2], [dist, dist], c='black')

        nuevo = n + i
        pos[nuevo] = (x1 + x2) / 2
        altura[nuevo] = dist

        if dist > max_dist:
            max_dist = dist

    # Dibujo líneas de corte
    if cortes:
        for c in cortes:
            y = max_dist * c
            ax.axhline(y=y, color='red', linestyle='--')

    # Etiquetas
    plt.title("Dendrograma - Método de Ward")
    plt.xlabel("Datos")
    plt.ylabel("Distancia")

    # Guardo y muestro
    plt.savefig("dendrograma_con_cortes.png")
    plt.show()


# ================================
# MAIN
# ================================

# Leo datos
datos, clases = leer_csv("DATA (1).csv")

# Normalizo
datos = normalizar(datos)

# Genero dendrograma
historial = ward_clustering(datos)
dibujar_dendrograma(historial, [0.051, 0.15, 0.30])

# Genero clusters
clusters_3 = obtener_clusters(datos, 3)
clusters_5 = obtener_clusters(datos, 5)
clusters_8 = obtener_clusters(datos, 8)

# Guardo resultados
guardar_clusters("clusters_3.csv", clusters_3)
guardar_clusters("clusters_5.csv", clusters_5)
guardar_clusters("clusters_8.csv", clusters_8)

# Matriz de confusión
matriz, etiquetas = matriz_confusion(clusters_8, clases)

# Imprimo resultados
print("\nMatriz de Confusión:")
for fila in matriz:
    print(fila)