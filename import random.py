import random

# Definir el número de generaciones y el tamaño de la población
NUM_GENERACIONES = 100
TAM_POBLACION = 50

# Definir los límites de los parámetros del sistema híbrido
LIMITE_PANELSOLAR = (0, 50)  # Número de paneles solares
LIMITE_BATERIAS = (0, 30)    # Número de baterías
LIMITE_EOLO = (0, 10)        # Número de turbinas eólicas

# Función de evaluación del sistema híbrido (ficticia para este ejemplo)
def evaluar_sistema(num_paneles, num_baterias, num_turbinas):
    costo = num_paneles * 500 + num_baterias * 200 + num_turbinas * 1000
    energia_generada = num_paneles * 5 + num_baterias * 2 + num_turbinas * 10
    eficiencia = energia_generada / costo
    return eficiencia

# Generar una población inicial
def generar_poblacion():
    return [(random.randint(*LIMITE_PANELSOLAR),
             random.randint(*LIMITE_BATERIAS),
             random.randint(*LIMITE_EOLO)) for _ in range(TAM_POBLACION)]

# Selección de los mejores individuos
def seleccionar_mejores(poblacion):
    return sorted(poblacion, key=lambda x: evaluar_sistema(*x), reverse=True)[:TAM_POBLACION // 2]

# Cruzar dos individuos para crear descendencia
def cruzar(padre, madre):
    hijo1 = (padre[0], madre[1], madre[2])
    hijo2 = (madre[0], padre[1], padre[2])
    return hijo1, hijo2

# Mutar un individuo
def mutar(individuo):
    idx = random.randint(0, 2)
    if idx == 0:
        return (random.randint(*LIMITE_PANELSOLAR), individuo[1], individuo[2])
    elif idx == 1:
        return (individuo[0], random.randint(*LIMITE_BATERIAS), individuo[2])
    else:
        return (individuo[0], individuo[1], random.randint(*LIMITE_EOLO))

# Algoritmo genético
def algoritmo_genetico():
    poblacion = generar_poblacion()
    for _ in range(NUM_GENERACIONES):
        mejores = seleccionar_mejores(poblacion)
        nueva_poblacion = []
        while len(nueva_poblacion) < TAM_POBLACION:
            padre, madre = random.sample(mejores, 2)
            hijos = cruzar(padre, madre)
            nueva_poblacion.extend(hijos)
        poblacion = [mutar(individuo) for individuo in nueva_poblacion]
    return seleccionar_mejores(poblacion)[0]

# Ejecutar el algoritmo genético y obtener el mejor dimensionamiento
mejor_dimensionamiento = algoritmo_genetico()
print(f"Mejor dimensionamiento: {mejor_dimensionamiento}")
