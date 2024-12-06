import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from scipy import stats
import matplotlib.pyplot as plt

# Definir los límites de los parámetros del sistema híbrido
LIMITE_PANELSOLAR = (0, 50)  # Número de paneles solares
LIMITE_BATERIAS = (0, 30)    # Número de baterías
LIMITE_EOLO = (0, 10)        # Número de turbinas eólicas

# Función de evaluación del sistema híbrido (ficticia para este ejemplo)
def evaluar_sistema(individuo):
    num_paneles, num_baterias, num_turbinas = individuo
    costo = num_paneles * 500 + num_baterias * 200 + num_turbinas * 1000
    energia_generada = num_paneles * 5 + num_baterias * 2 + num_turbinas * 10
    eficiencia = energia_generada / costo
    return eficiencia,

# Configurar el entorno de DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_paneles", random.randint, *LIMITE_PANELSOLAR)
toolbox.register("attr_baterias", random.randint, *LIMITE_BATERIAS)
toolbox.register("attr_turbinas", random.randint, *LIMITE_EOLO)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_paneles, toolbox.attr_baterias, toolbox.attr_turbinas), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluar_sistema)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[0, 0, 0], up=[50, 30, 10], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Parámetros del algoritmo genético
NUM_GENERACIONES = 100
TAM_POBLACION = 50
PROB_CRUCE = 0.5
PROB_MUTACION = 0.2

def main():
    random.seed(42)
    poblacion = toolbox.population(n=TAM_POBLACION)
    
    # Estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Algoritmo genético
    for gen in range(NUM_GENERACIONES):
        # Selección y variación
        offspring = toolbox.select(poblacion, len(poblacion))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < PROB_CRUCE:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < PROB_MUTACION:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluación
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        poblacion[:] = offspring

        # Registro de estadísticas
        record = stats.compile(poblacion) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)
    
    # Obtener el mejor individuo
    mejor_individuo = tools.selBest(poblacion, k=1)[0]
    
    # Convertir a DataFrame para visualización
    df = pd.DataFrame(logbook)
    df.set_index('gen', inplace=True)
    
    # Visualización de resultados
    df[['avg', 'min', 'max']].plot()
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title('Evolución del Fitness a lo largo de las Generaciones')
    plt.show()
    
    return mejor_individuo

# Ejecutar el algoritmo genético y obtener el mejor dimensionamiento
mejor_dimensionamiento = main()
print(f"Mejor dimensionamiento: {mejor_dimensionamiento}")
