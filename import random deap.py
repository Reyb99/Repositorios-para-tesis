import random
import numpy as np
from deap import base, creator, tools, algorithms

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
    
    # Algoritmo genético
    poblacion, logbook = algorithms.eaSimple(poblacion, toolbox, cxpb=PROB_CRUCE, mutpb=PROB_MUTACION,
                                             ngen=NUM_GENERACIONES, stats=stats, verbose=True)
    
    # Obtener el mejor individuo
    mejor_individuo = tools.selBest(poblacion, k=1)[0]
    return mejor_individuo

# Ejecutar el algoritmo genético y obtener el mejor dimensionamiento
mejor_dimensionamiento = main()
print(f"Mejor dimensionamiento: {mejor_dimensionamiento}")
