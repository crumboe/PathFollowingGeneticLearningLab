#========================================
# No Changes need to be made in this cell
#========================================

# Genetic Algorithm with tracking
from config import *
import numpy as np

from ast import If

class GeneticAlgorithm:
    def __init__(self, population_size=None, generations=None, visualize_best=None,
                 evaluate = None,individual_ranges=None):
        """
        Args:
            population_size: Number of individuals in each generation
            generations: Number of generations to evolve
            visualize_best: Whether to show visualization during evolution

        """
        # Use hyperparameters if not specified
        self.evaluate = evaluate
        self.population_size = population_size if population_size is not None else POPULATION_SIZE
        self.generations = generations if generations is not None else GENERATIONS
        self.individuala_ranges = individual_ranges
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.individual_fitness_history = []
        self.current_generation_fitness = None
        self.mutation_rate = MUTATION_RATE
        self.mutation_effect = MUTATION_EFFECT

    def initialize_population(self):
        # Smarter initialization - moderate values in a reasonable range
        population = []

        # Get range sizes
        

        # Initialize around moderate values (30-60% of range from minimum)
        for _ in range(self.population_size):
            individual = []
            for r in self.individuala_ranges:
                individual.append(np.random.uniform(r[0], r[1]))    
            population.append(individual)
        return np.array(population)

    def evaluate_population(self, population, current_gen):
        fitness = []
        for idx, ind in enumerate(population):
            # Determine if we should render this individual
            # render = self.visualize_best and (idx == 0 or current_gen == self.generations)
            # render = True
            render = VISUALIZE_ALL_INDIVIDUALS_DURING_TRAINING
            fit = self.evaluate(ind, render=render, ga_instance=self, current_gen=current_gen,episodes=EPISODES_PER_EVAL,current_individual=idx+1)
            fitness.append(fit)

            # Update current generation fitness for visualization
            self.current_generation_fitness = np.array(fitness + [0] * (self.population_size - len(fitness)))
        best_inx = np.argmax(fitness)
        if VISUALIZE_ALL_INDIVIDUALS_DURING_TRAINING == False and VISUALIZE_BEST:  # If False, only the best individual per generation is visualized
 #render the best individual of the generation
            self.evaluate(population[best_inx], render=True,
                        ga_instance=self, current_gen=current_gen, current_individual=best_inx+1,episodes=1)
        return np.array(fitness)

    def selection(self, population, fitness):
        # Tournament selection
        selected = []
        for _ in range(self.population_size):
            i, j = np.random.choice(self.population_size, 2, replace=False)
            selected.append(population[i] if fitness[i] > fitness[j] else population[j])
        return np.array(selected)

    def crossover(self, parent1, parent2):
        # Real-valued BLX-alpha crossover (better for continuous PID gains)
        if np.random.rand() >= CROSSOVER_RATE:
            return parent1.copy(), parent2.copy()
        alpha = self.mutation_effect  # exploration expansion factor
        ranges = self.individuala_ranges
        child1 = parent1.copy()
        child2 = parent2.copy()
        for i in range(len(parent1)):
            cmin = min(parent1[i], parent2[i])
            cmax = max(parent1[i], parent2[i])
            interval = cmax - cmin
            lower = cmin - interval * alpha
            upper = cmax + interval * alpha
            g1 = np.random.uniform(lower, upper)
            g2 = np.random.uniform(lower, upper)
            child1[i] = np.clip(g1, ranges[i][0], ranges[i][1])
            child2[i] = np.clip(g2, ranges[i][0], ranges[i][1])
        # Occasionally inject averaged stability child
        if np.random.rand() < 0.1:
            child1 = np.clip((parent1 + parent2) / 2, [r[0] for r in ranges], [r[1] for r in ranges])
        return child1, child2

    def mutate(self, individual):
        # Gaussian mutation with respect to the exploration ranges
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                # Mutate with Gaussian noise scaled by MUTATION_EFFECT
                individual[i] += np.random.normal(0, abs(individual[i]) * self.mutation_effect)
                # Clip to stay within exploration ranges
                ranges = self.individuala_ranges
                individual[i] = np.clip(individual[i], ranges[i][0], ranges[i][1])
        return individual

    def evolve(self):
        population = self.initialize_population()

        for gen in range(1, self.generations + 1):
            fitness = self.evaluate_population(population, gen)

            best_idx = np.argmax(fitness)
            best_fitness = fitness[best_idx]
            avg_fitness = np.mean(fitness)

            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            self.individual_fitness_history.append(fitness)
            print(f"\nGeneration {gen}/{self.generations}")
            print(f"Best Fitness: {best_fitness:.2f}, Avg Fitness: {avg_fitness:.2f}")
            print("-" * 60)
            if best_fitness - avg_fitness < .1:
                break
            # Selection

            # Crossover and Mutation
            new_population = []
            if ELITISM:
                # Keep the top N best individuals without mutation
                num_elites = min(ELITE_COUNT if 'ELITE_COUNT' in globals() else 3, self.population_size)
                elite_indices = np.argsort(fitness)[-num_elites:][::-1]  # Get indices of top N individuals
                for elite_idx in elite_indices:
                    elite = population[elite_idx].copy()
                    new_population.append(elite)

            selected = self.selection(population, fitness)

            for i in range(0, self.population_size - len(new_population), 2):
                parent1, parent2 = selected[i], selected[i+1] if i+1 < self.population_size else selected[0]
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])

            population = np.array(new_population[:self.population_size])
            self.mutation_rate *=MUTATION_EFFECT_DECAY
            # self.mutation_effect *=MUTATION_EFFECT_DECAY

        fitness = self.evaluate_population(population, self.generations)
        best_idx = np.argmax(fitness)
        return population[best_idx], fitness[best_idx]