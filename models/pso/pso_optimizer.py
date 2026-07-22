"""
Particle Swarm Optimization (PSO) for TFT Hyperparameter Tuning.
Custom implementation of continuous and discrete parameter search from scratch.
"""

import numpy as np
import logging
from typing import Dict, Any, Callable, List, Tuple

logger = logging.getLogger(__name__)


class Particle:
    """Represents an individual particle in the hyperparameter search space."""

    def __init__(self, param_bounds: Dict[str, Tuple[float, float, str]]):
        """
        Args:
            param_bounds: Dictionary defining hyperparameter limits.
                          Format: {'param_name': (min_val, max_val, type)}
                          Types: 'float' or 'int'
        """
        self.param_bounds = param_bounds
        self.position = {}
        self.velocity = {}
        self.pbest_position = {}
        self.pbest_fitness = float('inf')
        self.current_fitness = float('inf')

        # Random initialization within bounds
        for param, (min_val, max_val, param_type) in param_bounds.items():
            if param_type == 'int':
                self.position[param] = float(np.random.randint(min_val, max_val + 1))
            else: # float
                self.position[param] = np.random.uniform(min_val, max_val)

            # Initialize small random velocity
            velocity_range = (max_val - min_val) * 0.1
            self.velocity[param] = np.random.uniform(-velocity_range, velocity_range)

            self.pbest_position[param] = self.position[param]

    def update_position(self) -> None:
        """Updates particle position based on current velocity and clamps to bounds."""
        for param, (min_val, max_val, param_type) in self.param_bounds.items():
            self.position[param] += self.velocity[param]

            # Clamping position within lower and upper bounds
            self.position[param] = max(min_val, min(max_val, self.position[param]))

    def get_discrete_position(self) -> Dict[str, Any]:
        """Converts internal float positions into discrete types suitable for model training."""
        converted = {}
        for param, (min_val, max_val, param_type) in self.param_bounds.items():
            val = self.position[param]
            if param_type == 'int':
                converted[param] = int(round(val))
            else:
                converted[param] = float(val)
        return converted


class PSOOptimizer:
    """Particle Swarm Optimization Engine."""

    def __init__(
        self,
        objective_func: Callable[[Dict[str, Any]], float],
        param_bounds: Dict[str, Tuple[float, float, str]],
        num_particles: int = 5,
        max_iterations: int = 5,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5
    ):
        """
        Args:
            objective_func: Callback function evaluating a particle and returning validation loss.
            param_bounds: Search boundary limits.
            num_particles: Size of swarm.
            max_iterations: Number of swarm movement steps.
            w: Inertia weight.
            c1: Cognitive acceleration coefficient.
            c2: Social acceleration coefficient.
        """
        self.objective_func = objective_func
        self.param_bounds = param_bounds
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.swarm = [Particle(param_bounds) for _ in range(num_particles)]
        self.gbest_position = {}
        self.gbest_fitness = float('inf')
        self.convergence_history = []

    def optimize(self) -> Tuple[Dict[str, Any], float, List[float]]:
        """
        Executes the PSO optimization loop across iterations.
        
        Returns:
            Tuple of (Best Hyperparameters, Best Fitness Score, Fitness History)
        """
        logger.info(f"Initializing PSO with {self.num_particles} particles across {self.max_iterations} iterations...")

        for it in range(self.max_iterations):
            logger.info(f"--- PSO Iteration {it + 1}/{self.max_iterations} ---")

            for i, particle in enumerate(self.swarm):
                params = particle.get_discrete_position()
                logger.info(f"Evaluating Particle {i + 1}/{self.num_particles} with params: {params}")

                # Evaluate fitness (validation loss)
                fitness = self.objective_func(params)
                particle.current_fitness = fitness

                # Update Personal Best (pbest)
                if fitness < particle.pbest_fitness:
                    particle.pbest_fitness = fitness
                    particle.pbest_position = particle.position.copy()

                # Update Global Best (gbest)
                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = particle.get_discrete_position()
                    logger.info(f"🎉 New Global Best Found! Fitness: {fitness:.5f}")

            self.convergence_history.append(self.gbest_fitness)

            # Velocity & Position Update step for all particles
            for particle in self.swarm:
                for param in self.param_bounds.keys():
                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    # Velocity Update Equation
                    cognitive = self.c1 * r1 * (particle.pbest_position[param] - particle.position[param])
                    social = self.c2 * r2 * (self.gbest_position[param] - particle.position[param])
                    particle.velocity[param] = self.w * particle.velocity[param] + cognitive + social

                    # Position Update Equation
                    particle.update_position()

            logger.info(f"Iteration {it + 1} complete. Current Best Fitness: {self.gbest_fitness:.5f}")

        return self.gbest_position, self.gbest_fitness, self.convergence_history