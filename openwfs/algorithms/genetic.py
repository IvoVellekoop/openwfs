import itertools

import numpy as np

from .utilities import WFSResult, DummyProgressBar
from ..core import Detector, PhaseSLM


class SimpleGenetic:
    """Simple genetic algorithm for wavefront shaping.

    This algorithm is included for illustrative purposes. It is based on the algorithm described
    in [1] and [2].

    The algorithm performs the following steps:

    1. Initialize all wavefronts in the population with random phases

    2. For each generation:

      2.1. Determine the feedback signal for each wavefront.

      2.2. Select the 'elite_size' best wavefronts to keep. Replace the rest with new wavefronts.
        2.2.1 If the elite wavefronts are too similar (> 97% identical elements),
              randomly generate the new wavefronts.

        2.2.2 Otherwise, generate new wavefronts by randomly selecting two elite wavefronts
              and mixing them randomly (element-wise).
              Then perform a mutation that replaces a fraction (`mutation_probability`) of the
              elements by a new value.

    References
    ----------
    [^1]: Conkey D B, Brown A N, Caravaca-Aguirre A M and Piestun R 'Genetic algorithm optimization
          for focusing through turbid media in noisy environments' Opt. Express 20 4840–9 (2012).
    [^2]: Benjamin R Anderson et al. 'A modular GUI-based program for genetic algorithm-based
          feedback-assisted wavefront shaping', J. Phys. Photonics 6 045008 (2024).
    """

    def __init__(
        self,
        feedback: Detector,
        slm: PhaseSLM,
        shape: tuple[int, int] = (500, 500),
        population_size: int = 30,
        elite_size: int = 5,
        generations: int = 100,
        mutation_probability: float = 0.005,
        generator=None,
    ):
        """
        Args:
            feedback: Source of feedback
            slm: The spatial light modulator
            shape: Width × height (in segments) of the wavefront
            population_size (int): The number of individuals in the population
            elite_size (int): The number of individuals in the elite pool
            generations (int): The number of generations
            mutation_probability (int): Fraction of elements in the offspring to mutate
            generator: a `np.random.Generator`, defaults to np.random.default_rng()

        """
        if np.prod(feedback.data_shape) != 1:
            raise ValueError("Only scalar feedback is supported")
        self.feedback = feedback
        self.slm = slm
        self.shape = shape
        self.population_size = population_size
        self.elite_size = elite_size
        self.generations = generations
        self.generator = generator or np.random.default_rng()
        self.mutation_count = round((population_size - elite_size) * np.prod(shape) * mutation_probability)

    def _generate_random_phases(self, shape):
        return self.generator.random(size=shape, dtype=np.float32) * (2 * np.pi)

    def execute(self, *, progress_bar=DummyProgressBar()) -> WFSResult:
        """Executes the algorithm.
        Args:
            progress_bar: Optional tqdm-like progress bar for displaying progress
        """

        # Initialize the population
        population = self._generate_random_phases((self.population_size, *self.shape))

        # initialize the progress bar if available
        progress_bar.total = self.generations * self.population_size

        for i in itertools.count():
            # Try all phase patterns
            measurements = np.zeros(self.population_size, dtype=np.float32)
            for p in range(self.population_size):
                self.slm.set_phases(population[p])
                self.feedback.trigger(out=measurements[p, ...])
                progress_bar.update()

            self.feedback.wait()

            # Sort the measurements in ascending order
            sorted_indices = np.argsort(measurements)
            elite = sorted_indices[-self.elite_size :]
            plebs = sorted_indices[: -self.elite_size]

            # Terminate after the specified number of generations, return the best wavefront
            if i >= self.generations:
                return WFSResult(t=np.exp(-1.0j * population[sorted_indices[-1]]), axis=2)

            # We keep the elite individuals, and regenerate the rest by mixing the elite
            # For this mixing, the probability of selecting an individual is proportional to its measured intensity.
            probabilities = measurements[elite]
            probabilities /= np.sum(probabilities)
            couples = self.generator.choice(elite, size=(2, len(plebs)), p=probabilities)

            if np.mean(population[couples[0]] == population[couples[1]]) > 0.97:
                # if the parents are too similar, randomly generate the plebs
                population[plebs] = self._generate_random_phases((len(plebs), *self.shape))
            else:
                # otherwise, mix and mutate the parents
                mix_masks = self.generator.integers(1, size=(len(plebs), *self.shape), dtype=bool)
                offspring = np.where(mix_masks, population[couples[0]], population[couples[1]])
                mutations = self.generator.integers(offspring.size, size=self.mutation_count)
                offspring.ravel()[mutations] = self._generate_random_phases(self.mutation_count)
                population[plebs] = offspring
