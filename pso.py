import numpy as np
from scipy.interpolate import splprep, splev

from obstacles import ObstacleManager


class PSOPathPlanner:

    def __init__(self, start, end, n_obstacles, max_iterations, bounds_x, bounds_y, scenario=None):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)
        self.n_obstacles = n_obstacles
        self.max_iterations = max_iterations
        self.space_bounds_x = bounds_x
        self.space_bounds_y = bounds_y

        # Optional research scenario
        self.scenario = scenario
        if self.scenario is not None:
            self.random_seed = self.scenario.get(
                "seed",
                42,
            )
        else:
            self.random_seed = None

        # PSO parameters
        self.r_min = 5
        self.r_max = 15
        self.n_waypoints = 4
        self.n_particles = 100
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5

        # Obstacle initialization
        if self.scenario is not None:
            self._initialize_scenario_obstacles()
        else:
            self.obstacle_manager = ObstacleManager(
                self.n_obstacles, self.r_min, self.r_max, self.start, self.end
            )
            self.obstacles, self.radii = self.obstacle_manager.generate_obstacles()

    def _initialize_scenario_obstacles(self):
        scenario_obstacles = self.scenario.get("obstacles", [])
        self.obstacles = []
        self.radii = []

        for obstacle in scenario_obstacles:
            position = np.asarray(obstacle["initial_position"], dtype=float)
            velocity = np.asarray(obstacle.get("velocity", [0.0, 0.0]), dtype=float)
            radius = float(obstacle.get("radius", 3.0))

            self.obstacles.append({
                "id": obstacle.get("id", f"vehicle_{len(self.obstacles) + 1}"),
                "position": position.copy(),
                "velocity": velocity.copy(),
                "radius": radius,
            })
            self.radii.append(radius)

    def run(self):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        dim = self.n_waypoints * 2
        positions = np.zeros((self.n_particles, dim))

        for i in range(self.n_particles):
            for j in range(self.n_waypoints):
                positions[i, j * 2] = np.random.uniform(
                    self.space_bounds_x[0], self.space_bounds_x[1]
                )
                positions[i, j * 2 + 1] = np.random.uniform(
                    self.space_bounds_y[0], self.space_bounds_y[1]
                )

        velocities = np.random.uniform(-1, 1, (self.n_particles, dim))
        p_best_pos = positions.copy()
        p_best_cost = np.full(self.n_particles, np.inf)
        g_best_pos = None
        g_best_cost = np.inf
        g_best_iter = -1
        convergence = []

        for iteration in range(self.max_iterations):
            # Evaluate particles
            for i in range(self.n_particles):
                waypoints = positions[i].reshape(self.n_waypoints, 2)
                x_spline, y_spline = self._spline_path(waypoints)
                if x_spline is None:
                    continue

                cost = self._path_cost(x_spline, y_spline, iteration=iteration)

                if cost < p_best_cost[i]:
                    p_best_cost[i] = cost
                    p_best_pos[i] = positions[i].copy()

                if cost < g_best_cost:
                    g_best_cost = cost
                    g_best_pos = positions[i].copy()
                    g_best_iter = iteration

            convergence.append(g_best_cost)
            print(f"Iteration {iteration + 1}/{self.max_iterations}, Best Cost: {g_best_cost:.4f}")

            # Update particles
            if g_best_pos is None:
                continue

            for i in range(self.n_particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (p_best_pos[i] - positions[i])
                    + self.c2 * r2 * (g_best_pos - positions[i])
                )
                positions[i] += velocities[i]

                # Keep waypoints inside search space
                for j in range(self.n_waypoints):
                    positions[i, j * 2] = np.clip(
                        positions[i, j * 2],
                        self.space_bounds_x[0], self.space_bounds_x[1],
                    )
                    positions[i, j * 2 + 1] = np.clip(
                        positions[i, j * 2 + 1],
                        self.space_bounds_y[0], self.space_bounds_y[1],
                    )

        # Return best solution
        if g_best_pos is not None:
            best_waypoints = g_best_pos.reshape(self.n_waypoints, 2)
            return (
                self._spline_path(best_waypoints),
                g_best_cost,
                g_best_iter,
                convergence,
            )

        return (None, np.inf, -1, convergence)

    def generate_candidates(self, n_candidates=5):
        """
        Generate multiple feasible trajectory candidates.

        Each candidate is produced by running PSO with
        a different deterministic random seed.

        Returns
        -------
        candidates : list
            Each item contains:
                - candidate_id
                - path
                - cost
                - best_iteration
                - convergence
                - seed
        """
        if self.scenario is None:
            base_seed = 42
        else:
            base_seed = self.scenario.get("seed", 42)

        candidates = []

        for candidate_index in range(n_candidates):
            candidate_seed = base_seed + candidate_index
            self.random_seed = candidate_seed

            path, cost, best_iteration, convergence = self.run()

            if path is None:
                continue

            x, y = path

            candidates.append({
                "candidate_id": f"candidate_{candidate_index + 1:02d}",
                "path": (x.copy(), y.copy()),
                "cost": float(cost),
                "best_iteration": int(best_iteration),
                "convergence": convergence.copy(),
                "seed": candidate_seed,
            })

        # Restore original scenario seed
        if self.scenario is not None:
            self.random_seed = self.scenario.get("seed", 42)
        else:
            self.random_seed = 42

        return candidates

    def _spline_path(self, waypoints):
        points = np.vstack([self.start, waypoints, self.end]).T

        try:
            tck, _ = splprep(points, s=2.0)
            u = np.linspace(0, 1, 100)
            return splev(u, tck)
        except Exception:
            return None, None

    def _path_cost(self, x_spline, y_spline, iteration=0):
        if not self._is_valid_path(x_spline, y_spline, iteration):
            return 1e6

        dx = np.diff(x_spline)
        dy = np.diff(y_spline)
        return np.sum(np.sqrt(dx ** 2 + dy ** 2))

    def _is_valid_path(self, x_spline, y_spline, iteration=0, dynamic=True):
        # Original repository obstacle manager
        if self.scenario is None:
            for i, (x, y) in enumerate(zip(x_spline, y_spline)):
                if self.obstacle_manager.check_collision(x, y, i):
                    return False
            return True

        # Scenario-based vehicle collision checking
        for vehicle in self.obstacles:
            initial_position = vehicle["position"]
            velocity = vehicle["velocity"]
            radius = vehicle["radius"]

            # The current version assumes that trajectory sample index
            # represents elapsed planning time. This is intentionally simple.
            # Later we will replace this with an explicit vehicle/trajectory
            # time model.

            for i, (x, y) in enumerate(zip(x_spline, y_spline)):
                if dynamic:
                    obstacle_position = initial_position + velocity * i
                else:
                    obstacle_position = initial_position

                distance = np.sqrt(
                    (x - obstacle_position[0]) ** 2
                    + (y - obstacle_position[1]) ** 2
                )
                if distance <= radius:
                    return False

        return True