import numpy as np


class MOPSOPathPlanner:
    """
    Multi-Objective Particle Swarm Optimization (MOPSO)
    for autonomous vehicle path planning.

    Objectives:
        1. Safety      -> minimize safety-zone violation / collision risk
        2. Efficiency  -> minimize normalized path length
        3. Comfort     -> minimize curvature and trajectory variation

    All objectives are minimization objectives.

    The planner maintains an external Pareto archive containing
    non-dominated solutions.

    Time model:
        Trajectory samples are spatial samples rather than direct
        time samples.

        Estimated trajectory time is calculated from:

            time = cumulative_distance / reference_speed

        This allows moving obstacles to be evaluated using an
        interpretable physical time scale.
    """

    def __init__(
        self, start, end, bounds_x, bounds_y, scenario=None,
        n_waypoints=4, n_particles=100, max_iterations=50,
        inertia_weight=0.5, cognitive_coefficient=1.5,
        social_coefficient=1.5, archive_size=50, seed=42,
    ):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
        self.scenario = scenario
        self.n_waypoints = int(n_waypoints)
        self.n_particles = int(n_particles)
        self.max_iterations = int(max_iterations)
        self.w = float(inertia_weight)
        self.c1 = float(cognitive_coefficient)
        self.c2 = float(social_coefficient)
        self.archive_size = int(archive_size)
        self.random_seed = int(seed)
        self.n_variables = self.n_waypoints * 2
        self.x_min = float(bounds_x[0])
        self.x_max = float(bounds_x[1])
        self.y_min = float(bounds_y[0])
        self.y_max = float(bounds_y[1])
        self.archive = []

        # Straight-line distance is used to normalize efficiency.
        self.straight_line_distance = float(np.linalg.norm(self.end - self.start))
        if self.straight_line_distance <= 1e-12:
            self.straight_line_distance = 1.0

    def _reference_speed(self):
        """
        Return the reference ego-vehicle speed.

        Units:
            scenario-distance-units / second

        A positive default is used when the scenario does not
        explicitly define a reference speed.
        """
        if self.scenario is None:
            return 1.0

        simulation = self.scenario.get("simulation", {})
        reference_speed = float(simulation.get("reference_speed", 1.0))

        if reference_speed <= 1e-12:
            reference_speed = 1.0

        return reference_speed

    def _trajectory_times(self, x, y):
        """
        Estimate physical time for every trajectory sample.

        The trajectory is represented spatially. Therefore the
        time associated with each point is estimated from the
        cumulative distance traveled by the ego vehicle:

            t_i = cumulative_distance_i / reference_speed

        Returns:
            numpy array with one time value per trajectory point.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if len(x) == 0:
            return np.array([], dtype=float)

        if len(x) == 1:
            return np.array([0.0], dtype=float)

        dx = np.diff(x)
        dy = np.diff(y)
        segment_lengths = np.sqrt(dx ** 2 + dy ** 2)
        cumulative_distance = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        reference_speed = self._reference_speed()
        times = cumulative_distance / reference_speed
        return times.astype(float)

    def _decode_particle(self, particle):
        """
        Convert a particle vector into a trajectory.

        Particle:
            [x1, y1, x2, y2, ..., xn, yn]

        Trajectory:
            start -> waypoints -> end
        """
        particle = np.asarray(particle, dtype=float)
        waypoints = particle.reshape(self.n_waypoints, 2)
        points = np.vstack([self.start, waypoints, self.end])
        return self._interpolate_path(points, n_points=100)

    @staticmethod
    def _interpolate_path(points, n_points=100):
        """
        Interpolate waypoint path into a fixed number of
        spatial trajectory samples.

        The samples are uniformly distributed according to
        distance along the waypoint polyline.
        """
        points = np.asarray(points, dtype=float)
        segment_vectors = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        total_length = float(np.sum(segment_lengths))

        if total_length <= 1e-12:
            return (
                np.full(n_points, points[0, 0]),
                np.full(n_points, points[0, 1]),
            )

        cumulative_distance = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        target_distance = np.linspace(0.0, total_length, n_points)
        x = np.interp(target_distance, cumulative_distance, points[:, 0])
        y = np.interp(target_distance, cumulative_distance, points[:, 1])
        return x, y

    def _moving_obstacle_position(self, obstacle, time):
        """
        Calculate obstacle position at a physical time.

        Position model:

            p(t) = p_0 + v * t
        """
        initial_position = np.asarray(obstacle["initial_position"], dtype=float)
        velocity = np.asarray(obstacle.get("velocity", [0.0, 0.0]), dtype=float)
        return initial_position + velocity * float(time)

    def _distance_to_obstacle(self, point, obstacle, time):
        """
        Calculate center-to-center distance between a trajectory
        point and an obstacle at a physical time.
        """
        obstacle_position = self._moving_obstacle_position(obstacle, time)
        return float(np.linalg.norm(point - obstacle_position))

    def objective_safety(self, x, y):
        """
        Safety objective.

        Lower is better.

        The objective consists of:

            normalized safety-zone violation
            +
            bounded collision penalty

        Dynamic obstacles are evaluated using estimated physical
        trajectory time rather than trajectory sample index.
        """
        if self.scenario is None:
            return 0.0

        obstacles = self.scenario.get("obstacles", [])
        if not obstacles:
            return 0.0

        vehicle = self.scenario.get("vehicle", {})
        desired_safety_distance = float(vehicle.get("safety_distance", 0.0))
        if desired_safety_distance <= 0.0:
            desired_safety_distance = 1.0

        times = self._trajectory_times(x, y)
        total_violation = 0.0
        collision_count = 0
        total_points = max(len(x) * len(obstacles), 1)

        for time, px, py in zip(times, x, y):
            point = np.array([px, py], dtype=float)

            for obstacle in obstacles:
                radius = float(obstacle.get("radius", 0.0))
                distance = self._distance_to_obstacle(point, obstacle, time)
                collision_clearance = distance - radius

                if collision_clearance <= 0.0:
                    collision_count += 1

                if distance < desired_safety_distance:
                    violation = (desired_safety_distance - distance) / desired_safety_distance
                    total_violation += violation ** 2

        normalized_violation = total_violation / total_points
        collision_ratio = collision_count / total_points
        collision_penalty = 10.0 * collision_ratio
        safety = normalized_violation + collision_penalty
        return float(safety)

    def objective_efficiency(self, x, y):
        """
        Efficiency objective.

        Lower is better.

        Path length is normalized by the straight-line distance
        between start and goal.
        """
        dx = np.diff(x)
        dy = np.diff(y)
        segment_lengths = np.sqrt(dx ** 2 + dy ** 2)
        path_length = float(np.sum(segment_lengths))
        normalized_length = path_length / self.straight_line_distance
        return float(normalized_length)

    def objective_comfort(self, x, y):
        """
        Comfort objective.

        Lower is better.

        Current baseline uses:

            mean absolute curvature
            +
            curvature variation

        A future dynamic version can incorporate acceleration,
        jerk, steering-rate variation, and other vehicle-dynamics
        measures.
        """
        if len(x) < 3:
            return 0.0

        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        denominator = (dx ** 2 + dy ** 2) ** 1.5
        denominator = np.maximum(denominator, 1e-8)
        curvature = np.abs(dx * ddy - dy * ddx) / denominator
        curvature = np.nan_to_num(curvature, nan=0.0, posinf=1e6, neginf=0.0)

        mean_curvature = float(np.mean(curvature))

        if len(curvature) > 1:
            curvature_variation = float(np.mean(np.abs(np.diff(curvature))))
        else:
            curvature_variation = 0.0

        comfort = mean_curvature + curvature_variation
        return float(comfort)

    def evaluate_particle(self, particle):
        """
        Evaluate one particle.

        Returns:
            path
            objective vector

        Objective vector:

            [safety, efficiency, comfort]
        """
        x, y = self._decode_particle(particle)
        safety = self.objective_safety(x, y)
        efficiency = self.objective_efficiency(x, y)
        comfort = self.objective_comfort(x, y)
        objectives = np.array([safety, efficiency, comfort], dtype=float)
        return (x, y), objectives

    @staticmethod
    def dominates(objectives_a, objectives_b):
        """
        Return True if A Pareto-dominates B.

        All objectives are minimized.

        A dominates B when:

            A <= B for every objective

        and

            A < B for at least one objective.
        """
        objectives_a = np.asarray(objectives_a, dtype=float)
        objectives_b = np.asarray(objectives_b, dtype=float)
        no_worse = np.all(objectives_a <= objectives_b)
        strictly_better = np.any(objectives_a < objectives_b)
        return bool(no_worse and strictly_better)

    @staticmethod
    def _remove_duplicate_objectives(solutions):
        """Remove solutions with identical objective vectors."""
        unique = []
        seen = set()

        for solution in solutions:
            key = tuple(np.round(solution["objectives"], decimals=10))

            if key not in seen:
                seen.add(key)
                unique.append(solution)

        return unique

    def _get_non_dominated(self, solutions):
        """Extract Pareto-non-dominated solutions."""
        non_dominated = []

        for i, candidate in enumerate(solutions):
            dominated = False

            for j, other in enumerate(solutions):
                if i == j:
                    continue

                if self.dominates(other["objectives"], candidate["objectives"]):
                    dominated = True
                    break

            if not dominated:
                non_dominated.append(candidate)

        return self._remove_duplicate_objectives(non_dominated)

    @staticmethod
    def _crowding_distances(solutions):
        """
        Calculate crowding distance.

        Larger values indicate a less crowded region of the
        Pareto front.
        """
        n = len(solutions)

        if n == 0:
            return np.array([], dtype=float)

        if n <= 2:
            return np.full(n, np.inf, dtype=float)

        objectives = np.array(
            [solution["objectives"] for solution in solutions], dtype=float
        )
        n_objectives = objectives.shape[1]
        distances = np.zeros(n, dtype=float)

        for objective_index in range(n_objectives):
            order = np.argsort(objectives[:, objective_index])
            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf
            minimum = objectives[order[0], objective_index]
            maximum = objectives[order[-1], objective_index]
            scale = maximum - minimum

            if scale <= 1e-12:
                continue

            for position in range(1, n - 1):
                index = order[position]

                if np.isinf(distances[index]):
                    continue

                previous_value = objectives[order[position - 1], objective_index]
                next_value = objectives[order[position + 1], objective_index]
                distances[index] += (next_value - previous_value) / scale

        return distances

    def _update_archive(self, candidates):
        """
        Merge new candidates with the archive and retain only
        Pareto-non-dominated solutions.
        """
        combined = self.archive + candidates
        non_dominated = self._get_non_dominated(combined)

        if len(non_dominated) <= self.archive_size:
            self.archive = non_dominated
            return

        distances = self._crowding_distances(non_dominated)
        order = np.argsort(-distances)
        selected_indices = order[:self.archive_size]
        self.archive = [non_dominated[index] for index in selected_indices]

    def _select_leader(self):
        """
        Select a global guide from the Pareto archive.

        Solutions in less crowded regions receive greater
        probability of being selected.
        """
        if not self.archive:
            raise RuntimeError("Pareto archive is empty.")

        if len(self.archive) == 1:
            return self.archive[0]

        distances = self._crowding_distances(self.archive)
        finite = distances[np.isfinite(distances)]

        if len(finite) == 0:
            probabilities = np.ones(len(self.archive)) / len(self.archive)
        else:
            adjusted = distances.copy()
            infinite_mask = np.isinf(adjusted)

            if np.any(infinite_mask):
                maximum_finite = np.max(finite)
                adjusted[infinite_mask] = maximum_finite + 1.0

            adjusted += 1e-12
            probabilities = adjusted / np.sum(adjusted)

        index = np.random.choice(len(self.archive), p=probabilities)
        return self.archive[index]

    def _initialize_particles(self):
        """Initialize particle positions and velocities."""
        positions = np.random.uniform(
            low=0.0, high=1.0, size=(self.n_particles, self.n_variables)
        )
        positions[:, 0::2] = self.x_min + positions[:, 0::2] * (self.x_max - self.x_min)
        positions[:, 1::2] = self.y_min + positions[:, 1::2] * (self.y_max - self.y_min)

        velocity_scale = np.array(
            [
                (self.x_max - self.x_min) if index % 2 == 0
                else (self.y_max - self.y_min)
                for index in range(self.n_variables)
            ],
            dtype=float,
        )
        velocities = np.random.uniform(
            low=-0.1, high=0.1, size=(self.n_particles, self.n_variables)
        ) * velocity_scale

        return positions, velocities

    def _clip_position(self, position):
        """Keep waypoints inside scenario boundaries."""
        position = position.copy()
        position[0::2] = np.clip(position[0::2], self.x_min, self.x_max)
        position[1::2] = np.clip(position[1::2], self.y_min, self.y_max)
        return position

    def _clip_velocity(self, velocity):
        """Limit particle velocity."""
        velocity = velocity.copy()

        for index in range(self.n_variables):
            if index % 2 == 0:
                maximum = self.x_max - self.x_min
            else:
                maximum = self.y_max - self.y_min

            maximum *= 0.20
            velocity[index] = np.clip(velocity[index], -maximum, maximum)

        return velocity

    def run(self):
        """
        Run MOPSO.

        Returns:
            list of Pareto-optimal trajectory candidates.
        """
        np.random.seed(self.random_seed)
        self.archive = []
        positions, velocities = self._initialize_particles()
        personal_best_positions = positions.copy()
        personal_best_objectives = []
        initial_candidates = []

        for particle_index in range(self.n_particles):
            path, objectives = self.evaluate_particle(positions[particle_index])
            personal_best_objectives.append(objectives.copy())
            initial_candidates.append({
                "particle_index": particle_index,
                "position": positions[particle_index].copy(),
                "path": path,
                "objectives": objectives,
                "iteration": 0,
            })

        personal_best_objectives = np.asarray(personal_best_objectives, dtype=float)
        self._update_archive(initial_candidates)

        for iteration in range(1, self.max_iterations + 1):
            for particle_index in range(self.n_particles):
                leader = self._select_leader()
                leader_position = leader["position"]
                r1 = np.random.random(self.n_variables)
                r2 = np.random.random(self.n_variables)

                velocities[particle_index] = (
                    self.w * velocities[particle_index]
                    + self.c1 * r1 * (personal_best_positions[particle_index] - positions[particle_index])
                    + self.c2 * r2 * (leader_position - positions[particle_index])
                )
                velocities[particle_index] = self._clip_velocity(velocities[particle_index])
                positions[particle_index] += velocities[particle_index]
                positions[particle_index] = self._clip_position(positions[particle_index])

            iteration_candidates = []

            for particle_index in range(self.n_particles):
                path, objectives = self.evaluate_particle(positions[particle_index])
                current_solution = {
                    "particle_index": particle_index,
                    "position": positions[particle_index].copy(),
                    "path": path,
                    "objectives": objectives,
                    "iteration": iteration,
                }
                iteration_candidates.append(current_solution)

                current_best = personal_best_objectives[particle_index]

                if self.dominates(objectives, current_best):
                    personal_best_positions[particle_index] = positions[particle_index].copy()
                    personal_best_objectives[particle_index] = objectives.copy()
                elif not self.dominates(current_best, objectives):
                    if np.random.random() < 0.5:
                        personal_best_positions[particle_index] = positions[particle_index].copy()
                        personal_best_objectives[particle_index] = objectives.copy()

            self._update_archive(iteration_candidates)

        results = []

        for index, solution in enumerate(self.archive):
            results.append({
                "candidate_id": f"pareto_{index + 1:03d}",
                "path": solution["path"],
                "objectives": solution["objectives"].copy(),
                "position": solution["position"].copy(),
                "seed": self.random_seed,
                "iteration": solution.get("iteration", self.max_iterations),
            })

        return results


def generate_pareto_candidates(
    scenario,
    n_particles=100,
    max_iterations=50,
    archive_size=20,
    seed=42,
):
    """Convenience function for generating Pareto candidates."""
    planner = MOPSOPathPlanner(
        start=scenario["start"],
        end=scenario["end"],
        bounds_x=scenario["bounds_x"],
        bounds_y=scenario["bounds_y"],
        scenario=scenario,
        n_waypoints=4,
        n_particles=n_particles,
        max_iterations=max_iterations,
        inertia_weight=0.5,
        cognitive_coefficient=1.5,
        social_coefficient=1.5,
        archive_size=archive_size,
        seed=seed,
    )
    return planner.run()