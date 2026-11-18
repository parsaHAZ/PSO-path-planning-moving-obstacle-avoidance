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
        preference_model=None, preference_guidance_strength=0.35,
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
        # Optional preference guidance is disabled unless a fitted model is supplied.
        self.preference_model = preference_model
        self.preference_guidance_strength = float(np.clip(preference_guidance_strength, 0.0, 1.0))
        self._preference_model_estimator = (
            preference_model.get("model") if isinstance(preference_model, dict)
            else preference_model
        )
        if (preference_model is not None and self.preference_guidance_strength > 0.0
                and self._preference_model_estimator is None):
            raise ValueError("Preference payload was supplied without a fitted model.")
        self.preference_guidance_diagnostics = {
            "leader_selections": 0,
            "preference_guided_selections": 0,
        }

        # Straight-line distance is used to normalize efficiency.
        self.straight_line_distance = float(np.linalg.norm(self.end - self.start))
        if self.straight_line_distance <= 1e-12:
            self.straight_line_distance = 1.0

    def _reference_speed(self):
        """
        Return the reference ego-vehicle speed.

        Units:
            scenario-distance-units / second

        The preferred location is the top-level scenario field
        ``reference_speed`` created by the GNN/MOPSO integration.

        A nested ``simulation.reference_speed`` is supported for
        backward compatibility.

        A positive default is used when the scenario does not
        explicitly define a reference speed.
        """
        if self.scenario is None:
            return 1.0

        if "reference_speed" in self.scenario:
            reference_speed = float(
                self.scenario["reference_speed"]
            )
        else:
            simulation = self.scenario.get("simulation", {})
            reference_speed = float(
                simulation.get("reference_speed", 1.0)
            )

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
        Safety objective evaluated at spatial intervals no greater than 0.25 m.

        Dynamic obstacle positions are evaluated at estimated physical times:
            t = cumulative_path_distance / reference_speed

        The desired safety distance is a CENTER-TO-CENTER threshold.
        The separate collision penalty detects overlap with the obstacle
        radius, but does not model an ego-vehicle footprint.
        """
        if self.scenario is None:
            return 0.0

        obstacles = self.scenario.get("obstacles", [])
        if not obstacles:
            return 0.0

        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)

        if len(x) != len(y):
            raise ValueError("x and y must have the same number of samples.")
        if len(x) == 0:
            return 0.0
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            return float("inf")

        vehicle = self.scenario.get("vehicle", {})
        if "safety_distance" in self.scenario:
            desired_safety_distance = float(self.scenario["safety_distance"])
        else:
            desired_safety_distance = float(
                vehicle.get("safety_distance", 1.0)
            )

        if not np.isfinite(desired_safety_distance) or desired_safety_distance <= 0:
            desired_safety_distance = 1.0

        # Resample the decoded polyline densely. The maximum interval is 0.25 m.
        dx = np.diff(x)
        dy = np.diff(y)
        segment_lengths = np.sqrt(dx * dx + dy * dy)
        cumulative_distance = np.concatenate(
            ([0.0], np.cumsum(segment_lengths))
        )
        total_length = float(cumulative_distance[-1])

        if total_length > 1e-12:
            n_samples = max(2, int(np.ceil(total_length / 0.25)) + 1)
            sample_distance = np.linspace(0.0, total_length, n_samples)
            sample_x = np.interp(sample_distance, cumulative_distance, x)
            sample_y = np.interp(sample_distance, cumulative_distance, y)
        else:
            sample_distance = np.array([0.0])
            sample_x = np.array([x[0]])
            sample_y = np.array([y[0]])

        reference_speed = float(self._reference_speed())
        if not np.isfinite(reference_speed) or reference_speed <= 1e-12:
            raise ValueError(
                f"Invalid reference speed for safety timing: {reference_speed}"
            )

        sample_times = sample_distance / reference_speed
        total_violation = 0.0
        collision_count = 0
        total_points = max(len(sample_x) * len(obstacles), 1)

        for time, px, py in zip(sample_times, sample_x, sample_y):
            point = np.array([px, py], dtype=float)

            for obstacle in obstacles:
                radius = max(0.0, float(obstacle.get("radius", 0.0)))
                distance = self._distance_to_obstacle(point, obstacle, time)

                # Physical obstacle overlap; ego footprint is not included.
                if distance - radius <= 0.0:
                    collision_count += 1

                # Center-to-center safety-zone violation.
                if distance < desired_safety_distance:
                    violation = (
                        desired_safety_distance - distance
                    ) / desired_safety_distance
                    total_violation += violation ** 2

        normalized_violation = total_violation / total_points
        collision_ratio = collision_count / total_points
        collision_penalty = 10.0 * collision_ratio

        return float(normalized_violation + collision_penalty)

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
        Numerically stable discrete comfort objective.

        Lower is better. Curvature is estimated from heading changes
        divided by traveled distance. Zero-length segments are excluded.
        """
        points = np.column_stack((
            np.asarray(x, dtype=np.float64).reshape(-1),
            np.asarray(y, dtype=np.float64).reshape(-1),
        ))

        if len(points) < 3 or not np.isfinite(points).all():
            return 1e6

        segments = np.diff(points, axis=0)
        lengths = np.linalg.norm(segments, axis=1)

        # Ignore duplicate or numerically negligible consecutive points.
        mean_segment_length = float(np.mean(lengths)) if len(lengths) else 0.0
        eps = max(1e-8, 1e-6 * mean_segment_length)

        valid = (
            (lengths[:-1] > eps) &
            (lengths[1:] > eps)
        )

        if not np.any(valid):
            # A path without usable movement is not a comfortable path.
            return 1e6

        v1 = segments[:-1][valid]
        v2 = segments[1:][valid]
        l1 = lengths[:-1][valid]
        l2 = lengths[1:][valid]

        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        dot = np.sum(v1 * v2, axis=1)

        turning_angle = np.abs(np.arctan2(cross, dot))
        local_distance = 0.5 * (l1 + l2)

        curvature = turning_angle / np.maximum(local_distance, eps)
        curvature = curvature[np.isfinite(curvature)]

        if curvature.size == 0:
            return 1e6

        mean_curvature = float(np.mean(curvature))

        # Only measure variation between consecutive valid vertices.
        valid_indices = np.flatnonzero(valid)
        consecutive = np.diff(valid_indices) == 1

        if np.any(consecutive):
            curvature_variation = float(np.mean(
                np.abs(np.diff(curvature)[consecutive])
            ))
        else:
            curvature_variation = 0.0

        comfort = mean_curvature + curvature_variation

        if not np.isfinite(comfort):
            return 1e6

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

        # The fitted model was trained on pairwise differences A - B, not
        # absolute objective vectors. Estimate each archive candidate's human-
        # preference score by comparing it against every other archive member.
        # Positive class 1 means the first candidate (A) is preferred.
        strength = self.preference_guidance_strength
        estimator = self._preference_model_estimator
        if strength > 0.0 and estimator is not None:
            try:
                objective_matrix = np.asarray(
                    [solution["objectives"][:3] for solution in self.archive], dtype=float
                )
                n_candidates = len(self.archive)
                if (objective_matrix.shape == (n_candidates, 3)
                        and n_candidates > 1
                        and np.isfinite(objective_matrix).all()
                        and hasattr(estimator, "predict_proba")):
                    pairwise_differences = []
                    owner_indices = []
                    for candidate_index in range(n_candidates):
                        for other_index in range(n_candidates):
                            if candidate_index == other_index:
                                continue
                            pairwise_differences.append(
                                objective_matrix[candidate_index] - objective_matrix[other_index]
                            )
                            owner_indices.append(candidate_index)
                    pairwise_differences = np.asarray(pairwise_differences, dtype=float)
                    # Symmetrize pairwise predictions so swapping A/B
                    # produces complementary preference probabilities.
                    # This removes the systematic bias from the classifier
                    # intercept while preserving the learned feature effects.
                    if not hasattr(estimator, "classes_") or 1 not in list(estimator.classes_):
                        raise ValueError(
                            "Preference estimator must expose class 1 for A-preferred."
                        )

                    class_one_index = list(estimator.classes_).index(1)
                    forward = np.asarray(
                        estimator.predict_proba(pairwise_differences), dtype=float
                    )
                    reverse = np.asarray(
                        estimator.predict_proba(-pairwise_differences), dtype=float
                    )
                    expected_shape = (len(pairwise_differences), len(estimator.classes_))
                    if (forward.shape != expected_shape or reverse.shape != expected_shape
                            or not np.isfinite(forward).all() or not np.isfinite(reverse).all()
                            or np.any(forward < 0.0) or np.any(forward > 1.0)
                            or np.any(reverse < 0.0) or np.any(reverse > 1.0)):
                        raise ValueError("Preference model returned invalid probability rows.")
                    p_forward = forward[:, class_one_index]
                    p_reverse = reverse[:, class_one_index]

                    pairwise_probabilities = 0.5 * (
                        p_forward + 1.0 - p_reverse
                    )
                    if (len(pairwise_probabilities) != len(owner_indices)
                            or not np.isfinite(pairwise_probabilities).all()):
                        raise ValueError("Preference model returned incomplete pairwise predictions.")
                    if len(pairwise_probabilities) == len(owner_indices):
                        preference_scores = np.zeros(n_candidates, dtype=float)
                        comparison_counts = np.zeros(n_candidates, dtype=float)
                        for owner, score in zip(owner_indices, pairwise_probabilities):
                            preference_scores[owner] += score
                            comparison_counts[owner] += 1.0
                        preference_scores /= np.maximum(comparison_counts, 1.0)
                        preference_scores = np.clip(preference_scores, 1e-12, None)

                        preference_probs = preference_scores / preference_scores.sum()
                        probabilities = (
                            (1.0 - strength) * probabilities
                            + strength * preference_probs
                        )
                        probabilities /= probabilities.sum()
                        self.preference_guidance_diagnostics["preference_guided_selections"] += 1

            except (ValueError, TypeError, AttributeError, IndexError) as exc:
                raise RuntimeError(
                    "Preference guidance failed inside _select_leader(). "
                    "The run is stopped instead of silently falling back "
                    "to conventional MOPSO."
                ) from exc


        # Safety is the planner's normalized violation objective: zero means
        # every sampled point satisfies the configured clearance and overlap
        # terms. Preference scores may rank only feasible leaders when any
        # exist. If the archive contains no feasible leader, retain only the
        # minimum-violation tier so preference cannot increase safety cost.
        objective_matrix = np.asarray(
            [solution["objectives"][:3] for solution in self.archive], dtype=float
        )
        safety = objective_matrix[:, 0]
        if not np.isfinite(objective_matrix).all():
            raise RuntimeError("Cannot select a MOPSO leader with non-finite objectives.")
        feasible = safety <= 1e-12
        if np.any(feasible):
            allowed = feasible
        else:
            best_safety = float(np.min(safety))
            allowed = np.isclose(safety, best_safety, rtol=0.0, atol=1e-12)
        probabilities = probabilities * allowed.astype(float)
        probability_sum = float(probabilities.sum())
        if probability_sum <= 0.0 or not np.isfinite(probability_sum):
            probabilities = allowed.astype(float) / float(np.count_nonzero(allowed))
        else:
            probabilities /= probability_sum


        self.preference_guidance_diagnostics["leader_selections"] += 1

        if (
            self._preference_model_estimator is not None
            and self.preference_guidance_strength > 0.0
            and self.preference_guidance_diagnostics[
                "preference_guided_selections"
            ] == 0
            and self.preference_guidance_diagnostics["leader_selections"] >= 10
        ):
            raise RuntimeError(
                "Preference model has not guided any leader selections "
                "after 10 attempts. Check model compatibility and features."
            )

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
        """Run preference-guided MOPSO with safety-first personal-best updates."""
        np.random.seed(self.random_seed)
        self.archive = []

        self.preference_guidance_diagnostics = {
            "leader_selections": 0,
            "preference_guided_selections": 0,
            "preference_guided_personal_best_updates": 0,
        }

        positions, velocities = self._initialize_particles()
        personal_best_positions = positions.copy()
        personal_best_objectives = []
        initial_candidates = []

        for particle_index in range(self.n_particles):
            path, objectives = self.evaluate_particle(
                positions[particle_index]
            )
            objectives = np.asarray(objectives, dtype=float)

            personal_best_objectives.append(objectives.copy())
            initial_candidates.append({
                "particle_index": particle_index,
                "position": positions[particle_index].copy(),
                "path": path,
                "objectives": objectives.copy(),
                "iteration": 0,
            })

        personal_best_objectives = np.asarray(
            personal_best_objectives, dtype=float
        )
        self._update_archive(initial_candidates)

        estimator = getattr(self, "_preference_model_estimator", None)
        strength = float(
            np.clip(
                getattr(self, "preference_guidance_strength", 0.0),
                0.0,
                1.0,
            )
        )

        for iteration in range(1, self.max_iterations + 1):

            # Particle movement using archive leaders.
            for particle_index in range(self.n_particles):
                leader = self._select_leader()
                leader_position = leader["position"]

                r1 = np.random.random(self.n_variables)
                r2 = np.random.random(self.n_variables)

                velocities[particle_index] = (
                    self.w * velocities[particle_index]
                    + self.c1 * r1 * (
                        personal_best_positions[particle_index]
                        - positions[particle_index]
                    )
                    + self.c2 * r2 * (
                        leader_position - positions[particle_index]
                    )
                )

                velocities[particle_index] = self._clip_velocity(
                    velocities[particle_index]
                )
                positions[particle_index] += velocities[particle_index]
                positions[particle_index] = self._clip_position(
                    positions[particle_index]
                )

            iteration_candidates = []

            # Evaluate particles and update personal bests.
            for particle_index in range(self.n_particles):
                path, objectives = self.evaluate_particle(
                    positions[particle_index]
                )
                objectives = np.asarray(objectives, dtype=float)

                current_solution = {
                    "particle_index": particle_index,
                    "position": positions[particle_index].copy(),
                    "path": path,
                    "objectives": objectives.copy(),
                    "iteration": iteration,
                }
                iteration_candidates.append(current_solution)

                current_best = personal_best_objectives[particle_index]

                # Safety is objective 0; zero means no safety penalty.
                current_safe = objectives[0] <= 1e-12
                previous_safe = current_best[0] <= 1e-12

                replace = False

                if current_safe and not previous_safe:
                    replace = True

                elif previous_safe and not current_safe:
                    replace = False

                elif not current_safe and not previous_safe:
                    # Both candidates are unsafe: reduce safety penalty first.
                    replace = objectives[0] < current_best[0] - 1e-12

                elif self.dominates(objectives, current_best):
                    replace = True

                elif self.dominates(current_best, objectives):
                    replace = False

                else:
                    # Incomparable safe candidates: consult human preferences.
                    if estimator is not None and strength > 0.0:
                        difference = (
                            objectives[:3] - current_best[:3]
                        ).reshape(1, 3)

                        classes = list(estimator.classes_)
                        if 1 not in classes:
                            raise RuntimeError(
                                "Preference model must encode class 1 as "
                                "candidate A being preferred."
                            )

                        class_index = classes.index(1)

                        p_forward = float(
                            estimator.predict_proba(difference)[0, class_index]
                        )
                        p_reverse = float(
                            estimator.predict_proba(-difference)[0, class_index]
                        )

                        p_preferred = float(np.clip(
                            0.5 * (p_forward + 1.0 - p_reverse),
                            0.0,
                            1.0,
                        ))

                        p_replace = (
                            strength * p_preferred
                            + (1.0 - strength) * 0.5
                        )
                        replace = np.random.random() < p_replace

                        self.preference_guidance_diagnostics[
                            "preference_guided_personal_best_updates"
                        ] += 1

                    else:
                        # Original neutral tie-break if no model is configured.
                        replace = np.random.random() < 0.5

                if replace:
                    personal_best_positions[particle_index] = (
                        positions[particle_index].copy()
                    )
                    personal_best_objectives[particle_index] = (
                        objectives.copy()
                    )

            self._update_archive(iteration_candidates)

        if estimator is not None and strength > 0.0:
            guided_leaders = self.preference_guidance_diagnostics[
                "preference_guided_selections"
            ]
            attempted_leaders = self.preference_guidance_diagnostics[
                "leader_selections"
            ]

            if guided_leaders == 0:
                raise RuntimeError(
                    "MOPSO completed without using the preference model. "
                    f"Leader selections attempted: {attempted_leaders}. "
                    "Check the model, feature convention, and objective "
                    "order [safety, efficiency, comfort]."
                )

        results = []

        for index, solution in enumerate(self.archive):
            results.append({
                "candidate_id": f"pareto_{index + 1:03d}",
                "path": solution["path"],
                "objectives": solution["objectives"].copy(),
                "position": solution["position"].copy(),
                "seed": self.random_seed,
                "iteration": solution.get(
                    "iteration", self.max_iterations
                ),
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
