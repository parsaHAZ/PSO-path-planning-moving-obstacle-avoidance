# Contributing

## Branches

Do not work directly on `main` or `research-baseline`.

Create a separate branch for each feature, experiment, bug fix, or research task.

### Feature Branch

```bash
git switch research-baseline
git pull origin research-baseline
git switch -c feature/<short-description>
```

### Experiment Branch

```bash
git switch research-baseline
git pull origin research-baseline
git switch -c experiment/<short-description>
```

### Bug Fix Branch

```bash
git switch research-baseline
git pull origin research-baseline
git switch -c fix/<short-description>
```

---

## Commit Messages

Use clear and descriptive commit messages.

Preferred prefixes:

* `feat:` — new functionality
* `fix:` — bug or correctness fix
* `test:` — tests or validation
* `refactor:` — code restructuring without changing intended behavior
* `docs:` — documentation changes
* `experiment:` — experimental implementation or evaluation
* `research:` — research-analysis utilities or methodology-related work
* `chore:` — repository or development infrastructure

Examples:

```text
feat: add synthetic traffic generator
fix: correct trajectory time calculation
test: add moving obstacle validation
refactor: separate trajectory objectives
docs: update research methodology
experiment: evaluate preference model
research: add Pareto analysis visualizations
chore: configure continuous integration
```

Keep commits focused. A commit should ideally represent one logical change.

---

## Pull Requests

Every research or engineering change should be submitted through a pull request.

Do not merge experimental work directly into `main`.

A pull request should explain:

* what changed
* why the change was made
* how the change was validated
* whether existing experimental results changed
* whether datasets were affected
* whether reproducibility is affected

For research-related changes, the PR should make it possible to understand the relationship between:

```text
Research question
      ↓
Implementation
      ↓
Experiment
      ↓
Results
      ↓
Interpretation
```

---

## Testing

Before opening a pull request, run the complete test suite using Python 3.10:

```bash
python -m tests.test_vehicle_scenario
python -m tests.test_vehicle_pso
python -m tests.test_candidate_generation
python -m tests.test_mopso
python -m tests.test_preference_data
python -m tests.test_preference_dataset
python -m tests.test_preference_annotator
python -m tests.test_trajectory_features
```

All existing tests should pass before submitting a pull request.

If a test fails because of a deliberate research change, document the reason in the pull request rather than silently modifying or removing the test.

---

## Research Integrity

Research results must remain traceable and reproducible.

Do not silently modify previously generated experimental results.

If a code change alters previous results:

1. Document the change.
2. Identify the affected experiment.
3. Record the new configuration.
4. Record relevant parameter changes.
5. Explain why the results changed.
6. Clearly distinguish the new results from previous results.

Do not overwrite historical research artifacts unless there is a documented reason to do so.

### Human Data

Human-generated data must be clearly distinguished from synthetic or programmatically generated data.

Do not present synthetic traffic data as real human driving data.

Do not present synthetic preference labels as human annotations.

Human preference annotations should preserve their original labels and provenance.

---

## Dataset Policy

Large raw datasets should not be committed directly to Git unless explicitly approved.

Repository data should generally contain:

* small reproducible datasets
* scenario definitions
* annotations
* derived research artifacts
* metadata
* scripts required to reproduce processing

Large datasets should remain outside the repository when appropriate.

For every externally sourced dataset, document:

* dataset name
* dataset version
* acquisition method
* source
* preprocessing procedure
* relevant license or usage restrictions
* train/validation/test split information
* any filtering or sampling performed

---

## Generated Data

Generated data should be clearly identified.

For synthetic traffic data, record the parameters used to generate it, including when applicable:

* random seed
* scenario ID
* simulation parameters
* vehicle parameters
* traffic configuration
* behavioral parameters
* generation method

Synthetic data should be reproducible whenever practical.

---

## Experimental Reproducibility

When running an experiment, record the information necessary to reproduce it.

When appropriate, record:

* random seed
* Python version
* dependency versions
* Git commit hash
* branch name
* scenario ID
* dataset ID/version
* model configuration
* optimization parameters
* simulation parameters
* evaluation metrics

If an experiment depends on a particular configuration, keep that configuration version-controlled.

---

## Baseline Preservation

The `research-baseline` branch represents the current research development baseline.

Changes that intentionally alter baseline behavior should be documented.

Before modifying baseline behavior, consider recording:

```text
Baseline commit:
Previous configuration:
New configuration:
Reason for change:
Expected effect:
Observed effect:
```

Do not change baseline algorithms merely to improve a single experimental result without documenting the reason.

---

## Experimental Branches

Experimental branches are intended for temporary or exploratory work.

Examples:

```text
experiment/preference-model
experiment/synthetic-traffic
experiment/adaptive-mopso
experiment/uncertainty-analysis
experiment/highd-evaluation
```

Experimental code should not automatically become part of the baseline.

Before merging an experiment, verify:

* the implementation is understood
* tests pass
* experimental configuration is documented
* results are reproducible
* the research purpose is clear
* unintended baseline changes are excluded

---

## Feature Branches

Feature branches should contain focused implementation work.

Examples:

```text
feature/synthetic-data
feature/uncertainty-model
feature/preference-model
feature/adaptive-mopso
feature/real-dataset
```

Keep unrelated changes out of feature branches.

If a feature requires multiple independent changes, consider splitting them into separate commits or branches.

---

## Code Quality

Prefer readable and explicit Python code over unnecessarily complex implementations.

When modifying existing research code:

* preserve existing behavior unless the change is intentional
* avoid unnecessary refactoring
* keep functions focused
* use descriptive variable names
* document non-obvious research assumptions
* add tests for new functionality where practical

Do not introduce dependencies unless they are necessary for the research or development workflow.

---

## Reproducibility and Randomness

When randomness is used, make the random seed explicit whenever possible.

For example:

```python
seed = 42
```

Research experiments should avoid uncontrolled sources of randomness when deterministic reproduction is required.

If deterministic behavior is not possible, document the source of randomness and the expected variability.

---

## Research Artifacts

Research artifacts may include:

* scenario definitions
* trajectory candidates
* preference annotations
* processed datasets
* experiment configurations
* evaluation scripts
* numerical analysis
* visualization scripts
* reproducibility utilities

Artifacts should have clear names and should not obscure whether they are:

* raw data
* processed data
* human annotations
* synthetic data
* experimental outputs
* final results

---

## Documentation

Update documentation when a change affects:

* installation
* repository structure
* experiment execution
* datasets
* model configuration
* optimization parameters
* evaluation procedures
* reproducibility

Documentation should describe the current implementation rather than an outdated experimental state.

---

## Recommended Development Workflow

A typical research task should follow this workflow:

```text
1. Start from research-baseline
          ↓
2. Pull the latest changes
          ↓
3. Create a feature or experiment branch
          ↓
4. Implement the change
          ↓
5. Run the relevant tests
          ↓
6. Run the full test suite
          ↓
7. Record experimental configuration/results
          ↓
8. Commit with a descriptive message
          ↓
9. Push the branch
          ↓
10. Open a pull request
          ↓
11. Review CI and research impact
          ↓
12. Merge after approval
```

Example:

```bash
git switch research-baseline
git pull origin research-baseline

git switch -c feature/synthetic-data

# Implement changes

python -m tests.test_vehicle_scenario
python -m tests.test_vehicle_pso
python -m tests.test_candidate_generation
python -m tests.test_mopso
python -m tests.test_preference_data
python -m tests.test_preference_dataset
python -m tests.test_preference_annotator
python -m tests.test_trajectory_features

git add .
git commit -m "feat: add synthetic traffic generator"
git push -u origin feature/synthetic-data
```

Then open a pull request against:

```text
research-baseline
```

---

## Branch Roles

The repository uses the following branch roles:

| Branch              | Purpose                              |
| ------------------- | ------------------------------------ |
| `main`              | Stable/publication-ready code        |
| `research-baseline` | Active research development baseline |
| `feature/*`         | Focused implementation work          |
| `experiment/*`      | Experimental or exploratory work     |
| `fix/*`             | Bug fixes and correctness changes    |

The `main` branch should remain stable.

The `research-baseline` branch may contain active research development but should remain reproducible and testable.

---

## Final Principle

Every important research change should answer three questions:

1. **What changed?**
2. **Why did it change?**
3. **Can we reproduce and evaluate the effect of the change?**

The repository should preserve enough history and metadata to trace the project from the research question through implementation, experimentation, results, and final conclusions.
