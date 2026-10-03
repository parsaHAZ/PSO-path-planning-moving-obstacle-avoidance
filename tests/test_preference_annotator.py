from research.preference_annotator import PreferenceAnnotator


def main():
    annotator = PreferenceAnnotator(
        dataset_path="data/preference/scenario_0001_preferences.json",
        output_path="data/preference/scenario_0001_annotated.json",
        n_candidates=10,
    )
    annotator.run()


if __name__ == "__main__":
    main()