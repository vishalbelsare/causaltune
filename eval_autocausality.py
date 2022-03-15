import os
import sys
import warnings
import pickle

warnings.filterwarnings("ignore")

root_path = root_path = os.path.realpath("../")
sys.path.append(os.path.join(root_path, "auto-causality"))

from auto_causality import AutoCausality, datasets  # noqa F402


def main():
    """performs automl for each combination of dataset and metric
       that is available in this package
    """
    all_datasets = {
        "amazon_pos": datasets.amazon_reviews,
        "ihdp": datasets.synth_ihdp,
        "nhefs": datasets.nhefs,
        "lalonde": datasets.lalonde_nsw,
        "acic_1": datasets.synth_acic,
    }

    all_metrics = ["erupt", "qini", "auc", "ate", "r_score"]

    for k_ds, f_ds in all_datasets.items():
        for metric in all_metrics:
        
            print(f"now performing automl on {k_ds} dataset with {metric} metric")
            # load and preproc data
            data = f_ds()
            (
                data,
                features_X,
                features_W,
                targets,
                treatment,
            ) = datasets.preprocess_dataset(data)
            # perform automl
            ac = AutoCausality(
                time_budget=600,
                metric=metric,
                num_samples=20,
                verbose=1,
                components_verbose=1,
                components_time_budget=20,
                use_ray=False,
            )
            ac.fit(data, treatment, targets[0], features_W, features_X)

            # store results
            with open("results_" + k_ds + "_" + metric + ".pkl", "wb") as f:
                pickle.dump(ac.full_scores, f)
            

if __name__ == "__main__":
    main()
