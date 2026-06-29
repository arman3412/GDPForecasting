import copy
import preprocessing
import lstm
import transformer
import torch
import numpy as np
from collections import defaultdict

NUM_EPOCHS = 100


class IndicatorDataset(torch.utils.data.Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, country_idx: torch.Tensor, groups: list):
        self.X = X
        self.y = y
        self.country_idx = country_idx
        self.groups = groups

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.country_idx[idx], self.groups[idx]


def train(seed, hidden_dim=32, layer_dim=2, num_epochs=NUM_EPOCHS, lr=0.001, batch_size=32, patience=5):
    """Train LSTM and Transformer and return all tracked metrics.

    Returns:
        results[model_type][seed] = {
            "best_epoch"         : int,    # epoch of lowest val loss
            "early_stop_epoch"   : int,    # epoch training actually stopped at
            "best_val_mae"       : float,
            "baseline_mae"       : float,  # overall naive-mean baseline MAE (val)
            "per_country"        : {country: (model_mae, baseline_mae)},  # val, at best epoch
            "test_per_country"   : {country: (model_mae, baseline_mae)},  # held-out test set
            "train_loss_history" : [float, ...],
            "val_loss_history"   : [float, ...],
        }
    """
    torch.manual_seed(seed)
    # Load and split the data
    data_splits = preprocessing.split_data()

    # Convert to PyTorch tensors
    X_train = torch.tensor(data_splits["X_train"], dtype=torch.float32)
    y_train = torch.tensor(data_splits["y_train"], dtype=torch.float32)
    country_idx_train = torch.tensor(data_splits["numerical_labels_train"], dtype=torch.long)
    groups_train = data_splits["groups_train"]

    train_loader = torch.utils.data.DataLoader(
        IndicatorDataset(X_train, y_train, country_idx_train, groups_train),
        batch_size=batch_size, shuffle=True,
    )

    X_val = torch.tensor(data_splits["X_val"], dtype=torch.float32)
    y_val = torch.tensor(data_splits["y_val"], dtype=torch.float32)
    country_idx_val = torch.tensor(data_splits["numerical_labels_val"], dtype=torch.long)
    groups_val = data_splits["groups_val"]

    val_loader = torch.utils.data.DataLoader(
        IndicatorDataset(X_val, y_val, country_idx_val, groups_val),
        batch_size=batch_size, shuffle=False,
    )

    X_test = torch.tensor(data_splits["X_test"], dtype=torch.float32)
    y_test = torch.tensor(data_splits["y_test"], dtype=torch.float32)
    country_idx_test = torch.tensor(data_splits["numerical_labels_test"], dtype=torch.long)
    groups_test = data_splits["groups_test"]

    num_countries = data_splits["num_countries"]

    test_loader = torch.utils.data.DataLoader(
        IndicatorDataset(X_test, y_test, country_idx_test, groups_test),
        batch_size=batch_size, shuffle=False,
    )

    # Naive mean baseline: predict each country's train-set mean on the val set
    train_sum = defaultdict(float)
    train_count = defaultdict(int)
    for y, g in zip(data_splits["y_train"], data_splits["groups_train"]):
        train_sum[g] += float(y)
        train_count[g] += 1
    country_train_mean = {c: train_sum[c] / train_count[c] for c in train_sum}

    # Calculate baseline error to compare to model performance
    # Guessing the mean of the previous GDP growths
    baseline_abs_err = defaultdict(float)
    baseline_count = defaultdict(int)
    for y, g in zip(data_splits["y_val"], data_splits["groups_val"]):
        baseline_abs_err[g] += abs(float(y) - country_train_mean.get(g, 0.0))
        baseline_count[g] += 1
    baseline_mae = {c: baseline_abs_err[c] / baseline_count[c] for c in baseline_abs_err}
    total_baseline_MAE = sum(baseline_mae.values()) / num_countries

    results = {}

    for model_type in ["lstm", "transformer"]:
        if model_type == "lstm":
            model = lstm.LSTM(
                input_dim=X_train.shape[2], hidden_dim=hidden_dim,
                layer_dim=layer_dim, output_dim=1, num_countries=num_countries,
            )
        else:
            model = transformer.Transformer(
                input_dim=X_train.shape[2], layer_dim=layer_dim,
                output_dim=1, num_countries=num_countries,
            )

        criterion = torch.nn.L1Loss()  # MAE loss because some volatile countries are just being guessed on
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Track and record the best epoch.
        least_loss_MAE = float("inf")
        least_loss_epoch = 0
        least_loss_country_abs_err = {}
        least_loss_country_counts = {}
        best_state_dict = None

        train_loss_history = []
        val_loss_history = []

        epochs_without_improvement = 0  # Early stopping
        early_stop_epoch = num_epochs - 1

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            batch_count = 0
            for X_batch, y_batch, country_idx_batch, group_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch, country_idx_batch)
                batch_loss = criterion(outputs.squeeze(), y_batch)
                batch_loss.backward()
                optimizer.step()
                total_train_loss += batch_loss.item()
                batch_count += 1

            train_loss_history.append(total_train_loss / batch_count)

            # Val evaluation
            total_val_loss = 0
            batch_count = 0
            country_sq_err = {}
            country_abs_err = {}
            country_counts = {}

            model.eval()
            for X_batch, y_batch, country_idx_batch, group_batch in val_loader:
                with torch.no_grad():
                    outputs = model(X_batch, country_idx_batch)
                    val_loss = criterion(outputs.squeeze(), y_batch)
                total_val_loss += val_loss.item()
                batch_count += 1

                sq_err = (outputs.squeeze() - y_batch) ** 2
                abs_err = torch.abs(outputs.squeeze() - y_batch)
                for i, country in enumerate(group_batch):
                    country_sq_err[country] = country_sq_err.get(country, 0.0) + sq_err[i].item()
                    country_abs_err[country] = country_abs_err.get(country, 0.0) + abs_err[i].item()
                    country_counts[country] = country_counts.get(country, 0) + 1

            val_loss_history.append(total_val_loss / batch_count)

            # Track best performing epoch
            if total_val_loss / batch_count < least_loss_MAE:
                least_loss_MAE = total_val_loss / batch_count
                least_loss_epoch = epoch
                least_loss_country_abs_err = dict(country_abs_err)
                least_loss_country_counts = dict(country_counts)
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping: record the epoch and exit the training loop
            if epochs_without_improvement >= patience:
                early_stop_epoch = epoch
                break

        # Restore best weights after early stopping
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        # Test set evaluation (best weights, held-out period)
        test_country_abs_err = defaultdict(float)
        test_country_counts = defaultdict(int)
        model.eval()
        for X_batch, y_batch, country_idx_batch, group_batch in test_loader:
            with torch.no_grad():
                outputs = model(X_batch, country_idx_batch)
            abs_err = torch.abs(outputs.squeeze() - y_batch)
            for i, country in enumerate(group_batch):
                test_country_abs_err[country] += abs_err[i].item()
                test_country_counts[country] += 1

        # Test baseline: same train-mean predictor applied to the test period
        test_baseline_abs_err = defaultdict(float)
        test_baseline_count = defaultdict(int)
        for y, g in zip(data_splits["y_test"], data_splits["groups_test"]):
            test_baseline_abs_err[g] += abs(float(y) - country_train_mean.get(g, 0.0))
            test_baseline_count[g] += 1
        test_baseline_mae = {
            c: test_baseline_abs_err[c] / test_baseline_count[c]
            for c in test_baseline_abs_err
        }

        test_per_country = {
            c: (test_country_abs_err[c] / test_country_counts[c],
                test_baseline_mae.get(c, float("nan")))
            for c in sorted(test_country_abs_err)
        }

        least_loss_per_country = {
            c: least_loss_country_abs_err[c] / least_loss_country_counts[c]
            for c in sorted(least_loss_country_abs_err)
        }

        results[model_type] = {
            seed: {
                "best_epoch":          least_loss_epoch,
                "early_stop_epoch":    early_stop_epoch,
                "best_val_mae":        least_loss_MAE,
                "baseline_mae":        total_baseline_MAE,
                "per_country":         {
                    c: (least_loss_per_country[c], baseline_mae.get(c, float("nan")))
                    for c in least_loss_per_country
                },
                "test_per_country":    test_per_country,
                "train_loss_history":  train_loss_history,
                "val_loss_history":    val_loss_history,
            }
        }

    return results


def _print_per_country_table(per_country):
    print(f"  {'Country':<10} {'Model MAE':>10} {'Baseline':>10} {'Ratio':>8}")
    print(f"  {'-'*42}")
    for c, (m, b) in sorted(per_country.items()):
        ratio = m / b if b > 0 else float("nan")
        mark = "<" if m < b else ">"
        print(f"  {c:<10} {m:>10.3f} {b:>10.3f} {ratio:>7.2f}x {mark}")


if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 5]
    all_results = {}  # all_results[model_type][seed] = {...}

    # Per-seed training
    for seed in seeds:
        print(f"\nSeed {seed} {'─'*54}")
        r = train(seed=seed)
        for model_type, seed_dict in r.items():
            all_results.setdefault(model_type, {}).update(seed_dict)
            info = seed_dict[seed]
            print(f"\n  [{model_type.upper()}]  "
                  f"stopped epoch {info['early_stop_epoch']}  "
                  f"(best {info['best_epoch']})  |  "
                  f"val MAE {info['best_val_mae']:.4f}  |  "
                  f"baseline {info['baseline_mae']:.4f}")
            _print_per_country_table(info["per_country"])

    # Summary across seeds
    print(f"\n{'='*60}")
    print(f"Aggregate Summary — {len(seeds)} seeds: {seeds}")
    print(f"{'='*60}")

    for model_type in ["lstm", "transformer"]:
        seed_data = all_results[model_type]
        countries = sorted(next(iter(seed_data.values()))["per_country"])

        print(f"\n  [{model_type.upper()}]")
        print(f"  {'Country':<10} {'Mean MAE':>10} {'± Std':>8} {'Baseline':>10} {'Mean Ratio':>11}")
        print(f"  {'-'*52}")
        for c in countries:
            maes = [
                seed_data[s]["per_country"][c][0]
                for s in seeds if c in seed_data[s]["per_country"]
            ]
            baseline = seed_data[seeds[0]]["per_country"][c][1]
            mean_mae = np.mean(maes)
            std_mae = np.std(maes)
            mean_ratio = mean_mae / baseline if baseline > 0 else float("nan")
            mark = "<" if mean_mae < baseline else ">"
            print(f"  {c:<10} {mean_mae:>10.3f} {std_mae:>8.3f} {baseline:>10.3f} {mean_ratio:>10.2f}x {mark}")

        stop_epochs = [seed_data[s]["early_stop_epoch"] for s in seeds]
        best_epochs = [seed_data[s]["best_epoch"] for s in seeds]
        val_maes = [seed_data[s]["best_val_mae"] for s in seeds]
        print(f"\n  Stopped epoch : {np.mean(stop_epochs):.1f} ± {np.std(stop_epochs):.1f}  "
              f"(best {np.mean(best_epochs):.1f} ± {np.std(best_epochs):.1f})")
        print(f"  Val MAE       : {np.mean(val_maes):.4f} ± {np.std(val_maes):.4f}")
        print(f"  Baseline      : {seed_data[seeds[0]]['baseline_mae']:.4f}")

    print(f"\n{'='*60}")

    # Test set evaluation — seed 0 model
    TEST_SEED = 0
    print(f"\n{'='*60}")
    print(f"Test Set Evaluation — seed {TEST_SEED}")
    print(f"{'='*60}")
    for model_type in ["lstm", "transformer"]:
        info = all_results[model_type][TEST_SEED]
        test_maes = [m for m, _ in info["test_per_country"].values()]
        test_baselines = [b for _, b in info["test_per_country"].values()]
        overall_test_mae = np.mean(test_maes)
        overall_baseline = np.mean(test_baselines)
        print(f"\n  [{model_type.upper()}]  "
              f"test MAE {overall_test_mae:.4f}  |  "
              f"baseline {overall_baseline:.4f}  |  "
              f"ratio {overall_test_mae/overall_baseline:.2f}x")
        _print_per_country_table(info["test_per_country"])
    print(f"{'='*60}")
