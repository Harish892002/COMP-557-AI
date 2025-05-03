import os
from pgm import PGM
from utils import load_csv_data
import pickle
import argparse


def train_model(file_name):
    # Load the training data
    data_w, data_x, data_y = load_csv_data(file_name)

    model = PGM()
    model.train(data_w, data_x, data_y)

    with open('pgm_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model


def load_model():
    model_path = 'pgm_model.pkl'
    # if not exists:
    if not os.path.exists(model_path):
        return None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def test_q1(model: PGM, test_filename):
    test_w, test_x, test_y = load_csv_data(test_filename)
    diffs = []
    for idx in range(len(test_w)):
        y = model.predict_y(test_w[idx], test_x[idx])
        diffs.append(abs(y - test_y[idx]))

    passed = [1 if diff < 1 else 0 for diff in diffs]
    q1_points = int(round(sum(passed) / len(passed) * 30))
    print(f"Q1 Points: {q1_points}")
    return q1_points


def test_q2(model: PGM):
    true_w_stats = {2: (5.08, 0.96), 11: (5.27, 0.99)}
    estimated_w_stats = {}
    for z in true_w_stats:
        w_mean, w_std = model.infer_w_given_z(z)
        estimated_w_stats[z] = (w_mean, w_std)
        if (abs(w_mean - true_w_stats[z][0]) < 0.1
                and abs(w_std - true_w_stats[z][1]) < 0.1):
            print(
                f"For z={z}, the estimation (mean: {w_mean:.2f}, std: {w_std:.2f}) is correct."
                f" True values are (mean: {true_w_stats[z][0]}, std: {true_w_stats[z][1]})."
            )
        else:
            print(
                f"For z={z}, the estimation (mean: {w_mean:.2f}, std: {w_std:.2f}) is INCORRECT."
                f" True values are (mean: {true_w_stats[z][0]}, std: {true_w_stats[z][1]})."
            )


def test_q3(model: PGM):
    true_w_stats = {(2, 4): (4.99, 0.98), (-5, 10): (5.40, 0.74)}
    estimated_w_stats = {}
    for key in true_w_stats:
        x, z = key
        w_mean, w_std = model.infer_w_given_x_z(x, z)
        estimated_w_stats[key] = (w_mean, w_std)
        if (abs(w_mean - true_w_stats[key][0]) < 0.1
                and abs(w_std - true_w_stats[key][1]) < 0.1):
            print(
                f"For (x, z)=({x}, {z}), the estimation (mean: {w_mean:.2f}, std: {w_std:.2f}) is correct."
                f" True values are (mean: {true_w_stats[key][0]}, std: {true_w_stats[key][1]})."
            )
        else:
            print(
                f"For (x, z)=({x}, {z}), the estimation (mean: {w_mean:.2f}, std: {w_std:.2f}) is INCORRECT."
                f" True values are (mean: {true_w_stats[key][0]}, std: {true_w_stats[key][1]})."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file",
                        type=str,
                        help="Path to training data CSV",
                        default='training_data.csv')
    parser.add_argument("--validation_file",
                        type=str,
                        help="Path to validation data CSV",
                        default='validation_data.csv')
    parser.add_argument("--q1", action="store_true", help="Run test_q1")
    parser.add_argument("--q2", action="store_true", help="Run test_q2")
    parser.add_argument("--q3", action="store_true", help="Run test_q3")
    args = parser.parse_args()

    model = None
    if args.q1:
        model = train_model(args.train_file)
        test_q1(model, args.validation_file)

    if args.q2:
        if model is None:
            model = load_model()
        if model is None:
            print("Model not found. Please train the model first.")
            exit(1)
        test_q2(model)

    if args.q3:
        if model is None:
            model = load_model()
        if model is None:
            print("Model not found. Please train the model first.")
            exit(1)
        test_q3(model)
