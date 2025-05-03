import numpy as np
import csv


def generate_z_given_y(y):
    """Generates a value for the random variable Z given Y.
     Args:
        y (float): The value of the variable Y.
     Returns:
        int: The generated value of the variable Z.
  """
    if y > 100:
        z = np.random.randint(0, 3)
    elif y > 30:
        z = np.random.randint(2, 5)
    elif y > -30:
        z = np.random.randint(4, 7)
    elif y > -100:
        z = np.random.randint(7, 10)
    else:
        z = np.random.randint(9, 13)
    return z


def generate_w():
    return np.random.normal(5.0, 1.0)


def generate_x():
    return np.random.normal(0.0, 5.0)


def load_csv_data(file_name):
    """
    Loads data from a CSV file.
    Returns:
        tuple: A tuple containing three numpy arrays: w, x, and y.
    """

    x_vals, w_vals, y_vals = [], [], []
    with open(file_name, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            w_vals.append(float(row['w']))
            x_vals.append(float(row['x']))
            y_vals.append(float(row['y']))
    return (
        np.array(w_vals, dtype=np.float32),
        np.array(x_vals, dtype=np.float32),
        np.array(y_vals, dtype=np.float32),
    )
