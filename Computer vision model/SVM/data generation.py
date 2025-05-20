import pandas as pd
import numpy as np

n_samples_per_class = 3000  
total_samples = n_samples_per_class * 3



# Class 0: slope near zero (-0.5 to 0.5) + some noise
slopes_0 = np.random.uniform(-0.5, 0.5, n_samples_per_class) + np.random.normal(0, 0.1, n_samples_per_class)
lengths_0 = np.random.uniform(10, 100, n_samples_per_class)
mid_x_0 = np.random.uniform(-50, 50, n_samples_per_class)
mid_y_0 = np.random.uniform(-50, 50, n_samples_per_class)
labels_0 = np.zeros(n_samples_per_class, dtype=int)

# Class 1: slope > 1.0 (right)
slopes_1 = np.random.uniform(1.0, 10.0, n_samples_per_class) + np.random.normal(0, 0.5, n_samples_per_class)
lengths_1 = np.random.uniform(10, 100, n_samples_per_class)
mid_x_1 = np.random.uniform(-50, 50, n_samples_per_class)
mid_y_1 = np.random.uniform(-50, 50, n_samples_per_class)
labels_1 = np.ones(n_samples_per_class, dtype=int)

# Class 2: slope < -1.0 (left)
slopes_2 = np.random.uniform(-10.0, -1.0, n_samples_per_class) + np.random.normal(0, 0.5, n_samples_per_class)
lengths_2 = np.random.uniform(10, 100, n_samples_per_class)
mid_x_2 = np.random.uniform(-50, 50, n_samples_per_class)
mid_y_2 = np.random.uniform(-50, 50, n_samples_per_class)
labels_2 = np.full(n_samples_per_class, 2, dtype=int)

# Combine all classes
slopes = np.concatenate([slopes_0, slopes_1, slopes_2])
lengths = np.concatenate([lengths_0, lengths_1, lengths_2])
mid_xs = np.concatenate([mid_x_0, mid_x_1, mid_x_2])
mid_ys = np.concatenate([mid_y_0, mid_y_1, mid_y_2])
labels = np.concatenate([labels_0, labels_1, labels_2])

# Build DataFrame
data = {
    'slope': slopes,
    'length': lengths,
    'mid_x': mid_xs,
    'mid_y': mid_ys,
    'label': labels
}

df = pd.DataFrame(data)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv('labeled_line_data.csv', index=False)

print(df['label'].value_counts())
