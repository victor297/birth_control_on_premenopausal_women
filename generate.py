import pandas as pd
import numpy as np

# Example: Generate dataset
np.random.seed(0)
n_samples = 1000

data = {
    "Wife's age": np.random.randint(15, 45, n_samples),
    "Wife's education": np.random.choice([1, 2, 3, 4], n_samples),
    "Husband's education": np.random.choice([1, 2, 3, 4], n_samples),
    "Number of children ever born": np.random.randint(0, 6, n_samples),
    "Wife's religion": np.random.choice([0, 1], n_samples),
    "Wife's now working?": np.random.choice([0, 1], n_samples),
    "Husband's occupation": np.random.choice([1, 2, 3, 4], n_samples),
    "Standard-of-living index": np.random.choice([1, 2, 3, 4], n_samples),
    "Media exposure": np.random.choice([0, 1], n_samples),
    "Contraceptive method used": np.random.choice([1, 2, 3], n_samples),
    'Birth control drug type': np.random.randint(1, 5, n_samples),
    'Age when drug was used': np.random.randint(10, 35, n_samples),
    'Effect of the drug': np.random.randint(1, 4, n_samples),
    'Race': np.random.randint(1, 5, n_samples),
    'Stature': np.random.randint(150, 190, n_samples),
    'Complexion': np.random.randint(1, 4, n_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Generate random ages at which the test was taken
df['Age at test'] = np.random.randint(35, 50, n_samples)

# Save to CSV
df.to_csv('dataset_with_age_at_test.csv', index=False)

print("Dataset saved successfully as dataset_with_age_at_test.csv")
