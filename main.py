# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Create a sample dataset with students and their features
data = {
    "Student": ["A", "B", "C", "D"],
    "Attendance (%)": [40, 55, 60, 85],  # Class attendance percentage
    "Study Hours/Week": [2, 4, 6, 8],     # Weekly study hours
    "Attention Span (%)": [45, 55, 70, 80],  # Attention span as a percentage
    "Final Grade": ["C-", "B-", "B+", "A+"]  # Final grade as letter
}

# Convert data into a pandas DataFrame
df = pd.DataFrame(data)

# Step 2: Map final grades to numerical values for analysis (optional, not used in PCA itself)
grade_encoding = {"C-": 1, "B-": 2, "B+": 3, "A+": 4}
df["Final Grade Encoded"] = df["Final Grade"].map(grade_encoding)

# Step 3: Select the features for PCA
X = df[["Attendance (%)", "Study Hours/Week", "Attention Span (%)"]]  # These are the independent variables

# Step 4: Standardize the data so that each feature contributes equally
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Mean = 0, Standard Deviation = 1

# Step 5: Apply PCA
pca = PCA(n_components=3)  # We have 3 features, so we can extract up to 3 principal components
X_pca = pca.fit_transform(X_scaled)  # Transform the data into the new PCA space

# Step 6: Check how much variance is explained by each principal component
explained_variance = pca.explained_variance_ratio_  # Shows the importance of each PC

# Step 7: Display the loadings (contribution of each feature to each principal component)
loadings = pd.DataFrame(
    pca.components_,
    columns=X.columns,
    index=[f'PC{i+1}' for i in range(pca.n_components_)]
)

# Step 8: Convert PCA results into a DataFrame for inspection
pca_results = pd.DataFrame(
    X_pca,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)]
)
pca_results['Student'] = df['Student']
pca_results['Final Grade'] = df['Final Grade']

# Now we have:
# - 'explained_variance': the importance of each principal component
# - 'loadings': how each feature contributes to the principal components
# - 'pca_results': transformed student data in the PCA space

# These outputs can be printed or plotted depending on the teaching use case
