# iris-data-task6
Objective
This project aims to understand and implement the K-Nearest Neighbors (KNN) algorithm for a classification task using the well-known Iris dataset. The process involves several steps, including data normalization, model training, hyperparameter tuning (selecting the optimal K), model evaluation, and visualizing decision boundaries.

Concepts Covered
Instance-based Learning
Euclidean Distance Calculation
Normalization & Scaling
Model Evaluation Metrics
Hyperparameter Tuning (K selection)
Visualization of Decision Boundaries

Tools and Libraries Used
Python 3
Pandas for data handling
NumPy for numerical computations
Matplotlib for visualization
Scikit-learn (sklearn) for machine learning algorithms and tools

Steps Performed
1.  Load Dataset
The Iris dataset is loaded directly from sklearn.datasets.

2. Normalize Features
To ensure that the KNN algorithm performs optimally, we applied StandardScaler to normalize the feature values.

3. Train-Test Split
We used train_test_split() to divide the data into training (80%) and testing (20%) sets.

4.  Model Training with KNN
Implemented KNN using KNeighborsClassifier.
Trained models with various values of K (from 1 to 10).
Identified the best-performing K based on accuracy on the test data.
5.  Model Evaluation
Evaluated the model using Accuracy, Confusion Matrix, and Classification Report.
The final model demonstrated impressive performance, achieving 100% accuracy at K=3.
6. Visualize Decision Boundary
Plotted decision boundaries using the first two features.

The KNN algorithm classifies new data points by looking at the majority label of its K nearest neighbors, using distance measures like Euclidean distance. Finding the right K involves trying out different values and using methods like cross-validation to see which works best.
KNN also handles multi-class problems well by voting among neighbors, regardless of how many classes there are. The choice of distance metric is key, as it defines how close two points are, with common options being Euclidean, Manhattan, and Minkowski distances.

