import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy, SVDpp
from surprise.model_selection import train_test_split, cross_validate
import matplotlib.pyplot as plt

# Load the dataset
ratings = pd.read_csv("ML-100K/small_ratings.csv")

# Defining a Reader object
reader = Reader(rating_scale=(0.5, 5))  # MovieLens ratings are between 0.5 and 5

# Loading the dataset into Surprise's format
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Spliting the dataset into the trainset and testset
trainset, testset = train_test_split(data, test_size=0.25)

print(f"\nTrainset Shape: {trainset}")

# Using the SVD algorithm from Surprise
algorithm = SVD()

# Training the algorithm on the trainset
algorithm.fit(trainset)

# Predicting ratings for the testset
predictions = algorithm.test(testset)

# Printing individual predictions
for prediction in predictions:
    print(f"User: {prediction.uid}, Item: {prediction.iid}, Actual Rating: {prediction.r_ui}, Predicted Rating: {prediction.est}, Error: {prediction.est - prediction.r_ui:.2f}")


# Calculating RMSE
rmse = accuracy.rmse(predictions)
print(f'RMSE: {rmse}')

results = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=50, verbose=True)

# Plotting RMSE from cross-validation
plt.figure(figsize=(7, 5))
plt.plot(results['test_rmse'], label='Test RMSE')
plt.title('Root Mean Square Error (RMSE) Over 5-Folds')
plt.ylabel('Root Mean Square Error')
plt.xlabel('Fold')
plt.legend()
plt.show()

# Plotting MAE from cross-validation
plt.figure(figsize=(7, 5))
plt.plot(results['test_mae'], label='Test MAE')
plt.title('Mean Absolute Error (MAE) Over 5-Folds')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Fold')
plt.legend()
plt.show()