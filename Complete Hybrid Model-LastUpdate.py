import numpy as np
import pandas as pd
import random
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Embedding, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import RootMeanSquaredError
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
ratings = pd.read_csv("filtered_ratings.csv")
movies = pd.read_csv("filtered_movies.csv")

# Merge datasets
merged_df = pd.merge(ratings, movies, on='movieId', how='left')

# Creating a sparse matrix
user_ids = merged_df['userId'].values - 1
movie_ids = merged_df['movieId'].values - 1
ratings = merged_df['rating'].values

# Create the sparse matrix
interaction_matrix = coo_matrix((ratings, (user_ids, movie_ids)), shape=(user_ids.max() + 1, movie_ids.max() + 1))

# Perform SVD
k = 100  # Number of latent factors
U, sigma, Vt = svds(interaction_matrix, k=k)

# Encode categorical data
genre_encoder = LabelEncoder()
director_encoder = LabelEncoder()
merged_df['Genre_encoded'] = genre_encoder.fit_transform(merged_df['genres'])
merged_df['Director_encoded'] = director_encoder.fit_transform(merged_df['Director'])

# Normalaize imdbVotes
merged_df['imdbVotes'] = merged_df['imdbVotes'] / merged_df['imdbVotes'].max()

assert merged_df[
    ['Rotten Tomatoes', 'imdbRatings', 'imdbVotes']].notna().all().all(), "There are missing values in the data"

rotten_tomatoes_input = Input(shape=(1,), name='rotten_tomatoes')
imdb_rating_input = Input(shape=(1,), name='imdb_rating')
imdb_voting_input = Input(shape=(1,), name='imdb_voting')

# Filter merged_df to ensure it only includes the user_ids and movie_ids that were in the sparse matrix
valid_user_ids = np.unique(user_ids)  # This extracts the unique user_ids that have ratings
merged_df = merged_df[merged_df['userId'].isin(valid_user_ids + 1)]  # +1 to match the original IDs

# Prepare inputs for training, ensure to extract data only for valid users
genre_data = merged_df['Genre_encoded'].values.reshape(-1, 1)
director_data = merged_df['Director_encoded'].values.reshape(-1, 1)
ratings = merged_df['rating'].values

# Align user_features using an index that matches filtered merged_df
user_features = U[merged_df['userId'].values - 1, :]

rotten_tomatoes_data = merged_df['Rotten Tomatoes'].values.reshape(-1, 1)
imdb_rating_data = merged_df['imdbRatings'].values.reshape(-1, 1)
imdb_voting_data = merged_df['imdbVotes'].values.reshape(-1, 1)

X_train, X_test, genre_train, genre_test, director_train, director_test, rotten_tomatoes_train, rotten_tomatoes_test, imdb_rating_train, imdb_rating_test, imdb_voting_train, imdb_voting_test, y_train, y_test = train_test_split(
    user_features, genre_data, director_data, rotten_tomatoes_data, imdb_rating_data, imdb_voting_data, ratings,
    test_size=0.2, random_state=42
)

# Model Inputs
user_features_input = Input(shape=(k,), name='user_features')
genre_input = Input(shape=(1,), name='genre_input')
director_input = Input(shape=(1,), name='director_input')
genre_embedding = Embedding(input_dim=merged_df['Genre_encoded'].nunique(), output_dim=15)(genre_input)
director_embedding = Embedding(input_dim=merged_df['Director_encoded'].nunique(), output_dim=15)(director_input)
genre_vector = Flatten()(genre_embedding)
director_vector = Flatten()(director_embedding)
concatenated_features = Concatenate()([user_features_input, genre_vector, director_vector,
                                       rotten_tomatoes_input, imdb_rating_input, imdb_voting_input
                                       ])

assert len(user_features) == len(genre_data) == len(director_data) == len(ratings), "Data arrays must all be the same length"

print("\n\nStarting model training...")

print("--------------------------------------------------")

# Model architecture
x = Dense(256, activation='relu')(concatenated_features)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='linear')(x)

model = Model(inputs=[user_features_input, genre_input, director_input, rotten_tomatoes_input, imdb_rating_input,
                      imdb_voting_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['MAE', RootMeanSquaredError(name='RMSE')])
0
# Setup checkpoints and early stopping
checkpoint = ModelCheckpoint('Results/Models/Complete_Hybrid_Model_Run01.keras', monitor='val_loss',
                             save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

# Train the model
history = model.fit(
    [X_train, genre_train, director_train, rotten_tomatoes_train, imdb_rating_train, imdb_voting_train], y_train,
    validation_data=(
        [X_test, genre_test, director_test, rotten_tomatoes_test, imdb_rating_test, imdb_voting_test], y_test),
    epochs=150,
    batch_size=64,
    callbacks=[checkpoint, early_stopping]
)

# Evaluate the model
test_loss, test_mae, test_rmse = model.evaluate([X_test, genre_test, director_test, rotten_tomatoes_test, imdb_rating_test, imdb_voting_test], y_test)
predicted_ratings = model.predict([X_test, genre_test, director_test, rotten_tomatoes_test, imdb_rating_test, imdb_voting_test]).flatten()

predicted_ratings = model.predict(
    [X_test, genre_test, director_test, rotten_tomatoes_test, imdb_rating_test, imdb_voting_test]).flatten()


# Calculate MAE and RMSE
mae = mean_absolute_error(y_test, predicted_ratings)
rmse = np.sqrt(mean_squared_error(y_test, predicted_ratings))
print("------------------------------------------------------------------------------------")
print("Final results:")
print(f'\nTest Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f} \n')
print(f'Mean Absolute Error (MAE): {mae:.4f}\n')
print(f'Root Mean Square Error (RMSE): {rmse:.4f}')
print("------------------------------------------------------------------------------------")

# Plotting training and validation loss
plt.figure(figsize=(7, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plotting training and validation MAE
plt.figure(figsize=(7, 5))
plt.plot(history.history['MAE'], label='Train MAE')
plt.plot(history.history['val_MAE'], label='Validation MAE')
plt.title('Mean Absolute Error (MAE) Over Epochs')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plotting training and validation RMSE
plt.figure(figsize=(7, 5))
plt.plot(history.history['RMSE'], label='Train RMSE')
plt.plot(history.history['val_RMSE'], label='Validation RMSE')
plt.title('RMSE Over Epochs')
plt.ylabel('Root Mean Square Error')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# New steps to select 5 random users and recommend top 10 movies
# Step 1: Select 5 Random Users
random_users = random.sample(list(merged_df['userId'].unique()), 5)


# Step 2: Predict Ratings
def get_user_predictions(user_id):
    user_features = U[user_id - 1].reshape(1, -1)
    all_movie_ids = merged_df['movieId'].unique()

    genre_data = []
    director_data = []
    rotten_tomatoes_data = []
    imdb_rating_data = []
    imdb_voting_data = []

    for movie_id in all_movie_ids:
        movie_data = merged_df[merged_df['movieId'] == movie_id].iloc[0]
        genre_data.append(movie_data['Genre_encoded'])
        director_data.append(movie_data['Director_encoded'])
        rotten_tomatoes_data.append(movie_data['Rotten Tomatoes'])
        imdb_rating_data.append(movie_data['imdbRatings'])
        imdb_voting_data.append(movie_data['imdbVotes'])

    genre_data = np.array(genre_data).reshape(-1, 1)
    director_data = np.array(director_data).reshape(-1, 1)
    rotten_tomatoes_data = np.array(rotten_tomatoes_data).reshape(-1, 1)
    imdb_rating_data = np.array(imdb_rating_data).reshape(-1, 1)
    imdb_voting_data = np.array(imdb_voting_data).reshape(-1, 1)

    predictions = model.predict([np.repeat(user_features, len(all_movie_ids), axis=0),
                                 genre_data, director_data,
                                 rotten_tomatoes_data, imdb_rating_data, imdb_voting_data])

    predictions = predictions.flatten()
    movie_predictions = list(zip(all_movie_ids, predictions))
    return movie_predictions


# Step 3: Filter Movies Already Rated by the User
def filter_already_rated(user_id, predictions):
    rated_movie_ids = merged_df[merged_df['userId'] == user_id]['movieId'].tolist()
    filtered_predictions = [(movie_id, rating) for movie_id, rating in predictions if movie_id not in rated_movie_ids]
    return filtered_predictions


# Step 4: Sort Movies According to Predicted Ratings
def sort_predictions(predictions):
    return sorted(predictions, key=lambda x: x[1], reverse=True)


# Step 5: Extract Top 10 Recommendations for Each User
def get_top_10_recommendations(user_id):
    predictions = get_user_predictions(user_id)
    filtered_predictions = filter_already_rated(user_id, predictions)
    sorted_predictions = sort_predictions(filtered_predictions)
    return sorted_predictions[:10]


def compare_with_actual_ratings(user_id, top_n=10):
    predictions = get_user_predictions(user_id)
    sorted_predictions = sort_predictions(predictions)
    top_predictions = sorted_predictions[:top_n]

    actual_ratings = merged_df[merged_df['userId'] == user_id][['movieId', 'rating']]

    print(f"\nComparing top {top_n} predicted ratings with actual ratings for User {user_id}:\n")
    for movie_id, predicted_rating in top_predictions:
        actual_rating = actual_ratings[actual_ratings['movieId'] == movie_id]['rating'].values
        if len(actual_rating) > 0:
            actual_rating = actual_rating[0]
            print(f"Movie ({movie_id}): Predicted Rating: {predicted_rating:.2f}, Actual Rating: {actual_rating:.2f}")
        else:
            print(f"Movie ({movie_id}): Predicted Rating: {predicted_rating:.2f}, Actual Rating: Not Rated by User")
    print("\n")


# Step 6: Display Results
for user_id in random_users:
    print(f"\nTop 10 recommendations for User {user_id}:")
    top_10_recommendations = get_top_10_recommendations(user_id)
    for movie_id, predicted_rating in top_10_recommendations:
        movie_title = merged_df[merged_df['movieId'] == movie_id]['title'].values[0]
        print(f"Movie ({movie_id}): {movie_title}, Predicted Rating: {predicted_rating:.2f}")
    print("\n")
    compare_with_actual_ratings(user_id)


def recommend_movies_for_new_user(top_n=10, min_votes=0.001):
    # Include title in the features dataframe
    movie_features_df = merged_df[['movieId', 'title', 'genres', 'Director', 'Rotten Tomatoes', 'imdbRatings', 'imdbVotes']].drop_duplicates()
    movie_features_df.set_index('movieId', inplace=True)

    # Encoding 'genres' and 'Director' columns (Assuming you have some method to encode them)
    # For simplicity, let's use a simple encoding method here:
    movie_features_df['Genre_encoded'] = movie_features_df['genres'].factorize()[0]
    movie_features_df['Director_encoded'] = movie_features_df['Director'].factorize()[0]

    # Filter for popular movies with high ratings
    popular_movies_df = movie_features_df[(movie_features_df['imdbVotes'] >= min_votes) &
                                          (movie_features_df['Rotten Tomatoes'] >= 8) &
                                          (movie_features_df['imdbRatings'] >= 8)]

    # Check if we have enough movies after filtering
    if len(popular_movies_df) < top_n:
        raise ValueError(f"Not enough movies meet the criteria. Only {len(popular_movies_df)} movies available after filtering.")

    # Standardize features
    scaler = StandardScaler()
    movie_features_scaled = scaler.fit_transform(popular_movies_df[['Genre_encoded', 'Director_encoded', 'Rotten Tomatoes', 'imdbRatings', 'imdbVotes']])

    # Compute cosine similarity between all movies
    similarity_matrix = cosine_similarity(movie_features_scaled)

    # Aggregate similarity scores across all movies
    similarity_scores = similarity_matrix.sum(axis=0)

    # Get top N recommendations
    top_n_indices = np.argsort(similarity_scores)[-top_n:][::-1]
    recommendations = popular_movies_df.iloc[top_n_indices].reset_index()

    # Display recommendations with similarity scores
    print("\nRecommendations for a new user with no ratings:")
    for idx, row in recommendations.iterrows():
        movie_id = row['movieId']
        movie_title = row['title']
        similarity_score = similarity_scores[top_n_indices[idx]]
        print(f"Movie ({movie_id}): {movie_title}, Similarity Score: {similarity_score:.2f}")

    return recommendations

# Example usage for new users
new_user_recommendations = recommend_movies_for_new_user(top_n=10, min_votes=0.001)
