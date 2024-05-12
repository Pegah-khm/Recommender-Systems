import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Embedding, Flatten, Dropout, Lambda
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import MeanAbsoluteError, RootMeanSquaredError
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Loading datasets
ratings = pd.read_csv('Final Implementation/ML-25M/filtered_ratings.csv')
movies = pd.read_csv('Final Implementation/ML-25M/filtered_movies.csv')

# Merge ratings and movies datasets on movieID
merged_df = pd.merge(ratings, movies, on='movieId', how='left')

# Extracting Values from merged_df
user_ids = merged_df['userId'].values - 1
movie_ids = merged_df['movieId'].values - 1
ratings = merged_df['rating'].values

# Defining the dimensions of the sparse matrix
num_users = user_ids.max() + 1
num_movies = movie_ids.max() + 1

# Creating a sparse matrix
interaction_matrix = coo_matrix((ratings, (user_ids, movie_ids)), shape=(num_users, num_movies))

# Performing SVD
k = 100  # Number of latent factors
U, sigma, Vt = svds(interaction_matrix, k=k)
print(f'U Shape: {U.shape}')

# Filtering merged_df to ensure it only includes the user_ids and movie_ids that were in the sparse matrix
valid_user_ids = np.unique(user_ids)  # This extracts the unique user_ids that have ratings
merged_df = merged_df[merged_df['userId'].isin(valid_user_ids + 1)]  # +1 to match the original IDs

# Encode Categorical Data (genres and director)
genre_encoder = LabelEncoder()
director_encoder = LabelEncoder()
merged_df['Genre_encoded'] = genre_encoder.fit_transform(merged_df['genres'])
merged_df['Director_encoded'] = director_encoder.fit_transform(merged_df['Director'])

# Preparing inputs for training
genre_data = merged_df['Genre_encoded'].values.reshape(-1, 1)
director_data = merged_df['Director_encoded'].values.reshape(-1, 1)
ratings = merged_df['rating'].values

# Aligning user_features using an index that matches filtered merged_df
user_features = U[merged_df['userId'].values - 1, :]

# Splitting the dataset (80%-20%)
X_train, X_test, genre_train, genre_test, director_train, director_test, y_train, y_test = train_test_split(
    user_features, genre_data, director_data, ratings, test_size=0.2, random_state=42
)


# Defining input layers for user features and movie features (genre and director)
user_features_input = Input(shape=(k,), name='user_features')
genre_input = Input(shape=(1,), name='genre_input')
director_input = Input(shape=(1,), name='director_input')

# Embed genre and director categorical data into a continuous, dense vector space
genre_embedding = Embedding(input_dim=merged_df['Genre_encoded'].nunique(), output_dim=15)(genre_input)
director_embedding = Embedding(input_dim=merged_df['Director_encoded'].nunique(), output_dim=15)(director_input)

# Flatten the embeddings to collapse the extra dimension
genre_vector = Flatten()(genre_embedding)
director_vector = Flatten()(director_embedding)

# Concatenating the user features with the flattened genre and director vectors
concatenated_features = Concatenate()([user_features_input, genre_vector, director_vector])

# Assertion to ensure that all feature arrays have the same length before proceeding
assert len(user_features) == len(genre_data) == len(director_data) == len(ratings), "Data arrays must all be the same length"

# Signal the beginning of model training
print("Starting model training...")

# Define a neural network for predicting ratings, with dense layers and dropout for regularisation
x = Dense(256, activation='relu')(concatenated_features)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='linear')(x)  # Output layer with linear activation for regression

# Assembling the full model, specifying inputs and outputs
model = Model(inputs=[user_features_input, genre_input, director_input], outputs=output)

# Compiling the model with an optimiser, loss function, learning rate, and metrics to evaluate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error',
              metrics=['mean_absolute_error', RootMeanSquaredError(name='rmse')])

# Setup checkpoints and early stopping
checkpoint = ModelCheckpoint('ml-25m-cleaned/best_model_test.keras', monitor='val_loss', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Training the model
history = model.fit(
    [X_train, genre_train, director_train], y_train,
    validation_data=([X_test, genre_test, director_test], y_test),
    epochs=100,
    batch_size=64,
    callbacks=[checkpoint, early_stopping]
)

# Evaluating the model
test_loss = model.evaluate([X_test, genre_test, director_test], y_test)
predicted_ratings = model.predict([X_test, genre_test, director_test]).flatten()

# Calculate MAE and RMSE
mae = mean_absolute_error(y_test, predicted_ratings)
rmse = np.sqrt(mean_squared_error(y_test, predicted_ratings))
# print(f'\nTest Loss: {test_loss}\n')
print(f'{"Metric":<20}{"Value":<10}')
print(f'{"-"*30}')
print(f'Mean Absolute Error (MAE): {mae:.4f}\n')
print(f'Root Mean Square Error (RMSE): {rmse:.4f}')


# Plotting training and validation loss
plt.figure(figsize=(7, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plotting training and validation MAE
plt.figure(figsize=(7, 5))
plt.plot(history.history['mean_absolute_error'], label='Train MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Mean Absolute Error (MAE) Over Epochs')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plotting training and validation RMSE
plt.figure(figsize=(7, 5))
plt.plot(history.history['rmse'], label='Train RMSE')
plt.plot(history.history['val_rmse'], label='Validation RMSE')
plt.title('RMSE Over Epochs')
plt.ylabel('Root Mean Square Error')
plt.xlabel('Epoch')
plt.legend()
plt.show()


