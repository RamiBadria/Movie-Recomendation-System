import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from fm_model import FactorizationMachine
import pymysql
import os
import time
from scipy.sparse import lil_matrix
from tqdm import tqdm

def create_sparse_feature_matrix(data):
    """
    Converts user-movie ratings into a sparse feature matrix efficiently.
    Handles potentially non-consecutive user/movie IDs by creating mappings.
    """
    n_samples = data.shape[0]
    
    # Create mappings to ensure continuous IDs from 0 to N-1
    user_ids = sorted(data['user_id'].unique())
    movie_ids = sorted(data['movie_id'].unique())
    user_map = {id_: i for i, id_ in enumerate(user_ids)}
    movie_map = {id_: i for i, id_ in enumerate(movie_ids)}
    
    n_users = len(user_map)
    n_movies = len(movie_map)
    n_features = n_users + n_movies
    
    print(f"Creating sparse matrix for {n_samples} samples with {n_features} features...")
    
    # Use lil_matrix for efficient incremental construction
    X = lil_matrix((n_samples, n_features), dtype=np.int32)
    y = data['rating'].values.astype(np.float32)

    # Fill the matrix
    # Using itertuples is much faster than iterrows for large DataFrames
    for i, row in enumerate(tqdm(data.itertuples(index=False), total=n_samples, desc="Building Matrix")):
        u_idx = user_map[row.user_id]
        m_idx = movie_map[row.movie_id]
        X[i, u_idx] = 1
        X[i, n_users + m_idx] = 1 # Offset movie features by number of users

    # Convert to CSR format for efficient arithmetic operations during training
    return X.tocsr(), y, user_map, movie_map, n_users, n_movies

def main():
    # Database connection details (ensure these match your docker-compose setup)
    db_host = os.environ.get('DB_HOST', 'db')
    db_user = os.environ.get('DB_USER', 'user')
    db_password = os.environ.get('DB_PASSWORD', 'password')
    db_name = os.environ.get('DB_NAME', 'movielens')

    connection = None
    # Wait for the database to be ready and populated
    max_retries = 30
    for i in range(max_retries):
        try:
            print(f"Attempt {i+1}/{max_retries}: Connecting to database...")
            connection = pymysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
            
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM ratings")
                count = cursor.fetchone()[0]
            
            if count > 0:
                print(f"Success! Found {count} ratings in the database.")
                break
            else:
                print("Connected, but table is empty. Waiting for data ingestion...")
                connection.close()
                connection = None
                time.sleep(10)

        except pymysql.err.OperationalError as e:
            print(f"Database not ready yet, waiting... Error: {e}")
            time.sleep(10)
    
    if not connection:
        print("Error: Could not connect to DB or find data after many retries.")
        return

    print("Fetching all ratings from database...")
    # For 1M rows, this might take a moment and require some RAM.
    # If it crashes, we might need to implement chunked loading.
    ratings = pd.read_sql("SELECT user_id, movie_id, rating FROM ratings", connection)
    connection.close()
    
    print("Data fetched. Starting sparse matrix conversion...")
    X, y, user_map, movie_map, n_users, n_movies = create_sparse_feature_matrix(ratings)

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Initializing Factorization Machine...")
    fm = FactorizationMachine(
        n_features=n_users + n_movies,
        k=10,
        learning_rate=0.005, # Slightly lower LR often better for larger datasets
        l2_reg=0.1,
        n_epochs=5 # Start with fewer epochs to test pipeline speed
    )
    
    fm.fit(X_train, y_train, X_test, y_test)

    # Save the trained model AND the mappings
    model_path = '/app/saved_model/fm_model.npz'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    print(f"Saving model and mappings to {model_path}...")
    # We must save user_map and movie_map so the serving API knows how to encode new requests
    np.savez(
        model_path,
        w0=fm.w0,
        w=fm.w,
        V=fm.V,
        n_users=n_users,
        n_movies=n_movies,
        user_map=user_map,
        movie_map=movie_map
    )
    print("Model saved successfully.")

if __name__ == "__main__":
    main()