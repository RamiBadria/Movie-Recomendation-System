import numpy as np
import pandas as pd
import pymysql
import os
from scipy.sparse import csr_matrix
import time

class Recommender:
    def __init__(self, model_path='/app/saved_model/fm_model.npz'):
        self.MODEL_PATH = model_path
        self.MODEL = None
        self.USER_MAP = None
        self.MOVIE_MAP = None
        # We will load movies dataframe once as it is relatively small
        self.MOVIES_DF = None
        
        # Database connection details (ensure these match your docker-compose)
        self.DB_CONFIG = {
            'host': os.environ.get('DB_HOST', 'db'),
            'user': os.environ.get('DB_USER', 'user'),
            'password': os.environ.get('DB_PASSWORD', 'password'),
            'database': os.environ.get('DB_NAME', 'movielens'),
            'cursorclass': pymysql.cursors.DictCursor
        }

        self.load_model()
        self.load_movies_metadata()

    def load_model(self):
        """Loads the trained model parameters and mappings."""
        if not os.path.exists(self.MODEL_PATH):
            print(f"Warning: Model file not found at {self.MODEL_PATH}. API will serve cold-start only.")
            self.MODEL = None
            return

        try:
            print(f"Loading model from {self.MODEL_PATH}...")
            start_time = time.time()
            model_data = np.load(self.MODEL_PATH, allow_pickle=True)
            
            self.MODEL = {
                'w0': model_data['w0'],
                'w': model_data['w'],
                'V': model_data['V'],
                'n_users': int(model_data['n_users']),
                'n_movies': int(model_data['n_movies']),
            }
            # Load mappings (saved as object arrays, need .item() to get dict back)
            self.USER_MAP = model_data['user_map'].item()
            self.MOVIE_MAP = model_data['movie_map'].item()
            
            # Create reverse mapping for fast movie ID lookup later
            self.MOVIE_ID_REVERSE_MAP = {v: k for k, v in self.MOVIE_MAP.items()}
            
            print(f"Model loaded successfully in {time.time() - start_time:.2f}s.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.MODEL = None

    def load_movies_metadata(self):
        """Loads movie metadata (ID, title) from DB once."""
        try:
            conn = pymysql.connect(**self.DB_CONFIG)
            with conn.cursor() as cursor:
                cursor.execute("SELECT movie_id, title, genres FROM movies")
                movies_data = cursor.fetchall()
            conn.close()
            self.MOVIES_DF = pd.DataFrame(movies_data)
            # Ensure movie_id is int for merging later
            self.MOVIES_DF['movie_id'] = self.MOVIES_DF['movie_id'].astype(int)
            print(f"Loaded metadata for {len(self.MOVIES_DF)} movies.")
        except Exception as e:
            print(f"Error loading movie metadata from DB: {e}")
            self.MOVIES_DF = pd.DataFrame(columns=['movie_id', 'title', 'genres'])

    def _get_user_seen_movies(self, user_id):
        """Fetches list of movie IDs already seen by the user from DB."""
        try:
            conn = pymysql.connect(**self.DB_CONFIG)
            with conn.cursor() as cursor:
                # Fetch only movie_ids for this specific user
                cursor.execute("SELECT movie_id FROM ratings WHERE user_id = %s", (user_id,))
                result = cursor.fetchall()
            conn.close()
            return {row['movie_id'] for row in result}
        except Exception as e:
            print(f"Error fetching seen movies for user {user_id}: {e}")
            return set()

    def _predict_batch(self, X):
        """
        Optimized batch prediction for sparse matrix X.
        Re-implements the FM predict logic using numpy vectorization.
        """
        # Linear term
        pred = np.full(X.shape[0], self.MODEL['w0']) + X.dot(self.MODEL['w'])
        
        # Interaction term
        k = self.MODEL['V'].shape[1]
        if k > 0:
            term1 = (X.dot(self.MODEL['V'])) ** 2
            term2 = X.dot(self.MODEL['V'] ** 2)
            interaction = 0.5 * np.sum(term1 - term2, axis=1)
            pred += interaction
        return pred

    def get_cold_start_recommendations(self, n=5):
        """Generates non-personalized recommendations (e.g., most popular movies)."""
        # In a real system, you might cache this popular list.
        # For simplicity, we'll do a quick DB query if possible, or fallback.
        try:
            conn = pymysql.connect(**self.DB_CONFIG)
            with conn.cursor() as cursor:
                query = """
                    SELECT m.movie_id, m.title, COUNT(r.rating) as rating_count
                    FROM movies m
                    LEFT JOIN ratings r ON m.movie_id = r.movie_id
                    GROUP BY m.movie_id, m.title
                    ORDER BY rating_count DESC
                    LIMIT %s
                """
                cursor.execute(query, (n,))
                recs = cursor.fetchall()
            conn.close()
            return [{'movie_id': r['movie_id'], 'title': r['title']} for r in recs], "cold_start_popular"
        except Exception as e:
            print(f"Error generating cold start recs: {e}")
            # Fallback if DB fails: return random movies from loaded metadata
            if not self.MOVIES_DF.empty:
                 return self.MOVIES_DF.sample(min(n, len(self.MOVIES_DF)))[['movie_id', 'title']].to_dict('records'), "cold_start_fallback"
            return [], "error"

    def get_recommendations(self, user_id, n=5):
        """
        Main recommendation function.
        """
        # 1. Check if we can do personalized recommendations
        if self.MODEL is None or self.USER_MAP is None or user_id not in self.USER_MAP:
            print(f"User {user_id} unknown or model not loaded. Returning cold-start recs.")
            return self.get_cold_start_recommendations(n)

        # 2. Identify candidate movies (unseen by user)
        seen_movie_ids = self._get_user_seen_movies(user_id)
        # Filter to only movies that exist in our model's training mapping
        candidate_movie_ids = [mid for mid in self.MOVIE_MAP.keys() if mid not in seen_movie_ids]
        
        if not candidate_movie_ids:
            print("User has seen everything! Returning popular.")
            return self.get_cold_start_recommendations(n)

        # 3. Prepare batch prediction input (Sparse Matrix)
        n_candidates = len(candidate_movie_ids)
        u_idx = self.USER_MAP[user_id]
        m_indices = [self.MOVIE_MAP[mid] for mid in candidate_movie_ids]
        
        # Build CSR matrix efficiently:
        # Each row has 2 active features: [user_idx, movie_idx + n_users]
        rows = np.repeat(np.arange(n_candidates), 2)
        cols = np.empty(n_candidates * 2, dtype=np.int32)
        cols[0::2] = u_idx                                       # Even indices are user
        cols[1::2] = np.array(m_indices) + self.MODEL['n_users'] # Odd indices are movies
        data = np.ones(n_candidates * 2, dtype=np.float32)
        
        X_pred = csr_matrix((data, (rows, cols)), shape=(n_candidates, len(self.MODEL['w'])))

        # 4. Predict scores
        scores = self._predict_batch(X_pred)

        # 5. Rank and format results
        # Combine movie IDs with their scores
        scored_candidates = list(zip(candidate_movie_ids, scores))
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N movie IDs
        top_ids = [mid for mid, score in scored_candidates[:n]]
        
        # Retrieve titles from our pre-loaded movies dataframe
        # set_index/reindex preserves the sorted order
        recs_df = self.MOVIES_DF.set_index('movie_id').reindex(top_ids).reset_index()
        
        return recs_df[['movie_id', 'title']].to_dict('records'), "personalized"