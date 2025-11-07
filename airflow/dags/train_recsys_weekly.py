from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from docker.types import Mount
import pandas as pd
import os
import kagglehub

# Define the DAG start date and schedule
START_DATE = pendulum.datetime(2023, 1, 1, tz="UTC")
SCHEDULE_INTERVAL = "@weekly" # Weekly retraining is usually sufficient for this dataset size

def _load_1m_data_to_mysql():
    """
    Downloads MovieLens 1M data from Kaggle and loads it into the MySQL database.
    replaces the old 100k data loading function.
    """
    # 1. Download data using kagglehub
    print("Downloading MovieLens 1M dataset...")
    path = kagglehub.dataset_download("odedgolden/movielens-1m-dataset")
    print(f"Dataset downloaded to: {path}")

    # 2. Prepare Database Connection
    mysql_hook = MySqlHook(mysql_conn_id='movielens_mysql_conn')
    engine = mysql_hook.get_sqlalchemy_engine()
    
    # 3. Define new 1M Schema and drop old tables
    with engine.begin() as connection:
        print("Clearing existing data and dropping old tables...")
        # Disable FK checks temporarily to allow dropping tables in any order
        connection.execute("SET FOREIGN_KEY_CHECKS = 0;")
        connection.execute("DROP TABLE IF EXISTS ratings;")
        connection.execute("DROP TABLE IF EXISTS movies;")
        connection.execute("DROP TABLE IF EXISTS users;")
        connection.execute("SET FOREIGN_KEY_CHECKS = 1;")
        print("Old tables dropped.")
        
        print("Creating new schemas for MovieLens 1M...")
        # Users table adapted for 1M format (UserID::Gender::Age::Occupation::Zip-code)
        connection.execute("""
            CREATE TABLE users (
                user_id INT PRIMARY KEY,
                gender CHAR(1),
                age INT,
                occupation INT,
                zip_code VARCHAR(20)
            );
        """)
        # Movies table adapted for 1M format (MovieID::Title::Genres)
        connection.execute("""
            CREATE TABLE movies (
                movie_id INT PRIMARY KEY,
                title VARCHAR(255),
                genres VARCHAR(255)
            );
        """)
        # Ratings table (UserID::MovieID::Rating::Timestamp)
        connection.execute("""
            CREATE TABLE ratings (
                user_id INT,
                movie_id INT,
                rating INT,
                timestamp BIGINT,
                PRIMARY KEY (user_id, movie_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
            );
        """)
        print("New tables created successfully.")

    # 4. Load data from files into Pandas DataFrames
    print("Reading 1M .dat files...")
    # Define column names for 1M dataset
    ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    movies_cols = ['movie_id', 'title', 'genres']
    users_cols = ['user_id', 'gender', 'age', 'occupation', 'zip_code']

    # Read files using '::' separator and latin-1 encoding
    ratings_df = pd.read_csv(os.path.join(path, 'ratings.dat'), sep='::', header=None, names=ratings_cols, engine='python', encoding='latin-1')
    movies_df = pd.read_csv(os.path.join(path, 'movies.dat'), sep='::', header=None, names=movies_cols, engine='python', encoding='latin-1')
    users_df = pd.read_csv(os.path.join(path, 'users.dat'), sep='::', header=None, names=users_cols, engine='python', encoding='latin-1')

    # 5. Bulk insert into MySQL
    print("Inserting data into MySQL (this may take a minute for 1M ratings)...")
    # Use chunksize to avoid overwhelming the DB connection with huge insert queries
    users_df.to_sql('users', con=engine, if_exists='append', index=False, chunksize=1000)
    movies_df.to_sql('movies', con=engine, if_exists='append', index=False, chunksize=1000)
    ratings_df.to_sql('ratings', con=engine, if_exists='append', index=False, chunksize=10000)
    
    print("Data loading complete.")

# Define the DAG
with DAG(
    dag_id="movielens_retraining_pipeline",
    start_date=START_DATE,
    schedule=SCHEDULE_INTERVAL,
    catchup=False,
    tags=['movielens', '1m', 'training']
) as dag:

    # Task 1: Update Database with fresh 1M data
    update_database_task = PythonOperator(
        task_id="update_database_1m",
        python_callable=_load_1m_data_to_mysql,
    )

    # Task 2: Run the Training Container
    train_model_task = DockerOperator(
        task_id="train_model_1m",
        image="movie-recommender-trainer:latest", # Ensure this matches your built image name
        container_name="model_trainer_1m_task",
        api_version="auto",
        auto_remove=True,
        # Necessary if Airflow itself is running inside Docker to spawn sibling containers
        docker_url="unix://var/run/docker.sock",
        # Network to allow communication with the Database container
        network_mode="movie-recomendation-system_movielens_network",
        mount_tmp_dir=False,
        # Mount volume to persist the trained model for the API to use
        mounts=[
            Mount(
                # השתמש בנתיב אבסולוטי מלא לתיקייה במחשב שלך
                source="/home/rami/GitHub/Movie-Recomendation-System/shared_model",
                target="/app/saved_model",
                type="bind" # שנה מ-volume ל-bind
            )
        ],
        # Environment variables for the training script to connect to DB
        environment={
            'DB_HOST': 'db',
            'DB_USER': 'user',
            'DB_PASSWORD': 'password',
            'DB_NAME': 'movielens'
        },
        # Increased memory limit for 1M dataset training
        mem_limit="4g"
    )

    # Set task dependencies
    update_database_task >> train_model_task