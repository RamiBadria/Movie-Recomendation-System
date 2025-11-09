# End-to-End Movie Recommendation System

A robust, end-to-end movie recommendation system built using **Factorization Machines**. This project goes beyond simple collaborative filtering by implementing a full data pipeline with Airflow, serving predictions via an API, and containerizing the entire application with Docker.

## 📖 About

This system uses Factorization Machines (based on the paper [Rendle 2010](./Rendle2010FM.pdf)) to predict user preferences and recommend movies. Factorization Machines are particularly effective for recommendation tasks as they can model interactions between features even in highly sparse datasets.

The project is designed as a complete ML engineering pipeline, moving from generic notebooks to a production-ready system orchestrated by Airflow and served via a dedicated API.

## ✨ Key Features

* **Advanced Algorithm:** Utilizes Factorization Machines for high-quality recommendations.
* **End-to-End Pipeline:** Automated data ingestion and processing workflows using **Apache Airflow**.
* **Model Serving:** dedicated **API** for requesting real-time recommendations.
* **Containerized:** Fully dockerized environment ensuring consistency across development and production using `docker-compose`.
* **Reproducible Research:** Jupyter `notebooks` included for Exploratory Data Analysis (EDA) and model prototyping.

## 🛠️ Tech Stack

* **Language:** Python
* **Machine Learning:** Factorization Machines (generic implementation details based on repo)
* **Orchestration:** Apache Airflow
* **Containerization:** Docker & Docker Compose
* **API Framework:** Flask
* **Data Analysis:** Pandas, Jupyter Notebooks

## 📂 Project Structure

```bash
├── airflow/          # DAGs and configuration for Airflow pipelines
├── api/              # Backend code for serving the model
├── model/            # Scripts for training and managing the FM model
├── notebooks/        # Jupyter notebooks for EDA and prototyping
├── docker-compose.yml# Orchestration for all services
├── Rendle2010FM.pdf  # Reference paper for the algorithm
└── README.md         # Project documentation
