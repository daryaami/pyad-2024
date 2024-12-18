import pandas as pd
import pickle

from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv"""
    ratings = df[df["Book-Rating"] != 0]
    ratings = ratings[ratings.groupby("ISBN")["User-ID"].transform("count") > 1]
    ratings = ratings[ratings.groupby("User-ID")["Book-Rating"].transform("count") > 1]
    ratings = ratings.reset_index(drop=True).dropna()
    return ratings


def modeling(ratings: pd.DataFrame) -> None:
    """В этой функции нужно выполнить следующие шаги:
    1. Разбить данные на тренировочную и обучающую выборки
    2. Обучить и протестировать SVD
    3. Подобрать гиперпараметры (при необходимости)
    4. Сохранить модель"""
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(ratings, reader)

    svd = SVD(n_factors=500, random_state=42, reg_all=0.05, lr_all=0.01)

    trainset, testset = train_test_split(data, test_size=0.2)

    svd.fit(trainset)
    predictions = svd.test(testset)

    acc = accuracy.mae(predictions)
    print(acc)

    with open("svd.pkl", "wb") as file:
        pickle.dump(svd, file)

if __name__ == '__main__':
    ratings = pd.read_csv("Ratings.csv")

    ratings = ratings_preprocessing(ratings)

    modeling(ratings)
