import pickle
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder

# nltk.download("stopwords")
# nltk.download("punkt")
# nltk.download('punkt_tab')


def books_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Books.scv"""
    books_df = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']].copy()

    locs = books_df[books_df["Year-Of-Publication"].map(str).str.match("[^0-9]")].index
    books_df.loc[locs, 'Publisher'] = books_df.loc[locs, 'Year-Of-Publication']
    books_df.loc[locs, 'Year-Of-Publication'] = books_df.loc[locs, 'Book-Author']

    for idx in locs:
        title = books_df.at[idx, 'Book-Title']
        split_title = title.split(';')

        if len(split_title) > 1:
            books_df.at[idx, 'Book-Author'] = split_title[1].strip()[:-1]
            books_df.at[idx, 'Book-Title'] = split_title[0].strip()

    books_df['Year-Of-Publication'] = books_df['Year-Of-Publication'].map(int)
    books_df.loc[books_df['Year-Of-Publication'] > 2019, 'Year-Of-Publication'] = books_df['Year-Of-Publication'].median()

    books_df['Book-Title'] = books_df['Book-Title'].map(title_preprocessing)

    books_df['Book-Author'] = books_df['Book-Author'].fillna('Unknown')
    books_df['Publisher'] = books_df['Publisher'].fillna('Unknown')

    books_df = books_df.drop(books_df[books_df['Book-Title'] == ''].index, axis=0)
    books_df = books_df.drop(books_df[books_df['Year-Of-Publication'] == 0].index, axis=0)
    print('Finished preprocessing books;')
    return books_df


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv
    Целевой переменной в этой задаче будет средний рейтинг книги,
    поэтому в предобработку (помимо прочего) нужно включить:
    1. Замену оценки книги пользователем на среднюю оценку книги всеми пользователями.
    2. Расчет числа оценок для каждой книги (опционально)."""
    rating_df = df[df["Book-Rating"] != 0].copy()
    rating_df = rating_df[rating_df.groupby("ISBN")["User-ID"].transform("count") > 1]
    rating_df = rating_df[rating_df.groupby("User-ID")["Book-Rating"].transform("count") > 1]
    rating_df = rating_df.reset_index(drop=True).dropna()
    rating_prep = rating_df.groupby('ISBN')['Book-Rating'].mean().reset_index(name='Average-Rating')
    print('Finished preprocessing ratings;')
    return rating_prep

def title_preprocessing(text: str) -> str:
    """Функция для нормализации текстовых данных в стобце Book-Title:
    - токенизация
    - удаление стоп-слов
    - удаление пунктуации
    Опционально можно убрать шаги или добавить дополнительные.
    """
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    processed_text = ' '.join(tokens)

    return processed_text


def modeling(books: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """В этой функции нужно выполнить следующие шаги:
    1. Бинаризовать или представить в виде чисел категориальные столбцы (кроме названий)
    2. Разбить данные на тренировочную и обучающую выборки
    3. Векторизовать подвыборки и создать датафреймы из векторов (размер вектора названия в тестах – 1000)
    4. Сформировать итоговые X_train, X_test, y_train, y_test
    5. Обучить и протестировать SGDRegressor
    6. Подобрать гиперпараметры (при необходимости)
    7. Сохранить модель"""
    # Объединяем датасеты
    merged_df = pd.merge(left=books_prep, right=ratings_prep, how='left', on='ISBN').dropna()
    X = merged_df[['Book-Author', 'Publisher', 'Year-Of-Publication', 'Book-Title']]
    y = merged_df['Average-Rating']

    # Делим датасет на тренировочный и тестовый
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Кодируем категориальные столбцы
    target_encoder = TargetEncoder(cols=['Book-Author', 'Publisher'])
    X_train[['Book-Author', 'Publisher']] = target_encoder.fit_transform(X_train[['Book-Author', 'Publisher']],
                                                                         y_train)
    X_test[['Book-Author', 'Publisher']] = target_encoder.transform(X_test[['Book-Author', 'Publisher']])
    with open("target_encoder.pkl", "wb") as file:
        pickle.dump(target_encoder, file)

    # Векторизуем Book-Title
    vect_titles = TfidfVectorizer(stop_words='english', max_features=1000)
    X_train_vector = vect_titles.fit_transform(X_train["Book-Title"])
    X_test_vector = vect_titles.transform(X_test["Book-Title"])
    with open("tfidf_vectorizer.pkl", "wb") as file:
        pickle.dump(vect_titles, file)

    train_vec_df = pd.DataFrame.sparse.from_spmatrix(X_train_vector, index=X_train.index)
    test_vec_df = pd.DataFrame.sparse.from_spmatrix(X_test_vector, index=X_test.index)

    # Удаляем столбец с названиями книг
    X_train = X_train.drop(columns=['Book-Title'])
    X_test = X_test.drop(columns=['Book-Title'])

    # Объединяем обработанные данные
    X_train_final = pd.concat([X_train, train_vec_df], axis=1)
    X_test_final = pd.concat([X_test, test_vec_df], axis=1)

    X_train_final.columns = X_train_final.columns.astype(str)
    X_test_final.columns = X_test_final.columns.astype(str)

    # Нормализуем
    scaler = StandardScaler()
    cols = X_train_final.columns
    X_train_final[cols] = scaler.fit_transform(X_train_final[cols])
    X_test_final[cols] = scaler.transform(X_test_final[cols])
    with open("scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    # Обучаем модель
    linreg = SGDRegressor(random_state=42, learning_rate='adaptive', early_stopping=True)
    linreg.fit(X_train_final, y_train)

    # Оценка
    y_pred = linreg.predict(X_test_final)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"MAE: {mae}")

    with open("linreg.pkl", "wb") as file:
        pickle.dump(linreg, file)


if __name__ == '__main__':
    ratings = pd.read_csv("Ratings.csv", low_memory=False)
    books = pd.read_csv("Books.csv", low_memory=False)

    ratings_prep = ratings_preprocessing(ratings)
    books_prep = books_preprocessing(books)

    modeling(books_prep, ratings_prep)