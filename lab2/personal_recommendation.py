import pandas as pd
import pickle


def load_model(file_path: str):
    """Загружаем сохраненную модель SVD"""
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model


def get_user_with_most_zeros(df: pd.DataFrame) -> int:
    """Находим пользователя с наибольшим количеством нулевых оценок"""
    user_zeros = df[df['Book-Rating'] == 0].groupby('User-ID').size()
    user_with_most_zeros = user_zeros.idxmax()
    return user_with_most_zeros


def preprocess_dataset(df):
    with open("target_encoder.pkl", "rb") as file:
        target_encoder = pickle.load(file)

    with open("tfidf_vectorizer.pkl", "rb") as file:
        tfidf_vectorizer = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    # Применение к новым данным
    X_encoded = target_encoder.transform(df[['Book-Author', 'Publisher']])
    X_vectorized = tfidf_vectorizer.transform(df['Book-Title'])

    X_final = pd.concat([pd.DataFrame(X_encoded, index=df.index),
                         df[['Year-Of-Publication']],
                         pd.DataFrame(X_vectorized.toarray(), index=df.index)], axis=1)

    X_final.columns = X_final.columns.astype(str)
    X_final[X_final.columns] = scaler.transform(X_final[X_final.columns])

    return X_final


def make_svd_predictions(model, user_id, books_with_zeros):
    """Делаем предсказания SVD для книг с нулевым рейтингом"""
    predictions = []
    for book in books_with_zeros:
        pred = model.predict(user_id, book)
        predictions.append((book, pred.est))
    return predictions


def make_linreg_predictions(model, predictions, df):
    books = [book for book, _ in predictions]
    book_for_prediction = df[df['ISBN'].isin(books)].copy()

    data_prep = preprocess_dataset(book_for_prediction)

    predictions = model.predict(data_prep)

    book_for_prediction.loc[:, 'prediction'] = predictions
    return book_for_prediction


if __name__ == '__main__':
    ratings = pd.read_csv("Ratings.csv", low_memory=False)
    books = pd.read_csv("Books.csv", low_memory=False)

    svd = load_model('svd.pkl')

    user_id = get_user_with_most_zeros(ratings)
    books_with_zeros = ratings[(ratings['User-ID'] == user_id) & (ratings['Book-Rating'] == 0)]['ISBN'].tolist()

    svd_predictions = make_svd_predictions(svd, user_id, books_with_zeros)

    books_to_recommend = [pred for pred in svd_predictions if pred[1] >= 8]

    linreg_model = load_model('linreg.pkl')

    final_predictions = make_linreg_predictions(linreg_model, books_to_recommend, books)

    final_predictions = final_predictions.sort_values('prediction', ascending=False)

    for book in final_predictions.values:
        print(f'"{book[1]}" by {book[2]} ({book[3]})')


# RECOMMENDATIONS:

# "Matilda" by Roald Dahl (1996)
# "The Lion, the Witch and the Wardrobe (Full-Color Collector's Edition)" by C. S. Lewis (2000)
# "The Magician's Nephew (rack) (Narnia)" by C. S. Lewis (2002)
# "Harry Potter and the Chamber of Secrets (Book 2)" by J. K. Rowling (2000)
# "Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))" by J. K. Rowling (1999)
# "Unnatural Selections" by Gary Larson (1991)
# "Socks (Cleary Reissue)" by Beverly Cleary (1990)
# "For Better, For Worse, Forever" by LURLENE MCDANIEL (1997)
# "Milton's Christmas" by Hayde Ardalan (2000)
# "Charlotte's Web (Trophy Newbery)" by E. B. White (1974)
# "The Shining" by Stephen King (1997)
# "Goodnight Moon Board Book" by Margaret Wise Brown (1991)
# "Henry and the Clubhouse (Henry Huggins (Paperback))" by Beverly Cleary (1979)
# "The Bean Trees" by Barbara Kingsolver (1989)
# "Everything's Eventual : 14 Dark Tales" by Stephen King (2002)
# "Wizard and Glass (The Dark Tower, Book 4)" by Stephen King (1998)
# "Driving Force" by Dick Francis (1992)
# "Stuart Little" by E. B. White (1974)
# "The Killing Hour" by LISA GARDNER (2003)
# "A Wrinkle in Time" by Madeleine L'Engle (1976)
# "Little House on the Prairie" by Laura Ingalls Wilder (1953)
# "Sideways Stories from Wayside School (Wayside School)" by Louis Sachar (1985)
# "A Wrinkle In Time" by MADELEINE L'ENGLE (1998)
# "Man's Search For Meaning" by Viktor E. Frankl (1997)
# "A KNIGHT IN SHINING ARMOR PROMOTION" by Jude Deveraux (1996)
# "A Long Way from Chicago: A Novel in Stories" by Richard Peck (2000)
# "The Thorn Birds" by Colleen McCullough (1978)
# "Into Thin Air : A Personal Account of the Mount Everest Disaster" by JON KRAKAUER (1997)
# "Good Work, Amelia Bedelia" by Peggy Parish (1980)
# "Roll of Thunder, Hear My Cry" by Mildred D. Taylor (1991)
# "The RAGMAN'S SON" by Kirk Douglas (1989)
# "The Giver (21st Century Reference)" by LOIS LOWRY (1994)
# "There's a Wocket in My Pocket!" by DR SEUSS (1974)
# "Alexander And The Terrible, Horrible, No Good, Very Bad Day" by Judith Viorst (1987)
# "Samantha's Surprise: A Christmas Story (American Girls Collection (Paper))" by Maxine Rose Schur (1986)
# "Milk and Honey (Decker and Lazarus Series)" by Faye Kellerman (1998)
# "A Prayer for Owen Meany" by John Irving (1990)
# "The Power of Myth" by Joseph Campbell (1991)
# "Along Came a Spider (Alex Cross Novels)" by James Patterson (1993)
# "The Black Cauldron (Chronicles of Prydain (Paperback))" by LLOYD ALEXANDER (1985)
# "How to Murder Your Mother-In-Law" by DOROTHY CANNELL (1995)
# "Whispers In The Woods (Reader's Choice)" by Helen R Myers (2002)
# "The Color Purple" by Alice Walker (1985)
# "Lightning" by Dean R. Koontz (1996)
# "The Poet" by Michael Connelly (1997)
# "The General's Daughter" by Nelson DeMille (1993)
# "The Andalite's Gift (Animorphs : Megamorphs 1)" by K. A. Applegate (1997)
# "The Prince of Tides" by Pat Conroy (1987)
# "The Cider House Rules" by John Irving (1986)
# "Lonesome Dove" by Larry McMurtry (1988)
# "Divine Secrets of the Ya-Ya Sisterhood: A Novel" by Rebecca Wells (1997)
# "October Sky: A Memoir" by Homer Hickam (1999)
# "HERBAL HEALING FOR WOMEN" by Rosemary Gladstar (1993)
# "Who Am I? (My First Hello Reader!)" by Nancy Christensen (1993)
# "E Is for Evidence: A Kinsey Millhone Mystery (Kinsey Millhone Mysteries (Paperback))" by Sue Grafton (1989)
# "The Encounter (Animorphs , No 3)" by K. A. Applegate (1996)
# "Meet Samantha: An American Girl (American Girls Collection (Paper))" by Susan S. Adler (1986)
# "Until Proven Guilty" by J.A. Jance (1985)
# "Inferno (Mentor)" by Dante Alighieri (1993)
# "The Crystal Cave" by Mary Stewart (1989)
# "She Said Yes : The Unlikely Martyrdom of Cassie Bernall" by Misty Bernall (2000)
# "A Time to Kill" by John Grisham (1992)
# "Where Echoes Live" by Marcia Muller (1992)
# "Deadlock (V.I. Warshawski Novels (Paperback))" by SARA PARETSKY (1992)
# "Night" by Elie Wiesel (1982)
# "DEAD BY SUNSET : DEAD BY SUNSET" by Ann Rule (1996)
# "The HUNTER (FORBIDDEN GAME 1): THE HUNTER" by L.J. Smith (1994)
# "Girl with a Pearl Earring" by Tracy Chevalier (2001)
# "All Creatures Great and Small" by James Herriot (0)
# "Lake Wobegon days" by Garrison Keillor (1986)
# "Star wars: From the adventures of Luke Skywalker : a novel" by George Lucas (1976)
# "All I Really Need to Know" by ROBERT FULGHUM (1989)
# "Snow White and the Seven Dwarfs" by Little Golden Staff (1994)
# "Island of the Blue Dolphins (Laurel Leaf Books)" by Scott O'Dell (1978)
# "The Hunt for Red October" by Tom Clancy (1985)
# "Three Wishes" by Barbara Delinsky (2003)
# "Christmas Treats" by Lindsay Armstrong (1998)
# "A Greek God at the Ladies' Club (Avon Romance)" by Jenna McKnight (2003)
# "Night Shield (Intimate Moments, 1027)" by Nora Roberts (2000)
# "Flowers In The Attic (Dollanganger)" by V.C. Andrews (1990)
# "Winter Solstice" by Rosamunde Pilcher (2001)
# "Angels Everywhere: A Season of Angels/Touched by Angels (Avon Romance)" by Debbie MacOmber (2002)
# "The Cat Who Talked to Ghosts" by Lilian Jackson Braun (1996)
# "HOW TO BE A VAMPIRE: R L STINES GHOSTS OF FEAR STREET #13" by R.L. Stine (1996)