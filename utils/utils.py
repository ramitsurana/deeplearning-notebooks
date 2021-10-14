import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import os
import datetime
import zipfile


class DataScienceUtils:

    @staticmethod
    def tensorboard_callback(log_dir_name):
        log_dir_path = os.path.join(log_dir_name, datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
        tensorboard_callback = TensorBoard(log_dir=log_dir_path, histogram_freq=1, write_graph=True, write_images=True,
                                           update_freq='epoch', profile_batch=2, embeddings_freq=1)
        return tensorboard_callback

    @staticmethod
    def unzip_all(zip_file_name):
        zip_ref = zipfile.ZipFile(zip_file_name, "r")
        zip_ref.extractall()
        zip_ref.close()

    @staticmethod
    def walk_through_dirs(folder_path):
        for dirpath, dirnames, filenames in os.walk(folder_path):
            print(f"There are {len(dirnames)} directory and {len(filenames)} files in {dirpath}.")

    @staticmethod
    def calculate_results(y_true, y_pred):
        model_accuracy = accuracy_score(y_true, y_pred) * 100
        model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        model_results = {"accuracy": model_accuracy,
                         "precision": model_precision,
                         "recall": model_recall,
                         "f1": model_f1}
        return model_results

    @staticmethod
    def resize_image(img_path, img_shape=224, channels=3):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=channels)
        img = tf.image.resize(img, size=[img_shape, img_shape])
        img = img/255
        # Resizing the image to include batch size as the first parameter
        img = tf.expand_dims(img, axis=0)
        return img

    @staticmethod
    def tf_idf_accuracy(train_sentences, train_labels, val_sentences, val_labels):
        # Create tokenization and modelling pipeline
        model_0 = Pipeline([
            ("tfidf", TfidfVectorizer()),  # convert words to numbers using tfidf
            ("clf", MultinomialNB())  # model the text
        ])
        # Fit the pipeline to the training data
        model_0.fit(train_sentences, train_labels)
        baseline_score = model_0.score(val_sentences, val_labels)
        print(f"Our baseline model achieves an accuracy of: {baseline_score * 100:.2f}%")
        return baseline_score
