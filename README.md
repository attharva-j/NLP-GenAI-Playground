# NaturalLanguageProcessing
A code repository for the things I am trying out on the NLP front.

- Notebook 1: Sentiment Analysis of Movie Reviews Using Word2Vec and Random Forest - Aug 21, 2024 <br>
  - In this project, I implemented a sentiment analysis model to classify movie reviews as positive or negative using Word2Vec embeddings and a Random Forest classifier on the [IMDB Movie Reviews 50K dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). I transformed the raw text reviews into numerical vectors using Word2Vec, which captures the syntactic as well as semantic meaning of words based on their context. These vectors were then used to train a Random Forest classifier, a robust ensemble learning method, to predict the sentiment of the reviews.
  - The model achieved an accuracy of `83.44%`, demonstrating its effectiveness in distinguishing between positive and negative sentiments, which can be further improved by tuning the hyper-parameters for the Random Forest Classifier OR changing the Word2Vec pre-trained model OR training an customized Word2Vec model altogether from scratch. This project highlights the power of combining Word2Vec's language understanding with Random Forest's predictive capabilities, offering valuable insights into the sentiment hidden in text data.

