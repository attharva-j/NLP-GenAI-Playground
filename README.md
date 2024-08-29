# NaturalLanguageProcessing
A code repository for the things I am trying out on the NLP front.

- Notebook 1: Sentiment Analysis of Movie Reviews Using Word2Vec and Random Forest - Aug 21, 2024 <br>
  - In this project, I implemented a sentiment analysis model to classify movie reviews as positive or negative using Word2Vec embeddings and a Random Forest classifier on the [IMDB Movie Reviews 50K dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). I transformed the raw text reviews into numerical vectors using Word2Vec, which captures the syntactic as well as semantic meaning of words based on their context. These vectors were then used to train a Random Forest classifier, a robust ensemble learning method, to predict the sentiment of the reviews.
  - The model achieved an accuracy of `83.44%`, demonstrating its effectiveness in distinguishing between positive and negative sentiments, which can be further improved by tuning the hyper-parameters for the Random Forest Classifier OR changing the Word2Vec pre-trained model OR training an customized Word2Vec model altogether from scratch. This project highlights the power of combining Word2Vec's language understanding with Random Forest's predictive capabilities, offering valuable insights into the sentiment hidden in text data.
  - A **limitation** of this approach:
    - Efforts taken to generate the Word2Vec representations seem to go in vain as in the end, I took the mean of vector representations of each word in each row / document to bring it in a consistent 300-element vector representation.
    - This kind of resulted in the loss of the weights, importances and other information regarding the sequence of words which were captured in the vec representations of these words.
    - I had to do it because the models being trained are not adaptable to the variable length of different words.
  - How can this Word2Vec (which is actually a pretty good way to represent words) be used effeciently?
    - RNNs
    - LSTMs
    - GRUs

