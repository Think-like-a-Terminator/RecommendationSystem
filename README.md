**🎯 TensorFlow Recommenders: Building a Powerful Recommendation System**

This project demonstrates the creation of a robust recommendation engine, capable of providing personalized suggestions by leveraging the power of TensorFlow Recommenders. 

<br/>

**🚀 What Are Recommendation Systems?**

Recommendation systems are at the core of many modern platforms, providing users with personalized content such as movies, songs, products, and more. These systems predict and suggest items a user may like based on various factors, using the following common approaches:

**User-Item Similarities:** Recommendations based on users with similar preferences. If users A and B have similar taste profiles, items that A likes are recommended to B.

**Item-Item Similarities:** Similarity between items themselves. If a user likes item X, the system will recommend item Y, which has a similar pattern of interactions from other users.

<br/>

**💡 Key Features**

State-of-the-art recommendation engine using TensorFlow Recommenders.

Utilizes nearest neighbor search and matrix factorization for generating accurate, personalized recommendations.

<br/>

**🔍 How Does It Work?**

The underlying magic of recommendation systems comes from mathematical models that analyze the relationships between users and items. The two primary techniques include Nearest Neighbor Search and Matrix Factorization.


**Nearest Neighbor Search in N-Dimensional Space**

Imagine that we represent every user and item as a point in a multi-dimensional space, where each dimension corresponds to a feature, such as age, genre preference, or price range. For instance:

User Features: Age, preferred genres, time spent watching/reading a type of content, etc.

Item Features: Category, average rating, tags, etc.

The goal is to find the items that are closest to a user in this space, which would imply those are the items the user is likely to prefer.

For example, if user preferences and item features are plotted in an n-dimensional array, the system calculates the nearest neighbors to the user. The distance between points reflects the similarity: smaller distances mean closer matches, translating into better recommendations. Essentially, the system searches for the closest items to the user's preference vector.

Matrix Factorization
Matrix Factorization is another cornerstone technique of modern recommendation engines. This method reduces the user-item interaction matrix into lower-dimensional representations of users and items.

User Matrix: Encodes user preferences.
Item Matrix: Encodes item characteristics.
By decomposing the large matrix of user-item interactions, matrix factorization identifies hidden factors that influence user behavior. These factors can represent abstract concepts, such as a preference for action movies or a tendency to buy products during a sale. When these low-dimensional representations are multiplied together, the result is a prediction of how much a given user might like a particular item.

<br/>

**TensorFlow Recommenders Implementation**

This project uses a hybrid model of TensorFlow Recommenders (TFRS) and custom rules to recommend webinar sessions to different users.  The recommendation system was created to provide recommended webinar sessions to users based on their attributes and the webinar's attributes. It allowed us to quickly experiment with different models, data, and techniques and expand further. This repository includes the initial code and some dummy data for training, and predictions. 4 attributes of users and 4 attributes of sessions were used initially. 

This project demonstrates:

User and Session Embeddings: Transform user and session data into embeddings that capture meaningful relationships.

Two-Tower Models: One tower for user embeddings and another for session embeddings, connected by a dot product to produce predictions/recommendations.

Nearest Neighbor Retrieval: Efficiently find the top sessions for a given user by retrieving the nearest neighbors in the embedding space.

<br/>

**🔧 Installation and Usage**

Clone the repo:
git clone ttps://github.com/Think-like-a-Terminator/RecommendationSystem.git

Install the dependencies:
pip install -r requirements.txt

Train the recommendation model:
run train_main function from model.py

Generate recommendations:
python main.py

<br/>

Feel free to explore the code, swap out the sample data with your own data, run the training, and provide recommendations!
