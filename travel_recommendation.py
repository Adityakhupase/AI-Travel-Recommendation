import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'Destination': ['Goa', 'Bali', 'Manali', 'Jaipur', 'Andaman Islands'],
    'Description': [
        'Beaches, water sports, nightlife, seafood',
        'Tropical beaches, temples, adventure, surfing',
        'Mountains, snow, trekking, adventure',
        'Palaces, history, culture, food',
        'Beaches, scuba diving, coral reefs, relaxation'
    ]
}

df = pd.DataFrame(data)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Description'])

user_input = input("Enter your interests: ")
user_vector = tfidf.transform([user_input])

similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
top_indices = similarities.argsort()[-3:][::-1]

print("\nTop Recommended Destinations:")
for i in top_indices:
    print(df.iloc[i]['Destination'])
