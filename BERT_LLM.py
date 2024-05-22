from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import numpy as np



#This is the importing of the data from our cloud hosted API we built

request2 = requests.get('https://dementia-7yf6.onrender.com/all')

data = request2.json()

websites = [item['attributes'].get('WEBSITE') for item in data['features']]
addresses = [item['attributes'].get('FULL_ADDRESS') for item in data['features']]
descriptions = [item['attributes'].get('DESCRIPTION_OF_SERVICE') for item in data['features']]


#This portion created the LLM Model locally and sets up the framework for feeding in data
sentences = descriptions


model = SentenceTransformer('bert-base-nli-mean-tokens')


#User input for what the information from the database they want
user_input = input("Enter your query: ")

#Uses the LLM to encode strings into matrices and vectors such that a cosine similarity function may be applied to check which data point is most similar to the query
sentences[0] = user_input

sentence_embeddings = model.encode(sentences)


similarity_scores = cosine_similarity([sentence_embeddings[0]], sentence_embeddings[1:])

similarity_scores = np.array(similarity_scores)

#Finds the maximum similarity with the query and returns that datapoint
max_index = np.argmax(similarity_scores)


#Prints all the information about the datapoint closest to the query of the user
print(websites[max_index])
print(addresses[max_index])
print(descriptions[max_index+1])
