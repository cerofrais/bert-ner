from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances,cosine_similarity

model = SentenceTransformer('bert-base-nli-mean-tokens')
sentences = ['CHILDRENS CUTLERY RETROSPOT RED'.lower(),'CHILDRENS CUTLERY POLKADOT GREEN'.lower(), 'SET OF 4 ENGLISH ROSE PLACEMATS'.lower()]
sentence_embeddings = model.encode(sentences)

print(cosine_similarity(sentence_embeddings))

