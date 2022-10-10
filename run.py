import argparse
import pandas as pd
import numpy as np
import torch
import faiss
from transformers import BertTokenizer, BertModel

#extract vector
def extract_vec(i, text=None):
  if text is None:
        text = data.iloc[i]['question1']
  inputs = tokenizer(text, return_tensors="pt")
  outputs = model(**inputs)
  last_hidden_states = outputs[0] 
  mean_vec = last_hidden_states[:, 0, :]#.mean(dim=1)#.detach().numpy()
  return mean_vec, text

def make_query(text): 
  queries = []
  vec, text = extract_vec(0, text)
  query_embeddings = vec.detach().numpy()
  queries.append(text)
  return query_embeddings, queries

#make example case
def make_docvec(length):
  total_vec = []
  queries = []
  for i in range(length):
    vec, text = extract_vec(i)
    total_vec.append(vec)
    queries.append(text)
  #vec, text = extract_vec(0, "Should I buy tiaaa?")
  #queries.append(text)
  total_vec.append(vec)
  total_vec = torch.cat(total_vec, dim=0).detach().numpy()
  return total_vec, queries

def prepare_faiss(total_vec):
  d = 768
  index = faiss.IndexFlatL2(d)
  print(index.is_trained)
  index.add(np.stack(total_vec, axis=0))
  print(index.ntotal)
  return index

def search_query(index, query_embeddings, k=5):
  # we want to see 4 nearest neighbors
  D, I = index.search(np.stack(query_embeddings, axis=0), k)     # actual search
  print(I)        
  return D, I

def search_data(t, total_vec, k):
  query_embeddings, queries = make_query(t)
  index = prepare_faiss(total_vec)
  D, I = search_query(index, query_embeddings, k)
  return query_embeddings, queries, I, D, index


def run(data_dir, query, k, length):
    global data
    global tokenizer
    global model
    data = pd.read_csv(data_dir)
    data.head()
    #prepare model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    total_vec = None
    t = query

    if total_vec is None:
        total_vec, total_docs = make_docvec(length)
    query_embeddings, queries, I, D, index = search_data(t, total_vec, k)

    for query, query_embedding in zip(queries, query_embeddings):
        distances, indices = index.search(np.asarray(query_embedding).reshape(1,768),k)
        print("\n======================\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")
        for idx in range(0,5):
            print(total_docs[indices[0,idx]], "(Distance: %.4f)" % distances[0,idx])
    return I, D

if __name__=="__main__":
    # 인자값을 받을 수 있는 인스턴스 생성
    parser = argparse.ArgumentParser(description='Argparse Tutorial')

    # 입력받을 인자값 설정 (default 값 설정가능)
    parser.add_argument('--data_dir', type=str,   default="question.csv")
    parser.add_argument('--query', type=str)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--length', type=int, default=10)

    # args 에 위의 내용 저장
    args = parser.parse_args()
    I, D = run(args.data_dir, args.query, args.k, args.length)
    print("Index", I)
    print("Distance", D)