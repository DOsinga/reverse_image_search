#!/usr/bin/env python
from sklearn.neighbors import NearestNeighbors
import pickle

INFO_PATH = 'images_info.pickle'
NEIGHBORS = 'neighbors.pickle'

def main():
  print('creating model(s)')
  info, image_vecs = pickle.load(open(INFO_PATH, 'rb'))
  nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(image_vecs)
  with open(NEIGHBORS, 'wb') as fout:
    pickle.dump(nbrs, fout)


if __name__ == '__main__':
  main()





