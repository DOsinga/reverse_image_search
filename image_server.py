#!/usr/bin/env python
import argparse
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import os
from flask import Flask, request, redirect, flash, jsonify

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
INFO_PATH = 'images_info.pickle'
NEIGHBORS = 'neighbors.pickle'

app = Flask(__name__)


nbrs = None
image_infos = None
bottleneck_tensor = None
jpeg_data_tensor = None


def patch_rec(rec):
  rec['wiki_url'] = 'https://en.wikipedia.org/wiki/' + rec['wiki_name'].replace(' ', '_')
  rec['image_url'] = rec['image_url'].replace('http://', 'https://')
  return rec

@app.route('/', methods=['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
    # check if the post request has the file part
    if 'file' not in request.files:
      flash('No file part')
      return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
      flash('No selected file')
      return redirect(request.url)

    if file:
      image_data = file.read()
      bottleneck_values = sess.run(bottleneck_tensor, {jpeg_data_tensor: image_data})
      bottleneck_values = np.squeeze(bottleneck_values)
      distances, indices = nbrs.kneighbors([bottleneck_values])
      res = [(patch_rec(image_infos[idx]), dist) for idx, dist in zip(indices[0], distances[0])]
      return jsonify(results=res)


  return '''
  <!doctype html>
  <title>Upload new File</title>
  <h1>Upload new File</h1>
  <form action="" method=post enctype=multipart/form-data>
    <p><input type=file name=file>
       <input type=submit value=Upload>
  </form>
  '''


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Index images using inception')
  parser.add_argument('--model_path', type=str, default='inception2015',
                      help='Where the unpacked model dump is')
  parser.add_argument('--image_match_model', type=str, default='image_match_model.pickle',
                      help='Where to save the resulting model')
  args = parser.parse_args()

  print('setting up tensor flow')
  sess = tf.Session()
  model_filename = os.path.join(args.model_path, 'classify_image_graph_def.pb')

  with gfile.FastGFile(model_filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    bottleneck_tensor, jpeg_data_tensor = (
        tf.import_graph_def(graph_def, name='', return_elements=[
            BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME]))

  print('loading image model')
  image_infos, image_vecs = pickle.load(open(INFO_PATH, 'rb'))
  nbrs = pickle.load(open(NEIGHBORS, 'rb'))

  app.run(port=5050, host='0.0.0.0')
