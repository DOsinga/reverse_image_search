#!/usr/bin/env python
import argparse
import os

import mwparserfromhell
import numpy as np
import psycopg2
import psycopg2.extras
import requests
import urllib.parse
from hashlib import md5
from PIL import Image
from io import BytesIO
import json
import pickle
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError
from tensorflow.python.platform import gfile
import status

CLASSES_PATH = 'classes.json'
INFO_PATH = 'images_info.pickle'
IMAGE_PATH_EN = 'http://upload.wikimedia.org/wikipedia/en/%s/%s/%s'
IMAGE_PATH_COMMONS = 'http://upload.wikimedia.org/wikipedia/commons/%s/%s/%s'
IMAGE_MARKERS = ['Size of this preview: <a href="', '<div class="fullMedia"><a href="']
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'


def fetch_image_content(image_name):
  m = md5()
  m.update(image_name.encode('utf-8'))
  c = m.hexdigest()
  url = IMAGE_PATH_EN % (c[0], c[0:2], image_name)
  try:
    r = requests.get(url)
    if r.status_code == 404:
      url = IMAGE_PATH_COMMONS % (c[0], c[0:2], image_name)
      r = requests.get(url)
      if r.status_code != 404:
        return r.content, url
  except IOError:
    pass

  for prefix in 'http://en.wikipedia.org/wiki/', 'http://commons.wikimedia.org/wiki/', 'http://commons.wikimedia.org/wiki/File:':
    image_url = prefix + image_name
    request = requests.get(image_url)
    if request.status_code == 404:
      continue
    html = request.text
    for marker in IMAGE_MARKERS:
      p = html.find(marker)
      if p == -1:
        continue
      p += len(marker)
      p2 = html.find('"', p)
      url = html[p: p2]
      if url.startswith('//'):
        url = 'http:' + url
      r = requests.get(url)
      if r.status_code != 404:
        try:
          return r.content, url
        except IOError:
          continue
  return None


def fetch_image(image_name, image_cache):
  if not image_name:
    return None
  file_path = os.path.join(image_cache, urllib.parse.quote(image_name.encode('utf8')) + '.jpg')
  image_name = image_name.strip('[')
  image_name = image_name.replace(' ', '_')
  if image_name[0].upper() != image_name[0]:
    image_name = image_name.capitalize()
  image_name = urllib.parse.quote(image_name.encode('utf-8'))
  if image_name.startswith('%3C%21--_'):
    image_name = image_name[len('%3C%21--_'):]
  res = fetch_image_content(image_name)
  if res:
    try:
      image = Image.open(BytesIO(res[0]))
      image.save(open(file_path, 'wb'), format='jpeg')
    except OSError:
      return None
    return file_path, res[1]
  return None


def fetch_image_for_wikipedia_id(wikipedia_id, img_dir, status, postgress_cursor):
  sql = 'select wikitext from wikipedia where title = %s'
  postgress_cursor.execute(sql, (wikipedia_id,))
  rec = postgress_cursor.fetchone()
  if rec:
    wikicode = mwparserfromhell.parse(rec['wikitext'])
    for link in wikicode.filter_wikilinks():
      if link.title.lower().endswith('.jpg'):
        res = fetch_image(link.title, img_dir)
        if res:
          return res
  else:
    status.count('wp-not-found')

def main(postgres_cursor, img_dir, model_path):
  print('loading model')
  sess = tf.Session()
  model_filename = os.path.join(model_path, 'classify_image_graph_def.pb')

  with gfile.FastGFile(model_filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    bottleneck_tensor, jpeg_data_tensor = (
        tf.import_graph_def(graph_def, name='', return_elements=[
            BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME]))

  classes = json.load(open(CLASSES_PATH))

  info = []
  image_vecs = []

  stat = status.Status(qps_field='class')
  for wikipedia_id, description, image in classes:
    stat.count('class').report()
    res = fetch_image(image, img_dir)
    if not res:
      res = fetch_image_for_wikipedia_id(wikipedia_id, img_dir, stat, postgres_cursor)
      if not res:
        stat.count('no-image')
        continue
      else:
        stat.count('wikipedia')
    else:
      stat.count('direct')
    img_path, image_url = res
    image_data = gfile.FastGFile(img_path, 'rb').read()
    try:
      bottleneck_values = sess.run(bottleneck_tensor, {jpeg_data_tensor: image_data})
      bottleneck_values = np.squeeze(bottleneck_values)
    except InvalidArgumentError:
      stat.count('tf-error')
      continue
    image_vecs.append(bottleneck_values)
    info.append({'wiki_name': wikipedia_id,
                 'img_path': img_path,
                 'image_url': image_url,
                 'description': description})

  with open(INFO_PATH, 'wb') as fout:
    pickle.dump((info, np.array(image_vecs)), fout)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Fetch movies from a previously processed wikipedia dump out of postgres')
  parser.add_argument('--postgres', type=str,
                      help='postgres connection string')
  parser.add_argument('--img_dir', type=str,
                      default='images',
                      help='where the images go')
  parser.add_argument('--model_path', type=str, default='inception2015',
                      help='Where the unpacked model dump is')

  args = parser.parse_args()

  postgres_conn = psycopg2.connect(args.postgres)
  postgres_cursor = postgres_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

  if not os.path.isdir(args.img_dir):
    os.makedirs(args.img_dir)
  main(postgres_cursor, args.img_dir, args.model_path)
