#!/usr/bin/env python
import argparse
import psycopg2
import psycopg2.extras
import json

INFO_PATH = 'classes.json'


def main(postgres_cursor):
  print('running db query')
  sql = "select properties->>'instance of' as thing, count(*) as c from wikidata group by thing"
  postgres_cursor.execute(sql)
  class_ids = [rec['thing'] for rec in postgres_cursor]
  print('class ids found: %d' % len(class_ids))
  sql = "select wikipedia_id, description, properties->>'image' as image from wikidata where wikipedia_id in %s"
  postgres_cursor.execute(sql, (tuple(class_ids),))
  res = list(postgres_cursor)
  print('images found: %d' % len(res))
  with open(INFO_PATH, 'w') as fout:
    json.dump(res, fout, indent=2)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Fetch movies from a previously processed wikipedia dump out of postgres')
  parser.add_argument('--postgres', type=str,
                      help='postgres connection string')

  args = parser.parse_args()

  postgres_conn = psycopg2.connect(args.postgres)
  postgres_cursor = postgres_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

  main(postgres_cursor)

