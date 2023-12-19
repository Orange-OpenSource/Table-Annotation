"""
* Software Name : TableAnnotation
* Author: Viet-Phi Huynh, Jixiong Liu, Yoan Chabot, Frédéric Deuzé and Raphael Troncy
* Software description: TableAnnotation (a.k.a DAGOBAH) is a semantic annotation tool for tables leveraging three steps: 1) Table Preprocessing: a set of comprehensive heuristic to clean the table (e.g. fix encoding error), determine table orientation, data types of columns. 2) Entity Lookup: retrieve a number of entity candidates for mentions in the table, using an elastic search-based entity lookup. 3) Annotation: disambiguate retrieved entity candidates, select the most relevant entity for each mention. This consists of three tasks, namely Cell-Entity Annotation, Column-Type Annotation, Column-Pair Annotation.
* Version: <1.0.0>
* SPDX-FileCopyrightText: Copyright (c) 2023 Orange
* SPDX-License-Identifier: GPL-3.0-or-later
* Licensed under the GNU-GPL v3 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.gnu.org/licenses/gpl-3.0.html#license-text
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""
import os
from elasticsearch import Elasticsearch, helpers
import json
import gzip
import logging
import time
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

es_host = os.getenv("ELASTICSEARCH_HOST", "localhost")
es_port = os.getenv("ELASTICSEARCH_PORT", 9200)
es_user = os.getenv("ELASTICSEARCH_USER", "")
es_pwd = os.getenv("ELASTICSEARCH_PWD", "")

wd_lookup_dump_url = os.getenv("WD_LOOKUP_DUMP_URL", "")
index_name = "dagobah_lookup"
dump_file = "/data/label_dump.json.gz"

if not os.path.isfile(dump_file):
    os.system(f"wget {wd_lookup_dump_url} -O {dump_file} ")

logging.basicConfig(
    filename='/data/indexing.log',
    format='%(asctime)s [%(module)s] %(levelname)s: %(message)s',
    level=logging.INFO
)

logging.info("BEGIN")

es=Elasticsearch([{'host': es_host,'port': es_port}], timeout=30, max_retries=10, retry_on_timeout=True, http_auth=(es_user, es_pwd))
if es.indices.exists(index=index_name):
    logging.info("Index exists, skipping this step.")
    logging.info("END")

else:
    es.indices.create(index=index_name, body={
    'settings' : {
            'index' : {
                'number_of_shards':3
            }
    }
    })

    try:
        connected = False
        while not connected:
            try:
                logging.info("Connecting to elasticsearch: %s:%s", es_host, es_port)
                res = es.cluster.health()
                logging.info("Connection successful to %s", res["cluster_name"])
                connected = True
            except Exception:
                logging.warning("Error during connection to elasticsearch, retry in 10 seconds")
                time.sleep(10)
        bulk = []
        count = 0
        for line in gzip.open(dump_file, "r"):
            count += 1
            line = line.strip()
            if line[-1] == 44:#comma
                line = line[:-1]
            item = json.loads(line)
            qid = item["ID"]
            page_rank = item["page_rank"]
            labels = item["labels"]
            main_aliases = item["main_aliases"]
            sub_aliases = item["sub_aliases"]
            for label in labels:
                data = {"entity": qid, "label": label, "length" : len(label), "origin": "LABEL", "PR": page_rank}
                bulk.append({"_index": index_name, "_source": data})
            for alias in main_aliases:
                if alias not in labels:
                    data = {"entity": qid, "label": alias, "length" : len(alias), "origin": "MAIN_ALIAS", "PR": page_rank}
                    bulk.append({"_index": index_name, "_source": data})
            for alias in sub_aliases:
                if alias not in labels and alias not in main_aliases:
                    data = {"entity": qid, "label": alias,  "length" : len(alias), "origin": "SUB_ALIAS","PR": page_rank}
                    bulk.append({"_index": index_name, "_source": data})

            if len(bulk) >= 1000000:
                res = helpers.bulk(es,bulk)
                logging.info(res)
                logging.info(count)
                bulk.clear()

        if len(bulk) > 0:
            res = helpers.bulk(es,bulk)

        logging.info(count)
    except Exception as e:
        logging.exception(e)

    logging.info("END")
