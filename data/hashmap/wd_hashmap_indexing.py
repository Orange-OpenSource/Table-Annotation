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
import lmdb
import json
import gzip
import pickle
import logging

wd_lookup_dump_url = os.getenv("WD_HASHMAP_URL", "")
index_name = "/data/edges"
os.makedirs(index_name, exist_ok=True)
dump_file = "/data/graph_dump.json.gz"

if not os.path.isfile(dump_file):
    os.system(f"wget {wd_lookup_dump_url} -O {dump_file}")

logging.basicConfig(
    filename='/data/indexing.log',
    format='%(asctime)s [%(module)s] %(levelname)s: %(message)s',
    level=logging.INFO
)
logging.info("BEGIN")

if os.listdir(index_name):
    logging.info("Index exists, Skpipping this step.")
else:
    edge_lmdb_writer = lmdb.open(index_name, map_size=248000000000)
    count_item = 0
    with edge_lmdb_writer.begin(write=True) as e_txn:
        with gzip.open(dump_file, "r") as f:
            for line in f:
                count_item += 1
                try:
                    json_line = json.loads(line[:-2])
                except:
                    json_line = json.loads(line)

                item_QID = list(json_line.keys())[0]
                item_infos = json_line[item_QID]
                # print(item_infos)
                new_item_infos = {}
                for pid, qid_list in item_infos.items(): 
                    if pid in ["labels", "descriptions", "aliases"]:
                        new_item_infos[pid] = qid_list.get("en-us", [])
                    else:
                        if "P1889" not in pid:
                            if "(-)" not in pid:
                                new_item_infos[pid] = {}
                                for qid, qtype in qid_list.items():
                                    if isinstance(qtype, str) and qtype.split("-")[0] == "DateTime":
                                        new_qid = qid.replace("-00-00", "").replace("-01-01", "")
                                        new_item_infos[pid][new_qid] = qtype  
                                    else:
                                        new_item_infos[pid][qid] = qtype
                            else:
                                new_item_infos[pid] = qid_list
                e_txn.put(item_QID.encode("ascii"), pickle.dumps(new_item_infos))
                if (count_item%100000 == 0):
                    logging.info("... Processed " + str(count_item) + " wikidata items.")
                if count_item == 1000000:
                    break
    edge_lmdb_writer.close()
logging.info("Done")