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
import time
import threading
from .es_lookup import LookupES
from . import settings

class LookupManager:
    def __init__(self):
        self.lookup = LookupES()
        self.lookup.connect()

    def search(self, labels, KG):
        """Search switch entry point according to mode"""
        start_time = time.time()
        index_name = KG
        result = None
        if isinstance(labels, str):
            labels = [labels]
        nb = len(labels)
        if nb > settings.PARALLEL_MIN and settings.PARALLEL_MODE:
            #Split list in 2 and execute lookup in half labels in different tasks
            half = int(nb/2)
            threads = []
            quarter = int(half/2)
            threads.append(LookupThread(self.lookup, index_name, labels[:quarter]))
            threads.append(LookupThread(self.lookup, index_name, labels[quarter:half]))
            threads.append(LookupThread(self.lookup, index_name, labels[half:half+quarter]))
            threads.append(LookupThread(self.lookup, index_name, labels[half+quarter:]))
            # Start all threads
            for t in threads:
                t.start()
            # Wait for all threads to complete
            for t in threads:
                t.join()
            result = [item for sub_list in threads for item in sub_list.get_result()]
        else:
            result = self.lookup.flat_search(index_name, labels)
        end_time = time.time()
        return {"executionTimeSec": round(end_time-start_time,2), "output": result}

class LookupThread(threading.Thread):
    def __init__(self, es_lookup, index_name, labels):
        threading.Thread.__init__(self)
        self.es_lookup = es_lookup
        self.index_name = index_name
        self.labels = labels
        self.result = []

    def run(self):
        for label in self.labels:
            self.result.append(self.es_lookup.flat_search_item(self.index_name, label))

    def get_result(self):
        return self.result

def entity_lookup(labels, KG):
    lkp_manager = LookupManager()
    return lkp_manager.search(labels, KG)

if __name__ == "__main__":
    print(entity_lookup(labels=["belgium"], KG="dagobah_lookup"))