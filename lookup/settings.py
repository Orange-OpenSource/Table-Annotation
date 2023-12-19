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
# Elasticsearch server information
ELASTICSEARCH_HOST = os.getenv('ELASTICSEARCH_HOST', 'localhost').strip()
ELASTICSEARCH_PORT = os.getenv('ELASTICSEARCH_PORT', 9200)
ELASTICSEARCH_AUTH = os.getenv('ELASTICSEARCH_AUTH', False)
ELASTICSEARCH_USER = os.getenv('ELASTICSEARCH_USER', '').strip()
ELASTICSEARCH_PWD = os.getenv('ELASTICSEARCH_PWD', '').strip()

# Adaptative threshold ratio
ADAPTATIVE_RATIO_MIN_THRESHOLD = float(os.getenv('ADAPTATIVE_RATIO_MIN_THRESHOLD', 0.70))
ADAPTATIVE_RATIO_MAX_GAP = float(os.getenv('ADAPTATIVE_RATIO_MAX_GAP', 0.25))

# Parallel mode
PARALLEL_MODE = os.getenv('PARALLEL_MODE', True)
PARALLEL_MIN = os.getenv('PARALLEL_MIN', 5)

# Lookup score factors
MAIN_ALIAS_FACTOR = float(os.getenv('MAIN_ALIAS_FACTOR', 0.94))
SUB_ALIAS_FACTOR = float(os.getenv('SUB_ALIAS_FACTOR', 0.88))

## Page rank for wikidata
PAGE_RANK_FACTOR = float(os.getenv('PAGE_RANK_FACTOR', 0.1))

## Filter on label length
LABEL_LENGTH_MIN_FACTOR = float(os.getenv('LABEL_LENGTH_MIN_FACTOR', 0.25))
LABEL_LENGTH_MAX_FACTOR = float(os.getenv('LABEL_LENGTH_MAX_FACTOR', 4))
LABEL_TOKEN_DIFF = float(os.getenv('LABEL_TOKEN_DIFF_FACTOR', 4))

## BM25 (TDIDF) score factor.
BM25_FACTOR = float(os.getenv('BM25_FACTOR', 0.20))