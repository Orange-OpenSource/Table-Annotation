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
from quantulum3 import parser as qt_unit_parser

def unit_parser(list_cell):
    ner_per_label = {}
    for label in list_cell:
        ner_per_label[label] = []
        unit_res = qt_unit_parser.parse(label)
        surface = ""
        entity = []
        for res in unit_res:
            surface += res.surface
            entity.append(res.unit.entity.name)
        """ v1.0.1 """
        ## if a label is covered enough by entities. we keep all possible entities detected on this label.
        if 1.4*len(surface) >= len(label.replace(" ", "")):
            for a_ner in entity:
                if a_ner not in ["unknown", "dimensionless"]:
                    if a_ner == "time":
                        a_ner = "DURATION"
                    elif a_ner == "length":
                        a_ner = "DISTANCE"
                    elif a_ner == "currency":
                        a_ner = "MONEY"
                    else:
                        a_ner = a_ner.upper()
                    if a_ner not in ner_per_label[label]:
                        ner_per_label[label].append(a_ner)
    return ner_per_label
