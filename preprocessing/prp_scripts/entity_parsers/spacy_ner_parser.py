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
import spacy

def is_concept(label: str):
    ## verify if a typing represents a named entity.
    concept_list =  ["EVENT", "FAC", "GPE", "LAW", "LOC", "NORP", "ORG", "PERSON", "PRODUCT", "WORK_OF_ART", "LANGUAGE", "MONEY", "PERCENT", "UNKNOWN"]
    for concept in concept_list:
        if concept in label:
            return True
    return False

spacy_model = {"trf": spacy.load("en_core_web_sm", disable=["parser", "textcat"])}

def spacy_parser(list_cell):
    ner_per_label = {}
    for doc in spacy_model["trf"].pipe(list_cell):
        label = str(doc)
        ner_per_label[label] = []
        covered_label = ''.join([t.text for t in doc.ents]) ## record which parts of input label are covered by an named entity.
        if 1.4*len(covered_label) >= len(label): ## if a label is covered enough by named entities. we have all possible entities detected on this label.
            concept_exist = False
            for a_ner in doc.ents:
                if is_concept(a_ner.label_):
                    concept_exist = True
                if a_ner.label_ not in ner_per_label[label]:
                    ner_per_label[label].append(a_ner.label_)
                if concept_exist:
                    for num_enity in ["CARDINAL", "ORDINAL", "DATE"]:
                        if num_enity in ner_per_label[label]:
                            ner_per_label[label].remove(num_enity)
    return ner_per_label


    



