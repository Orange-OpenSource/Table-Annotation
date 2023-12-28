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
import numpy as np
import time
import math
import operator
from string import punctuation
import unidecode

from .abstract_classes import Candidate_Entity, Column, Cell, Column_Pair, Relation, AbstractAnnotationModel, Edge
from .knowledge_bases import Wikidata_KB
from . import utils
from preprocessing import table_preprocessing
from lookup import entity_lookup

class Baseline_Model(AbstractAnnotationModel):
    """
    This API annotates semantically a table (Cell Entity Annotation CEA, Column Type Annotation CTA, Column Pair Annotation CPA) using baseline model.
    Reference baseline model: TODO
    """
    def __init__(self, table, target_kb={"kb_path":  "./data/hashmap", "lookup_index": "dagobah_lookup"},  
                        params={}):
        """
        Initializing the Model. This includes following steps:
        + Initialize the KB interface
        + Preprocessing the table
            + Clean table
            + Header detection
            + Find semantic columns, literal columns
        + Entity lookup for sementic columns
        + Initialize entity scores
        + Set entity_subgraph, literal_subgraph for each candidate lookup.

        If all those steps go correctly, the model init operation return True
        Args:
            table: 2D list
            target_kb: dictionary with {"kb_path": path of KB,
                                        "lookup_index": associated lookup index}
            preprocessing_backend: url of preprocessing service.
            lookup_backend: url of lookup service.
        """
        self.params = params
        self.target_kb = target_kb
        ## initialize the KB
        self.KB = Wikidata_KB(self.target_kb["kb_path"])

        ## If all init operation below go correctly, this flag to return True to indicate that further annotation can be performed
        ## Otherwise, no annotation returned to user.
        self.is_model_init_success = False
        ## for debugging
        self.lookup_time = 0.0
        self.preprocessing_time = 0.0
        self.avg_lookup_candidate = 0.0
        self.subgraph_construction_time = 0.0
        self.entity_scoring_time = 0.0
        self.cea_task_time = 0.0
        self.cta_task_time = 0.0
        self.cpa_task_time = 0.0
        self.abnormal_lookup_mentions = [] ## contains abnormal mentions in which the lookup works incorrectly.
  
        ## preprocessing the table.
        start_time = time.time()
        self.table_infos = self.preprocessing_task(table)["preprocessed"]
        self.preprocessing_time = round(time.time() - start_time, 2)
        ## it only makes sense to continue if preprocessing succeed.
        if self.table_infos:
            self.table = self.table_infos["tableDataRevised"]
            self.num_columns = len(self.table[0])
            self.num_rows = len(self.table)
            if self.table_infos["headerInfo"]["hasHeader"]:
                self.first_data_row = 1
            else:
                self.first_data_row = 0
            ## find semantic columns
            self.entity_cols = self._find_semantic_columns() 
            ## others are literal columns
            self.literal_cols = list(set(range(self.num_columns)) - set(self.entity_cols)) 
            ## categorize literal columns
            self.date_cols, self.numeral_cols, self.textual_cols, self.index_col = self._disambiguate_literal_columns()
            ## entity lookup on entity columns
            self.lookup = {}
            self.lookup_scores = {} ## store score of entity w.r.t target mention exported from lookup API.
            is_lkp_success = self.lookup_task()
            ## it only make sense to initialize the annotation if lookup succeed.
            if is_lkp_success:     
                self.multiHop_context = params["multiHop_context"]
                self.transitivePropertyOnly_path = params["transitivePropertyOnly_path"] # only consider transitive property: (a) s -> p1 -> p2 -> o or (b) s <- p1 <- p2 <- o
                self.soft_scoring = params["soft_scoring"]
                ## initialize the scores
                ##      literal context is more sensible to noise, due to difficulty in the detection, normalization, comparison of the type 
                ##          for e.g 5 kg != 5 m/s or 05/06/2021 can also be 06/05/2021 or even 05/06/2021 != 06/05/2021
                ##      so the weight of literal context in the calculation of context score is 0.15.
                self.semantic_context_weight = 1.0
                self.literal_context_weight = 0.3
                self.entity_context_scores = {} ## context score
                self.entity_sim_scores = {} ## literal similarity score.
                self.entity_scores = {} ## final score: f(context_score, literal_similarity_score, potential disambiguation_score)
                self._initialize_scores()
                
                ## set the entity_subgraph, literal_subgraph for each candidate lookup
                ## and save to G_memory
                self.G_memory = {}
                self.type_graph = {}
                ## remember popular entities (num_edges > 1000000) on which caching mechanism is applied (see self.unrelated_candidate_pairs)
                ## note: we do not apply caching mechanism on all of entity candidates in order to avoid memory explosion.
                self.popular_entities = set()
                self._set_subgraph()

                self.node_generality = {}

                ## cache relations between pairs of candidate
                ##    since graph intersection operation's performance depends on number of candidate lookups.
                ##    caching relations found between entity pairs avoid repeating intersection operation for same entity pairs.
                self.cached_cpa_candidates = {}
                self.unrelated_candidate_pairs = set() ## to store entity pairs that do not have any relation.

                ## to store unrelated-column pairs (avoid CPA calculation)
                self.unrelated_col_pairs = set() 

                ## cache hierarchical types of a candidate entity.
                self.cached_cta_candidates = {}

                ## store cells that do not have any valid contexts, for this kind of cell, put more weight on CTA disambiguation.
                self.contextless_cells = {}

                ## for a contextless cell, apart from CTA disambiguation of higher weight,
                ## candidates whose neighbor set contains a relation which is indeed a CPA in table should be more considerable 
                ## even the object of this relation is not found in the table. At least, the existence of this relation in the graph of 
                ## a candidate entity tells us that this candidate shares some similar features with other table rows.
                self.potential_candidates = {}

                ## Annotations
                ##   in cta taxonomy: ancestor type (second_level, third level) is less preferable (lower weight) than descendant type or direct type
                self.cta_taxonomy_weights = {"first_level": 1.0, "second_level": 0.7, "third_level": 0.2}
                self.cta_annot = {}
                self.cea_annot = {}
                self.cpa_annot = {}

                ## until here, all init operations success, set flag to True
                self.is_model_init_success = True
    
    def _find_semantic_columns(self):
        """ Find semantic columns in table: they are object columns which may refer to KG entities. """
        semantic_columns = []
        for column_idx in range(self.num_columns):
            object_typing_score = 0.0 ## quantify how much column represents an object.
            num_long_cellcontent = 0 ## count column cells represented by very long string.
            num_punctuated_cellcontent = 0
            ## a cell content that is too long or contains too much punctuations is assumed to be unlookupable 
            for row_idx in range(self.first_data_row, self.num_rows):
                if len(self.table[row_idx][column_idx]) > 150:
                    num_long_cellcontent += 1
                puncs = {}
                for p in punctuation:
                    s = self.table[row_idx][column_idx].count(p)
                    if s > 0:
                        puncs[p] = s
                if len(puncs) > 3:
                    num_punctuated_cellcontent += 1

            ## among column's typings, if a typing reprensents an object (e.g. PERSON, ORG, GPE...), 
            ##     then accumulate its corresponding score.
            for column_type in self.table_infos["primitiveTyping"][column_idx]["typing"]:
                if utils.named_entity_related_typing(column_type["typingLabel"]):
                    object_typing_score += column_type["typingScore"]

            ## if the score of object-like typing > 0.5 and the cell strings are not too long, then column is considered as semantic column.
            if object_typing_score > 0.5 and (num_long_cellcontent/(self.num_rows-self.first_data_row)) < 0.5  and (num_punctuated_cellcontent/(self.num_rows-self.first_data_row)) < 0.5:
                semantic_columns.append(column_idx)
        return semantic_columns
    
    def _disambiguate_literal_columns(self):
        """ 
        Among literal (non-object) columns, we categorize them into 3 classes: "date" columns, "numeral" columns and "textual" columns. 
            "date" column: contains date values 
            "numeral" column: contains number values (plain number or unit values)
            "textual" column: others.
        """
        date_columns = []
        numeral_columns = {"with_unit": [], "without_unit": []}
        textual_columns = []
        index_column = None
        for column_idx in self.literal_cols:
            ## the column is considered as DATE if DATE type's score > 0.5
            if utils.date_related_typing(self.table_infos["primitiveTyping"][column_idx]["typing"][0]["typingLabel"]):
                if self.table_infos["primitiveTyping"][column_idx]["typing"][0]["typingScore"] > 0.5:
                    date_columns.append(column_idx)
            else:
                if column_idx == 0 and self.table_infos["primitiveTyping"][column_idx]["typing"][0]["typingLabel"] == "ORDINAL":
                    index_column = 0
                else:
                    worth_tobe_unit_numeral_column_score = 0.0
                    worth_tobe_nonunit_numeral_column_score = 0.0
                    for column_type in self.table_infos["primitiveTyping"][column_idx]["typing"]:
                        if utils.numerical_typing_with_unit(column_type["typingLabel"]):
                            worth_tobe_unit_numeral_column_score += column_type["typingScore"]
                        elif utils.numerical_typing_without_unit(column_type["typingLabel"]):
                            worth_tobe_nonunit_numeral_column_score += column_type["typingScore"]
                    ## ## the column is considered as numeral if numeral related type's scores > 0.5
                    if worth_tobe_unit_numeral_column_score > 0.5:
                        numeral_columns["with_unit"].append(column_idx)
                    elif worth_tobe_nonunit_numeral_column_score > 0.5:
                        numeral_columns["without_unit"].append(column_idx)
                    else:
                        ## other cases are considered as textual columns.
                        textual_columns.append(column_idx)

        return date_columns, numeral_columns, textual_columns, index_column

    def preprocessing_task(self, table):
        """ 
        Preprocessing the table: detect header, orientation, column typing... 
        Return True is task is succeeded.    
        """
        return table_preprocessing(table)

    def lookup_task(self):
        """
        Entity Lookup task:
        + Get all valid mentions from entity columns in table
        + Do the lookup on those mentions
        Return True is task is succeeded.    
        """
        ## Get all mentions from entity columns for lookup.
        lookup_inputs = set()
        for column_idx in self.entity_cols:
            for row_idx in range(self.first_data_row, self.num_rows):
                if len(self.table[row_idx][column_idx].lower()) > 1:
                    lookup_inputs.add(self.table[row_idx][column_idx].lower())
        lookup_inputs = list(lookup_inputs)
        ## Lookup
        response = entity_lookup(labels=lookup_inputs, KG=self.target_kb["lookup_index"])
        lookup_results = {}
        for item in response["output"]:
            if "entities" in item:
                lookup_results[item["label"]] = item["entities"][:self.params["K"]]
            else: ## erreur in the output
                self.abnormal_lookup_mentions.append(item["label"])

        # wg_logger.info(f" - Lookup time: {time.time() - start:.2f} s")
        if lookup_results:
            self.lookup_time = response["executionTimeSec"]
            ## Lookup successfully. Distribute the candidates to each table mention.
            for column_idx in self.entity_cols:
                col_coverage = 0 ## to track %cells in this column has candidate entities.
                for row_idx in range(self.first_data_row, self.num_rows):
                    if self.table[row_idx][column_idx].lower() in lookup_results:
                        col_coverage += 1/(self.num_rows-self.first_data_row)
                        cell = Cell(row_index=row_idx, col_index=column_idx)
                        self.lookup[cell] = []
                        for a_candidate in lookup_results[self.table[row_idx][column_idx].lower()]:
                            self.lookup[cell].append(a_candidate["entity"])
                            candidate_entity = Candidate_Entity(row_index=row_idx, col_index=column_idx, id=a_candidate["entity"]) 
                            self.lookup_scores[candidate_entity] = a_candidate["score"]
                ## if > 70% cells of column does not have any candidate, 
                ##        the column is not considered as semantic column, but rather a literal (or textual) column.
                if col_coverage < 0.3:
                    self.entity_cols.remove(column_idx)
                    self.textual_cols.append(column_idx)
                    self.literal_cols.append(column_idx)
                    ## delete lookup in invalid column
                    for row_idx in range(self.first_data_row, self.num_rows):
                        cell = Cell(row_index=row_idx, col_index=column_idx)
                        self.lookup.pop(cell, None)
                        for a_candidate in lookup_results.get(self.table[row_idx][column_idx].lower(), []):
                            candidate_entity = Candidate_Entity(row_index=row_idx, col_index=column_idx, id=a_candidate["entity"])                         
                            self.lookup_scores.pop(candidate_entity)
            ## calculate average lookup candidates per mention.
            denom = 0
            for cell in self.lookup:
                denom += 1
                self.avg_lookup_candidate += len(self.lookup[cell])
            if denom > 0:
                self.avg_lookup_candidate = round(self.avg_lookup_candidate/denom, 2)

            return True
        else:
            return False

    def _initialize_scores(self):
        """ Intializa the scores for each candidate entity: context_score, similarity_score, final_score """
        for cell, entity_list in self.lookup.items():
            for entity_id in entity_list:
                candidate = Candidate_Entity(row_index=cell.row_index, col_index=cell.col_index, id=entity_id) 
                self.entity_context_scores[candidate] = {}
                self.entity_sim_scores[candidate] = 0.0
                self.entity_scores[candidate] = 0.0

    def _set_subgraph(self):
        """
        Set subgraphs {entity subgraph, literal subgraph, list_of_all_predicateds} for each candidate entity.
        A subgraph of an entity stores its neighbor nodes associated with the predicates found along the path in KG. 
        We only consider one-hop subgraph, meaning that neighbor nodes are directly connected to the target entity.
        If node is entity, it is stored in entity_subgraph, otherwise it is stored in literal_subgraph.
        Ex. 
            + entity_subgraph of "Q90 (Paris)" : {"Q2851133 (Anne Hildago)": [{"PID": "P6 (head of goverment)", "INFO": "entity"}...]...}
                neighbor node "Q2851133" relates to target "Q90" via path "P6" and it is an "entity".
            + literal_subgraph of "Q90 (Paris): {"75": [{"PID": "P395 (licence plate code)", "INFO": "String"}]}
                neighbor node "75" relates to target "Q90" via "P395" and it is a "String".
            Note: for neighbors nodes found in reverse direction, we add "(-)" to the predicate.
        """
        start_time = time.time()
        for cell, lookups in self.lookup.items():           
            for candidate_id in lookups:
                if candidate_id not in self.G_memory:
                    ## loading subgraph of candidate entity
                    subgraph = {"entity": {}, "literal": {}, "pids": set()}
                    ##  browsing 1-hop forward neighbors
                    neighbors = self.KB.get_subgraph_of_entity(candidate_id)
                    date_items = {}
                    for pid, objs in neighbors.items():
                        """ deprecated
                        if "::" in pid: ## pid contains qualifier
                            new_pid = new_pid.split("::")
                            ## remove the entity QID in pid.
                            ## e.g. ElonMusk -> educatedAt PensylvaniaUniv academicDegree -> BachelorOfScience 
                            ##           becomes ElonMusk -> educatedAt::academicDegree -> BachelorOfScience
                            new_pid = new_pid[0] + "::" + new_pid[2] 
                        """ 
                        subgraph["pids"].add(pid)
                        if pid[:3] == "(-)": ## backward property, subject is always entity.
                            edge = Edge(pid=pid, info="entity")
                            for obj in objs:
                                subgraph["entity"].setdefault(obj, []).append(edge)
                        else:
                            for obj, obj_type in objs.items(): 
                                if obj_type in ["NORMAL", "PREFERRED", "DEPRECATED"]: ## object is entity (expressed by its rank)
                                    subgraph["entity"].setdefault(obj, []).append(Edge(pid=pid, info="entity"))
                                else:
                                    subgraph["literal"].setdefault(obj, []).append(Edge(pid=pid, info=obj_type))
                    self.G_memory[candidate_id] = subgraph
        end_time = time.time()
        self.subgraph_construction_time = round(end_time-start_time, 2)

    def update_context_weight(self, onlyLiteralContext=False):
        """ Update the weight of each context in table according to the CPA of associated column.
        At the end of annotation pipeline, *onlyLiteralContext* is set to True which focuses on updating
        the weight of literal context only. Specifically, at this step, a literal column is connected to at most
        1 entity column, it is not likely that a literal column has more than 1 connections 
        (note: this is not always true for entity column)
        """
        # if self.cpa_annot:
        if not onlyLiteralContext:
            for candidate in self.entity_context_scores:
                for col_idx, a_context in self.entity_context_scores[candidate].items():
                    if col_idx < candidate.col_index and col_idx in self.entity_cols:
                        col_pair = Column_Pair(head_col_index=col_idx, tail_col_index=candidate.col_index)
                    else:
                        col_pair = Column_Pair(head_col_index=candidate.col_index, tail_col_index=col_idx)    
                    if col_pair in self.cpa_annot:
                        cnt_col = self.cpa_annot[col_pair][0]["coverage"]
                        df_col = (1+4*min(abs(col_idx - min(self.entity_cols)), abs(candidate.col_index - min(self.entity_cols))))**-1
                        tau_col = self.cpa_annot[col_pair][0]["semantic_proximity"]
                        if col_idx in self.entity_cols:
                            a_context["weight"] = max(0.05, self.semantic_context_weight*cnt_col*tau_col*df_col)
                        else:
                            a_context["weight"] = max(0.01, self.literal_context_weight*cnt_col*tau_col*df_col)
                    else:
                        if col_idx in self.entity_cols:
                            a_context["weight"] = 0.05
                        else:
                            a_context["weight"] = 0.01
        else:
            ## find entity column that best matches with a literal column.
            for literal_col in self.literal_cols:
                match_score = 0
                match_column = None
                for entity_col in self.entity_cols:
                    self.unrelated_col_pairs.add(Column_Pair(head_col_index=entity_col, tail_col_index=literal_col)) ## related col pair will be discared later.
                    col_pair = Column_Pair(head_col_index=entity_col, tail_col_index=literal_col)
                    if col_pair in self.cpa_annot:
                        cnt_col = self.cpa_annot[col_pair][0]["coverage"]
                        if cnt_col > match_score:
                            match_score = cnt_col
                            match_column = entity_col
                if match_column is not None:
                    ## discard related column pair
                    self.unrelated_col_pairs.remove(Column_Pair(head_col_index=match_column, tail_col_index=literal_col))
                        
    def _context_scoring(self):
        """
        Calculate context scores for all candidate entities of all semantic cells.
        Context score is composed of component scores calculated at any columns other than target column.
        + if context is semantic entity, use "_pairwise_semantic_context_scoring" for the calculation
            the score is weighted by self.semantic_context_weight 
        + if context is literal, use "_literal_context_scoring" for the calculation 
            the score is weighted by self.literal_context_weight, normally, this value is small (0.15) as we do not trust much in literal context 
                since it maybe usually noisy and we lack an effective strategie for detecting, normalizing type, comparing literal value.
        """
        ## browsing all table cells to calculate context scores.
        for row_idx in range(self.first_data_row, self.num_rows):   
            # print(f"Entity Scoring step: finished {row_idx+self.first_data_row}/{self.num_rows} table rows.")
            ## semantic context calculation
            for i in range(len(self.entity_cols)-1):
                head_col = self.entity_cols[i]
                head_mention = self.table[row_idx][head_col]
                head_cell = Cell(row_index=row_idx, col_index=head_col)
                if not self.lookup.get(head_cell, []):
                    ## tail candidate has no context at head_col, just set its context score at this column to 0.1
                    for j in range(i+1, len(self.entity_cols)):
                        tail_col = self.entity_cols[j]
                        tail_mention = self.table[row_idx][tail_col]
                        tail_cell = Cell(row_index=row_idx, col_index=tail_col)
                        for tail_id in self.lookup.get(tail_cell, []):
                            tail_candidate = Candidate_Entity(row_index=row_idx, col_index=tail_col, id=tail_id) 
                            ## initialize the context score of tail_candidate at column "head_col".
                            ## "context" entry stores relations.
                            self.entity_context_scores[tail_candidate][head_col] = {"weight": self.semantic_context_weight, "score": 0.1, "context": []}  
                else:
                    for head_id in self.lookup[head_cell]:
                        head_candidate = Candidate_Entity(row_index=head_cell.row_index, col_index=head_cell.col_index, id=head_id) 
                        ## get entity_subgraph of head candidate entity.
                        G_head = {}      
                        if head_id in self.G_memory:
                            G_head = self.G_memory[head_id]["entity"]
                        for j in range(i+1, len(self.entity_cols)):
                            tail_col = self.entity_cols[j]
                            tail_mention = self.table[row_idx][tail_col]
                            tail_cell = Cell(row_index=row_idx, col_index=tail_col)
                            ## initialize the context score of head_candidate at column "tail_col".
                            ## "context" entry stores relations.
                            self.entity_context_scores[head_candidate][tail_col] = {"weight": self.semantic_context_weight, "score": 0.1, "context": []}
                            if tail_cell in self.lookup:
                                for tail_id in self.lookup[tail_cell]:
                                    tail_candidate = Candidate_Entity(row_index=tail_cell.row_index, col_index=tail_cell.col_index, id=tail_id) 
                                    if head_col not in self.entity_context_scores[tail_candidate]:
                                        ## initialize the context score of tail_candidate at column "head_col".
                                        ## "context" entry stores relations.
                                        self.entity_context_scores[tail_candidate][head_col] = {"weight": self.semantic_context_weight, "score": 0.1, "context": []}

                                    ## calculate context score of head candidate w.r.t tail column and score of tail candidate w.r.t. head column.
                                    # Since the context score is calculated based (head_candidate, tail_candidate) subgraph intersections, we can, at the same time, update the score for both 
                                    #         head candidate and tail_candidate.
                                    # Specifically, this functio calculates context score between:
                                    #     + entity candidate in head column and the semantic context cell in tail column.
                                    #     + entity candidate in tail column and the semantic context cell in head column.
                                    # We also cache the predicate paths linking head candidate to tail candidate. The label (ID) and the "semantic proximity" of predicate path is stored.
                                    # "Sementic Proximity" of predicate path measures its level of preference as CPA and is computed as function of aggregated popularity of entities along the path.
                                    # For e.g. 1-length predicate path is more preferable than 2-length path...
                                    #     Ref: Ciampaglia GL, Shiralkar P, Rocha LM, Bollen J, Menczer F, et al. (2015) Correction: Computational Fact Checking from Knowledge Networks. 
                                    if head_id != tail_id:
                                        ## cache predicate paths for CPA along with its semantic proximity
                                        semantic_proximities = {}
                                        # many predicate paths can be found between a candidate pair. 
                                        # The final semantic proximity is max of all semantic proximities of predicate paths existing beetween candidate pair.
                                        best_semantic_proximity = 0.0
                                        if (head_id, tail_id) in self.cached_cpa_candidates: 
                                            ## (head_candidate, tail_candidate) is cached
                                            cpa_candidates = self.cached_cpa_candidates[(head_id, tail_id)]
                                            for a_cpa_candidate in cpa_candidates:
                                                best_semantic_proximity = max(best_semantic_proximity, a_cpa_candidate.semantic_proximity)
                                                semantic_proximities[a_cpa_candidate.id] = a_cpa_candidate.semantic_proximity
                                        else:
                                            ## (head_candidate, tail_candidate) is not cached
                                            ## get entity_subgraph of tail candidate entity.
                                            G_tail = {}      
                                            if tail_id in self.G_memory:
                                                G_tail = self.G_memory[tail_id]["entity"]
                                            ##  find the intersection of subgraph of head and tail.
                                            ## do subgraph intersection to see whether head candidate entity and tail candidate entity are connected
                                            ##    if yes, tail mention is a potential context of head candidate entity and similarly, head mention is a context of tail candidate entity.
                                            ##      so, we can update their context score.    
                                            if tail_id in G_head:
                                                ## tail candidate found in subgraph of head candidate --> they are directly connected.
                                                ## we cache the connecting relation for CPA.
                                                ##   as direct relation, it gets highest semantic proximity 1.0.
                                                best_semantic_proximity = 1.0
                                                # if (head_candidate.id, tail_candidate.id) not in self.cached_cpa_candidates:
                                                # self.cached_cpa_candidates[(head_candidate.id, tail_candidate.id)] = []
                                                for prop in G_head[tail_id]:
                                                    semantic_proximities[prop.pid] = best_semantic_proximity
                                            elif self.multiHop_context:
                                                ## check whether head candidate and tail candidate are connected via intermediate nodes.
                                                ##  find the intersection of subgraph of head and tail.     
                                                G_intersect = G_head.keys() & G_tail.keys()           
                                                if G_intersect:
                                                    ## subgraphs of head and tail candidate are overlapping.
                                                    ## retrieve predicate paths linking head candidate to tail candidate.
                                                    ## we use "::" to seperate two predicates in the path.
                                                    ## browsing the subgraph intersection to find the predicate paths.
                                                    for node in G_intersect:
                                                        num_edges = self.KB.get_num_edges(node)
                                                        if num_edges:
                                                            node_popularity = 1/(2 + math.log10(2+num_edges))
                                                        else:
                                                            node_popularity = 0.0
                                                        if node_popularity > 0:                                
                                                            for prop_head in G_head[node]:
                                                                rel_head = prop_head.pid
                                                                for prop_tail in G_tail[node]:
                                                                    rel_tail = prop_tail.pid
                                                                    ## reverse the direction of rel_tail
                                                                    if rel_tail[:3] == "(-)": ## if rel_tail is a backward relation
                                                                        rel_tail = rel_tail.replace("(-)","") 
                                                                    else:
                                                                        rel_tail = "(-)" + rel_tail        
                                                                    ## check 
                                                                    if rel_head == rel_tail:
                                                                        if rel_head.replace("(-)", "") in self.KB.transitivePID: 
                                                                            a_cpa_candidate = rel_head
                                                                            semantic_proximity = 1.0 ## transitive path get highest weight
                                                                        else:
                                                                            a_cpa_candidate = rel_head + "::" + rel_tail
                                                                            semantic_proximity = node_popularity
                                                                    else:
                                                                        a_cpa_candidate = rel_head + "::" + rel_tail
                                                                        if (rel_head[:3] == "(-)" and rel_tail[:3] != "(-)") or (rel_head[:3] != "(-)" and rel_tail[:3] == "(-)"):
                                                                            semantic_proximity = node_popularity/1.75
                                                                        else:
                                                                            semantic_proximity = node_popularity
                                                                    best_semantic_proximity = max(best_semantic_proximity, semantic_proximity)
                                                                    semantic_proximities[a_cpa_candidate] = min(semantic_proximities.get(a_cpa_candidate, semantic_proximity), semantic_proximity)

                                        ## cache predicate paths for CPA.
                                        ## and update the final semantic proximity for (head_candidate, tail_candidate) pair since a candidate pair may have
                                        ##  a lot of cpa candidates or predicate paths. The final semantic proximity is max of all semantic proximities of predicate paths existing beetween candidate pair.
                                        # if (head_candidate, tail_candidate) pair is semantically close, or one is a potential context text of another,
                                        #    we update their context score.
                                        if best_semantic_proximity > 0.0:
                                            ## context score for head candidate
                                            if len(tail_mention) > 5:
                                                threshold = 0.7
                                            else:
                                                threshold = 0.9
                                            if self.entity_sim_scores[tail_candidate] >= threshold:
                                                head_score = max(0.1, best_semantic_proximity*self.entity_sim_scores[tail_candidate])
                                            else:
                                                head_score = 0.1
                                            self.entity_context_scores[head_candidate][tail_candidate.col_index]["score"] = max(self.entity_context_scores[head_candidate][tail_candidate.col_index]["score"], head_score)
                                            ## context score for tail candidate
                                            if len(head_mention) > 5:
                                                threshold = 0.7
                                            else:
                                                threshold = 0.9
                                            if self.entity_sim_scores[head_candidate] >= threshold:
                                                tail_score = max(0.1, best_semantic_proximity*self.entity_sim_scores[head_candidate])
                                            else:
                                                tail_score = 0.1
                                            self.entity_context_scores[tail_candidate][head_candidate.col_index]["score"] = max(self.entity_context_scores[tail_candidate][head_candidate.col_index]["score"], tail_score) 
                                            if head_score > 0.1 or tail_score > 0.1:
                                                ## two graphs are considered as reliably connected, cache predicate paths.
                                                if (head_id, tail_id) not in self.cached_cpa_candidates:
                                                    self.cached_cpa_candidates[(head_id, tail_id)] = []
                                                    for a_cpa_candidate, cpa_score in semantic_proximities.items():                            
                                                        self.cached_cpa_candidates[(head_id, tail_id)].append(Relation(id=a_cpa_candidate, semantic_proximity=cpa_score))   
                                                ## cache the contexts from which the score is calculated.
                                                for a_cpa_candidate, cpa_score in semantic_proximities.items():                            
                                                    self.entity_context_scores[head_candidate][tail_col]["context"].append(a_cpa_candidate)
                                                    self.entity_context_scores[tail_candidate][head_col]["context"].append(a_cpa_candidate)

            ## literal context calculation
            for entity_col in self.entity_cols:
                entity_cell = Cell(row_index=row_idx, col_index=entity_col)
                if entity_cell in self.lookup:
                    for entity_id in self.lookup[entity_cell]:
                        ## get the literal subgraph of the entity.
                        G_literal_entity = {}      
                        if entity_id in self.G_memory:
                            G_literal_entity = self.G_memory[entity_id]["literal"]
                        entity_candidate = Candidate_Entity(row_index=row_idx, col_index=entity_col, id=entity_id)               
                        for literal_col in self.literal_cols:
                            if literal_col < entity_col: ## literal column should stay after entity column
                                continue
                            ## initialize the context score of head_candidate at literal column "tail_col".
                            ## "context" entry stores relations.
                            self.entity_context_scores[entity_candidate][literal_col] = {"weight": self.literal_context_weight, "score": 0.1, "context": []}
                            literal_mention = self.table[row_idx][literal_col]
                            ## calculate context score of head candidate w.r.t this literal column
                            # Calculating context score between the literal subgraph of the entity and the literal context cell.
                            # we consider: + date time comparison
                            #             + quantity comparison (number)
                            #             + string comparison  
                            ## browsing the subgraph and compare each literal node with the literal context cell.
                            for obj, props in G_literal_entity.items():
                                for prop in props:
                                    ## if a literal node is considered to be valid (is_possible_valid_rel is True), we cache its associated property for the CPA task.
                                    ##  in order to avoid refinding the property from KB.
                                    # is_possible_valid_rel = False
                                    matching_score = 0.0
                                    # if "::" not in prop.pid:
                                    if True:
                                        if prop.info.split("-")[0] == "DateTime" and literal_col in self.date_cols:
                                            if prop.info.split("-")[1] != "Period":
                                                ## compare two date values.
                                                if utils.date_similarity(obj, literal_mention, operator.eq):
                                                    matching_score = 1.0
                                                else: ## approximately compare two date values by its years (ignoring day, month, others), this case gets lower matching score.
                                                    year_obj = utils.get_year_from_date(obj)
                                                    year_cell = utils.get_year_from_date(literal_mention)
                                                    if utils.date_similarity(year_obj, year_cell, operator.eq):
                                                        matching_score = 0.8                        
                                            else:
                                                ## compare two period of time
                                                obj_start_date, obj_end_date = obj.split(":")
                                                new_literal_date = literal_mention.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
                                                new_literal_date = unidecode.unidecode(new_literal_date).split("-") ## use unidecode to normalize variants of dash splitter to unicode 
                                                if len(new_literal_date) == 2:
                                                    if utils.date_similarity(obj_start_date, new_literal_date[0], operator.eq) and \
                                                                        utils.date_similarity(obj_end_date, new_literal_date[1], operator.eq):
                                                        matching_score = 1.0
                                            if matching_score:
                                                self.entity_context_scores[entity_candidate][literal_col]["score"] = matching_score
                                                self.entity_context_scores[entity_candidate][literal_col]["context"].append(prop.pid)

                                        elif prop.info.split("-")[0] == "String" and literal_col in self.textual_cols:
                                            ## compare two string values.
                                            sim = utils.textual_similarity(obj, literal_mention)
                                            if sim > 0.9: ## high threshold for the selection, since textual context is not very trustable.
                                                # is_possible_valid_rel = True
                                                matching_score = sim
                                                self.entity_context_scores[entity_candidate][literal_col]["score"] = max(self.entity_context_scores[entity_candidate][literal_col]["score"], sim)    
                                                self.entity_context_scores[entity_candidate][literal_col]["context"].append(prop.pid)
                                        ## compare two dimensionless numeral values
                                        elif prop.info.split("-")[0] == "Quantity": ## "1" indicates property in Wikidata has no unit. 
                                            prop_unit = prop.info.split("-")[1].replace("http://www.wikidata.org/entity/", "")
                                            if literal_col in self.numeral_cols["without_unit"]:
                                                pass
                                                ## compare two quantity values.
                                                # sim = utils.dimensionless_quantity_similarity(obj, literal_mention)
                                                # if sim >= 0.98: ## high threshold for the selection, since numeral context is not very trustable.
                                                #     matching_score = sim
                                                #     self.entity_context_scores[entity_candidate][literal_col]["score"] = max(self.entity_context_scores[entity_candidate][literal_col]["score"], sim) 
                                                #     self.entity_context_scores[entity_candidate][literal_col]["context"].append(prop.pid)
                                            else:
                                                if literal_col in self.numeral_cols["with_unit"] and prop_unit != "1" : ## != "1" indicates that property in Wikidata has unit. 
                                                    ## compare two quantity values
                                                    ## get the dimension of unit entity in KG.
                                                    unit_dim_of_obj = self.KB.get_symbol_of_unit_entity(prop_unit)
                                                    ## standardize measurement in obj to base unit
                                                    standardized_obj = utils.standardize_to_base_unit({"value": obj, "unit": unit_dim_of_obj})
                                                    ## parse and standardize the unit form of literal table cell.
                                                    standardized_literal_cell = utils.standardize_to_base_unit(literal_mention)
                                                    if standardized_obj:
                                                        base_unit_of_obj = list(standardized_obj.keys())[0]
                                                        if len(standardized_literal_cell) == 1:
                                                            ## only 1 base unit in literal cell
                                                            if base_unit_of_obj in standardized_literal_cell and len(standardized_literal_cell[base_unit_of_obj]) == 1:
                                                                ## if literal cell has same base unit as KG obj and has only one measurement.
                                                                ##  then compare two corresponding measurements.
                                                                sim = utils.dimensionless_quantity_similarity(standardized_obj[base_unit_of_obj][0], standardized_literal_cell[base_unit_of_obj][0])
                                                                ##  if the comparison involves currency, it should be more tolerant of dissimilarity since money quantities usually change over times.
                                                                if base_unit_of_obj == "dollar": ## to check if is is money quantities, look at its base unit (dollar)
                                                                    matching_threshold = 0.75
                                                                else:
                                                                    matching_threshold = 0.95
                                                                if sim > matching_threshold: ## high threshold for the selection, since numeral context is not very trustable.
                                                                    matching_score = sim
                                                                    self.entity_context_scores[entity_candidate][literal_col]["score"] = max(self.entity_context_scores[entity_candidate][literal_col]["score"], sim) 
                                                                    self.entity_context_scores[entity_candidate][literal_col]["context"].append(prop.pid)

                                        ## if the neighbor node is a valid, we cache the property pointing to it for CPA.
                                        ##     apart from ID, we also store the its "semantic_proximity" which plays as its level of preference as CPA.
                                        ##     in literal context, all relations have same preference.
                                        if matching_score:
                                            a_cpa_candidate = prop.pid
                                            if (entity_id, literal_mention) not in self.cached_cpa_candidates:
                                                self.cached_cpa_candidates[(entity_id, literal_mention)] = []
                                            if Relation(id=a_cpa_candidate, semantic_proximity=self.literal_context_weight) not in self.cached_cpa_candidates[(entity_id, literal_mention)]:
                                                self.cached_cpa_candidates[(entity_id, literal_mention)].append(Relation(id=a_cpa_candidate, semantic_proximity=1.0))
                            
    def _literal_similarity_scoring(self):
        """
        Calculate the literal similarity scores w.r.t target mention for all candidate entities of all semantic cells 
        Score computed from label should be slightly more important than the one calculated from aliases, we put weight 0.9 on score calculated from aliases.
        """
        if self.lookup_scores:
            ## if scores are already calculated in lookup API, no need to recalculate.
            for candidate in self.entity_sim_scores:
                self.entity_sim_scores[candidate] = self.lookup_scores[candidate]
            ## lookup_scores is not needed anymore, clear it to save memory
            self.lookup_scores = {}

        else:
            for candidate in self.entity_sim_scores:
                mention = self.table[candidate.row_index][candidate.col_index]
                candidate_labels_and_aliases = self.KB.get_label_of_entity(candidate.id)
                candidate_labels, candidate_aliases = candidate_labels_and_aliases["labels"], candidate_labels_and_aliases["aliases"]
                score_wrt_labels = max([utils.textual_similarity(mention, label) for label in candidate_labels], default=0.0)
                score_wrt_aliases = max([utils.textual_similarity(mention, aliase) for aliase in candidate_aliases], default=0.0)
                sim_score = max(score_wrt_labels, 0.9*score_wrt_aliases)
                self.entity_sim_scores[candidate] = sim_score

    def entity_scoring_task(self, first_step=True, last_step=False):
        """
        Calculate before-cta,cpa-disambiguation score for all candidate entities of all semantic cells.
        This score is a combination of context score and literal similarity score.
        if there is no context (table has only one column), context_score is set 0.25 by defaut.
        """
        if first_step:
            ## calculate component score: context score and literal similarity score
            start_time = time.time()
            self._literal_similarity_scoring()
            self._context_scoring()
            end_time = time.time()
            self.entity_scoring_time = round(end_time-start_time, 2)
        ## browsing all candidate.
        for candidate in self.entity_scores:
            cell = Cell(row_index=candidate.row_index, col_index=candidate.col_index)
            if self.num_columns > 1 and (self.entity_cols or self.literal_cols):
                ## table contexts exist.
                ## weighted aggregate all columnar component of context scores into an unique score
                ## semetic context's weight is higher than literal context's weight.
                context_score = 0.0
                context_weight = 0.0
                max_context_weight = 0.0
                if self.entity_context_scores[candidate]:
                    for col_idx, a_context in self.entity_context_scores[candidate].items():
                        if col_idx < candidate.col_index and col_idx in self.entity_cols:
                            col_pair = Column_Pair(head_col_index=col_idx, tail_col_index=candidate.col_index)
                        else:
                            col_pair = Column_Pair(head_col_index=candidate.col_index, tail_col_index=col_idx)  
                        if col_pair not in self.unrelated_col_pairs and col_pair in self.cpa_annot:
                            if first_step:
                                scale_factor = 1.0
                            else:
                                ## CPA disambiguation
                                ## context score updated by CPAs
                                scale_factor = 0.0
                                if col_pair in self.cpa_annot:
                                    for a_cpa in self.cpa_annot[col_pair]:
                                        if a_cpa["id"] in a_context["context"]:
                                            scale_factor = a_cpa["coverage"]*a_cpa["semantic_proximity"]
                                            break
                            scaled_score = max(0.1, scale_factor*a_context["score"])
                            context_score += a_context["weight"]*scaled_score
                            if last_step: ## only store contextless cells at the last stage of annotation to reduce the noise.
                                if cell not in self.contextless_cells:
                                    self.contextless_cells[cell] = scaled_score
                                else:
                                    self.contextless_cells[cell] = max(self.contextless_cells[cell], scaled_score)
                                for a_cpa in self.cpa_annot[col_pair]:
                                    is_candidate_contain_cpa = False
                                    if col_idx < candidate.col_index and col_idx in self.entity_cols:
                                        if "(-)" in a_cpa["id"]:
                                            if a_cpa["id"].replace("(-)", "") in self.G_memory[candidate.id]["pids"]:
                                                is_candidate_contain_cpa = True
                                        else:
                                            if "(-)"+a_cpa["id"] in self.G_memory[candidate.id]["pids"]:
                                                is_candidate_contain_cpa = True
                                    else:
                                        if a_cpa["id"] in self.G_memory[candidate.id]["pids"]:
                                            is_candidate_contain_cpa = True
                                    ## cache the potential candidate
                                    if is_candidate_contain_cpa:
                                        if candidate not in self.potential_candidates:
                                            self.potential_candidates[candidate] = [{"cpa_coeff": a_cpa["coverage"], "cpa_score": a_cpa["score"], "cpa_id": a_cpa["id"]}]
                                        else:
                                            self.potential_candidates[candidate].append({"cpa_coeff": a_cpa["coverage"], "cpa_score": a_cpa["score"], "cpa_id": a_cpa["id"]})
    
                            max_context_weight = max(max_context_weight, a_context["weight"])
                            if col_idx in self.entity_cols:
                                context_weight += self.semantic_context_weight
                            elif col_idx in self.literal_cols:
                                context_weight += self.literal_context_weight
                    if context_weight:
                        context_score = context_score/context_weight
                    else:
                        context_score = 0.01
                else:
                    context_score = 0.01
                    if last_step:
                        if cell not in self.contextless_cells:
                            self.contextless_cells[cell] = 0.1                        
                ## if table context is clear,
                if max_context_weight > 0.1:
                    if (self.num_rows-self.first_data_row) > 3:
                        ampli_factor = 2
                    else: # if is very short, put more importance on textual similarity
                        ampli_factor = 5.0
                    ## calculate final score for this candidate as a combination of context score and literal similarity score.
                    # self.entity_scores[candidate] = context_score * math.exp(ampli_factor*(self.entity_sim_scores[candidate]-1.0))
                    self.entity_scores[candidate] = context_score * 1/(1+math.exp(-(self.entity_sim_scores[candidate]**2.5/0.5-1.0)/0.2))
                else:
                    ## if table context has ambigous, only consider textual similarity
                    self.entity_scores[candidate] = 0.1*self.entity_sim_scores[candidate]
            else:
                ## if table has only 1 column or has neither entity columns nor literal column, it has no context score.
                self.entity_scores[candidate] = self.entity_sim_scores[candidate] 
                if last_step:
                    if cell not in self.contextless_cells:
                        self.contextless_cells[cell] = 0.1

    def cta_task(self, col_index, only_one=True):
        """
        This task identifies the representative types for target column "col_index".
        We employ multi-level hierarchy (higher level has lower weight, weights are specifed at "self.cta_taxonomy_weights") and carefully select the most suitable type according to its role 
        in the tree (ancestor, descendant) and its coverage in the table.
        Args:
            col_index: index of target column.
            only_one: in case that there are many CTAs of same score, return only one or return all.
            multi_properties: by defaut, "instance_of", "sub_class_of" are properties used to get the types of an entity.
                if multi_properties is True, we consider additional properties specified in env variable "WIKIDATA_TYPE_PROPERTIES", e.g. "occupation", "position_held".
        Return:
            column types (CTA): ID, score, coverage.
        """
        start_time = time.time()
        candidate_types = {} 
        ## browsing all candidates in target column.
        for row_index in range(self.first_data_row, self.num_rows):
            cell = Cell(row_index=row_index, col_index=col_index)
            if cell in self.cea_annot:
                types_in_current_row = {}
                for cea in self.cea_annot[cell]:
                    candidate_id, candidate_score = cea["id"], cea["score"]                  
                    ## update candidate types
                    ## we cache the types of seen candidates, in order to avoid reloading its types from KB.
                    if candidate_id not in self.cached_cta_candidates:   
                        ## get hierachical types of this candidate   
                        hierachical_types = self.KB.get_types_of_entity(entity_id=candidate_id, num_level=3)
                        ## cache candidate types
                        self.cached_cta_candidates[candidate_id] = hierachical_types  
                    else:
                        ## get candidate types from cache rather reloading from KB.
                        hierachical_types = self.cached_cta_candidates[candidate_id]
                    
                    ## browsing candidate types and update their score, their rank.
                    ## type at lower level has higher weight.
                    for t1, rank in hierachical_types["level_1"].items():
                        if t1 in types_in_current_row:
                            types_in_current_row[t1] = {"score": max(types_in_current_row[t1].get("score", 0.0), self.cta_taxonomy_weights["first_level"]*candidate_score),
                                                "rank": max(types_in_current_row[t1].get("rank", 0), self.KB.map_rank(rank[0]))}
                        else:
                            types_in_current_row[t1] = {"rank": self.KB.map_rank(rank[0]), "score": self.cta_taxonomy_weights["first_level"]*candidate_score}                          
                    for t2, rank in hierachical_types["level_2"].items():
                        if t2 in types_in_current_row:
                            types_in_current_row[t2] = {"score": max(types_in_current_row[t2].get("score", 0.0), self.cta_taxonomy_weights["second_level"]*candidate_score),
                                                "rank": max(types_in_current_row[t1].get("rank", 0), self.KB.map_rank(rank[0]))}
                        else:
                            types_in_current_row[t2] = {"rank": self.KB.map_rank(rank[0]), "score": self.cta_taxonomy_weights["second_level"]*candidate_score}
                    for t3, rank in hierachical_types["level_3"].items():
                        if t3 in types_in_current_row:
                            types_in_current_row[t3] = {"score": max(types_in_current_row[t3].get("score", 0.0), self.cta_taxonomy_weights["third_level"]*candidate_score),
                                                "rank": max(types_in_current_row[t3].get("rank", 0), self.KB.map_rank(rank[0]))}
                        else:
                            types_in_current_row[t3] = {"rank": self.KB.map_rank(rank[0]), "score": self.cta_taxonomy_weights["third_level"]*candidate_score}

                ## update final candidate_types for target column by candidate type just found from current row_index.                                                                                                  ""
                for t in types_in_current_row:
                    if t in candidate_types:
                        candidate_types[t]["count"] += 1
                        candidate_types[t]["total_scores"] += types_in_current_row[t]["score"]
                        candidate_types[t]["total_ranks"] += types_in_current_row[t]["rank"]                           
                    else:
                        candidate_types[t] = {"count": 1, "total_scores": types_in_current_row[t]["score"],
                                                 "total_ranks": types_in_current_row[t]["rank"]}
        ## After scanning all rows, we have a final list of candidate types.
        if candidate_types:
            ## Sort the list of candidate types and find the best types
            ## Order: total_scores*count, total_rank (if types have same score, higher rank type is more preferable)                  
            sorted_candidate_types = sorted(candidate_types.items(), 
                                                key=lambda it: (it[1]["count"]*it[1]["total_scores"], it[1]["total_ranks"]), reverse=True)
            ## retain some best candidate types of highest scores or highest coverage.
            threshold = sorted_candidate_types[0][1] 
            column = Column(col_index=col_index)
            self.cta_annot[column] = []
            if only_one:
                ## for final annotation, return the most relevant CTAs of same score. 
                supertypes = set() ## also take super types into account.
                for candidate_type in sorted_candidate_types:
                    if candidate_type[1]["count"]*candidate_type[1]["total_scores"] == threshold["count"]*threshold["total_scores"]:
                        ## reformat the cta annotation with "id", average "score", "coverage".
                        self.cta_annot[column].append({"id": candidate_type[0], "score": candidate_type[1]["total_scores"]/(self.num_rows-self.first_data_row),
                                        "coverage": candidate_type[1]["count"]/(self.num_rows-self.first_data_row)})
                        supertypes.update(list(self.KB.get_supertypes_of_type(candidate_type[0])))
                ## get the super types of relevant types.
                for candidate_type in sorted_candidate_types:
                    if candidate_type[0] in supertypes and candidate_type[0] not in [t["id"] for t in self.cta_annot[column]]:
                        self.cta_annot[column].append({"id": candidate_type[0], "score": candidate_type[1]["total_scores"]/(self.num_rows-self.first_data_row),
                                        "coverage": candidate_type[1]["count"]/(self.num_rows-self.first_data_row)})  
            else:
                ## for intermediate disambiguation steps (CEA disambiguation), return many CTAs of highest score or highest coverage which maybe useful for disambiguation.
                for candidate_type in sorted_candidate_types:
                    if candidate_type[1]["count"] >= threshold["count"]:
                        ## reformat the cta annotation with "id", average "score", "coverage".
                        self.cta_annot[column].append({"id": candidate_type[0], "score": candidate_type[1]["total_scores"]/(self.num_rows-self.first_data_row),
                                        "coverage": candidate_type[1]["count"]/(self.num_rows-self.first_data_row)})

            end_time = time.time()
            self.cta_task_time += round(end_time-start_time, 2)
            return self.cta_annot[column]
        else:
            ## no type returned
            end_time = time.time()
            self.cta_task_time += round(end_time-start_time, 2)
            return ""

    def cea_task(self, col_index, row_index, only_one=True):
        """
        This task annotates the tables cells with KG entities. 
        Scoring of a candidate entity is updated by the CPA and CTA (disambiguation step).
            --> final_score(cea) = [score(cea) + coeff_cpa(cpa_disambiguation_score) + coeff_cta(cta_disambiguation_score)]/(1+coeff_cta+coeff_cpa)
            - coeff_cpa is defined as the averaged coverage of CPAs over column pairs which contain the target cea.
            - cpa_disambiguation_score is the weighted score of CPAs attached to target cea.
            - coeff_cta is defined as the coverage of CTA of the column containing  target cea.
            - cta_disambiguation_score is score of the CTA of the column containing target cea.
            - Higher coeff is, more trustable the disambiguation is .
        The candidates with highest score are chosen as CEA output.
        Args:
            col_index, row_index: position of the cell in the table.
            only_one: in case that there are many CEAs of same score, return only one or return all.
        Return:
            cell annotation (CEA): ID, score, coverage.
        """
        def is_neighbor_of_entity(target_graph, query_entity_ids):
            """ Verify whether a query_entity or a list of query_entities is a neighbor of tartget_entity """
            for query_e in query_entity_ids:
                if query_e in target_graph:
                    return True
            return False
        start_time = time.time()
        cea_candidates = []
        cell = Cell(row_index=row_index, col_index=col_index)
        ## browsing all lookup candidates of current cell.
        if cell in self.lookup:
            ## get the before-cta,cpa-disambiguation score for each cea candidate
            for candidate_id in self.lookup[cell]:
                candidate = Candidate_Entity(row_index=row_index, col_index=col_index, id=candidate_id)
                if candidate in self.entity_scores:
                    cea_candidates.append({"id": candidate_id, 
                                        "score": self.entity_scores[candidate]})
            if cea_candidates:                         
                ## update cea candidate scores by the CTAs
                ## in other word, the score of CTA at the column containing current cea candidate will participate in the score of this cea.
                cta_disabg_applied = False ## flag indicates whether cta disambiguation was applied.
                if self.cta_annot:
                    ## store scores of CTA attached to current cea candidate.
                    cta_disambiguation_scores = {}
                    ## store the coverage of CTA attached to current cea, they will used to weight the cta score in the update of cea score.
                    cta_disambiguation_weights = []
                    ## browsing all CTA results.
                    for col, cta in self.cta_annot.items():
                        ## only consider CTA which is attached to target column.
                        if col.col_index == col_index:
                            cta_disabg_applied = True
                            for a_cta in cta:
                                cta_type = a_cta["id"]
                                if cta_type not in self.type_graph:
                                    self.type_graph[cta_type] = {}
                                    graph_tmp = self.KB.get_subgraph_of_entity(cta_type)
                                    graph_tmp.pop("(-)P31", None)
                                    for pid, objs in graph_tmp.items():
                                        if pid[:3] == "(-)":
                                            for obj in objs:
                                                self.type_graph[cta_type][obj] = ''
                                        else:
                                            for obj, obj_type in objs.items(): 
                                                if obj_type in ["NORMAL", "PREFERRED", "DEPRECATED"]: #    
                                                    self.type_graph[cta_type][obj] = ''
                                cta_score = a_cta["score"]
                                cta_coverage = a_cta["coverage"]
                                ## store the coverage (weight) of CTA at target column.
                                cta_disambiguation_weights.append(cta_coverage)
                                ## browsing cea candidates to find ones that have type CTA.
                                ##  the score of those cea candidates will be updated by current CTA score. 
                                for cea in cea_candidates:
                                    if cea["id"] not in cta_disambiguation_scores:
                                        cta_disambiguation_scores[cea["id"]] = 0.0
                                    hierachical_types = self.cached_cta_candidates[cea["id"]]
                                    if cta_type in hierachical_types["level_1"]:
                                        ## direct type of current cea is indeed CTA.
                                        ## then remember this CTA score for the score update of cea candidate.
                                        cta_disambiguation_scores[cea["id"]]= max(cta_disambiguation_scores[cea["id"]], self.cta_taxonomy_weights["first_level"]*cta_score)
                                    elif cta_type in hierachical_types["level_2"] or is_neighbor_of_entity(self.type_graph[cta_type], list(hierachical_types["level_1"])):
                                    # elif cta_type in hierachical_types["level_2"]:
                                        ## super type of current cea is indeed CTA.
                                        ## then remember this CTA score for the score update of cea candidate.
                                        cta_disambiguation_scores[cea["id"]]= max(cta_disambiguation_scores[cea["id"]], self.cta_taxonomy_weights["second_level"]*cta_score)
                                    elif cta_type in hierachical_types["level_3"] or is_neighbor_of_entity(self.type_graph[cta_type], list(hierachical_types["level_2"])):
                                    # elif cta_type in hierachical_types["level_3"]:
                                        ## super type of current cea is indeed CTA.
                                        ## then remember this CTA score for the score update of cea candidate.
                                        cta_disambiguation_scores[cea["id"]]= max(cta_disambiguation_scores[cea["id"]], self.cta_taxonomy_weights["third_level"]*cta_score)        
                if cta_disabg_applied:
                    ## cta_coeff in cea score update function is the coverage of the CTA at current column.
                    if self.soft_scoring:
                        if self.contextless_cells and self.contextless_cells.get(cell, 0.1) == 0.1: ## in case a cell has no valid context, cta disambiguation receives more weight.
                            cta_coeff = np.mean(cta_disambiguation_weights)
                            for cea in cea_candidates: ## in case a candidate has a cpa as its predicates, its score is augemented.
                                if Candidate_Entity(row_index=row_index, col_index=col_index, id=cea["id"])  in self.potential_candidates:
                                    cpa_coeff = max([it["cpa_coeff"] for it in self.potential_candidates[Candidate_Entity(row_index=row_index, col_index=col_index, id=cea["id"])]])
                                    cea["score"] = min(1.0, cea["score"]*(1+cpa_coeff))
                        else:
                            cta_coeff = np.mean(cta_disambiguation_weights)/2
                    else:
                        cta_coeff = 0.25

                ## update cea score by cta, cpa score.
                ## final_score(cea) = [score(cea) + coeff_cpa(cpa_disambiguation_score) + coeff_cta(cta_disambiguation_score)]/(1+coeff_cta+coeff_cpa)                            
                for cea in cea_candidates:
                    total_coeff = 1
                    if cta_disabg_applied:
                        total_coeff += cta_coeff
                        cea["score"] += cta_coeff*cta_disambiguation_scores[cea["id"]]
                    cea["score"] = cea["score"]/total_coeff
                    candidate = Candidate_Entity(row_index=row_index, col_index=col_index, id=cea["id"])
                    # self.entity_scores[candidate] = cea["score"] 

                ## Sort the list of cea candidates and find the best ceas
                ## if there are many best ceas of same score 
                sorted_cea_candidates = sorted(cea_candidates, key=lambda t: (t["score"], len(self.potential_candidates.get(Candidate_Entity(row_index=row_index, col_index=col_index, id=t["id"]), []))), reverse=True)
                self.cea_annot[cell] = []
                if only_one:
                    for candidate_cea in sorted_cea_candidates:
                        if candidate_cea["score"] == sorted_cea_candidates[0]["score"]:
                            self.cea_annot[cell].append(candidate_cea)
                else:
                    self.cea_annot[cell] = sorted_cea_candidates
                end_time = time.time()
                self.cea_task_time += round(end_time-start_time, 2)
                return self.cea_annot[cell]
        else:
            end_time = time.time()
            self.cea_task_time += round(end_time-start_time, 2)
            return ""
                                                            
    def cpa_task(self, head_col_index, tail_col_index, only_one=True):
        """
        This task finds a semantic relation between an ordered pair of column (head_column, tail_column)
        We try to figure out the most suitable relation (highest score, shortest length (1-hope relation is more preferable than 2-hope relation))
                among the ones connecting an entity candidate 
                            in head column to an entity candidate in tail column.
        Args:
            head and tail column index.
            only_one: in case that there are many CPAs of same score, return only one or return all.
        Return:
            column pair relations (CPA): ID, score, coverage.
        """
        start_time = time.time()
        cpa_candidates = {}
        head_cells = {}
        tail_cells = {}
        if Column_Pair(head_col_index=head_col_index, tail_col_index=tail_col_index) in self.unrelated_col_pairs or (tail_col_index in self.literal_cols and tail_col_index < head_col_index):
            ## no need to calculate CPA for 2 unrelated columns.
            end_time = time.time()
            self.cpa_task_time += round(end_time-start_time, 2)
            return ""

        ## get CEAs for head column and tail column. a CEA has its ID and its score.
        for row_index in range(self.first_data_row,self.num_rows):
            ## get CEAs for head entity column.
            cell = Cell(row_index=row_index, col_index=head_col_index)
            if cell in self.cea_annot:
                head_cells[row_index] = self.cea_annot[cell]

            ## tail column can be entity column or literal column.
            if tail_col_index in self.entity_cols:
                ## get CEAs for entity tail column
                cell = Cell(row_index=row_index, col_index=tail_col_index)
                if cell in self.cea_annot:
                    tail_cells[row_index] = self.cea_annot[cell]
            else:
                ## get mentions for literal column. a CEA is supposed to be the cell's mention and its score is 0.0
                tail_cells[row_index] = [{"id": self.table[row_index][tail_col_index], "score": 0.0}]

        ## browsing all row to find the CPA candidates.
        for row_index in set(head_cells.keys()) & set(tail_cells.keys()):
            relation_in_current_row = {}
            for head_candidate in head_cells[row_index]:
                head_id = head_candidate["id"]
                head_conf = head_candidate["score"]        
                for tail_candidate in tail_cells[row_index]:
                    tail_id = tail_candidate["id"]
                    tail_conf = tail_candidate["score"]
                    col_cpa_candidates = self.cached_cpa_candidates.get((head_id, tail_id), {})
                    for a_cpa_candidate in col_cpa_candidates:
                        ## a cpa candidate is found between a head candidate entity and a tail candidate entity in current row.
                        ## we update the score of this cpa candidate: score = max(head_candidate_entity_score, tail_candidate_entity_score)
                        ##     the score is weighted by the semantic proximity of involved relation.
                        if a_cpa_candidate.id in relation_in_current_row:
                            relation_in_current_row[a_cpa_candidate.id]["score"] = max(relation_in_current_row[a_cpa_candidate.id]["score"], a_cpa_candidate.semantic_proximity*max(head_conf,tail_conf))
                            relation_in_current_row[a_cpa_candidate.id]["semantic_proximity"] = min(relation_in_current_row[a_cpa_candidate.id]["semantic_proximity"], a_cpa_candidate.semantic_proximity)
                        else:
                            relation_in_current_row[a_cpa_candidate.id] =  {"semantic_proximity": a_cpa_candidate.semantic_proximity, "score": a_cpa_candidate.semantic_proximity*max(head_conf,tail_conf)}                                        
            ## update the final list of cpa candidates with ones found in current row.                                                                   
            for prop, score_info in relation_in_current_row.items():
                if prop not in cpa_candidates:
                    cpa_candidates[prop] = {"count": 1, "total_scores": score_info["score"], "semantic_proximity": score_info["semantic_proximity"]}
                else:
                    cpa_candidates[prop]["count"] = cpa_candidates[prop]["count"] + 1
                    cpa_candidates[prop]["total_scores"] = cpa_candidates[prop]["total_scores"] + score_info["score"]
                    cpa_candidates[prop]["semantic_proximity"] = min(cpa_candidates[prop]["semantic_proximity"], score_info["semantic_proximity"]) 

        ## After scanning all rows, we have a final list of candidate cpas.          
        if cpa_candidates :      
            ## Sort the list of candidate cpas and find the best cpas
            ## Order: total_scores*count, count (if cpas have same score, more frequent cpa is more preferable) 
            sorted_cpa_candidates = sorted(cpa_candidates.items(), key=lambda it: (it[1]["count"]*it[1]["total_scores"], it[1]["count"], it[1]["semantic_proximity"],
                                                                                    "::" not in it[0], "(-)" not in it[0]), reverse=True)

            ## get best candidate cpas (there maybe many of same score)
            threshold = sorted_cpa_candidates[0][1]
            col_pair = Column_Pair(head_col_index=head_col_index, tail_col_index=tail_col_index)
            self.cpa_annot[col_pair] = []
            if only_one:
                ## for final annotation, return the most relevant CPAs of same score. 
                for a_cpa_candidate in sorted_cpa_candidates:
                    if a_cpa_candidate[1]["count"]*a_cpa_candidate[1]["total_scores"] >= threshold["count"]*threshold["total_scores"]:
                        ## reformat the cpa annotation with "id", average "score", "coverage", "semantic_proximity"
                        self.cpa_annot[col_pair].append({"id": a_cpa_candidate[0], "score": a_cpa_candidate[1]["total_scores"]/(self.num_rows-self.first_data_row), 
                                    "semantic_proximity": a_cpa_candidate[1]["semantic_proximity"], "coverage": a_cpa_candidate[1]["count"]/(self.num_rows-self.first_data_row) })    
            else:
                ## for intermediate disambiguation steps (CEA disambiguation), return many CPAs of highest score or highest coverage which maybe useful for disambiguation.
                for a_cpa_candidate in sorted_cpa_candidates:
                    if a_cpa_candidate[1]["count"] >= threshold["count"]:
                        ## reformat the cpa annotation with "id", average "score", "coverage", "semantic_proximity"
                        self.cpa_annot[col_pair].append({"id": a_cpa_candidate[0], "score": a_cpa_candidate[1]["total_scores"]/(self.num_rows-self.first_data_row), 
                                    "semantic_proximity": a_cpa_candidate[1]["semantic_proximity"], "coverage": a_cpa_candidate[1]["count"]/(self.num_rows-self.first_data_row) })  
            end_time = time.time()
            self.cpa_task_time += round(end_time-start_time, 2)
            return self.cpa_annot[col_pair]
        else:
            end_time = time.time()
            self.cpa_task_time += round(end_time-start_time, 2)
            return ""

if __name__ == '__main__':
    pass
                       
