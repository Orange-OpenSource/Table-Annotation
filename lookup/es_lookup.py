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
from elasticsearch import Elasticsearch
from rapidfuzz import fuzz
import copy
import math
from . import settings

class LookupES:
    FLAT_QUERY_STRING = {
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "should": [
                                {
                                    "bool": {
                                        "must": [
                                            {"match": 
                                                {"label":  {"query": "ABC", "fuzziness": "AUTO" }}
                                            }
                                        ],
                                        "filter": {
                                            "range": { "length" : { "gte" : 0, "lte": 0 }}
                                        }

                                    }
                                },
                                {
                                    "bool": {
                                        "must": [
                                            {"match": 
                                                {"label.keyword":  {"query": "ABC", "fuzziness": "AUTO" }}
                                            }
                                        ],
                                        "filter": {
                                            "range": { "length" : { "gte" : 0, "lte": 0 }}
                                        }

                                    }
                                }                                

                            ]

                        }
                    },
                    "functions": [
                        {
                            "filter": { "match": { "origin": "MAIN_ALIAS" }},
                            "weight": settings.MAIN_ALIAS_FACTOR
                        },
                        {
                            "filter": { "match": { "origin": "SUB_ALIAS" }},
                            "weight": settings.SUB_ALIAS_FACTOR
                        }
                    ]
                }
            },
            "size": 10000
        }
    def __init__(self):
        self.es = None
 
    def connect(self):
        """Connect to ElasticSearch cluster"""
        host = settings.ELASTICSEARCH_HOST
        port = settings.ELASTICSEARCH_PORT
        auth = settings.ELASTICSEARCH_AUTH
        if auth:
            user = settings.ELASTICSEARCH_USER
            pwd = settings.ELASTICSEARCH_PWD
            self.es = Elasticsearch([{'host':host,'port':port}], timeout=300, http_auth=(user, pwd), verify_certs=False, use_ssl=True)
        else:
            self.es = Elasticsearch([{'host':host,'port':port}], timeout=300)
        # connect to ES cluster 
        self.es.cluster.health()

    def flat_search(self, index_name, labels):
        """Search candidate entities for labels in flat index"""
        result = []
        for label in labels:
            result.append(self.flat_search_item(index_name, label))
        return result

    def flat_search_item(self, index_name, label):
        try:
            def es_search(request):
                result = self.es.search(index=index_name, body=request)
                return result

            def filter_result(label, result):
                entities_result = []
                total = result["hits"]["total"]["value"]
                bm25_max = result["hits"]["max_score"] ## tdidf score max             
                if (total > 0):
                    #Calculate max ratio for all entities returned by ES
                    entities_set = set()
                    entity_fuzzy_ratio = {} ## store fuzzy matching score
                    entity_bm25_ratio = {} ## store keyword matching (tfidf) score (retrieved from ES)
                    entity_pr_ratio = {}
                    entity_partial_matching = set() ## since partial exact matching sometimes not fit well with levenshtein, 
                                                    ## we do not use levenshtein distances to evaluate the partial matching entity.
                                                    ## for e.g. "YANKEES" vs. "NEW YORK YANKEES"
                    entity_best_label = {}
                    max_ratio = 0.0
                    for hit in result["hits"]["hits"]:
                        #Calculate ratio for label of the entity
                        entity_label = hit["_source"]["label"]
                        entity_origin = hit["_source"]["origin"]
                        bm25_score = hit["_score"]/bm25_max
                        entity_pr_ratio[hit["_source"]["entity"]] = hit["_source"].get("PR", 0.0)
                        entity_bm25_ratio[hit["_source"]["entity"]] = max(entity_bm25_ratio.get(hit["_source"]["entity"], bm25_score), bm25_score)
                        
                        entity_label_lower = entity_label.lower()
                        ## ratio components
                        char_based_ratio = 0.9*fuzz.ratio(label_lower, entity_label_lower)/100 + 0.1*fuzz.ratio(new_label, entity_label)/100
                        token_sort_based_ratio = 0.9*fuzz.token_sort_ratio(label_lower, entity_label_lower)/100 + 0.1*fuzz.token_sort_ratio(new_label, entity_label)/100
                        if 0.5 < len(label_lower)/len(entity_label_lower) < 2.0: ## token set ratio is noisy, only apply on two labels of similar lengths.
                            token_set_based_ratio = 0.9*fuzz.token_set_ratio(label_lower, entity_label_lower)/100 + 0.1*fuzz.token_set_ratio(new_label, entity_label)/100
                        else:
                            token_set_based_ratio = 0.0
                        ## find entities that have partial exact matching, we put them directly in output without evaluating levenshtein distances
                        ## since levenshtein does not fit well with partial exact matching.
                        ## to avoid extracting too much irrelevant entities, the entity label should not be too long or too short w.r.t. input label
                        partial_ratio = 0.9*fuzz.partial_ratio(label_lower, entity_label_lower)/100 + 0.1*fuzz.partial_ratio(new_label, entity_label)/100
                        token_diff = abs(len(label_lower.split(" ")) - len(entity_label_lower.split(" "))) ## token difference between 2 labels
                            ## partial matching ratio and token set ratio are noisy, only apply on two labels of similar lengths.
                        if (partial_ratio > 0.9 and token_diff <= 2) or \
                                (token_set_based_ratio > 0.9 and 0.5 < len(label_lower)/len(entity_label_lower) < 2.0):                  
                            entity_partial_matching.add(hit["_source"]["entity"])
                        ## the final ratio is the mean of two maximum ratios among three ratios. 
                        ## to avoid that 2 ratios of same values dominate the other.
                        ## e.g. char_based_ratio("universal", "universal picture") = token_sort_based_ratio("universal", "universal picture") = 0.66
                        ##        so including both ratios in the final ratio will decrease the significance of token_set_based_ratio("universal", "universal picture") which is 1.0
                        ratio = sum(sorted([char_based_ratio, token_sort_based_ratio, token_set_based_ratio], reverse=True)[:2])/2

                        #Apply factor according to label origin
                        label_origin = hit["_source"].get("origin")
                        factor = 1
                        if label_origin == "MAIN_ALIAS":
                            factor = settings.MAIN_ALIAS_FACTOR
                        elif label_origin == "SUB_ALIAS":
                            factor = settings.SUB_ALIAS_FACTOR
                        ratio *= factor

                        #Store max ratio of the queried label
                        max_ratio = max(max_ratio, ratio)
                        #print(hit["_source"]["entity"] + ": " + entity_label + ": " + str(ratio) + " ("+str(hit["_score"])+")")
                        if entity_fuzzy_ratio.get(hit["_source"]["entity"]):
                            if ratio > entity_fuzzy_ratio[hit["_source"]["entity"]]:
                                #Store max ratio of the entity
                                entity_fuzzy_ratio[hit["_source"]["entity"]] = ratio
                                entity_best_label[hit["_source"]["entity"]] = entity_label
                        else:
                            #Store ratio of the entity
                            entity_fuzzy_ratio[hit["_source"]["entity"]] = ratio
                            entity_best_label[hit["_source"]["entity"]] = entity_label

                    ratio_threshold = max(settings.ADAPTATIVE_RATIO_MIN_THRESHOLD, max_ratio-settings.ADAPTATIVE_RATIO_MAX_GAP)

                    ## in wikidata, we use pagerank to re-rank relevant candidates.
                    #Filter entities
                    filtered = 0
                    ## in wikidata, we use pagerank to re-rank relevant candidates.
                    ## fist, find the max page rank among candidates.
                    max_page_rank = 0
                    # with self.wikidata_stats_reader.begin() as stat_txn:
                    for entity in entity_fuzzy_ratio:
                        if entity_fuzzy_ratio[entity] >= ratio_threshold or entity in entity_partial_matching:
                            entities_set.add(entity)
                            max_page_rank = max(max_page_rank, entity_pr_ratio[entity])
                            filtered += 1    
                    if max_page_rank == 0.0:
                        max_page_rank = 1.0  

                    ## re-rank relevant candidates with locally log-normalized page rank score.
                    for entity in entities_set:
                        # entity_score = (1-settings.PAGE_RANK_FACTOR-settings.BM25_FACTOR)* entity_fuzzy_ratio[entity] + settings.PAGE_RANK_FACTOR*math.log2(list_page_rank[entity]+1)/math.log2(max_page_rank+1) + settings.BM25_FACTOR * entity_bm25_ratio[entity]
                        entity_score = (1-settings.PAGE_RANK_FACTOR-settings.BM25_FACTOR)* entity_fuzzy_ratio[entity] + settings.PAGE_RANK_FACTOR*math.log2(entity_pr_ratio[entity]+1)/math.log2(max_page_rank+1) + settings.BM25_FACTOR * entity_bm25_ratio[entity]
                        entities_result.append({"entity": entity, "label": entity_best_label[entity], "score": entity_score, "origin": entity_origin})
                    entities_result.sort(key=lambda x: x["score"], reverse=True)
                return {"label": label, "entities": entities_result}

            request = copy.deepcopy(self.FLAT_QUERY_STRING)
            new_label = label.replace('"', '').strip() ## only vital preprocessing on input label: replace '"" by '' since ES does not accept double quote and remove last spaces.
            new_label = " ".join(new_label.split()) # remove multiple space in string
            label_lower = new_label.lower()
            request["query"]["function_score"]["query"]["bool"]["should"][0]["bool"]["must"][0]["match"]["label"]["query"] = new_label
            request["query"]["function_score"]["query"]["bool"]["should"][1]["bool"]["must"][0]["match"]["label.keyword"]["query"] = new_label
            request["query"]["function_score"]["query"]["bool"]["should"][0]["bool"]["filter"]["range"]["length"]["gte"] = int(len(new_label) * settings.LABEL_LENGTH_MIN_FACTOR)
            request["query"]["function_score"]["query"]["bool"]["should"][0]["bool"]["filter"]["range"]["length"]["lte"] = int(len(new_label) * settings.LABEL_LENGTH_MAX_FACTOR)
            request["query"]["function_score"]["query"]["bool"]["should"][1]["bool"]["filter"]["range"]["length"]["gte"] = int(max(0, len(new_label) - settings.LABEL_TOKEN_DIFF))
            request["query"]["function_score"]["query"]["bool"]["should"][1]["bool"]["filter"]["range"]["length"]["lte"] = int(len(new_label) + settings.LABEL_TOKEN_DIFF)

            result = es_search(request)
            entities = filter_result(label, result)

            return entities
        except Exception as e:
            return {"label": label, "error": str(e)}
