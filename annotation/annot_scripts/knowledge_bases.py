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
import lmdb
import json
import pickle
from .abstract_classes import AbstractKnowledgeBase

class Wikidata_KB(AbstractKnowledgeBase):
    """ Wikidata KB Interface """
    def __init__(self, dump_path):
        """ Initialize the Wikidata KB """
        ## list of properties to be considered in CTA
        self.type_properties =  ["P31", "P106", "P39", "P105"]
        ## PID of subclass property in Wikidata KB.
        self.instanceOfPID = "P31"
        self.subClassPID = "P279"
        ## PID of unit symbol in Wikidata KB
        self.unitSymbolPID = "P5061"
        ## pairs of time periods
        self.timePeriodPID = [
            ("P571", "P576"), ("P571", "P2699"), ("P571", "P730"), ("P571", "P3999"),
            ("P1619", "P3999"), ("P729", "P730"), ("P5204", "P2669"), ("P5204", "P576"),
            ("P580", "P582"), ("P2031", "P2032"), ("P3415", "P3416"), ("P3027", "P3028"),
            ("P7103", "P7104"), ("P2310", "P2311"), ("P7124", "P7125"), ("P569", "P570"),
            ("P1636", "P570")
        ]
        
        ## transitive property (https://www.wikidata.org/wiki/Wikidata:List_of_properties/transitive_relation)
        self.transitivePID = ["P131", "P276", "P279", "P361", "P403", "P460", "P527", "P706", "P927", "P1647", "P2094",
                                "P3373", "P3403", "P5607", "P5973", "P171"]
        
        ## edge reader, entity reader, property reader, unit dictionary reader
        try:
            ## load the unit_entity dictionary
            self.unit_entity_mapping = json.load(open(dump_path + "/units.json", "r"))
            ## wikidata edges reader
            self.edge_reader = lmdb.open(dump_path + "/edges", readonly=True, readahead=False, lock=False)
            self.edge_txn = self.edge_reader.begin()
        except Exception as e:
            print(f" Error loading knowledge dumps. Details: {e} !! ")

    def close_dump(self):
        """ Close the graph readers """
        self.edge_reader.close()

    def is_valid_ID(self, entity_id):
        """ Heuristic way to verify whether an id is a valid entityID wrt. Wikidata """
        if (len(entity_id) > 1) and (entity_id[0] in ["P", "Q"]) and (entity_id[1:].isdigit()):
            return True
        return False

    def get_subgraph_of_entity(self, entity_id):
        """ Get forward nodes (predicate->object) and backward nodes (subject->predicate) of an entity in KG. """
        key = self.edge_txn.get(entity_id.encode("ascii"))
        if key:
            prop_obj_dict = pickle.loads(key)   
            del prop_obj_dict["labels"], prop_obj_dict["aliases"],  prop_obj_dict["descriptions"]  
        else:
            prop_obj_dict = {}
        return prop_obj_dict
    
    def get_label_of_entity(self, entity_id):
        """ Get the labels and aliases of an entity. Language info is not returned 
            If only_one = True, return the default en label """
        en_label = ""
        key = self.edge_txn.get(entity_id.encode("ascii"))
        if key:
            prop_obj_dict = pickle.loads(key)
            if prop_obj_dict["labels"]:
                en_label = prop_obj_dict["labels"][0]
            else:
                en_label = "No English Label"
        return en_label

    def get_num_edges(self, entity_id):
        """ Get number of incoming edges of an entity in KG """
        key = self.edge_txn.get(entity_id.encode("ascii"))
        num_edges = 0
        if key:
            prop_obj_dict = pickle.loads(key)
            for prop, obj_dict in prop_obj_dict.items():
                if prop not in ["descriptions", "labels", "aliases"]:
                    num_edges += len(obj_dict)
        return num_edges
            
    def get_symbol_of_unit_entity(self, unit_entity_id):
        """ get the unit symbol of an unit entity. E.g. unit symbol of Q11573 (metre) is m """
        key = self.edge_txn.get(unit_entity_id.encode("ascii"))
        if key:
            prop_obj_dict = pickle.loads(key)
            ## since Pint does not support officially currency, we should handle it ourself by defining Currency in Pint. First, convert currency symbol to name (e.g. € to euro)
            ##     since Pint does not accept special symbols like currency symbols. 
            ## Currently, we only support: dollar, euro, japanese_yen, chinese_yuan, pound_sterling, south_korean_won, russian_ruble, australian_dollar"
            if "Q8142" in prop_obj_dict.get(self.instanceOfPID, {}): ## if unit indicates currency (Q8142)
                return "_".join(self.get_label_of_entity(unit_entity_id, only_one=True).lower().split(" "))
            else:
                if self.unitSymbolPID in prop_obj_dict:
                    return list(prop_obj_dict[self.unitSymbolPID].keys())[0]
            ## not an unit entity
            return None
        else:
            ## not an entity
            return None      

    def map_unit_dimension_to_entity(self, unit_dim):
        """ return the corresponding entity of a given unit dimension. E.g. "microsecond" --> Q842015 """
        return self.unit_entity_mapping.get(unit_dim, {}).get("wikidataID", None).replace("http://www.wikidata.org/entity/", "")

    def get_supertypes_of_type(self, type_id):
        """ return supertype of an entity type """
        key = self.edge_txn.get(type_id.encode("ascii"))
        if key:
            prop_obj_dict = pickle.loads(key)
        else:
            prop_obj_dict = {}    
        super_type = prop_obj_dict.get(self.subClassPID, {})
        return super_type

    def get_types_of_entity(self, entity_id, num_level=1):
        """ get the hierachical types of an entity upto num_level. (Level 0 is direct types).
            If muti_properties is True, other properties in self.type_properties, 
                                                        apart from P31, are also considered."""
        hierachical_types = {}
        if num_level > 0:
            key = self.edge_txn.get(entity_id.encode("ascii"))
            if key:
                prop_obj_dict = pickle.loads(key)
            else:
                prop_obj_dict = {}

            instanceOf_types = {}
            others_types = {}
            for prop in self.type_properties:
                a_type = prop_obj_dict.get(prop, None)
                if a_type:
                    if prop == self.instanceOfPID:
                        instanceOf_types.update(a_type)
                    else:
                        others_types.update(a_type)
            if others_types:
                hierachical_types[f"level_1"] = others_types 
                # hierachical_types[f"level_2"] = instanceOf_types 
            else:
                hierachical_types[f"level_1"] = instanceOf_types 

            if num_level > 1:
                inter_types = hierachical_types[f"level_1"]
                for i in range(2, num_level+1):
                    if f"level_{i}" not in hierachical_types:
                        hierachical_types[f"level_{i}"] = {}
                    types = {}
                    for t in inter_types:
                        key = self.edge_txn.get(t.encode("ascii"))
                        if key:
                            prop_obj_dict = pickle.loads(key)
                        else:
                            prop_obj_dict = {}
                        super_type = prop_obj_dict.get(self.subClassPID, None)
                        if super_type:
                            types.update(super_type)                    
                    hierachical_types[f"level_{i}"].update(types)
                    inter_types = types
        return hierachical_types

    def map_rank(self, rank):
        """ 
            Each wikidata attribute has a rank {"PREFERRED", "NORMAL", "DEPRECATED"}.
            We numerize it as relevance value (2: high, 0: low)
        """
        if rank == "PREFERRED":
            return 2
        elif rank == "NORMAL":
            return 1
        else:
            return 0

    def prefixing_entity(self, entity):
        """ Appending prefix to entity """
        if entity[0] == "Q": ## entity
            # return "https://www.wikidata.org/wiki/" + entity
            return "http://www.wikidata.org/entity/" + entity ## avoid wiki redirect, use real identifier of property
        elif entity[0] == "P": ## property
            # return "https://www.wikidata.org/wiki/Property:" + entity
            return "http://www.wikidata.org/prop/direct/" + entity ## avoid wiki redirect, use real identifier of entity
        else:
            return entity

