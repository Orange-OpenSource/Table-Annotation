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
from abc import ABC, abstractmethod
import collections

### For definition of KB ###
class AbstractKnowledgeBase(ABC):
    """ KB Abstract class """ 
    @abstractmethod
    def __init__(self, dump_path):
        """ Intialize a KB: reading hashmaps... """
        pass

    @abstractmethod
    def is_valid_ID(self, entity_id):
        """ Check whether an id is a valid ID w.r.p the KB """
        pass

    @abstractmethod
    def get_subgraph_of_entity(self, entity_id):
        """ get (subject, predicate, object) edges involving entity_id
                where object is target entity """
        pass

    @abstractmethod
    def get_types_of_entity(self, entity_id, num_level):
        """ Get the types of an entity in hierachical levels. 
            A list of relevant properties for entity types (for ex. occupation) is also allowed."""
        pass 

    @abstractmethod
    def get_label_of_entity(self, entity_id):
        """ Get the labels + aliases of an entity """
        pass
        
    @abstractmethod
    def get_num_edges(self, entity_id):
        """ Get number of incoming edges of an entity in KG """
        pass

    @abstractmethod
    def prefixing_entity(self, entity):
        """ Adding prefix to an entity """
        pass

### Table component definition ###
class Column(collections.namedtuple(
    "Column", ("col_index"))):
    """ Definition of table column """
    pass

class Cell(collections.namedtuple(
    "Cell", ("row_index", "col_index"))):
    """ Definition of table cell. """
    pass

class Column_Pair(collections.namedtuple(
    "Column_Pair", ("head_col_index", "tail_col_index"))):
    """ Definition of column pair. """
    pass

class Candidate_Entity(collections.namedtuple(
    "Candidate_Entity", ("row_index", "col_index", "id"))):
    """ Definition of a candidate entity for a table cell. """
    pass

### Abstract annotation components ###
class Relation(collections.namedtuple(
    "Relation", ("id", "semantic_proximity"))):
    """ Definition of a relation candidate"""
    pass

class Edge(collections.namedtuple(
    "Edge", ("pid", "info"))):
    """ Definition of an edge in KG: info field indicates the type of object that the edge points to: entity, literal value """
    pass

### Abstract annotation model ###
class AbstractAnnotationModel(ABC):
    """ Table annotation abstract class."""
    def __init__(self, table, target_kb, preprocessing_backend, lookup_backend):
        """ Initialize the annotation model: KB specified at target_kb, preprocessing using preprocessing_backend, lookup using lookup_backend... """
        pass 
    
    def preprocessing_task(self):
        """ table preprocessing task """
        pass

    def lookup_task(self):
        """ lookup task for semantic cells in table """
        pass

    def cta_task(self, column_idx, only_one):
        """ type annotation for a column index. Return many or more types. """
        pass

    def cea_task(self, column_idx, row_idx, only_one):
        """ entity annotation for a cell at (row_index, column_index). Return many or more annotation. """
        pass

    def cpa_task(self, column_head, column_tail, only_one):
        """ relation annotation for a column pair (head_column, tail_column). Return many or more annotation. """
        pass


