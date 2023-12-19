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
'''
TABLE UTILITY FUNCTIONS
Developper: Huynh Viet Phi (vietphi.huynh@orange.com)
'''
import json
import numpy as np
import ftfy
from collections import Counter, namedtuple
from itertools import combinations

from .entity_parsers.spacy_ner_parser import spacy_parser
from .entity_parsers.unit_parser import unit_parser
from .entity_parsers.regex_parser import regex_parser
from .entity_parsers.phoneNumber_parser import phonenumber_parser

""" Named entity and Unit entity parsers """
def is_concept(label):
    concept_list =  ["EVENT", "FAC", "GPE", "LAW", "LOC", "NORP", "ORG", "PERSON", "PRODUCT", "WORK_OF_ART", "LANGUAGE", "UNKNOWN"]
    for concept in concept_list:
        if concept in label:
            return True
    return False

def typing_priority(t):
    if t != "CARDINAL":
        return 1
    else:
        return 0

def get_string_type(label):
    if len(label) >=100:
        return "String_Normal"      

    ## string whose large portion is numer is considered as number string
    elif 2*len([char for char in label if char.isdigit()])>=len(label):  
        return "String_Number"
   
    elif label.upper() == label:
        return "String_Uppercase"

    else:
        ## Unsolved case assumes normal string
        return "String_Normal"

def text_parser(list_cell):
    entity_per_cell = {}
    remain_cells = []
    for cell in list_cell:
        if (cell == "") or (cell[0] in ".@_!#$%^&*()<>?/\|}{~:\'-+~~_°¨" and cell == cell[0]*len(cell)):  
            continue ## "" means no data type
        ## string contains a single char that is neither alpha nor digit has no type
        if (len(cell) == 1) and \
             (((not cell.isalpha()) and (not cell.isdigit())) or len(cell.encode("utf-8")) > 1):
            continue
        entity_per_cell[cell] = []
        if len(cell) > 70:
            entity_per_cell[cell].append("UNKNOWN")
            continue 
        remain_cells.append(cell)

    if remain_cells:
        ## perform typing parsers
        regex_entity_per_cell = regex_parser(remain_cells)
        unit_entity_per_cell = unit_parser(remain_cells)
        phone_entity_per_cell = phonenumber_parser(remain_cells)  
        named_entity_cell = spacy_parser(remain_cells)

        ## collect typings
        for ner_list in [phone_entity_per_cell, regex_entity_per_cell, 
                            unit_entity_per_cell, named_entity_cell]:
            for cell in ner_list:
                for a_ner in ner_list[cell]:
                    if a_ner not in entity_per_cell[cell]:
                        entity_per_cell[cell].append(a_ner)

    for cell in entity_per_cell:
        if not entity_per_cell[cell]:
            entity_per_cell[cell].append("UNKNOWN")

    ## datatype parsers (from typing)
    datatype_per_cell = {}
    for cell, entities in entity_per_cell.items():
        datatype_per_cell[cell] = []
        for entity in entities:
            if is_concept(entity):
                dtype = get_string_type(cell)
                if dtype not in datatype_per_cell[cell]:
                    datatype_per_cell[cell].append(dtype)
            else:
                if entity not in datatype_per_cell[cell]:
                    datatype_per_cell[cell].append(entity)

    return entity_per_cell, datatype_per_cell

""" Utility functions specifically for table processing """
#### type related functions ####
def header_related_datatype(t):
    """
    Verify whether a type could appear in header.
    """
    if t in ["String_Normal", "String_Uppercase"]:
        return True 
    else:
        return False  

def keyColumn_related_datatype(t):
    """
    Verify whether a type could appear in header.
    """
    if t in ["String_Normal", "String_Uppercase", "String_Number"]:
        return True 
    else:
        return False  

#### Table-related functions ####
def recover_poorly_encoded_cell(broke_cell):  
    """ Recovering a badly-encoded utf-8 cell """
    ## Poorly encoded cells: encode by utf-8 but decode by unicode
    try:
        cell = broke_cell
        # Solved: 
        # 1. Convert string to byte keeping content, in other word, encode string by latin1
        byte_cell = bytes(cell.encode('latin1'))
        # 2.  Decode Cell Byte by unicode
        new_cell = byte_cell.decode('unicode-escape')
        return ftfy.fix_text(new_cell)
    except:
        return ftfy.fix_text(broke_cell)

def table_filtering(table):
    """ 
        Possibly filtering empty rows and information rows (title rows) 
        Args:
            input table
        Return:
            filtered table and possible information rows.
    """
    title = []
    new_table = []
    max_width = max([len(row) for row in table])
    for row in table:
        if row:
            new_row = []
            num_nonmissing_cell = 0       
            for cell in row:
                if cell != "" and cell != " "*len(cell):
                    num_nonmissing_cell += 1
                try:
                    new_row.append(recover_poorly_encoded_cell(cell))
                except:
                    new_row.append(cell)
            if num_nonmissing_cell > 0:
                new_table.append(new_row)
    ## padding short rows
    table_padding(new_table, max_width)
    ## remove empty columns
    new_table = table_null_column_removing(new_table)
    return new_table, title

def table_padding(table, tab_width):
    """
        Padding "" to short rows to balance the table.
    """
    for line in table:
        for i in range(tab_width-len(line)):
            line.append("")

def table_null_column_removing(table):
    """
        Remove null columns in table.
    """
    table_T = list(map(list, zip(*table)))
    final_table = []
    for line in table_T:
        if line != [""]*(len(table)):
            final_table.append(line)
    final_table =  list(map(list, zip(*final_table)))    
    return final_table

def transpose_heterogeneous_table(table):
    """ 
        Transpose a heterogeneous table (rows with possible different widths).
        We natively padding "" for short lines and perform normal transpose.
    """
    table_T = []
    end_of_tab = False
    i = 0
    while not end_of_tab:
        end_of_tab = True
        line_T = []
        for line in table:
            if i < len(line):
                line_T.append(line[i])
                end_of_tab = False
            else:
                line_T.append("")
        i += 1
        table_T.append(line_T)
    return table_T[:-1]

#### Type-related functions ####
def parse_table(table):
    list_cells = list(set([item for line in table for item in line]))
    entity_list, dataType_list = text_parser(list_cells)
    return entity_list, dataType_list

def datatype_per_column(table, table_Datatype, top_k=1):
    """ 
        Return the top_k data types of each column in table. 
        A type counter is applied to each column and the top_k most frequent type 
        in the counter is the top_k data type of corresponding column.

        Args:
            table: horizontally oriented-table.
                   nested list whose sublist is a row.
            top_k: default 1, the most frequent type 
        Return:
            nested list whose a sublist is a data type of a column.
    """
    ## transpose the table in order to retrieve its columns more easily
    table_T = transpose_heterogeneous_table(table)
    ## apply Datatype counter to each column.
    type_per_col = {}
    for col_idx, col in enumerate(table_T):
        dtypes = {}
        sum_type = 0
        for cell in col:
            if cell in table_Datatype:
                for a_dt in table_Datatype[cell]:
                    dtypes[a_dt] = dtypes.get(a_dt, 0) + 1
            sum_type += 1

        for cell in col:
            if len(table_Datatype.get(cell, [])) > 1:
                sorted_ts = sorted(table_Datatype[cell], key=lambda x: (dtypes[x], typing_priority(x)), reverse=True)
                for other_type in sorted_ts[1:]:
                    dtypes[other_type] -= 1
                    if dtypes[other_type] == 0:
                        dtypes.pop(other_type)

        if dtypes:
            sorted_types = Counter(dtypes)
            top_k_freq = sorted_types.most_common(top_k)
            type_per_col[col_idx] = [{"type": item[0], "score": item[1]/sum_type} for item in top_k_freq if item[1] > 0.0]
        else:
            type_per_col[col_idx] = [{"type": "", "score": 1.0}]
    return type_per_col

def typing_per_column(table, table_Typing, top_k=1):
    """ 
        Return the top_k data types of each column in table. 
        A type counter is applied to each column and the top_k most frequent type 
        in the counter is the top_k data type of corresponding column.

        Args:
            table: horizontally oriented-table.
                   nested list whose sublist is a row.
            top_k: default 1, the most frequent type 
        Return:
            nested list whose a sublist is a data type of a column.
    """
    ## transpose the table in order to retrieve its columns more easily
    table_T = transpose_heterogeneous_table(table)
    ## apply Datatype counter to each column.
    type_per_col = {}
    for col_idx, col in enumerate(table_T):
        dtypes = {}
        sum_type = 0
        for cell in col:
            if cell in table_Typing:
                for a_tp in table_Typing[cell]:
                    dtypes[a_tp] = dtypes.get(a_tp, 0) + 1
            sum_type += 1

        for cell in col:
            if len(table_Typing.get(cell, set())) > 1:
                sorted_ts = sorted(table_Typing[cell], key=lambda x: (dtypes[x], typing_priority(x)), reverse=True)
                for other_type in sorted_ts[1:]:
                    dtypes[other_type] -= 1
                    if dtypes[other_type] == 0:
                        dtypes.pop(other_type)
        if dtypes:
            sorted_types = Counter(dtypes)
            top_k_freq = sorted_types.most_common(top_k)
            type_per_col[col_idx] = [{"type": item[0], "score": item[1]/sum_type} for item in top_k_freq if item[1] > 0.0]
        else:
            type_per_col[col_idx] = [{"type": "", "score": 1.0}]

        if col_idx == 0 and  type_per_col[col_idx][0]["type"] == "CARDINAL":
            ## verify if the first column is index column
            current_index = None
            is_index_column = True
            tolerate = 0
            for cell in col:
                try:
                    idx = int(float(cell))
                    if current_index:
                        if idx == current_index + 1:
                            current_index += 1
                        elif idx == current_index:
                            pass
                        else:
                            is_index_column = False
                            break
                    else:
                        current_index = idx
                except:
                    ## no index detected in this cell
                    current_index = None
                    tolerate += 1
                    if tolerate > 4: ## number of noindex-detected tolerance, exceed this number, a column is not ordinal.
                        is_index_column = False
                        break
            if is_index_column:
                type_per_col[col_idx][0]["type"] = "ORDINAL"

    return type_per_col

#### Orientation-related functions ####
def homogeneity_compute(table, table_Datatype, direction="horizontal") -> float:
    """
        Compute the horizontal (column-wise)/vertical (row-wise) homogeneity.
        The homogeneity represents the uniqueness of a data type across the table's lines.
        Args:
            table: input table
            direction: horizontal/vertical
        Return:
            mean and std of the homogeneity across the table's lines. .

    """
    computed_table = []
    if direction == "horizontal":
        computed_table = table
    elif direction == "vertical":
        computed_table = transpose_heterogeneous_table(table)

    ## compute homogeneity for each line
    homogeneity_per_line = []
    for line in computed_table:
        dtypes = {}
        sum_type = 0
        for cell in line:
            if cell in table_Datatype:
                for a_dt in table_Datatype[cell]:
                    dtypes[a_dt] = dtypes.get(a_dt, 0) + 1
                sum_type += 1    
        for cell in line:
            if len(table_Datatype.get(cell, [])) > 1:
                sorted_ts = sorted(table_Datatype[cell], key=lambda x: (dtypes[x], typing_priority(x)), reverse=True)
                for other_type in sorted_ts[1:]:
                    dtypes[other_type] -= 1
                    if dtypes[other_type] == 0:
                        dtypes.pop(other_type)

        ## if a line contains too much missing value, it is not trustable to take it into
        ## count in homogeneity calculation.
        if sum_type/len(line) >= 0.25:
            type_homoCoef = 0
            for t in dtypes: 
                popularity_score = 1 - np.square(1 - 2 * (dtypes[t] / sum_type))
                type_homoCoef = type_homoCoef + popularity_score               
            homogeneity = np.square(type_homoCoef / len(dtypes)) 
            homogeneity_per_line.append(homogeneity)
    
    ## it is not trustable to  
    if len(homogeneity_per_line) > 1:
        ## compute mean and std accross different lines
        mean_homogeneity = np.mean(homogeneity_per_line)
        std_homogeneity = np.std(homogeneity_per_line, ddof=1)
        return mean_homogeneity, std_homogeneity
    else:
        return None, None

def std_column_wordLength(table, direction="horizontal") -> float:
    computed_table = []
    if direction == "horizontal":
        computed_table = table
    elif direction == "vertical":
        computed_table = transpose_heterogeneous_table(table)

    standardDeviationofAllRows = []
    for line in computed_table:
        cell_lens = []
        for cell in line:
            if cell:
                cell_lens.append(len(cell))
        if 2*len(cell_lens) >= len(line):
            standardDeviationofAllRows.append(np.std(cell_lens))
    if standardDeviationofAllRows:      
        return np.mean(standardDeviationofAllRows)
    else:
        return 0

#### Realignment-related functions ####
def re_align_short_row(line, table_Datatype, column_dataTypes):
    ''' TODO '''

    dtype_of_currline = []
    for element in line:
        if element in table_Datatype:
            dtype_of_currline.append(table_Datatype[element])
        else:
            dtype_of_currline.append("")

    alignable = True
    if "" in dtype_of_currline:
        alignable = False
    else:
        for col_idx, col_datatype in column_dataTypes.items():
            if col_datatype[0]["type"] == "" or col_datatype[0]["score"] < 0.75:
                alignable = False
                break
    if alignable:
        index_comb = combinations(range(len(column_dataTypes)), len(line))
        index_comb = list(index_comb)
        valid_alignments = []
        for idx_set in index_comb:
            target_col_types = [column_dataTypes[idx][0]["type"] for idx in idx_set]
            if target_col_types == dtype_of_currline:
                valid_alignments.append(idx_set)

        if len(valid_alignments) == 1:
            new_line = [""]*len(column_dataTypes)
            for idx, new_val in zip(valid_alignments[0], line):
                new_line[idx] = new_val
            return new_line
    return line
   
""" OUTPUT """
class Table_Informations(namedtuple(
    "Table_Information", ("raw_table", "structured_table", "title",
                          "orientation", "header", "keyColumn", "primitiveTyping"))):
    """ To allow for flexibility in returning different outputs.  """
    pass

def save_table_info(output_path, table_info):
    """ Save json file of table infos. """
    with open(output_path, 'w') as f:
        json.dump(table_info._asdict(), f)

def output_jsonlize(table_info):
    # convert named tupled table_info --> dictionary table_info
    # to better display in Json way.
    table_info_dict = { "raw_table": table_info.raw_table,
                        "restructured_table": table_info.structured_table,
                        "orientation": {"label": table_info.orientation.orientation, "score": round(table_info.orientation.score,2)},
                        "header":{"label": table_info.header.header, "score": round(table_info.header.score,2)},
                        "keyColumn": {"label": table_info.keyColumn.keyColumn, "score": round(table_info.keyColumn.score,2)},
                        "primitiveTyping": [{"column": i_col, "typing": [{"label": t["type"], "score": round(t["score"],2)} for t in ts]} for i_col, ts in table_info.primitiveTyping.type_list.items()]
                      }
    return table_info_dict

if __name__ == '__main__':
    # table = [["Col 0","Col 1","Col 2","Col 3"],
    #           ["1.","Nguyen An","an@gmail.com","0654893215"],
    #           ["2.","Tran Binh","binh@orange.com","+33(0)624759812"],
    #           ["3.","Huynh Cong","cong@eurecom.fr","+840641896315"],
    #           ["12","1888","6 kilo","093-456-123"]]
    table = [["United States", "2015 National Women's Soccer League", "FC Kansas City" , "2nd", "2014"]]
    print(parse_table([["vietphi.huynh@orange.com", "2 m/s", "Orange Labs", "France"], ["(2-3)"]]))
    # print(parse_table(table))
    # table_Typing,table_Datatype = parse_table(table)
    # print(table_Typing)
    # for line in transpose_heterogeneous_table(table):
    # print(typing_per_column(table[1:], table_Typing, 3))
    # typing_per_column(table[1:], table_Typing, 3)



           