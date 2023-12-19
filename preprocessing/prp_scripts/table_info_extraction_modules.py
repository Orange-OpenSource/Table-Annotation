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
TABLE INFO EXTRACTION MODULES/
+ Orientation
+ Header 
+ Primitive Typing
+ Data Type
+ Key Column 

'''
import numpy as np
import collections
from typing import List
from . import utils

__all__ = ["table_orientation_detection", "table_header_detection", "table_reshaping",
                "table_primitive_typing", "table_key_column_detection"]

class Table_Orientation(collections.namedtuple(
    "Table_Orientation", ("orientation", "score"))):
    """ To allow for flexibility in returning different outputs. """
    pass

class Table_Header(collections.namedtuple(
    "Table_Header", ("has_header", "header", "score"))):
    """ To allow for flexibility in returning different outputs. """
    pass

class Table_primitive_Typing(collections.namedtuple(
    "Table_primitive_Typing", ("type_list"))):
    """ To allow for flexibility in returning different outputs. """
    pass

class Table_keyColumn(collections.namedtuple(
    "Table_keyColumn", ("keyColumn", "score"))):
    """ To allow for flexibility in returning different outputs. """
    pass

def table_orientation_detection(targetTable: List[List[str]], table_Datatype, table_Typings) -> Table_Orientation:
    """
        Detect the orientation of a table based on column-wise and row-wise dataType homogeneity.
        The homogeneities are calculated starting from the second row and second column, 
            to avoid the unexpected affect of the header.
        Args:
            targetTable: 2D nested list
            ignore_first_row_col: set True by default. If False, first row/rol are also taken into 
                account in calculation of the homogeneity.

        Return:
            Orientation with a confidence score. 
            Orientation can be "VERTICAL", "HORIZONTAL", "Unknow/In this case, we assume HORIZONTAL since it
                                                          is the case usual.
    """
    orientation = ""
    orientation_score = 0.0
    smooth_coef = 0.0
    table_rows = len(targetTable)
    table_cols = len(targetTable[0])

    ## step 1: ignoring first row/first column. The horizontal/vertical data type based heterogeneity score are calculated.
    ## to assure a confidential homogeneity calculation, table shoud be large enough (num_row, num_col > 2)
    is_step1_success = False
    if table_rows > 2 and table_cols > 2:
        starting_row = 1          
        starting_col = 1
        ## homogeneity mean and std for each row
        homogeneity_horizontal, std_horiz = utils.homogeneity_compute([line[starting_col:] for line in targetTable[starting_row:]], table_Datatype,
                                                            direction="horizontal")

        ## homogeneity mean and std for each column
        homogeneity_vertical, std_verti = utils.homogeneity_compute([line[starting_col:] for line in targetTable[starting_row:]], table_Datatype,
                                                        direction="vertical")
        if homogeneity_horizontal is not None and homogeneity_vertical is not None:
            ## To alleviate the impact of noise, we employ a soft margin in the comparison of horizontal/vertical homogeneity 
            ## by considering lower- and upper-confidence bounds.
            ## a confidence score is computed as a function of the mean and dev of homogeneity.
            if homogeneity_horizontal + 0.5*std_horiz/table_rows**0.5 + 0.01 < homogeneity_vertical - 0.5*std_verti/table_cols**0.5 :
                is_step1_success = True
                if homogeneity_horizontal < 0.1:
                    smooth_coef = 0.1 ## smooth coef is necessary to avoid resulting a high confidence score
                                    ## when both horizontal and vertical homogeneity are too small (<0.1)
                                    
                y_score = homogeneity_vertical - 0.5*std_verti/table_cols**0.5
                x_score = homogeneity_horizontal + 0.5*std_horiz/table_rows**0.5
                orientation = "VERTICAL"
                orientation_score = (y_score - x_score)/(y_score + smooth_coef)
                

            elif homogeneity_horizontal - 0.5*std_horiz/table_rows**0.5 >= homogeneity_vertical + 0.5*std_verti/table_cols**0.5 + 0.01 :
                is_step1_success = True
                if homogeneity_vertical < 0.1:
                    smooth_coef = 0.1
                y_score = homogeneity_vertical + 0.5*std_verti/table_cols**0.5
                x_score = homogeneity_horizontal - 0.5*std_horiz/table_rows**0.5
                orientation = "HORIZONTAL"
                orientation_score = (x_score-y_score)/(x_score + smooth_coef)
    else:
        orientation = "HORIZONTAL"
        orientation_score = 0.1     
        is_step1_success = True
    
    if not is_step1_success:
        ## step 2: working on first row/column. We impose a strict constraint: a header can not contain cells exposing a primitive typing. 
        ## store typing of each cell in top row
        top_row_typings = []
        for element in targetTable[0][1:]:
            if element in table_Typings:
                top_row_typings.append(table_Typings[element])
            else:
                top_row_typings.append("")

        ## store typing of each cell in left column
        left_col_typings = []
        for line in targetTable[1:]:
            element = line[0]
            if element in table_Typings:
                left_col_typings.append(table_Typings[element])
            else:
                left_col_typings.append("")
        ## count how many cells in top row that contain a typing.
        ratio_header_related_typing_top_row = 0
        for ts in top_row_typings:
            for t in ts:
                if t not in ["", "UNKNOWN"]:
                    ratio_header_related_typing_top_row += 1
                    break
        if top_row_typings:
            ratio_header_related_typing_top_row /= len(top_row_typings)
        else:
            ratio_header_related_typing_top_row = 0.0

        ## count how many cells in left col that contain a typing.
        ratio_header_related_typing_left_col = 0
        for ts in left_col_typings:
            for t in ts:
                if t not in ["", "UNKNOWN"]:
                    ratio_header_related_typing_left_col += 1
                    break
        if left_col_typings:
            ratio_header_related_typing_left_col /= len(left_col_typings)
        else:
            ratio_header_related_typing_left_col = 0.0

        ## If first column has nearly no primitive typings in its cells and first row has significant number of primitive typings in its cells
        if ratio_header_related_typing_top_row > 0.5 and ratio_header_related_typing_left_col < 0.05:
            orientation = "VERTICAL"
            orientation_score = 0.2
        ## If first row has nearly no primitive typings in its cells and first column has significant number of primitive typings in its cells
        elif ratio_header_related_typing_left_col > 0.5 and ratio_header_related_typing_top_row < 0.05:
            orientation = "HORIZONTAL"
            orientation_score = 0.2
        else:
            ## step 3:  working on complete primitive typings of the all table cells. 
            # We rely on the assumption that if a table is incorrectly oriented, no column in it has a consistent (homogenous) primitive typing, 
            # whereas if the table is correctly oriented, it has at least 1 column from which a consistent primitive typing can be retrieved.
            horizontal_typings = table_primitive_typing(targetTable, table_Typings, top_k = 1)
            vertical_typings = table_primitive_typing(utils.transpose_heterogeneous_table(targetTable), table_Typings, top_k = 1)

            ## check if horizontal table contains at least 1 column from which a consistent primitive typing can be retrieved.
            homo_hori_typing_exist = False
            for col_idx, typings in horizontal_typings.type_list.items():
                if typings[0]["type"] not in ["", "UNKNOWN"] and typings[0]["score"] > 0.8:
                    homo_hori_typing_exist = True
                    break
            ## check if vertical table contains at least 1 column from which a consistent primitive typing can be retrieved.
            homo_verti_typing_exist = False
            for col_idx, typings in vertical_typings.type_list.items():
                if typings[0]["type"] not in ["", "UNKNOWN"] and typings[0]["score"] > 0.8:
                    homo_verti_typing_exist = True
                    break
            ## decide orientation 
            if table_rows > 2 and table_cols > 2 and homo_hori_typing_exist and not homo_verti_typing_exist:
                orientation = "HORIZONTAL"
                orientation_score = 0.15 
            elif table_rows > 2 and table_cols > 2 and homo_verti_typing_exist and not homo_hori_typing_exist:     
                orientation = "VERTICAL"
                orientation_score = 0.15    
            else:            
                ## step 4: a very long+thin table is horizontal and a very short+fat table is vertical.
                if table_rows/table_cols <= 0.25 or table_rows/table_cols >= 4.0:
                    if table_rows >= table_cols:
                        orientation = "HORIZONTAL"
                        orientation_score = 0.1
                    else:
                        orientation = "VERTICAL"
                        orientation_score = 0.1
                else:
                    ## step 4: WTC string length-based calculation.
                    std_row_wordLength = utils.std_column_wordLength(targetTable, 
                                                                            direction="horizontal")  
                    std_col_wordLength = utils.std_column_wordLength(targetTable, 
                                                                            direction="vertical")    
                    print(std_row_wordLength, std_col_wordLength)
                    if std_row_wordLength >= std_col_wordLength:
                        orientation = "HORIZONTAL"
                        orientation_score = 0.1
                    else:             
                        orientation = "VERTICAL"
                        orientation_score = 0.1
                
    return Table_Orientation(orientation=orientation,
                                     score=orientation_score)

def table_header_detection(targetTable: List[List[str]], table_orientation_score, table_Typings) -> Table_Header:
    """
    Header detection using primitive typings. We impose a strict constraint: a header can not contain cells exposing a primitive typing. 
    """
    ## we currently consider 2 cases: no_header or single header (first table row)
    potential_header = targetTable[0]
    ## get primitive typing for each cell in first table row which is potential header.
    potential_header_typings = []
    for element in potential_header:
        if element in table_Typings:
            potential_header_typings.append(table_Typings[element])
        else:
            potential_header_typings.append("")
    ## get primitive typings represent for each column in table.
    if len(targetTable) > 1:        
        column_typings = utils.typing_per_column(targetTable[1:], table_Typings, 3)
    else:
        column_typings = utils.typing_per_column(targetTable, table_Typings, 3)
    ## Verify if first row is header by assumption: a header can not contain cells exposing a primitive typing. 
    noheader_score = 0.0
    for i_col, typings in column_typings.items():
        if potential_header_typings[i_col]:
            ## first case: if primitive typing of header cell is UNIT, MISC (not NAMED ENTITY).
            ##      if header cell and its column has same primitive typing. (threshold is set to low value 0.2 since UNIT, MISC is reliably parsed.)
            if sum([utils.is_concept(t) for t in potential_header_typings[i_col]]) == 0:
                if typings[0]["type"] in potential_header_typings[i_col] and typings[0]["score"] > 0.2:
                    noheader_score = max(noheader_score, typings[0]["score"])
            ## second case: if primitive typing of header cell is NAMED ENTITY, but we do not consider UNKNOWN and PERSON since UNKNOWN contains no information and PERSON is high false positive.
            ##      if header cell and its column has same primitive typing. (threshold is set to low value 0.2 since NAMED ENTITY is reliably parsed.)
            elif "UNKNOWN" not in potential_header_typings[i_col] and "PERSON" not in potential_header_typings[i_col]:
                if typings[0]["type"] in potential_header_typings[i_col] and typings[0]["score"] > 0.2:
                    noheader_score = max(noheader_score, typings[0]["score"])

    if noheader_score > 0.0:
        ## noheader detected.
        return Table_Header(has_header=False, header=[], score=noheader_score*table_orientation_score)
    else:
        ## header exist.
        hasheader_score = 0.0
        for i_col, typings in column_typings.items():
            if potential_header_typings[i_col]:
                for dt in typings:
                    if dt["type"] not in potential_header_typings[i_col]:              
                        hasheader_score += dt["score"]
        hasheader_score = hasheader_score / len(column_typings)       
        return Table_Header(has_header=True, header=potential_header, score=hasheader_score*table_orientation_score)

def table_primitive_typing(targetTable: List[List[str]], table_Typing, top_k: int = 1) -> Table_primitive_Typing:
    """
        Return the primitive typing (generic type + specific type) for each table column.
        Args:
            targetTable: 2D input table
            top_k: return top k frequent types.
        Return:
            two type dictionaries (generic and specific) under format: {"col_idx": types}
    """
    if len(targetTable) > 1:
        pri_typing_per_col = utils.typing_per_column(targetTable[1:], table_Typing, top_k)
    else:
        pri_typing_per_col = utils.typing_per_column(targetTable, table_Typing, top_k)
    return Table_primitive_Typing(type_list=pri_typing_per_col)

def table_key_column_detection(targetTable: List[List[str]], table_orientation_score, table_Datatype) -> Table_keyColumn:
    """
        The main assumption is that the key column covers a large amount of unique cell values.
        In addition to the requirement of having at least 50% unique values, it must be a column
        consisting primarily of objects (string values) ​​and an average length greater than 3.5% and less than 200.
        Args:
            targetTable: horizontal input table
            tablePrimitiveTyping: column typing used to decide whether the column is a potential key column.
                                  if typing is Object: Yes, otherwise, No
        Return:
            key column index and confidence score
    """
    if len(targetTable) > 1:
        column_dataTypes = utils.datatype_per_column(targetTable[1:], table_Datatype, 3)
    else:
        column_dataTypes = utils.datatype_per_column(targetTable, table_Datatype, 3)
    targetTable_T = utils.transpose_heterogeneous_table(targetTable)
    ## compute the uniqueness of each column
    column_scores = {}
    first_keyCol_candidate = None

    num_considered_cols = 0
    max_conisdered_cols = 2
    if len(targetTable_T) > 8:
        max_conisdered_cols = 3

    for col_idx, column in enumerate(targetTable_T):
        if column_dataTypes[col_idx][0]["type"]:
            if num_considered_cols <= max_conisdered_cols:
                num_considered_cols += 1
                keyCol_candidate_score = 0
                for dtype in column_dataTypes[col_idx]:
                    if utils.keyColumn_related_datatype(dtype["type"]):
                        keyCol_candidate_score += dtype["score"] 
                if keyCol_candidate_score > 0.5:
                    if not isinstance(first_keyCol_candidate, int):
                        first_keyCol_candidate = col_idx
                    unique_contents = []
                    empty_cells = []
                    number_of_words_per_cell = []
                    for cell in column:
                        number_of_words_per_cell.append(len([word for word in cell.split(" ") if word]))
                        if cell in table_Datatype:
                            new_cell = cell
                            for s in ".@_!#$%^&*()<>?/\|}{][~:\'-+~~_°¨":
                                new_cell = new_cell.replace(s, '')
                            for dtype in table_Datatype[cell]:
                                if utils.keyColumn_related_datatype(dtype) and (3 < len(new_cell) < 200):
                                    unique_contents.append(cell)
                                    break
                        else:
                            empty_cells.append(cell)
                    if unique_contents:
                        ratio_unique_content = len(set(unique_contents)) / len(column)
                        ratio_empty_cells = len(empty_cells) / len(column)
                        # avg_words_per_cell = sum(number_of_words_per_cell)/len(number_of_words_per_cell)
                        # column_scores[col_idx] = (3*ratio_unique_content+avg_words_per_cell-ratio_empty_cells)/np.sqrt(1+(col_idx-first_keyCol_candidate))
                        column_scores[col_idx] = (ratio_unique_content - ratio_empty_cells)/np.sqrt(1+2*(col_idx-first_keyCol_candidate))
                    else:
                        column_scores[col_idx] = 0.0
                else:
                    column_scores[col_idx] = 0.00
    ## sorted the list of uniqueness. In case of a lot of max_score, smaller index is more prefered.
    if column_scores:
        if len(column_scores) > 1:
            (key_col, max_score), (_, second_max_score) = sorted(column_scores.items(), key=lambda i: i[1], reverse=True)[:2]
            if max_score < 0.25: ## untrustable detection.
                return Table_keyColumn(keyColumn=None, score=0.0)
            else:
                return Table_keyColumn(keyColumn=key_col, score=(max_score - second_max_score)/(max_score+second_max_score)*table_orientation_score)
        else:
            if list(column_scores.items())[0][1] < 0.25: ## untrustable detection.
                return Table_keyColumn(keyColumn=None, score=0.0)
            else:

                return Table_keyColumn(keyColumn=list(column_scores.items())[0][0], score=table_orientation_score)
    else:
        return Table_keyColumn(keyColumn=None, score=0.0)

def table_reshaping(targetTable: List[List[str]], table_Datatype, table_Typing) -> List[List[str]]:
    """
        If table is not well-shaped (heterogeneous row lengths, col lengths), we try
        to reshape it padding "" or aligning short lines.
        Args:
            input table
        Return:
            reshaped table
    """
    list_row_lens = [len(row) for row in targetTable]
    if (min(list_row_lens) != max(list_row_lens)):
        tab_width = max(set(list_row_lens), key=list_row_lens.count)
        reduced_table = []
        for line in targetTable:
            if len(line) == tab_width:
                reduced_table.append(line)
        reshaped_table = []
        if len(reduced_table) > 1:
            ### Orientation detection on well-shaped part of table ###
            tableOrientation = table_orientation_detection(reduced_table, table_Datatype, table_Typing)   
            if tableOrientation.orientation == "HORIZONTAL":
                ### First row is potential header ###
                reshaped_table.append(targetTable[0])

                ### Column Datatype & Primitive Typing on well-shaped part of table ###
                column_dataTypes = utils.datatype_per_column(targetTable[1:], table_Datatype)

                ### Reshape content of Table ###
                for line in targetTable[1:]:       
                    if len(line) < tab_width:                  
                        ### try to reshape short line ###
                        new_line = utils.re_align_short_row(line, table_Datatype, column_dataTypes)  
                        reshaped_table.append(new_line)     
                    ### simply append other lines ###                                                                                
                    else:
                        reshaped_table.append(line)

            else:
                reshaped_table = targetTable
        else:
            reshaped_table = targetTable
        ## padding "" for unsolved short lines.
        utils.table_padding(reshaped_table, max(list_row_lens))
        return reshaped_table
    else:
        return targetTable

if __name__ == '__main__':
    pass

