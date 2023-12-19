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
MAIN PRE-PROCESSING
'''
from typing import List
from random import shuffle

from .prp_scripts import table_info_extraction_modules as tb_modu
from .prp_scripts import utils as tb_utils

def table_preprocessing(raw_table: List[List[str]]) :
    """
        Extracting basic information of a table: 
            + Potential Reshaping, Filtering table
            + Orientation
            + Header
            + Column Primitive Typing
            + Key Column

    """
    preprocessing_output = {"raw": {
                                    "tableDataRaw": raw_table
                                },
                            "preprocessed": {}}

    ## filtering table (encoding correction, blank line removing)
    table,_ = tb_utils.table_filtering(raw_table)
    if len(table) > 1:
        ## if table is two large, says > 400 lines (horizontal) or > 400 columns (vertical), 
        ##     random sub-sampling the table to 400 lines (or 400 columns) maximum to assure a reasonable preprocessing time (~<60s)
        sample_table = table
        if len(table) > 400: 
            sample_index = list(range(10,len(table)))
            shuffle(sample_index)
            sample_index = list(range(10)) + sample_index[:390] ## avoid sampling in first 10 lines due to existence of header.
            sample_index = sorted(sample_index)
            sample_table = [table[i] for i in sample_index]
            
        ### Table parsing: extract entities + datatypes ###
        tb_entity_list, tb_dataType_list = tb_utils.parse_table(sample_table)

        # print(tb_entity_list)
        ### Potentially reshaping exotic table ### TODO
        # table = tb_modu.table_reshaping(table, tb_dataType_list, tb_entity_list)
        ### Removing null column ### TODO
        # table = tb_utils.table_null_column_removing(table)

        ### Extracting table information ###
        ## Orientation
        tableOrientation = tb_modu.table_orientation_detection(sample_table, tb_dataType_list, tb_entity_list)      
        if tableOrientation.orientation == "VERTICAL":
            sample_table = tb_utils.transpose_heterogeneous_table(sample_table)
            table = tb_utils.transpose_heterogeneous_table(table)
        ## Column Primitive Typing
        tablePrimitiveTyping = tb_modu.table_primitive_typing(sample_table, tb_entity_list, top_k = 3)

        ## Key Column
        tableKeyColumn = tb_modu.table_key_column_detection(sample_table, tableOrientation.score, tb_dataType_list)               

        ## Header
        tableHeader = tb_modu.table_header_detection(sample_table, tableOrientation.score, tb_entity_list)

        preprocessing_output["preprocessed"]["tableDataRevised"] = table
        preprocessing_output["preprocessed"]["tableOrientation"] = {"orientationLabel": tableOrientation.orientation,
                                                                     "orientationScore": round(tableOrientation.score,2)
                                                                    }

        preprocessing_output["preprocessed"]["headerInfo"] = {"hasHeader": tableHeader.has_header,
                                                               "headerPosition": 0 if tableHeader.has_header else None,
                                                               "headerLabel": tableHeader.header,
                                                               "headerScore": round(tableHeader.score,2)
                                                              }

        preprocessing_output["preprocessed"]["primaryKeyInfo"] = {"hasPrimaryKey": bool(tableKeyColumn.keyColumn is not None),
                                                                  "primaryKeyPosition": tableKeyColumn.keyColumn,
                                                                  "primaryKeyScore": round(tableKeyColumn.score,2) 
                                                                 }

        preprocessing_output["preprocessed"]["primitiveTyping"] = [{"columnIndex": i_col, 
                                                                    "typing": [{"typingLabel": t["type"], "typingScore": round(t["score"],2)} for t in ts]} for i_col, ts in tablePrimitiveTyping.type_list.items()] 
                                                                        
    return preprocessing_output

if __name__ == "__main__":
    print("dkm")
