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
from .annot_scripts.annotation_models import Baseline_Model

def table_annotation(raw_table, K=20, kb_path="./data/hashmap", lookup_index="dagobah_lookup"):
	"""	
		Main table annotation:
		Input:
			+ 2D table
			+ K: number of entity candidates per table mention.
			+ kb_path: path to the KB hashmap.
			+ lookup_index: the index name of ES lookup.
	"""
	target_kb = {"kb_path": kb_path, "lookup_index": lookup_index}
	annotation_output = {"raw": {
                                    "tableDataRaw": raw_table
								},
						"annotated": {},
						"preprocessed": {},
						"preprocessingTime": 0.0,
						"lookupTime": 0.0,
						"entityScoringTime": 0.0,
						"subgraphConstructionTime": 0.0,
						"ctaTaskTime": 0.0,
						"ceaTaskTime": 0.0,
						"cpaTaskTime": 0.0,
						"avgLookupCandidate": 0.0}	

	params = {"multiHop_context": True, "transitivePropertyOnly_path": False, "soft_scoring": True, "K": K}
	baseline_model = Baseline_Model(table=raw_table, target_kb=target_kb, params=params)
	## record the size of subgraphs. Disabled in production due to time consuming.
	if baseline_model.is_model_init_success:
		revised_table = baseline_model.table
		baseline_model.entity_scoring_task()
		## First annotation loop
		# CEA
		for col_idx in baseline_model.entity_cols:
			for row_idx in range(baseline_model.first_data_row, baseline_model.num_rows):
				baseline_model.cea_task(col_index=col_idx, row_index=row_idx, only_one=False)
		# CPA
		for i in range(len(baseline_model.entity_cols)-1):
			head_col = baseline_model.entity_cols[i]
			for j in range(i+1, len(baseline_model.entity_cols)):
				tail_col = baseline_model.entity_cols[j]
				baseline_model.cpa_task(head_col_index=head_col,tail_col_index=tail_col, only_one=False)
		for head_col in baseline_model.entity_cols:
			for tail_col in baseline_model.literal_cols:
				baseline_model.cpa_task(head_col_index=head_col,tail_col_index=tail_col, only_one=False)  
		# Weight update: soft scoring
		baseline_model.update_context_weight()
		baseline_model.entity_scoring_task(first_step=False)
		## Second annotation loop: with updated score.
		baseline_model.cea_annot = {}
		for col_idx in baseline_model.entity_cols:
			for row_idx in range(baseline_model.first_data_row, baseline_model.num_rows):
				baseline_model.cea_task(col_index=col_idx, row_index=row_idx, only_one=False)
		for col_idx in baseline_model.entity_cols:
			baseline_model.cta_task(col_index=col_idx, only_one=False)
		## Third annotation loop: disambiguation.
		baseline_model.cea_annot = {}
		for col_idx in baseline_model.entity_cols:
			for row_idx in range(baseline_model.first_data_row, baseline_model.num_rows):
				baseline_model.cea_task(col_index=col_idx, row_index=row_idx, only_one=True)
		baseline_model.cta_annot = {}
		for col_idx in baseline_model.entity_cols:
			baseline_model.cta_task(col_index=col_idx, only_one=True)
		baseline_model.cpa_annot = {}
		for i in range(len(baseline_model.entity_cols)-1):
			head_col = baseline_model.entity_cols[i]
			for j in range(i+1, len(baseline_model.entity_cols)):
				tail_col = baseline_model.entity_cols[j]
				baseline_model.cpa_task(head_col_index=head_col,tail_col_index=tail_col, only_one=False)
		for head_col in baseline_model.entity_cols:
			for tail_col in baseline_model.literal_cols:
				baseline_model.cpa_task(head_col_index=head_col,tail_col_index=tail_col, only_one=False)
	
		# Fourth annotation loop: reinforced disambiguation
		baseline_model.update_context_weight(onlyLiteralContext=True)
		baseline_model.entity_scoring_task(first_step=False, last_step=True)
		baseline_model.cea_annot = {}
		for col_idx in baseline_model.entity_cols:
			for row_idx in range(baseline_model.first_data_row, baseline_model.num_rows):
				baseline_model.cea_task(col_index=col_idx, row_index=row_idx, only_one=True)
		baseline_model.cta_annot = {}
		for col_idx in baseline_model.entity_cols:
			baseline_model.cta_task(col_index=col_idx, only_one=True)
		baseline_model.cpa_annot = {}
		for i in range(len(baseline_model.entity_cols)-1):
			head_col = baseline_model.entity_cols[i]
			for j in range(i+1, len(baseline_model.entity_cols)):
				tail_col = baseline_model.entity_cols[j]
				baseline_model.cpa_task(head_col_index=head_col,tail_col_index=tail_col, only_one=True)
		for head_col in baseline_model.entity_cols:
			for tail_col in baseline_model.literal_cols:
				baseline_model.cpa_task(head_col_index=head_col,tail_col_index=tail_col, only_one=True)

		annotation_output["annotated"]["tableDataRevised"] = revised_table
		annotation_output["annotated"]["CEA"] = [{"row": cell.row_index, "column": cell.col_index, "annotation": {"label": baseline_model.KB.get_label_of_entity(cea[0]["id"], True), 
														"uri": baseline_model.KB.prefixing_entity(cea[0]["id"]),
															"score": round(cea[0]["score"],2)}} for cell, cea in baseline_model.cea_annot.items()]
		annotation_output["annotated"]["CTA"] = [{"column": col.col_index, "annotation": [{"label": baseline_model.KB.get_label_of_entity(cta["id"], True), 
													"uri":  baseline_model.KB.prefixing_entity(cta["id"]), "score": round(cta["score"],2), 
													"coverage": round(cta["coverage"],2)} for cta in cta_list]} for col, cta_list in baseline_model.cta_annot.items()]
		annotation_output["annotated"]["CPA"] = []
		for col_pair, cpa in baseline_model.cpa_annot.items():
			rel_id =  cpa[0]["id"]
			id_components = set(rel_id.replace("(-)", "").replace("(", "").replace(")", "").split("::"))
			rel_uri = rel_id
			rel_label = rel_id
			for a_id in id_components:
				if baseline_model.KB.is_valid_ID(a_id):
					rel_uri = rel_uri.replace(a_id, baseline_model.KB.prefixing_entity(a_id))
					rel_label = rel_label.replace(a_id,baseline_model.KB.get_label_of_entity(a_id))	
			annotation_output["annotated"]["CPA"].append({"headColumn": col_pair.head_col_index, "tailColumn": col_pair.tail_col_index, 
													"annotation": {"label": rel_label, "uri": rel_uri, "score": round(cpa[0]["score"],2), "coverage": round(cpa[0]["coverage"],2)}})

		annotation_output["preprocessed"] = baseline_model.preprocessing_infos["data"][0]["preprocessed"]
		annotation_output["preprocessed"].pop("tableDataRevised", None)
		annotation_output["preprocessingTime"] = baseline_model.preprocessing_time
		annotation_output["lookupTime"] = baseline_model.lookup_time
		annotation_output["entityScoringTime"] = baseline_model.entity_scoring_time
		annotation_output["subgraphConstructionTime"] = baseline_model.subgraph_construction_time
		annotation_output["ctaTaskTime"] = baseline_model.cta_task_time
		annotation_output["ceaTaskTime"] = baseline_model.cea_task_time
		annotation_output["cpaTaskTime"] = baseline_model.cpa_task_time
		annotation_output["avgLookupCandidate"] = baseline_model.avg_lookup_candidate

		## close graph readers
		baseline_model.KB.close_dump()

	return annotation_output
		
if __name__ == "__main__":
	table = [["Name", "Soundtrack", "Actors","Character"],
			 ["Pulp fiction", "Dick Dale", "Travolta", "Vincent Vega"], 
			 ["Casino Royal", "Chris Cornell", "Craig", "James Bond"],
			 ["Outsiders", "Carmine Coppola", "Dillon"],
	         ["Hearts of Darkness: A Filmmaker's Apocalypse","Todd Boekelheide","Coppola"],
	         ["Virgin Suicides","Thomas Mars","Dunst","Lux Lisbon"]] 

	print(table_annotation(raw_table=table, K=20, kb_path="./data/hashmap", lookup_index="dagobah_lookup"))



	


