# Table Annotation

The semantic annotation process of a table, performed by DAGOBAH, involves 3 steps:
1. <b>Table Preprocessing</b>: a set of comprehensive heuristic to clean the table (e.g. fix encoding error), determine table orientation, data types of columns.
2. <b>Entity Lookup</b>: retrieve a number of entity candidates for mentions in the table, using an elastic search-based entity lookup.
3. <b>Annotation</b>: disambiguate retrieved entity candidates, select the most relevant entity for each mention. This consists of three tasks:
    - Cell-Entity Annotation (CEA): assign an entity in a KG to a cell mention.
    - Column-Type Annotation (CTA): predicts semantic types for a column
    - Column-Pair Annotation (CPA): represents the relationship between two columns with property in KG

# Installation

- Environment: 

    ```conda create -n dagobah python=3.10```

-  <b>Indexing the entity label dump using [elasticsearch](https://github.com/elastic/elasticsearch-py) and KG dump using [lmdb](https://lmdb.readthedocs.io/en/release/):</b>

    *Resources requirement:* 500GB SSD.

    ```docker-compose up -d```

    It will download the dumps from [zenodo](https://zenodo.org/records/8426650) and index them using scripts from ```./table-annotation/data/```.

    This step will take several hours to finish (12-24h).

    Once finished, the elasticsearch endpoint for entity lookup is available at ```localhost:9200``` and the KG hashmap is accessible at ```./table-annotation/data/hashmap/```

- Install the system:

    Run 

    ```python setup.py install```

    then, download spaCy NER model, required by table preprocessing.
    ```python -m spacy download en_core_web_sm```

### Examples

- <b>Entity Lookup</b>:

    ```
    from lookup import entity_lookup
    entity_lookup(labels=["MUFC"], KG="dagobah_lookup")

    Output:
    {'executionTimeSec': 3.5, 'output': [{'label': 'MUFC', 'entities': [{'entity': 'Q18656', 'label': 'MUFC', 'score': 0.9314198500543867, 'origin': 'MAIN_ALIAS'}, {'entity': 'Q1764590', 'label': 'MUFC', 'score': 0.886259396137413, 'origin': 'MAIN_ALIAS'}, {'entity': 'Q1131109', 'label': 'MUFC', 'score': 0.8855865364578572, 'origin': 'MAIN_ALIAS'}, {'entity': 'Q19828435', 'label': 'MNUFC', 'score': 0.7682839133054579, 'origin': 'MAIN_ALIAS'}, ,...]}]}
    ```

- <b>Table Preprocessing</b>:

    ```
    from preprocessing import table_preprocessing
    table_preprocessing([["city", "country"],["Paris", "France"], ["Berlin", "Germany"], ["Madrid", "Spain"], ["Rome", "Italy"]])

    Output:
    {'raw': {'tableDataRaw': [['city', 'country'], ['Paris', 'France'], ['Berlin', 'Germany'], ['Madrid', 'Spain'], ['Rome', 'Italy']]}, 'preprocessed': {'tableDataRevised': [['city', 'country'], ['Paris', 'France'], ['Berlin', 'Germany'], ['Madrid', 'Spain'], ['Rome', 'Italy']], 'tableOrientation': {'orientationLabel': 'HORIZONTAL', 'orientationScore': 0.1}, 'headerInfo': {'hasHeader': True, 'headerPosition': 0, 'headerLabel': ['city', 'country'], 'headerScore': 0.09}, 'primaryKeyInfo': {'hasPrimaryKey': True, 'primaryKeyPosition': 0, 'primaryKeyScore': 0.03}, 'primitiveTyping': [{'columnIndex': 0, 'typing': [{'typingLabel': 'GPE', 'typingScore': 0.75}, {'typingLabel': 'UNKNOWN', 'typingScore': 0.25}]}, {'columnIndex': 1, 'typing': [{'typingLabel': 'GPE', 'typingScore': 1.0}]}]}}
    ```

- <b>Table Annotation</b>:
    ```
    from annotation import table_annotation
    table_annotation([["Title","Year","Cast","col3"], ["Pulp Fiction","1994","John Travolta","Gangster"], ["Casino Royale","1967","David Niven","James Bond"], ["Outsiders","1983","Matt Dillon","Drama"], ["Hearts of Darkness: A Filmmakers Apocalypse","1991","Marlon Brando","Docmuentary"], ["Virgin Suicides","1999","Kristen Dunst","Drama"]])

    Output:
    { "annotated": { "CEA": [ { "annotation": { "label": "Pulp Fiction", "score": 0.93, "uri": "https://www.wikidata.org/wiki/Q104123" }, "column": 0, "row": 1 }, { "annotation": { "label": "Casino Royale", "score": 0.98, "uri": "https://www.wikidata.org/wiki/Q591272" }, "column": 0, "row": 2 }, { "annotation": { "label": "The Outsiders", "score": 0.99, "uri": "https://www.wikidata.org/wiki/Q1055332" }, "column": 0, "row": 3 }, { "annotation": { "label": "Hearts of Darkness: A Filmmaker's Apocalypse", "score": 0.9, "uri": "https://www.wikidata.org/wiki/Q1962835" }, "column": 0, "row": 4 }, { "annotation": { "label": "The Virgin Suicides", "score": 0.95, "uri": "https://www.wikidata.org/wiki/Q1423971" }, "column": 0, "row": 5 }, { "annotation": { "label": "John Travolta", "score": 0.94, "uri": "https://www.wikidata.org/wiki/Q80938" }, "column": 2, "row": 1 }, { "annotation": { "label": "David Niven", "score": 0.94, "uri": "https://www.wikidata.org/wiki/Q181917" }, "column": 2, "row": 2 }, { "annotation": { "label": "Matt Dillon", "score": 0.91, "uri": "https://www.wikidata.org/wiki/Q193070" }, "column": 2, "row": 3 }, { "annotation": { "label": "Marlon Brando", "score": 0.9, "uri": "https://www.wikidata.org/wiki/Q34012" }, "column": 2, "row": 4 }, { "annotation": { "label": "Kirsten Dunst", "score": 0.81, "uri": "https://www.wikidata.org/wiki/Q76478" }, "column": 2, "row": 5 }, { "annotation": { "label": "gangster film", "score": 0.67, "uri": "https://www.wikidata.org/wiki/Q7444356" }, "column": 3, "row": 1 }, { "annotation": { "label": "James Bond", "score": 0.78, "uri": "https://www.wikidata.org/wiki/Q2009573" }, "column": 3, "row": 2 }, { "annotation": { "label": "drama", "score": 0.87, "uri": "https://www.wikidata.org/wiki/Q130232" }, "column": 3, "row": 3 }, { "annotation": { "label": "documentary film", "score": 0.64, "uri": "https://www.wikidata.org/wiki/Q93204" }, "column": 3, "row": 4 }, { "annotation": { "label": "drama", "score": 0.87, "uri": "https://www.wikidata.org/wiki/Q130232" }, "column": 3, "row": 5 } ], "CPA": [ { "annotation": { "coverage": 1.0, "label": "cast member", "score": 0.95, "uri": "https://www.wikidata.org/wiki/Property:P161" }, "headColumn": 0, "tailColumn": 2 }, { "annotation": { "coverage": 0.8, "label": "genre", "score": 0.75, "uri": "https://www.wikidata.org/wiki/Property:P136" }, "headColumn": 0, "tailColumn": 3 }, { "annotation": { "coverage": 0.8, "label": "(-)cast member::genre", "score": 0.09, "uri": "(-)https://www.wikidata.org/wiki/Property:P161::https://www.wikidata.org/wiki/Property:P136" }, "headColumn": 2, "tailColumn": 3 }, { "annotation": { "coverage": 0.4, "label": "publication date", "score": 0.06, "uri": "https://www.wikidata.org/wiki/Property:P577" }, "headColumn": 0, "tailColumn": 1 }, { "annotation": { "coverage": 0.2, "label": "publication date", "score": 0.0, "uri": "https://www.wikidata.org/wiki/Property:P577" }, "headColumn": 3, "tailColumn": 1 } ], "CTA": [ { "annotation": [ { "coverage": 1.0, "label": "film", "score": 0.95, "uri": "https://www.wikidata.org/wiki/Q11424" } ], "column": 0 }, { "annotation": [ { "coverage": 1.0, "label": "human", "score": 0.9, "uri": "https://www.wikidata.org/wiki/Q5" } ], "column": 2 }, { "annotation": [ { "coverage": 0.8, "label": "film genre", "score": 0.64, "uri": "https://www.wikidata.org/wiki/Q201658" } ], "column": 3 } ] }, "raw": { "tableContent": null, "tableEndOffset": null, "tableNum": null, "tableOffset": null }, "requestInfo": { "id": 1 } }
    ```
### Citation

```
@inproceedings{huynh2022heuristics,
  title={{From Heuristics to Language Models: A Journey Through the Universe of Semantic Table Interpretation with DAGOBAH}},
  author={Huynh, Viet-Phi and Chabot, Yoan and Labb{\'e}, Thomas and Liu, Jixiong and Troncy, Rapha{\"e}l},
  booktitle={Semantic Web Challenge on Tabular Data to Knowledge Graph Matching (SemTab)},
  year={2022}
}

or

@inproceedings{huynh2021dagobah,
  title={{DAGOBAH: Table and Graph Contexts for Efficient Semantic Annotation of Tabular Data}},
  author={Huynh, Viet-Phi and Liu, Jixiong and Chabot, Yoan and Deuz{\'e}, Fr{\'e}d{\'e}ric and Labb{\'e}, Thomas and Monnin, Pierre and Troncy, Rapha{\"e}l},
  booktitle={Semantic Web Challenge on Tabular Data to Knowledge Graph Matching (SemTab)},
  year={2021}
}
```