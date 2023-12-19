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
from setuptools import setup

setup(
    name='Table Annotation',
    version='1.0',
    description='DAGOBAH: A toolkit for semantic table annotation using heuristics',
    author='Orange SA',
    license='TODO',
    packages=['annotation', 'lookup', 'preprocessing'],
    install_requires=[
        'ftfy==6.0.3',
        'numpy==1.22.0',
        'scipy==1.10.0',
        'pandas==1.4.0',
        'quantulum3==0.7.9',
        'stemming==1.0.1',
        'phonenumbers==8.12.22',
        'spacy==3.7.2',
        'pydantic==2.5.2',
        'typing-extensions>=4.6.1',
        'tqdm==4.60.0',
        'lmdb==1.3.0',
        'rapidfuzz==1.9.1',
        'quantulum3==0.7.9',
        'openpyxl==3.0.9',
        'Pint==0.18',
        'Unidecode==1.3.4',
        'elasticsearch==7.15.1',
    ]
)