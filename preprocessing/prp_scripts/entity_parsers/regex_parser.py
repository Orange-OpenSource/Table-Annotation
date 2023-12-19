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
import re

def init_parser():
    regex_matcher = {} 
    range_pattern = ["^[\s\[\{\(]*[\s]*\d+[.,]?\d*[\s]*[-]+[\s]*\d+[.,]?\d*[\s]*[\s\]\)\}]*$",
                     "^[\[\{\(]+[\s]*\d+[.,]?\d*[\s]*[,]+[\s]*\d+[.,]?\d*[\s]*[\s\]\)\}]*$",
                     "^[\s\[\{\(]*[\s]*\d+[.,]?\d*[\s]*[,]+[\s]*\d+[.,]?\d*[\s]*[\]\)\}]+$",
                     "^[\s\[\{\(]*[\s]*\d+[.,]?\d*[\s]*[–]+[\s]*\d+[.,]?\d*[\s]*[\s\]\)\}]*$"]

    range_matcher = re.compile('|'.join(range_pattern))
    regex_matcher["RANGE"] = range_matcher

    cardinal_matcher = re.compile(r"^\s*[+,-]?\d+[.,]?\d*\s*$|^\s*[+,-]?\d*[\u2150-\u215E\u00BC-\u00BE]\s*$")
    regex_matcher["CARDINAL"] = cardinal_matcher

    percent_matcher = re.compile(r"^\s*(\d*(\.\d+)?[\s]*%)\s*$")
    regex_matcher["PERCENT"] = percent_matcher

    ip_pattern = "(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
    ip_matcher = re.compile(ip_pattern, re.IGNORECASE)
    regex_matcher["IP ADDRESS"] = ip_matcher

    ipv6_pattern = "\s*(?!.*::.*::)(?:(?!:)|:(?=:))(?:[0-9a-f]{0,4}(?:(?<=::)|(?<!::):)){6}(?:[0-9a-f]{0,4}(?:(?<=::)|(?<!::):)[0-9a-f]{0,4}(?:(?<=::)|(?<!:)|(?<=:)(?<!::):)|(?:25[0-4]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-4]|2[0-4]\d|1\d\d|[1-9]?\d)){3})\s*"
    ipv6_matcher = re.compile(ipv6_pattern, re.VERBOSE|re.IGNORECASE|re.DOTALL)
    regex_matcher["IPv6 ADDRESS"] = ipv6_matcher

    boolean_pattern = "^\s*true\s*$|^\s*false\s*$|^\s*on\s*$|^\s*off\s*$|^\s*yes\s*$|^\s*no\s*$"
    boolean_matcher = re.compile(boolean_pattern, re.IGNORECASE)
    regex_matcher["BOOLEAN"] = boolean_matcher

    return regex_matcher

## regrex parsers
regex_matcher = init_parser()
def regex_parser(list_cell):
    ner_per_label = {}
    for label in list_cell:
        ner_per_label[label] = []
        try:
            num = int(label)
            if 1000 <= num <= 2022:
                ner_per_label[label].append("DATE")
        except:
            pass

        for regex_label, matcher in regex_matcher.items():
            matching_res = re.match(matcher, label)
            if matching_res:
                if matching_res.group(0) == label:
                    ner_per_label[label].append(regex_label)
            
    return ner_per_label