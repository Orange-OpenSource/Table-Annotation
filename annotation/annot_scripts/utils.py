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
from dateutil.parser import parse
from rapidfuzz import fuzz
from quantulum3 import parser as qt_unit_parser
import pint
ureg = pint.UnitRegistry()
## since Pint does not officilly support currency, we define it by ourself.
## Currently, we only support: dollar, euro, japanese_yen, chinese_yuan, pound_sterling, south_korean_won, russian_ruble, australian_dollar"
ureg.load_definitions("""
    dollar = [currency] = united_states_dollar
    euro = 1.1 dollar 
    japanese_yen = 0.0082 dollar
    chinese_yuan = 0.16 dollar
    renminbi = 0.16 dollar
    pound_sterling = 1.32 dollar
    south_korean_won = 0.00082 dollar
    russian_ruble = 0.01 dollar
    australian_dollar = 0.75 dollar
""".splitlines())

def float_parse(value):
    """ Check whether an input str is a float, if yes, return the float"""
    if isinstance(value, float) or isinstance(value, int):
        return value
    elif isinstance(value, str):
        try:
            return float(value.replace(",", ""))
        except:
            return None

def date_similarity(s1, s2, operator):
    """ Check whether an 2 input str is 2 possible equal datetimes """
    try:
        if operator(parse(s1), parse(s2)):
            return True
        return False
    except:
        return False 

def get_year_from_date(d):
    " return year from date"
    try:
        return str(parse(d).year)
    except:
        return False      

def textual_similarity(s1, s2):
    """ Calculate the simliarity score between two textual values using three levenstein distances. """
    char_based_ratio = fuzz.ratio(s1.lower(), s2.lower())/100
    token_sort_based_ratio = fuzz.token_sort_ratio(s1.lower(), s2.lower())/100
    token_set_based_ratio = fuzz.token_set_ratio(s1.lower(), s2.lower())/100
    ## the final ratio is the mean of two maximum ratios among three ratios. 
    ## to avoid that 2 ratios of same values dominate the other.
    ## e.g. char_based_ratio("universal", "universal picture") = token_sort_based_ratio("universal", "universal picture") = 0.66
    ##    so including both ratios in the final ratio will decrease the significance of token_set_based_ratio("universal", "universal picture") which is 1.0
    final_ratio = sum(sorted([char_based_ratio, token_sort_based_ratio, token_set_based_ratio], reverse=True)[:2])/2
    return final_ratio
    # return (fuzz.ratio(s1.lower(), s2.lower())/100 + fuzz.token_sort_ratio(s1.lower(), s2.lower())/100 + fuzz.token_set_ratio(s1.lower(), s2.lower())/100)/3

def dimensionless_quantity_similarity(s1, s2):
    """ Calculate the similarity score between two dimensionless (quantity) values. """
    s1_float = float_parse(s1)
    s2_float = float_parse(s2)
    if s1_float is not None and s2_float is not None:
        sim = 1 - abs(s1_float - s2_float)/(abs(s1_float) + abs(s2_float) + 0.0001)
        return sim
    else:
        return 0.0

def standardize_to_base_unit(measure):
    """ standardize a measurement with unit to base unit. E.g. 5 km -> 5000 m given that metre is base unit of length """
    standardized_measure = {}
    if isinstance(measure, str): ## if inuput measure is plain text, e.g. "5 km"
        parsed_measure = qt_unit_parser.parse(measure)
        for a_unit in parsed_measure:
            if a_unit.unit.name != "dimensionless":
                try:
                    transformed_measure = float(a_unit.value)*ureg("_".join(a_unit.unit.name.lower().split(" "))).to_base_units()
                    if transformed_measure.units not in standardized_measure:
                        standardized_measure[transformed_measure.units] = [transformed_measure.magnitude]
                    else:
                        for magnitude in standardized_measure[transformed_measure.units]:
                            if 0.98 < magnitude/transformed_measure.magnitude < 0.98**-1:
                                ## this indicates that single measure has different units, hence, we dont need to append 
                                ##           duplicated measure into result
                                pass
                            else:
                                standardized_measure[transformed_measure.units].append(transformed_measure.magnitude)
                except:
                    pass
    elif isinstance(measure, dict) and "value" in measure and "unit" in measure:
        try:
            transformed_measure = float(measure["value"])*ureg(measure["unit"]).to_base_units()
            standardized_measure[transformed_measure.units] = [transformed_measure.magnitude]
        except:
            pass

    return standardized_measure

# def dimensional_quantity_similarity(s1, s1_unit, s2, s2_unit):
#     """ Calculate the similarity score between two dimensional (quantity) values. """
#     ## convert to base units
#     try:
#         converted_s1 = float(s1)*ureg(s1_unit).to_base_units()
#         converted_s2 = float(s2)*ureg(s2_unit).to_base_units()
#     except (pint.errors.UndefinedUnitError, AttributeError):
#         ## invalid or unsupported unit by Pint
#         return 0.0

#     if converted_s1.units == converted_s2.units:
#         converted_s1_val = converted_s1.magnitude
#         converted_s2_val = converted_s2.magnitude
#         ## compare s1 and s2 values
#         sim = 1 - abs(converted_s1_val - converted_s2_val)/(abs(converted_s1_val) + abs(converted_s2_val) + 0.0001)
#         return sim
#     else:
#         return 0.0

def named_entity_related_typing(t):
    """ Verify whether a type t talks about an entity. """
    named_entity_list = ["UNKNOWN", "PERSON", "ORG", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "NORP", "ORG", "PRODUCT", "WORK_OF_ART", "EVENT"]
    if t in named_entity_list:
        return True 
    else:
        return False
    
def date_related_typing(t):
    """ Verify whether a type t talks about a date. """
    if t == "DATE":
        return True 
    else:
        return False    

def numerical_typing_with_unit(t):
    """ Verify whether a type t talks about numerial entity that has an unit. """
    with_unit_list = ['PERCENT', 'DISTANCE', 'MASS', 'MONEY', 'DURATION',
                         'TEMPERATURE', 'CHARGE', 'ANGLE', 'DATA STORAGE',
                         'AMOUNT OF SUBSTANCE', 'CATALYTIC ACTIVITY', 'AREA',
                        'VOLUME','VOLUME (LUMBER)', 'FORCE', 'PRESSURE',
                        'ENERGY', 'POWER', 'SPEED', 'ACCELERATION',
                        'FUEL ECONOMY', 'FUEL CONSUMPTION', 'ANGULAR SPEED', 'ANGULAR ACCELERATION',
                        'DENSITY', 'SPECIFIC VOLUME', 'MOMENT OF INERTIA', 'TORQUE',
                        'THERMAL RESISTANCE', 'THERMAL CONDUCTIVITY', 'SPECIFIC HEAT CAPACITY', 'VOLUMETRIC FLOW',
                        'MASS FLOW', 'CONCENTRATION', 'DYNAMIC VISCOSITY', 'KINEMATIC VISCOSITY',
                        'FLUIDITY', 'SURFACE TENSION', 'PERMEABILITY', 'SOUND LEVEL',
                        'LUMINOUS INTENSITY', 'LUMINOUS FLUX', 'ILLUMINANCE', 'LUMINANCE',
                        'TYPOGRAPHICAL ELEMENT', 'IMAGE RESOLUTION', 'FREQUENCY', 'INSTANCE FREQUENCY',
                        'FLUX DENSITY', 'LINEAR MASS DENSITY', 'LINEAR CHARGE DENSITY', 'SURFACE CHARGE DENSITY',
                        'CHARGE DENSITY', 'CURRENT', 'LINEAR CURRENT DENSITY', 'SURFACE CURRENT DENSITY',
                        'ELECTRIC POTENTIAL', 'ELECTRIC FIELD', 'ELECTRICAL RESISTANCE', 'ELECTRICAL RESISTIVITY',
                        'ELECTRICAL CONDUCTANCE', 'ELECTRICAL CONDUCTIVITY', 'CAPACITANCE', 'INDUCTANCE',
                        'MAGNETIC FLUX', 'RELUCTANCE', 'MAGNETOMOTIVE FORCE', 'MAGNETIC FIELD',
                        'IRRADIANCE', 'RADIATION ABSORBED DOSE', 'RADIOACTIVITY', 'RADIATION EXPOSURE',
                        'RADIATION', 'DATA TRANSFER RATE']
    if t in with_unit_list:
        return True
    else:
        return False

def numerical_typing_without_unit(t):
    """ Verify whether a type t talks about numerical entity that doesn't have an unit. """
    without_unit_list = ["CARDINAL", "QUANTITY", "ORDINAL"]
    if t in without_unit_list:
        return True
    else:
        return False











