
import os
import imp
import sys

import cPickle as pkl
import xml.etree.ElementTree as ElementTree
import numpy as np
import math
from scipy import ndimage as nd

from operator import itemgetter

import common
import CommunFunctions.cpp_wrapping as cpp_wrapping
from CommunFunctions.ImageHandling import imread

#
#
#
#
#

monitoring = common.Monitoring()


########################################################################################
#
# classes
# - computation parameters
#
########################################################################################

class CellPropertiesParameters(object):

    def __init__(self):
        self.max_chunks_properties = None

    def write_parameters(self, log_file_name):
        with open(log_file_name, 'a') as logfile:
            logfile.write("\n")
            logfile.write('CellPropertiesParameters\n')
            logfile.write('- max_chunks_properties = ' + str(self.max_chunks_properties) + '\n')
            logfile.write("\n")
        return

    def print_parameters(self):
        print("")
        print('CellPropertiesParameters')
        print('- max_chunks_properties = ' + str(self.max_chunks_properties))
        print("")
        return

    def update_from_parameter_file(self, parameter_file):
        if parameter_file is None:
            return
        if not os.path.isfile(parameter_file):
            print("Error: '" + parameter_file + "' is not a valid file. Exiting.")
            sys.exit(1)

        parameters = imp.load_source('*', parameter_file)

        if hasattr(parameters, 'properties_nb_proc'):
            if parameters.properties_nb_proc is not None:
                self.max_chunks_properties = parameters.properties_nb_proc
        return


########################################################################################
#
# properties computation from a sequence
# calculation is done after sequence intra-registration
#
########################################################################################

def property_computation(experiment):
    """

    :param experiment:
    :return:
    """

    proc = 'property_computation'

    #
    # get directory name where to find co-registered images of the sequence
    # as well as the common image suffix
    #

    intrareg_path = os.path.join(experiment.intrareg_dir.get_directory(), experiment.post_dir.get_sub_directory())
    #
    # is there a post-segmentation directory in the intra-registration directory ?
    #
    if not os.path.isdir(intrareg_path):
        monitoring.to_log(proc + ": '" + str(intrareg_path) + "' does not exist")
        intrareg_path = os.path.join(experiment.intrareg_dir.get_directory(), experiment.astec_dir.get_sub_directory())
        #
        # if no, is there a segmentation directory in the intra-registration directory ?
        #
        if not os.path.isdir(intrareg_path):
            monitoring.to_log(proc + ": '" + str(intrareg_path) + "' does not exist")
            intrareg_path = experiment.intrareg_dir.get_directory()
            monitoring.to_log_and_console(proc + ": neither POST/ or SEG/ sub-directories in '"
                                          + str(intrareg_path) + "'", 0)
            monitoring.to_log_and_console("Exiting.", 0)
            return None
        else:
            working_dir = experiment.astec_dir
    else:
        working_dir = experiment.post_dir

    monitoring.to_log_and_console("... will compute sequence properties from '" + str(intrareg_path) + "'", 0)

    #
    # build name format for (post-corrected) segmentation images
    #
    name_format = experiment.intrareg_dir.get_file_prefix() + experiment.intrareg_dir.get_file_suffix() + \
                  working_dir.get_file_suffix() + experiment.intrareg_dir.get_time_prefix() + \
                  experiment.get_time_format()

    suffix = common.get_file_suffix(experiment, intrareg_path, name_format, flag_time=experiment.get_time_format())
    if suffix is None:
        monitoring.to_log_and_console(proc + ": no consistent naming was found in '"
                                      + str(intrareg_path) + "'", 1)
        monitoring.to_log_and_console("Exiting.", 0)
        sys.exit(1)
    name_format += "." + str(suffix)
    template_format = os.path.join(intrareg_path, name_format)

    #
    #
    #
    output_name = experiment.intrareg_dir.get_file_prefix() + experiment.intrareg_dir.get_file_suffix() + \
                  working_dir.get_file_suffix() + "_lineage"
    output_name = os.path.join(intrareg_path, output_name)

    if os.path.isfile(output_name + ".xml") and os.path.isfile(output_name + ".pkl"):
        if not monitoring.forceResultsToBeBuilt:
            monitoring.to_log_and_console('    xml file already existing', 2)
            return output_name + ".xml"
        else:
            monitoring.to_log_and_console('    xml file already existing, but forced', 2)

    first_time_point = experiment.first_time_point + experiment.delay_time_point
    last_time_point = experiment.last_time_point + experiment.delay_time_point

    cpp_wrapping.cell_properties(template_format, output_name + ".xml", first_time_point, last_time_point,
                                 diagnosis_file=output_name + ".txt", monitoring=monitoring)

    return output_name + ".xml"


########################################################################################
#
# key correspondences
#
# Examples
# - from 'full_properties_Samson_MN20.pkl', keys are
#   ['volumes information',
#    'Cells labels in time',
#    'Barycenters',
#    'Fate',
#    'All Cells',
#    'Principal values',
#    'Names',
#    'cell_cell_contact_information',
#    'Lineage tree',
#    'Cells history',
#    'Principal vectors']
# - from 'new_lineage_tree_MN20.pkl', keys are cell labels
# - from '140317-Patrick-St8_seg_lineage.pkl', keys are
#   ['h_mins_information', 'lin_tree', 'volumes_information', 'sigmas_information']
#
#
########################################################################################

keydictionary = {'lineage': {'output_key': 'cell_lineage',
                             'input_keys': ['lineage_tree', 'lin_tree', 'Lineage tree', 'cell_lineage']},
                 'h_min': {'output_key': 'cell_h_min',
                           'input_keys': ['cell_h_min', 'h_mins_information']},
                 'volume': {'output_key': 'cell_volume',
                            'input_keys': ['cell_volume', 'volumes_information', 'volumes information', 'vol']},
                 'surface': {'output_key': 'cell_surface',
                             'input_keys': ['cell_surface', 'cell surface']},
                 'compactness': {'output_key': 'cell_compactness',
                                 'input_keys': ['cell_compactness', 'Cell Compactness', 'compacity',
                                                'cell_sphericity']},
                 'sigma': {'output_key': 'cell_sigma',
                           'input_keys': ['cell_sigma', 'sigmas_information', 'sigmas']},
                 'label_in_time': {'output_key': 'cell_labels_in_time',
                                   'input_keys': ['cell_labels_in_time', 'Cells labels in time', 'time_labels']},
                 'barycenter': {'output_key': 'cell_barycenter',
                                'input_keys': ['cell_barycenter', 'Barycenters', 'barycenters']},
                 'fate': {'output_key': 'cell_fate',
                          'input_keys': ['cell_fate', 'Fate']},
                 'fate2': {'output_key': 'cell_fate_2',
                           'input_keys': ['cell_fate_2', 'Fate2']},
                 'fate3': {'output_key': 'cell_fate_3',
                           'input_keys': ['cell_fate_3', 'Fate3']},
                 'fate4': {'output_key': 'cell_fate_4',
                           'input_keys': ['cell_fate_4', 'Fate4']},
                 'all-cells': {'output_key': 'all_cells',
                               'input_keys': ['all_cells', 'All Cells', 'All_Cells', 'all cells', 'tot_cells']},
                 'principal-value': {'output_key': 'cell_principal_values',
                                     'input_keys': ['cell_principal_values', 'Principal values']},
                 'name': {'output_key': 'cell_name',
                          'input_keys': ['cell_name', 'Names', 'names', 'cell_names']},
                 'contact': {'output_key': 'cell_contact_surface',
                             'input_keys': ['cell_contact_surface', 'cell_cell_contact_information']},
                 'contact-edge': {'output_key': 'cell_contact_edge',
                             'input_keys': ['cell_contact_edge']},
                 'contact-edge-length': {'output_key': 'cell_contact_edge_length',
                             'input_keys': ['cell_contact_edge_length']},
                 'contact-edge-segment': {'output_key': 'cell_contact_edge_segment',
                             'input_keys': ['cell_contact_edge_segment']},
                 'history': {'output_key': 'cell_history',
                             'input_keys': ['cell_history', 'Cells history', 'cell_life', 'life']},
                 'principal-vector': {'output_key': 'cell_principal_vectors',
                                      'input_keys': ['cell_principal_vectors', 'Principal vectors']},
                 'name-score': {'output_key': 'cell_naming_score',
                                'input_keys': ['cell_naming_score', 'Scores', 'scores']},
                 'problems': {'output_key': 'problematic_cells',
                              'input_keys': ['problematic_cells']},
                 'urchin_apicobasal_length': {'output_key': 'urchin_cell_apicobasal_length',
                              'input_keys': ['urchin_cell_apicobasal_length']},
                 'urchin_apicobasal_segment': {'output_key': 'urchin_cell_apicobasal_segment',
                              'input_keys': ['urchin_cell_apicobasal_segment']},
                 'urchin_adjacency': {'output_key': 'urchin_cell_adjacency',
                                       'input_keys': ['urchin_cell_adjacency']},
                 'urchin_apical_surface': {'output_key': 'urchin_apical_surface',
                              'input_keys': ['urchin_apical_surface']},
                 'urchin_apical_surface_barycenter': {'output_key': 'urchin_apical_surface_barycenter',
                              'input_keys': ['urchin_apical_surface_barycenter']},
                 'urchin_basal_surface': {'output_key': 'urchin_basal_surface',
                              'input_keys': ['urchin_basal_surface']},
                 'urchin_basal_surface_barycenter': {'output_key': 'urchin_basal_surface_barycenter',
                              'input_keys': ['urchin_basal_surface_barycenter']},
                 'urchin_apical_contact_edge_length': {'output_key': 'urchin_apical_contact_edge_length',
                              'input_keys': ['urchin_apical_contact_edge_length']},
                 'urchin_apical_contact_edge_segment': {'output_key': 'urchin_apical_contact_edge_segment',
                              'input_keys': ['urchin_apical_contact_edge_segment']},
                 'urchin_basal_contact_edge_length': {'output_key': 'urchin_basal_contact_edge_length',
                              'input_keys': ['urchin_basal_contact_edge_length']},
                 'urchin_basal_contact_edge_segment': {'output_key': 'urchin_basal_contact_edge_segment',
                              'input_keys': ['urchin_basal_contact_edge_segment']},
                 'urchin_vegetal_distance': {'output_key': 'urchin_vegetal_distance',
                              'input_keys': ['urchin_vegetal_distance']},
                 'unknown': {'output_key': 'unknown_key',
                             'input_keys': ['unknown_key']}}


def normalize_dictionary_keys(inputdict):
    """

    :param inputdict:
    :return:
    """

    if inputdict == {}:
        return {}

    outputdict = {}

    for inputkey in inputdict:
        foundkey = False
        for k in keydictionary:
            # print "       compare '" + str(tmpkey) + "' with '" + str(k) + "'"
            if inputkey in keydictionary[k]['input_keys']:
                outputkey = keydictionary[k]['output_key']
                monitoring.to_log_and_console("   ... recognized key '" + str(outputkey) + "'", 3)
                #
                # update if key already exists, else just create the dictionary entry
                #
                outputdict[outputkey] = inputdict[inputkey]
                foundkey = True
                break

        if foundkey is False:
            outputdict[inputkey] = inputdict[inputkey]

    return outputdict


def get_dictionary_entry(inputdict, keystring):
    proc = 'get_dictionary_entry'
    if keystring not in keydictionary:
        monitoring.to_log_and_console(str(proc) + ": keystring must be in " + str(keydictionary.keys()), 1)
        return {}
    for k in keydictionary[keystring]['input_keys']:
        if k in inputdict:
            return inputdict[k]
    else:
        monitoring.to_log_and_console(str(proc) + ": '" + str(keystring) + "' was not found in input dictionary", 1)
        monitoring.to_log_and_console("    keys were: " + str(inputdict.keys()), 1)
        return {}

########################################################################################
#
# to translate a dictionary into XML
#
########################################################################################

#
# from stackoverflow.com
# questions/3095434/inserting-newlines-in-xml-file-generated-via-xml-etree-elementtree-in-python
#
#
# used for pretty printing
#


def _indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            _indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


########################################################################################
#
# types
# 'lineage':  'lineage_tree' :
#     dictionary de liste de int
#     lineage_tree.590002 = <type 'list'>
# 'h_min' : 'cell_h_min' :
# 'volume' : 'cell_volume' :
#     dictionary de int
#     cell_volume.590002 = <type 'int'>
#     590002: 236936
# 'sigma': 'cell_sigma':
# 'label_in_time': 'cell_labels_in_time'
#     dictionary de liste de numpy.int64
#     cell_labels_in_time.1 = <type 'list'>
#     1: [10002, 10003, 10004, ..., 10082]
# 'barycenter': 'cell_barycenter'
#     dictionary de numpy.ndarray de numpy.float64
#     cell_barycenter.590002 = <type 'numpy.ndarray'>
#     590002: array([ 258.41037242,  226.74975943,  303.67167927])
# 'fate': 'cell_fate'
#     dictionary de str
#     cell_fate.590002 = <type 'str'>
#     590002: 'Mesoderm Notochord 1'
# 'all-cells': 'all_cells'  # liste de toutes les cellules ?
#     liste de numpy.int64
#     all_cells = <type 'list'>
# 'principal-value': 'cell_principal_values'
#     dictionary de liste de numpy.float64
#     cell_principal_values.590002 = <type 'list'>
#     590002: [1526.0489371146978, 230.60881177650205, 91.063513300019849]
# 'name': 'cell_name'
#     dictionary de str
#     cell_name.590002 = <type 'str'>
#     590002: 'a9.0012_'
# 'contact': 'cell_contact_surface',
#     dictionary de dictionary de int
#     cell_contact_surface.590002.590019 = <type 'int'>
#     590002: {590001: 1808, 590003: 1436, 590004: 5012, ..., 590225: 2579}
# 'history': 'cell_history'
#     dictionary de numpy.ndarray de numpy.int64
#     cell_history.590002 = <type 'numpy.ndarray'>
#     590002: array([510002, 520002, 530002, 540002, 550002, 560002, 570002, 580002,
#         590002, 600002, 610002, 620002, 630002, 640002, 650002, 660002,
#         670002, 680002, 690002, 700002, 710002, 720002, 730002, 740002,
#         750002, 760002, 770002, 780002, 790002, 800002, 810002, 820002,
#         830002, 840002, 850002, 860002, 870002, 880002])
# 'principal-vector': 'cell_principal_vectors'    # liste de numpy.ndarray
#     dictionary de liste de numpy.ndarray de numpy.float64
#     cell_principal_vectors.590002 = <type 'list'>
#     590002: [array([ 0.17420991, -0.74923203,  0.63898534]),
#         array([-0.24877611,  0.59437038,  0.7647446 ]),
#         array([ 0.95276511,  0.29219037,  0.08284582])]
#
########################################################################################

def _set_xml_element_text(element, value):
    """

    :param element:
    :param value:
    :return:
    """
    proc = "_set_xml_element_text"

    #
    # dictionary : recursive call
    #   dictionary element may be list, int, numpy.ndarray, str
    # list : may be int, numpy.int64, numpy.float64, numpy.ndarray
    #

    if type(value) == dict:
        # print proc + ": type is dict"
        keylist = value.keys()
        keylist.sort()
        for k in keylist:
            _dict2xml(element, k, value[k])

    elif type(value) == list:

        #
        # empty list
        #

        if len(value) == 0:
            element.text = repr(value)
        #
        # 'lineage', 'label_in_time', 'all-cells', 'principal-value'
        #

        elif type(value[0]) in (int, float, np.int64, np.float64):
            # element.text = str(value)
            element.text = repr(value)

        #
        # 'principal-vector' case
        #  liste de numpy.ndarray de numpy.float64
        #
        elif type(value[0]) == np.ndarray:
            text = "["
            for i in range(len(value)):
                # text += str(list(value[i]))
                text += repr(list(value[i]))
                if i < len(value)-1:
                    text += ", "
                    if i > 0 and i % 10 == 0:
                        text += "\n  "
            text += "]"
            element.text = text
            del text

        else:
            monitoring.to_log_and_console(proc + ": error, element list type ('" + str(type(value[0]))
                                          + "') not handled yet")

    #
    # 'barycenter', 'cell_history'
    #
    elif type(value) == np.ndarray:
        # element.text = str(list(value))
        element.text = repr(list(value))

    #
    # 'volume', 'contact'
    #
    elif type(value) in (int, float, np.int64, np.float64):
        # element.text = str(value)
        element.text = repr(value)

    #
    # 'fate', 'name'
    #
    elif type(value) == str:
        element.text = repr(value)

    else:
        monitoring.to_log_and_console(proc + ": element type '" + str(type(value))
                                      + "' not handled yet, uncomplete translation")


#
#
#


def _dict2xml(parent, tag, value):
    """

    :param parent:
    :param tag:
    :param value:
    :return:
    """

    #
    # integers can not be XML tags
    #
    if type(tag) in (int, np.int64):
        child = ElementTree.Element('cell', attrib={'cell-id': str(tag)})
    else:
        child = ElementTree.Element(str(tag))

    _set_xml_element_text(child, value)

    parent.append(child)
    return parent


#
# procedure d'appel et creation de la racine
#


def dict2xml(dictionary, defaultroottag='data'):
    """

    :param dictionary:
    :param defaultroottag:
    :return:
    """

    proc = "dict2xml"

    if type(dictionary) is not dict:
        monitoring.to_log_and_console(proc + ": error, input is of type '" + str(type(dictionary)) + "'")
        return None

    #
    # s'il n'y a qu'un seul element dans le dictionnaire, on appelle la racine
    # d'apres celui-ci (eg lineage_tree), sinon on cree une racine qui contient
    # tous les elements
    #

    if len(dictionary) == 1:

        roottag = dictionary.keys()[0]
        root = ElementTree.Element(roottag)
        _set_xml_element_text(root, dictionary[roottag])

    elif len(dictionary) > 1:

        root = ElementTree.Element(defaultroottag)
        for k, v in dictionary.iteritems():
            _dict2xml(root, k, v)

    else:
        monitoring.to_log_and_console(proc + ": error, empty dictionary ?!")
        return None

    _indent(root)
    tree = ElementTree.ElementTree(root)

    return tree


########################################################################################
#
# to translate a XML tree into dictionary
#
########################################################################################


def _set_dictionary_value(root):
    """

    :param root:
    :return:
    """

    if len(root) == 0:

        #
        # pas de branche, on renvoie la valeur
        #

        # return ast.literal_eval(root.text)
        if root.text is None:
            return None
        else:
            return eval(root.text)

    else:

        dictionary = {}

        for child in root:

            # print "child.tag=" + str(child.tag)
            # print "len(child)=" + str(len(child))
            # print "child.text=" + str(child.text)

            key = child.tag
            if child.tag == 'cell':
                key = np.int64(child.attrib['cell-id'])
            dictionary[key] = _set_dictionary_value(child)

    return dictionary


def xml2dict(tree):
    """

    :param tree:
    :return:
    """

    proc = "xml2dict"

    root = tree.getroot()

    dictionary = {}

    for k, v in keydictionary.iteritems():

        if root.tag == v['output_key']:
            monitoring.to_log_and_console("   ... " + proc + ": process root.tag = '" + str(root.tag) + "'", 3)
            dictionary[str(root.tag)] = _set_dictionary_value(root)
            break
    else:
        for child in root:
            monitoring.to_log_and_console("   ... " + proc + ": process child.tag = '" + str(child.tag) + "'", 3)
            value = _set_dictionary_value(child)
            if value is None:
                monitoring.to_log_and_console("       " + proc + ": empty property '" + str(child.tag) + "' ?! "
                                              + " ... skip it", 1)
            else:
                dictionary[str(child.tag)] = value

    return dictionary


########################################################################################
#
# to read a set of files into a dictionary
#
########################################################################################


#
# update dictionary from what has been read
#

def _update_read_dictionary(propertiesdict, tmpdict, filename):
    """

    :param propertiesdict:
    :param tmpdict:
    :return:
    """
    proc = "_update_read_dictionary"
    unknownkeys = []

    for tmpkey in tmpdict:
        foundkey = False

        for k in keydictionary:
            # print "       compare '" + str(tmpkey) + "' with '" + str(k) + "'"
            if tmpkey in keydictionary[k]['input_keys']:
                outputkey = keydictionary[k]['output_key']
                monitoring.to_log_and_console("   ... recognized key '" + str(outputkey) + "'", 2)
                #
                # update if key already exists, else just create the dictionary entry
                #
                if outputkey in propertiesdict:
                    if type(propertiesdict[outputkey]) is dict and type(tmpdict[tmpkey]) is dict:
                        propertiesdict[outputkey].update(tmpdict[tmpkey])
                    elif type(propertiesdict[outputkey]) is list and type(tmpdict[tmpkey]) is list:
                        propertiesdict[outputkey] += tmpdict[tmpkey]
                    else:
                        monitoring.to_log_and_console(proc + ": error, can not update property '" + str(outputkey)
                                                      + "'")
                else:
                    propertiesdict[outputkey] = tmpdict[tmpkey]
                foundkey = True
                break

        if foundkey is False:
            unknownkeys.append(tmpkey)

    if len(unknownkeys) > 0 and len(unknownkeys) == len(tmpdict.keys()):
        #
        # no key was found
        # it is assumed it's a lineage tree: add some test here ?
        #
        monitoring.to_log_and_console("   ... assume '" + str(filename) + "' is a lineage", 1)
        outputkey = keydictionary['lineage']['output_key']
        if outputkey in propertiesdict:
            if type(propertiesdict[outputkey]) is dict and type(tmpdict) is dict:
                propertiesdict[outputkey].update(tmpdict)
            else:
                monitoring.to_log_and_console(proc + ": error, can not update property '" + str(outputkey) + "'")
        else:
            propertiesdict[outputkey] = tmpdict

    elif len(unknownkeys) > 0:
        #
        # some unknown keys were found
        #
        monitoring.to_log_and_console("   ... unrecognized key(s) are '" + str(unknownkeys) + "'", 1)

        # previous behavior: use keydictionary['unknown']['output_key'] as key
        # for *one* unknown property
        #
        #  outputkey = keydictionary['unknown']['output_key']
        # if len(unknownkeys) == 1:
        #     tmpkey = unknownkeys[0]
        #     if outputkey in propertiesdict:
        #         if type(propertiesdict[outputkey]) is dict and type(tmpdict[tmpkey]) is dict:
        #             propertiesdict[outputkey].update(tmpdict[tmpkey])
        #         elif type(propertiesdict[outputkey]) is list and type(tmpdict[tmpkey]) is list:
        #             propertiesdict[outputkey] += tmpdict[tmpkey]
        #         else:
        #             monitoring.to_log_and_console(proc + ": error, can not update property '" + str(outputkey)
        #                                           + "'")
        #     else:
        #         propertiesdict[outputkey] = tmpdict[tmpkey]
        # else:
        #     monitoring.to_log_and_console(proc + ": error, can not update many unknown properties")

        #
        # use unknown keys as such
        #
        for k in unknownkeys:
            propertiesdict[k] = tmpdict[k]

    return propertiesdict


#
# types issued from the reading of xml files may be erroneous
# fix it
#

def _set_types_from_xml(propertiesdict):
    """

    :param propertiesdict:
    :return:
    """

    if propertiesdict == {}:
        return {}

    if 'cell_barycenter' in propertiesdict:
        monitoring.to_log_and_console("   ... translate types of 'cell_barycenter'", 3)
        for c in propertiesdict['cell_barycenter']:
            propertiesdict['cell_barycenter'][c] = np.array(propertiesdict['cell_barycenter'][c])

    if 'cell_history' in propertiesdict:
        monitoring.to_log_and_console("   ... translate types of 'cell_history'", 3)
        for c in propertiesdict['cell_history']:
            propertiesdict['cell_history'][c] = np.array(propertiesdict['cell_history'][c])

    if 'cell_principal_vectors' in propertiesdict:
        monitoring.to_log_and_console("   ... translate types of 'cell_principal_vectors'", 3)
        for c in propertiesdict['cell_principal_vectors']:
            for v in range(len(propertiesdict['cell_principal_vectors'][c])):
                propertiesdict['cell_principal_vectors'][c][v] \
                    = np.array(propertiesdict['cell_principal_vectors'][c][v])

    return propertiesdict


#
#
#

def _read_xml_file(filename, propertiesdict):
    monitoring.to_log_and_console("... reading '" + str(filename) + "'", 1)
    inputxmltree = ElementTree.parse(filename)
    tmpdict = xml2dict(inputxmltree)
    propertiesdict = _update_read_dictionary(propertiesdict, tmpdict, filename)
    del tmpdict
    return propertiesdict


#
#
#

def _read_pkl_file(filename, propertiesdict):
    monitoring.to_log_and_console("... reading '" + str(filename) + "'", 1)
    inputfile = open(filename, 'r')
    tmpdict = pkl.load(inputfile)
    inputfile.close()
    propertiesdict = _update_read_dictionary(propertiesdict, tmpdict, filename)
    del tmpdict
    return propertiesdict


#
#
#

def read_dictionary(inputfilenames, inputpropertiesdict={}):
    """

    :param inputfilenames:
    :param inputpropertiesdict:
    :return:
    """
    proc = 'read_dictionary'

    if inputfilenames is None:
        monitoring.to_log_and_console(proc + ": error, no input files")
        return {}

    propertiesdict = inputpropertiesdict

    #
    #
    #

    if type(inputfilenames) == str:
        if not os.path.isfile(inputfilenames):
            monitoring.to_log_and_console(proc + ": error, file '" + str(inputfilenames) + "' does not exist")
            return {}

        if inputfilenames.endswith("xml") is True:
            propertiesdict = _read_xml_file(inputfilenames, propertiesdict)
            propertiesdict = _set_types_from_xml(propertiesdict)
        elif inputfilenames.endswith("pkl") is True:
            propertiesdict = _read_pkl_file(inputfilenames, propertiesdict)
        else:
            monitoring.to_log_and_console(proc + ": error: extension not recognized for '" + str(inputfilenames) + "'")

        propertiesdict = normalize_dictionary_keys(propertiesdict)
        return propertiesdict

    #
    # here, we assume type(inputfilenames) == list
    #

    #
    # read xml files
    #

    for filename in inputfilenames:

        if not os.path.isfile(filename):
            monitoring.to_log_and_console(proc + ": error, file '" + str(filename) + "' does not exist")
            continue

        if filename.endswith("xml") is True:
            propertiesdict = _read_xml_file(filename, propertiesdict)

    #
    # translation of xml may take place here
    #

    propertiesdict = _set_types_from_xml(propertiesdict)

    #
    # read pkl files
    #

    for filename in inputfilenames:

        if filename.endswith("pkl") is True:
            propertiesdict = _read_pkl_file(filename, propertiesdict)

    #
    #
    #

    for filename in inputfilenames:
        if filename[len(filename) - 3:len(filename)] == "xml":
            continue
        elif filename[len(filename) - 3:len(filename)] == "pkl":
            continue
        else:
            monitoring.to_log_and_console(proc + ": error: extension not recognized for '" + str(filename) + "'")

    propertiesdict = normalize_dictionary_keys(propertiesdict)
    return propertiesdict


def write_dictionary(inputfilename, inputpropertiesdict):
    """

    :param inputfilename:
    :param inputpropertiesdict:
    :return:
    """
    proc = 'write_dictionary'

    if inputfilename.endswith("pkl") is True:
        lineagefile = open(inputfilename, 'w')
        pkl.dump(inputpropertiesdict, lineagefile)
        lineagefile.close()
    elif inputfilename.endswith("xml") is True:
        xmltree = dict2xml(inputpropertiesdict)
        xmltree.write(inputfilename)
        del xmltree
    elif inputfilename.endswith("tlp") is True:
        write_tlp_file(inputfilename, inputpropertiesdict)
    else:
        monitoring.to_log_and_console(str(proc) + ": error when writing lineage file. Extension not recognized for '"
                                      + os.path.basename(inputfilename) + "'", 1)
    return


########################################################################################
#
# comparison of two dictionaries
#
########################################################################################


def _decode_cell_id(s, time_digits_for_cell_id=4):
    div = 10**time_digits_for_cell_id
    return "cell #{:4d}".format(int(s) % div) + " of image #{:4d}".format(int(s) / div)


def _get_time_interval_from_lineage(direct_lineage, time_digits_for_cell_id=4):

    nodes = list(set(direct_lineage.keys()).union(set([v for values in direct_lineage.values() for v in values])))
    first_time = min(nodes) / 10 ** time_digits_for_cell_id
    last_time = max(nodes) / 10 ** time_digits_for_cell_id
    # monitoring.to_log_and_console("  - estimated time interval = [" + str(first_time) + ", " + str(last_time) + "]", 1)

    return first_time, last_time


def _get_time_interval_from_properties(d, time_digits_for_cell_id=4):

    proc = "_get_time_interval_from_properties"
    keylineage = None

    for k in d:
        if k in keydictionary['lineage']['input_keys']:
            keylineage = k

    if keylineage is None:
        monitoring.to_log_and_console(str(proc) + ": no lineage in input dictionary", 1)
        return None, None

    return _get_time_interval_from_lineage(d[keylineage], time_digits_for_cell_id=time_digits_for_cell_id)


########################################################################################
#
# comparison of two dictionaries

########################################################################################


def _intersection_cell_keys(e1, e2, name1, name2):
    """

    :param e1:
    :param e2:
    :param name1:
    :param name2:
    :return:
    """
    intersection = list(set(e1.keys()).intersection(set(e2.keys())))
    difference1 = list(set(e1.keys()).difference(set(e2.keys())))
    difference2 = list(set(e2.keys()).difference(set(e1.keys())))

    monitoring.to_log_and_console("    ... " + str(len(e1.keys())) + " cells are in '" + str(name1) + "'")
    monitoring.to_log_and_console("    ... " + str(len(e2.keys())) + " cells are in '" + str(name2) + "'")
    if len(difference1) > 0:
        monitoring.to_log_and_console("    ... " + str(len(difference1)) + " cells are in '" + str(name1)
                                      + "' and not in '" + str(name2) + "'", 1)
        s = repr(difference1)
        monitoring.to_log_and_console("        " + s, 2)

    if len(difference2) > 0:
        monitoring.to_log_and_console("    ... " + str(len(difference2)) + " cells are not in '" + str(name1)
                                      + "' but in '" + str(name2) + "'", 1)
        s = repr(difference2)
        monitoring.to_log_and_console("        " + s, 2)

    return intersection


def _compare_lineage(e1, e2, name1, name2, description):

    monitoring.to_log_and_console("  === " + str(description) + " comparison === ", 1)

    intersection = _intersection_cell_keys(e1, e2, name1, name2)
    if len(intersection) > 0:
        n = 0
        for k in intersection:
            if e1[k] != e2[k]:
                n += 1
        monitoring.to_log_and_console("    ... " + str(n) + " cells have different lineages", 1)
        if n > 0:
            for k in intersection:
                if e1[k] != e2[k]:
                    s = "cell #'" + str(k) + "' has different lineage: "
                    s += str(e1[k]) + " and " + str(e2[k])
                    monitoring.to_log_and_console("        " + s, 2)

    return


# def _compare_h_min(e1, e2, name1, name2, description):
#     return


def _compare_volume(e1, e2, name1, name2, description):
    """

    :param e1:
    :param e2:
    :param name1:
    :param name2:
    :param description:
    :return:
    """
    #     dictionary de int
    #     cell_volume.590002 = <type 'int'>
    #     590002: 236936

    monitoring.to_log_and_console("  === " + str(description) + " comparison === ", 1)

    intersection = _intersection_cell_keys(e1, e2, name1, name2)
    if len(intersection) > 0:
        monitoring.to_log_and_console("    ... " + str(len(intersection)) + " cells have different volumes", 1)
        for k in intersection:
            if e1[k] != e2[k]:
                s = "cell #'" + str(k) + "' has different volumes: "
                s += str(e1[k]) + " and " + str(e2[k])
                monitoring.to_log_and_console("        " + s, 2)

    return


# def _compare_sigma(e1, e2, name1, name2, description):
#     return


# def _compare_label_in_time(e1, e2, name1, name2, description):
#     return


def _compare_barycenter(e1, e2, name1, name2, description):
    """

    :param e1:
    :param e2:
    :param name1:
    :param name2:
    :param description:
    :return:
    """

    # 'barycenter': 'cell_barycenter'
    #     dictionary de numpy.ndarray de numpy.float64
    #     cell_barycenter.590002 = <type 'numpy.ndarray'>
    #     590002: array([ 258.41037242,  226.74975943,  303.67167927])

    monitoring.to_log_and_console("  === " + str(description) + " comparison === ", 1)

    intersection = _intersection_cell_keys(e1, e2, name1, name2)

    residual = []

    if len(intersection) > 0:
        for k in intersection:
            residual.append([k, math.sqrt(sum((e1[k] - e2[k]) * (e1[k] - e2[k])))])

    residual = sorted(residual, key=itemgetter(1), reverse=True)
    monitoring.to_log_and_console("    ... largest residual at cell #" + str(residual[0][0]) + " is "
                                  + str(residual[0][1]), 1)
    s = str(e1[residual[0][0]]) + " <-> " + str(e2[residual[0][0]])
    monitoring.to_log_and_console("        " + s, 1)

    # for i in range(min(len(intersection),10)):
    #     print "#" + str(i) + ": " + str(residual[i])

    return


# def _compare_fate(e1, e2, name1, name2, description):
#     return


def _compare_all_cells(e1, e2, name1, name2, description):
    """

    :param e1:
    :param e2:
    :param name1:
    :param name2:
    :param description:
    :return:
    """

    # 'all-cells': 'all_cells'  # liste de toutes les cellules ?
    #     liste de numpy.int64
    #     all_cells = <type 'list'>

    monitoring.to_log_and_console("  === " + str(description) + " comparison === ", 1)

    difference1 = list(set(e1).difference(set(e2)))
    difference2 = list(set(e2).difference(set(e1)))

    if len(difference1) > 0:
        monitoring.to_log_and_console("    ... cells that are in '" + str(name1) + "' and not in '"
                                      + str(name2) + "'", 1)
        s = repr(difference1)
        monitoring.to_log_and_console("        " + s, 1)

    if len(difference2) > 0:
        monitoring.to_log_and_console("    ... cells that are not in '" + str(name1) + "' but in '"
                                      + str(name2) + "'", 1)
        s = repr(difference2)
        monitoring.to_log_and_console("        " + s, 1)

    return


def _compare_principal_value(e1, e2, name1, name2, description):
    """

    :param e1:
    :param e2:
    :param name1:
    :param name2:
    :param description:
    :return:
    """

    # 'principal-value': 'cell_principal_values'
    #     dictionary de liste de numpy.float64
    #     cell_principal_values.590002 = <type 'list'>
    #     590002: [1526.0489371146978, 230.60881177650205, 91.063513300019849]

    monitoring.to_log_and_console("  === " + str(description) + " comparison === ", 1)

    intersection = _intersection_cell_keys(e1, e2, name1, name2)

    residual = []

    if len(intersection) > 0:
        for k in intersection:
            residual.append([k, max(abs(np.array(e1[k]) - np.array(e2[k])))])

    residual = sorted(residual, key=itemgetter(1), reverse=True)
    monitoring.to_log_and_console("    ... largest residual at cell #" + str(residual[0][0]) + " is "
                                  + str(residual[0][1]), 1)
    s = str(e1[residual[0][0]]) + " <-> " + str(e2[residual[0][0]])
    monitoring.to_log_and_console("        " + s, 1)

    # for i in range(min(len(intersection),10)):
    #     print "#" + str(i) + ": " + str(residual[i])

    return


# def _compare_name(e1, e2, name1, name2, description):
#     return


def _compare_contact(e1, e2, name1, name2, description):
    """

    :param e1:
    :param e2:
    :param name1:
    :param name2:
    :param description:
    :return:
    """
    # 'contact': 'cell_contact_surface',
    #     dictionary de dictionary de int
    #     cell_contact_surface.590002.590019 = <type 'int'>

    monitoring.to_log_and_console("  === " + str(description) + " comparison === ", 1)

    intersection = _intersection_cell_keys(e1, e2, name1, name2)

    if len(intersection) > 0:
        for k in intersection:

            d = list(set(e1[k].keys()).symmetric_difference(set(e2[k].keys())))
            i = list(set(e1[k].keys()).intersection(set(e2[k].keys())))

            if len(d) > 0:
                s = "cell #" + str(k) + "has different contacts in '" + str(name1) + "' and '" + str(name2) + "'"
                monitoring.to_log_and_console("        " + s, 1)
                monitoring.to_log_and_console("        " + str(e1[k].keys()), 1)
                monitoring.to_log_and_console("        " + "<-> " + str(e2[k].keys()), 1)

            if len(i) > 0:
                for c in i:
                    if e1[k][c] != e2[k][c]:
                        s = "surface contact of cell #" + str(k) + " with cell #" + str(c)
                        s += " is " + str(e1[k][c]) + " in '" + str(name1) + "'"
                        s += " and " + str(e2[k][c]) + " in '" + str(name2) + "'"
                        monitoring.to_log_and_console("        " + s, 1)

    return


# def _compare_history(e1, e2, name1, name2, description):
#    return


def _compare_principal_vector(e1, e2, name1, name2, description):
    """

    :param e1:
    :param e2:
    :param name1:
    :param name2:
    :param description:
    :return:
    """
    # 'principal-vector': 'cell_principal_vectors'    # liste de numpy.ndarray
    #     dictionary de liste de numpy.ndarray de numpy.float64
    #     cell_principal_vectors.590002 = <type 'list'>
    #     590002: [array([ 0.17420991, -0.74923203,  0.63898534]),
    #         array([-0.24877611,  0.59437038,  0.7647446 ]),
    #         array([ 0.95276511,  0.29219037,  0.08284582])]

    monitoring.to_log_and_console("  === " + str(description) + " comparison === ", 1)

    intersection = _intersection_cell_keys(e1, e2, name1, name2)

    residual = []

    if len(intersection) > 0:
        for k in intersection:
            residual.append([k, max(math.sqrt(sum((e1[k][0] - e2[k][0]) * (e1[k][0] - e2[k][0]))),
                                    math.sqrt(sum((e1[k][1] - e2[k][1]) * (e1[k][1] - e2[k][1]))),
                                    math.sqrt(sum((e1[k][2] - e2[k][2]) * (e1[k][2] - e2[k][2]))))])

    residual = sorted(residual, key=itemgetter(1), reverse=True)
    monitoring.to_log_and_console("    ... largest residual at cell #" + str(residual[0][0]) + " is "
                                  + str(residual[0][1]), 1)
    s = str(e1[residual[0][0]]) + "\n" + "        " + " <-> " + str(e2[residual[0][0]])
    monitoring.to_log_and_console("        " + s, 1)

    return


def comparison(d1, d2, features, name1, name2):
    """

    :param d1:
    :param d2:
    :param features:
    :param name1:
    :param name2:
    :return:
    """

    monitoring.to_log_and_console("\n", 1)
    monitoring.to_log_and_console("... comparison between '" + str(name1) + "' and '" + str(name2) + "'", 1)

    #
    # 1. find common keys
    #

    unpairedkeys1 = []
    unpairedkeys2 = []
    pairedkeys = []
    unrecognizedkeys1 = []
    unrecognizedkeys2 = []

    for k1 in d1:

        #
        # loop on known dictionary
        #
        recognizedkey = False
        for k in keydictionary:

            if k1 in keydictionary[k]['input_keys']:
                recognizedkey = True
                pairedkey = False
                #
                # got it, try to pair it
                #
                for k2 in d2:
                    if k2 in keydictionary[k]['input_keys']:
                        pairedkey = True
                        pairedkeys.append([k1, k2])
                        break
                if pairedkey is False:
                    unpairedkeys1.append(k1)
                break

        if recognizedkey is False:
            unrecognizedkeys1.append(k1)

    #
    #
    #

    for k2 in d2:

        #
        # loop on known dictionary
        #
        recognizedkey = False
        for k in keydictionary:
            if k2 in keydictionary[k]['input_keys']:
                recognizedkey = True
                pairedkey = False
                #
                # got it, try to pair it
                #
                for k1 in d1:
                    if k1 in keydictionary[k]['input_keys']:
                        pairedkey = True
                        # pairedkeys.append([k1,k2])
                        break
                if pairedkey is False:
                    unpairedkeys2.append(k2)
                break

        if recognizedkey is False:
            unrecognizedkeys2.append(k2)

    #
    # first output, compare the dictionaries keys
    #

    print_summary = False

    if features is None or len(features) == 0:
        print_summary = True

    #
    #
    #

    if print_summary is True:
        monitoring.to_log_and_console("    found keys", 1)
        if len(pairedkeys) > 0:
            monitoring.to_log_and_console("    ... common keys to '" + str(name1) + "' and '" + str(name2) + "'", 1)
            for p in pairedkeys:
                monitoring.to_log_and_console("        " + str(p[0]) + " <-> " + str(p[1]), 1)
        if len(unpairedkeys1) > 0:
            monitoring.to_log_and_console("    ... keys in '" + str(name1) + "' and not in '" + str(name2) + "'", 1)
            for k in unpairedkeys1:
                monitoring.to_log_and_console("        " + str(k), 1)
        if len(unpairedkeys2) > 0:
            monitoring.to_log_and_console("    ... keys not in '" + str(name1) + "' and in '" + str(name2) + "'", 1)
            for k in unpairedkeys2:
                monitoring.to_log_and_console("        " + str(k), 1)
        if len(unrecognizedkeys1) > 0:
            monitoring.to_log_and_console("    ... keys in '" + str(name1) + "' not recognized", 1)
            for k in unrecognizedkeys1:
                monitoring.to_log_and_console("        " + str(k), 1)
        if len(unrecognizedkeys2) > 0:
            monitoring.to_log_and_console("    ... keys in '" + str(name2) + "' not recognized", 1)
            for k in unrecognizedkeys2:
                monitoring.to_log_and_console("        " + str(k), 1)

    #
    # 2. perform a comparison key by key
    #

    if len(pairedkeys) == 0:
        monitoring.to_log_and_console("... no common keys between '" + str(name1) + "' and '" + str(name2) + "'", 1)
        monitoring.to_log_and_console("    comparison is not possible", 1)
        return

    #
    # recall that the dictionary keys are the 'output_key' of the keydictionary
    #

    if features is None or len(features) == 0:

        monitoring.to_log_and_console("", 1)

        for pk in pairedkeys:
            if pk[0] == keydictionary['lineage']['output_key']:
                _compare_lineage(d1[pk[0]], d2[pk[1]], name1, name2, pk[0])
            elif pk[0] == keydictionary['h_min']['output_key']:
                pass
                # monitoring.to_log_and_console("    comparison of '" + str(pk[0]) + "' not implemented yet", 1)
            elif pk[0] == keydictionary['volume']['output_key']:
                _compare_volume(d1[pk[0]], d2[pk[1]], name1, name2, pk[0])
            elif pk[0] == keydictionary['sigma']['output_key']:
                pass
                # monitoring.to_log_and_console("    comparison of '" + str(pk[0]) + "' not implemented yet", 1)
            elif pk[0] == keydictionary['label_in_time']['output_key']:
                pass
                # monitoring.to_log_and_console("    comparison of '" + str(pk[0]) + "' not implemented yet", 1)
            elif pk[0] == keydictionary['barycenter']['output_key']:
                _compare_barycenter(d1[pk[0]], d2[pk[1]], name1, name2, pk[0])
            elif pk[0] == keydictionary['fate']['output_key']:
                pass
                # monitoring.to_log_and_console("    comparison of '" + str(pk[0]) + "' not implemented yet", 1)
            elif pk[0] == keydictionary['all-cells']['output_key']:
                _compare_all_cells(d1[pk[0]], d2[pk[1]], name1, name2, pk[0])
            elif pk[0] == keydictionary['principal-value']['output_key']:
                _compare_principal_value(d1[pk[0]], d2[pk[1]], name1, name2, pk[0])
            elif pk[0] == keydictionary['name']['output_key']:
                pass
                # monitoring.to_log_and_console("    comparison of '" + str(pk[0]) + "' not implemented yet", 1)
            elif pk[0] == keydictionary['contact']['output_key']:
                _compare_contact(d1[pk[0]], d2[pk[1]], name1, name2, pk[0])
            elif pk[0] == keydictionary['history']['output_key']:
                pass
                # monitoring.to_log_and_console("    comparison of '" + str(pk[0]) + "' not implemented yet", 1)
            elif pk[0] == keydictionary['principal-vector']['output_key']:
                _compare_principal_vector(d1[pk[0]], d2[pk[1]], name1, name2, pk[0])
            else:
                monitoring.to_log_and_console("    unknown key '" + str(pk[0]) + "' for comparison", 1)

    else:
        for f in features:
            if f not in keydictionary:
                monitoring.to_log_and_console("    unknown property '" + str(f) + "' for comparison", 1)
                continue

            outk = keydictionary[f]['output_key']

            for i in range(len(pairedkeys)):
                if pairedkeys[i][0] == outk:
                    if outk == keydictionary['lineage']['output_key']:
                        _compare_lineage(d1[outk], d2[outk], name1, name2, outk)
                    elif outk == keydictionary['h_min']['output_key']:
                        pass
                        # monitoring.to_log_and_console("    comparison of '" + str(outk) + "' not implemented yet", 1)
                    elif outk == keydictionary['volume']['output_key']:
                        _compare_volume(d1[outk], d2[outk], name1, name2, outk)
                    elif outk == keydictionary['sigma']['output_key']:
                        pass
                        # monitoring.to_log_and_console("    comparison of '" + str(outk) + "' not implemented yet", 1)
                    elif outk == keydictionary['label_in_time']['output_key']:
                        pass
                        # monitoring.to_log_and_console("    comparison of '" + str(outk) + "' not implemented yet", 1)
                    elif outk == keydictionary['barycenter']['output_key']:
                        _compare_barycenter(d1[outk], d2[outk], name1, name2, outk)
                    elif outk == keydictionary['fate']['output_key']:
                        pass
                        # monitoring.to_log_and_console("    comparison of '" + str(outk) + "' not implemented yet", 1)
                    elif outk == keydictionary['all-cells']['output_key']:
                        _compare_all_cells(d1[outk], d2[outk], name1, name2, outk)
                    elif outk == keydictionary['principal-value']['output_key']:
                        _compare_principal_value(d1[outk], d2[outk], name1, name2, outk)
                    elif outk == keydictionary['name']['output_key']:
                        pass
                        # monitoring.to_log_and_console("    comparison of '" + str(outk) + "' not implemented yet", 1)
                    elif outk == keydictionary['contact']['output_key']:
                        _compare_contact(d1[outk], d2[outk], name1, name2, outk)
                    elif outk == keydictionary['history']['output_key']:
                        pass
                        # monitoring.to_log_and_console("    comparison of '" + str(outk) + "' not implemented yet", 1)
                    elif outk == keydictionary['principal-vector']['output_key']:
                        _compare_principal_vector(d1[outk], d2[outk], name1, name2, outk)
                    else:
                        monitoring.to_log_and_console("    unknown key '" + str(outk) + "' for comparison", 1)
                    break

    return


########################################################################################
#
#
#
########################################################################################

class DiagnosisParameters(object):

    def __init__(self):
        #
        #
        #
        self.minimal_volume = 0
        self.items = 10

    def write_parameters(self, log_file_name):
        with open(log_file_name, 'a') as logfile:
            logfile.write("\n")
            logfile.write('DiagnosisParameters\n')

            logfile.write('- minimal_volume = ' + str(self.minimal_volume) + '\n')
            logfile.write('- items = '+str(self.items) + '\n')

            logfile.write("\n")
        return

    def print_parameters(self):
        print("")
        print('DiagnosisParameters')

        print('- minimal_volume = ' + str(self.minimal_volume))
        print('- items = ' + str(self.items))

        print("")

    def update_from_args(self, args):
        if hasattr(args, 'diagnosis_minimal_volume'):
            if args.diagnosis_minimal_volume is not None:
                self.minimal_volume = args.diagnosis_minimal_volume
        if hasattr(args, 'diagnosis_items'):
            if args.diagnosis_items is not None:
                self.items = args.diagnosis_items


def _diagnosis_lineage(direct_lineage, description, time_digits_for_cell_id=4):

    proc = "_diagnosis_lineage"

    monitoring.to_log_and_console("  === " + str(description) + " diagnosis === ", 1)

    first_time, last_time = _get_time_interval_from_lineage(direct_lineage,
                                                            time_digits_for_cell_id=time_digits_for_cell_id)
    monitoring.to_log_and_console("  - estimated time interval = [" + str(first_time) + ", " + str(last_time) + "]", 1)

    #
    # cell with more than 2 daughters
    #
    multiple_daughters = [cell for cell in direct_lineage if len(direct_lineage[cell]) > 2]
    divisions = [cell for cell in direct_lineage if len(direct_lineage[cell]) >= 2]

    #
    # get cells without daughters, remove cells from the last time point
    #
    direct_nodes = list(set(direct_lineage.keys()).union(set([v for values in direct_lineage.values() for v in values])))
    leaves = set(direct_nodes) - set(direct_lineage.keys())
    early_leaves = [leave for leave in leaves if (leave/10**time_digits_for_cell_id) < last_time]

    #
    # build a reverse lineage
    #
    reverse_lineage = {}
    for k, values in direct_lineage.iteritems():
        for v in values:
            if v not in reverse_lineage:
                reverse_lineage[v] = [k]
            else:
                reverse_lineage[v].append(k)

    #
    # histogram of lengths for terminal branches
    #
    branch_length_histogram = {}
    branch_lengths = []
    for leaf in leaves:
        le = leaf
        branch = [le]
        while True:
            if len(reverse_lineage.get(le, '')) == 1:
                le = reverse_lineage[le][0]
                if len(direct_lineage.get(le, '')) == 1:
                    branch.append(le)
                else:
                    break
            else:
                break
        length = len(branch)
        branch_lengths.append(length)
        if length in branch_length_histogram:
            branch_length_histogram[length] += 1
        else:
            branch_length_histogram[length] = 1
    #
    # get cells with more than 1 mother
    #
    multiple_mothers = [cell for cell in reverse_lineage if len(reverse_lineage[cell]) > 1]

    #
    # get cells without mother, remove cells from the first time point
    #
    reverse_nodes = list(set(reverse_lineage.keys()).union(set([v for values in reverse_lineage.values() for v in values])))
    orphans = set(reverse_nodes) - set(reverse_lineage.keys())
    late_orphans = [orphan for orphan in orphans if (orphan / 10 ** time_digits_for_cell_id) > first_time]

    monitoring.to_log_and_console("  - found " + str(len(direct_nodes)) + " cells", 1)

    if len(late_orphans) > 0:
        late_orphans.sort()
        monitoring.to_log_and_console("  - " + str(len(late_orphans))
                                      + " lineage branches starting after the first time point", 1)
        _print_list(late_orphans, time_digits_for_cell_id=time_digits_for_cell_id)
    if len(multiple_mothers) > 0:
        multiple_mothers.sort()
        monitoring.to_log_and_console("  - " + str(len(multiple_mothers)) + " cells with multiple mother cells", 1)
        _print_list(multiple_mothers, time_digits_for_cell_id=time_digits_for_cell_id)

    if len(leaves) > 0:
        monitoring.to_log_and_console("  - " + str(len(leaves))
                                      + " lineage terminal branches", 1)
    if len(early_leaves) > 0:
        early_leaves.sort()
        monitoring.to_log_and_console("  - " + str(len(early_leaves))
                                      + " lineage terminal branches ending before the last time point", 1)
        _print_list(early_leaves, time_digits_for_cell_id=time_digits_for_cell_id)

    if len(divisions) > 0:
        divisions.sort()
        monitoring.to_log_and_console("  - " + str(len(divisions)) + " cell divisions", 1)
    if len(multiple_daughters) > 0:
        multiple_daughters.sort()
        monitoring.to_log_and_console("  - " + str(len(multiple_daughters))
                                      + " divisions yielding more than 2 branches", 1)
        _print_list(multiple_daughters, time_digits_for_cell_id=time_digits_for_cell_id)

    monitoring.to_log("  - terminal branch lengths: ")
    monitoring.to_log("    " + str(branch_lengths))
    # lengths = branch_length_histogram.keys()
    # lengths.sort()
    # for le in lengths:
    #    monitoring.to_log_and_console("    length = " + str(le) + " : " + str(branch_length_histogram[le])
    #                                  + " branches ", 1)
    return


# def _diagnosis_h_min(d):
#     return


def _diagnosis_volume(dictionary, description, diagnosis_parameters=None, time_digits_for_cell_id=4):
    """

    :param dictionary:
    :param description:
    :param diagnosis_parameters:
    :return:
    """

    #     dictionary de int
    #     cell_volume.590002 = <type 'int'>
    #     590002: 236936

    monitoring.to_log_and_console("  === " + str(description) + " diagnosis === ", 1)

    all_cell_with_volume = set(dictionary.keys())
    cell_with_volume = [c for c in all_cell_with_volume if _cell_id(c, time_digits_for_cell_id) != 1]
    background_with_volume = [c for c in all_cell_with_volume if _cell_id(c, time_digits_for_cell_id) == 1]

    volume = [[c, dictionary[c]] for c in cell_with_volume]
    volume = sorted(volume, key=itemgetter(1))

    monitoring.to_log_and_console("    ... found " + str(len(background_with_volume))
                                  + " background cells with volume", 1)
    monitoring.to_log_and_console("    ... found " + str(len(cell_with_volume)) + " cells with volume", 1)

    monitoring.to_log_and_console("    ... smallest volumes", 1)

    d = DiagnosisParameters()
    n = int(d.items)
    v = int(d.minimal_volume)

    if isinstance(diagnosis_parameters, DiagnosisParameters):
        n = int(diagnosis_parameters.items)
        v = int(diagnosis_parameters.minimal_volume)

    for i in range(len(dictionary.keys())):
        if (n > 0 and i < n) or (v > 0 and int(volume[i][1]) <= v):
            s = _decode_cell_id(volume[i][0]) + " has volume = " + str(volume[i][1])
            monitoring.to_log_and_console("        " + s, 1)

    return


# def _diagnosis_sigma(d):
#     return


# def _diagnosis_label_in_time(d):
#     return


# def _diagnosis_barycenter(d):
#     return


# def _diagnosis_fate(d):
#     return


# def _diagnosis_all_cells(d):
#     return


# def _diagnosis_principal_value(d):
#     return


# def _diagnosis_name(d):
#     return


# def _diagnosis_contact(d):
#     return


# def _diagnosis_history(d):
#     return


# def _diagnosis_principal_vector(d):
#     return


def diagnosis(d, features, diagnosis_parameters):
    """

    :param d:
    :param features:
    :param diagnosis_parameters:
    :return:
    """

    # monitoring.to_log_and_console("\n", 1)
    monitoring.to_log_and_console("... diagnosis", 1)

    if features is None or len(features) == 0:
        for k in d:
            if k == keydictionary['lineage']['output_key']:
                _diagnosis_lineage(d[k], k)
            elif k == keydictionary['h_min']['output_key']:
                pass
                # monitoring.to_log_and_console("    diagnosis of '" + str(k) + "' not implemented yet", 1)
            elif k == keydictionary['volume']['output_key']:
                _diagnosis_volume(d[k], k, diagnosis_parameters=diagnosis_parameters)
            elif k == keydictionary['surface']['output_key']:
                pass
            elif k == keydictionary['sigma']['output_key']:
                pass
                # monitoring.to_log_and_console("    diagnosis of '" + str(k) + "' not implemented yet", 1)
            elif k == keydictionary['label_in_time']['output_key']:
                pass
                # monitoring.to_log_and_console("    diagnosis of '" + str(k) + "' not implemented yet", 1)
            elif k == keydictionary['barycenter']['output_key']:
                pass
                # monitoring.to_log_and_console("    diagnosis of '" + str(k) + "' not implemented yet", 1)
            elif k == keydictionary['fate']['output_key']:
                pass
                # monitoring.to_log_and_console("    diagnosis of '" + str(k) + "' not implemented yet", 1)
            elif k == keydictionary['all-cells']['output_key']:
                pass
                # monitoring.to_log_and_console("    diagnosis of '" + str(k) + "' not implemented yet", 1)
            elif k == keydictionary['principal-value']['output_key']:
                pass
                # monitoring.to_log_and_console("    diagnosis of '" + str(k) + "' not implemented yet", 1)
            elif k == keydictionary['name']['output_key']:
                pass
                # monitoring.to_log_and_console("    diagnosis of '" + str(k) + "' not implemented yet", 1)
            elif k == keydictionary['contact']['output_key']:
                pass
                # monitoring.to_log_and_console("    diagnosis of '" + str(k) + "' not implemented yet", 1)
            elif k == keydictionary['history']['output_key']:
                pass
                # monitoring.to_log_and_console("    diagnosis of '" + str(k) + "' not implemented yet", 1)
            elif k == keydictionary['principal-vector']['output_key']:
                pass
                # monitoring.to_log_and_console("    diagnosis of '" + str(k) + "' not implemented yet", 1)
            else:
                monitoring.to_log_and_console("    unknown key '" + str(k) + "' for diagnosis", 1)

    else:
        for f in features:
            if f not in keydictionary:
                monitoring.to_log_and_console("    unknown property '" + str(f) + "' for comparison", 1)
                continue

            outk = keydictionary[f]['output_key']

            for i in range(len(d.keys())):
                if d.keys()[i] == outk:
                    if outk == keydictionary['lineage']['output_key']:
                        _diagnosis_lineage(d[outk], outk)
                    elif outk == keydictionary['h_min']['output_key']:
                        pass
                        # monitoring.to_log_and_console("    diagnosis of '" + str(k) + "' not implemented yet", 1)
                    elif outk == keydictionary['volume']['output_key']:
                        _diagnosis_volume(d[outk], outk, diagnosis_parameters=diagnosis_parameters)
                    elif outk == keydictionary['surface']['output_key']:
                        pass
                    elif outk == keydictionary['sigma']['output_key']:
                        pass
                        # monitoring.to_log_and_console("    diagnosis of '" + str(k) + "' not implemented yet", 1)
                    elif outk == keydictionary['label_in_time']['output_key']:
                        pass
                        # monitoring.to_log_and_console("    diagnosis of '" + str(k) + "' not implemented yet", 1)
                    elif outk == keydictionary['barycenter']['output_key']:
                        pass
                        # monitoring.to_log_and_console("    diagnosis of '" + str(k) + "' not implemented yet", 1)
                    elif outk == keydictionary['fate']['output_key']:
                        pass
                        # monitoring.to_log_and_console("    diagnosis of '" + str(k) + "' not implemented yet", 1)
                    elif outk == keydictionary['all-cells']['output_key']:
                        pass
                        # monitoring.to_log_and_console("    diagnosis of '" + str(k) + "' not implemented yet", 1)
                    elif outk == keydictionary['principal-value']['output_key']:
                        pass
                        # monitoring.to_log_and_console("    diagnosis of '" + str(k) + "' not implemented yet", 1)
                    elif outk == keydictionary['name']['output_key']:
                        pass
                        # monitoring.to_log_and_console("    diagnosis of '" + str(k) + "' not implemented yet", 1)
                    elif outk == keydictionary['contact']['output_key']:
                        pass
                        # monitoring.to_log_and_console("    diagnosis of '" + str(k) + "' not implemented yet", 1)
                    elif outk == keydictionary['history']['output_key']:
                        pass
                        # monitoring.to_log_and_console("    diagnosis of '" + str(k) + "' not implemented yet", 1)
                    elif outk == keydictionary['principal-vector']['output_key']:
                        pass
                        # monitoring.to_log_and_console("    diagnosis of '" + str(k) + "' not implemented yet", 1)
                    else:
                        monitoring.to_log_and_console("    unknown key '" + str(outk) + "' for comparison", 1)
                    break
        pass

    return


########################################################################################
#
#
#
########################################################################################

def _cell_id(c, time_digits_for_cell_id=4):
    t = c // 10 ** time_digits_for_cell_id
    c -= t * 10 ** time_digits_for_cell_id
    return c


def _print_list(tab, time_digits_for_cell_id=4):
    for c in tab:
        t = c // 10**time_digits_for_cell_id
        c -= t * 10**time_digits_for_cell_id
        monitoring.to_log_and_console("    - cell #" + str(c) + " of time " + str(t), 2)


def check_volume_lineage(d, time_digits_for_cell_id=4):
    """
    Compare cells found in the volume dictionary with cells found in the lineage dictionary
    :param d:
    :param time_digits_for_cell_id:
    :return:
    """

    proc = "check_volume_lineage"

    direct_lineage = get_dictionary_entry(d, 'lineage')
    if direct_lineage == {}:
        monitoring.to_log_and_console(str(proc) + ": empty lineage information")
        sys.exit(1)

    dict_volume = get_dictionary_entry(d, 'volume')

    #
    # check whether cells in volume dictionary are in lineage and vice-versa
    #
    if dict_volume is not {}:
        all_nodes = list(set(direct_lineage.keys()).union(set([v for values in direct_lineage.values()
                                                               for v in values])))
        nodes = [c for c in all_nodes if _cell_id(c, time_digits_for_cell_id) != 1]
        #
        all_cell_with_volume = set(dict_volume.keys())
        cell_with_volume = [c for c in all_cell_with_volume if _cell_id(c, time_digits_for_cell_id) != 1]
        background_with_volume = [c for c in all_cell_with_volume if _cell_id(c, time_digits_for_cell_id) == 1]

        cell_in_volume_not_in_lineage = list(set(cell_with_volume).difference(set(nodes)))
        cell_in_lineage_not_in_volume = list(set(nodes).difference(set(cell_with_volume)))

        monitoring.to_log_and_console("", 1)
        monitoring.to_log_and_console("  === volume/lineage cross-checking === ", 1)

        monitoring.to_log_and_console("    ... found " + str(len(background_with_volume))
                                      + " background cells with volume", 1)
        monitoring.to_log_and_console("    ... found " + str(len(cell_with_volume)) + " cells with volume", 1)
        monitoring.to_log_and_console("    ... found " + str(len(nodes)) + " cells in lineage", 1)

        if len(cell_in_volume_not_in_lineage) > 0:
            cell_in_volume_not_in_lineage.sort()
            if len(cell_in_volume_not_in_lineage) > 0:
                monitoring.to_log_and_console("  - " + str(len(cell_in_volume_not_in_lineage))
                                              + " cells with volume not present in lineage: ", 1)
                _print_list(cell_in_volume_not_in_lineage, time_digits_for_cell_id=time_digits_for_cell_id)
        if len(cell_in_lineage_not_in_volume) > 0:
            cell_in_lineage_not_in_volume.sort()
            monitoring.to_log_and_console("  - " + str(len(cell_in_lineage_not_in_volume))
                                          + " cells present in lineage without volume: ", 1)
            _print_list(cell_in_lineage_not_in_volume, time_digits_for_cell_id=time_digits_for_cell_id)

        _diagnosis_volume(dict_volume, keydictionary['volume']['output_key'],
                          time_digits_for_cell_id=time_digits_for_cell_id)

    _diagnosis_lineage(direct_lineage, keydictionary['lineage']['output_key'],
                       time_digits_for_cell_id=time_digits_for_cell_id)
    monitoring.to_log_and_console("", 1)

    return


def check_volume_image(volume_from_lineage, image_name, current_time, time_digits_for_cell_id=4):
    """
    Compare cells found in the volume dictionary with cells found in one image
    :param volume_from_lineage:
    :param image_name:
    :param current_time:
    :param time_digits_for_cell_id:
    :return:
    """

    #
    # read volumes from image
    #
    readim = imread(image_name)
    labels_from_image = np.unique(readim)
    volume = nd.sum(np.ones_like(readim), readim, index=np.int16(labels_from_image))
    del readim

    labels_from_image = [current_time * 10 ** time_digits_for_cell_id + int(label) for label in labels_from_image]
    volume_from_image = dict(zip(labels_from_image, volume))

    labels_from_lineage = [label for label in volume_from_lineage.keys()
                           if label/10 ** time_digits_for_cell_id == current_time]

    labels_in_lineage_not_in_image = list(set(labels_from_lineage).difference(set(labels_from_image)))
    labels_in_image_not_in_lineage = list(set(labels_from_image).difference(set(labels_from_lineage)))
    labels_in_both = list(set(labels_from_image).intersection(set(labels_from_lineage)))

    volume_error = [label for label in labels_in_both if volume_from_image[label] != volume_from_lineage[label]]

    if len(labels_in_lineage_not_in_image) > 0:
        labels_in_lineage_not_in_image.sort()
        monitoring.to_log_and_console("  - " + str(len(labels_in_lineage_not_in_image))
                                      + " cells present in volume dictionary not in image: ", 1)
        _print_list(labels_in_lineage_not_in_image, time_digits_for_cell_id=time_digits_for_cell_id)

    if len(labels_in_image_not_in_lineage) > 0:
        labels_in_image_not_in_lineage.sort()
        monitoring.to_log_and_console("  - " + str(len(labels_in_image_not_in_lineage))
                                      + " cells present in image not in volume dictionary: ", 1)
        _print_list(labels_in_image_not_in_lineage, time_digits_for_cell_id=time_digits_for_cell_id)

    if len(volume_error) > 0:
        volume_error.sort()
        monitoring.to_log_and_console("  - " + str(len(volume_error))
                                      + " cells with different volume in image and volume dictionary: ", 1)
        _print_list(volume_error, time_digits_for_cell_id=time_digits_for_cell_id)
    return


########################################################################################
#
# utilities for debugging, etc.
#
########################################################################################

def print_keys(d, desc=None):

    monitoring.to_log_and_console("\n", 1)
    if desc is None:
        monitoring.to_log_and_console("... contents", 1)
    else:
        monitoring.to_log_and_console("... contents of '" + str(desc) + "'", 1)

    if type(d) is dict:
        if d == {}:
            monitoring.to_log_and_console("    " + "empty dictionary", 1)
        else:
            monitoring.to_log_and_console("    " + "keys are:", 1)
            for k in d:
                monitoring.to_log_and_console("    " + "- " + str(k), 1)
    else:
        monitoring.to_log_and_console("    " + "input is not a dictionary", 1)

    return


def print_type(d, t=None, desc=None):

    if desc is None:
        desc = ""
    if t is None:
        t = ""

    if type(d) is dict:

        print "type of " + desc + " is " + str(t) + str(type(d))
        for k in d:
            print_type(d[k], t + str(type(d)) + ".", desc + "." + str(k))

    elif type(d) in (list, np.array, np.ndarray):
        print "type of " + desc + " is " + str(t) + str(type(d))
        print_type(d[0], t + str(type(d)) + ".", desc + "[0]")

    else:
        print "type of " + desc + " is " + str(t) + str(type(d))

    return


########################################################################################
#
# write tlp file
# this was inspired from pkl2tlp from L. Guignard
#
########################################################################################

def write_tlp_file(tlpfilename, dictionary):
    """

    :param tlpfilename:
    :param dictionary:
    :return:
    """

    proc = "write_tlp_file"

    #
    # is there a lineage
    #
    if keydictionary['lineage']['output_key'] in dictionary:
        lineage = dictionary[keydictionary['lineage']['output_key']]
    else:
        monitoring.to_log_and_console(proc + ": no lineage was found.")
        return

    #
    # open file
    #
    f = open(tlpfilename, "w")
    f.write("(tlp \"2.0\"\n")

    #
    # write nodes = lineage.keys() + lineage.values()
    #
    nodes = set(lineage.keys()).union(set([v for values in lineage.values() for v in values]))
    f.write("(nodes ")
    for n in nodes:
        f.write(str(n) + " ")
    f.write(")\n")

    #
    # write edges
    #
    count_edges = 0
    for m, ds in lineage.iteritems():
        count_edges += 1
        for d in ds:
            f.write("(edge " + str(count_edges) + " " + str(m) + " " + str(d) + ")\n")

    #
    # write node ids
    #
    f.write("(property 0 int \"id\"\n")
    f.write("\t(default \"0\" \"0\")\n")
    for node in nodes:
        f.write("\t(node " + str(node) + str(" \"") + str(node) + "\")\n")
    f.write(")\n")

    #
    #
    #
    for p in dictionary:
        if p == keydictionary['lineage']['output_key']:
            pass
        elif p == keydictionary['all-cells']['output_key']:
            pass
        #
        # property as single double
        #
        elif p == keydictionary['volume']['output_key'] or p == keydictionary['surface']['output_key'] \
                or p == keydictionary['compactness']['output_key']:
            prop = dictionary[p]
            default = np.median(prop.values())
            f.write("(property 0 double \"" + str(p) + "\"\n")
            f.write("\t(default \"" + str(default) + "\" \"0\")\n")
            for node in nodes:
                f.write("\t(node " + str(node) + str(" \"") + str(prop.get(node, default)) + "\")\n")
            f.write(")\n")
        #
        # property as string
        #
        elif p == keydictionary['fate']['output_key'] or p == keydictionary['fate2']['output_key'] \
                or p == keydictionary['fate3']['output_key'] or p == keydictionary['fate4']['output_key'] \
                or p == keydictionary['name']['output_key']:
            prop = dictionary[p]
            f.write("(property 0 string \"" + str(p) + "\"\n")
            f.write("\t(default \"" + "no string" + "\" \"0\")\n")
            for node in nodes:
                f.write("\t(node " + str(node) + str(" \"") + str(prop.get(node, "no string")) + "\")\n")
            f.write(")\n")
        #
        #
        #
        elif p == keydictionary['h_min']['output_key'] or p == keydictionary['sigma']['output_key'] \
                or p == keydictionary['label_in_time']['output_key'] \
                or p == keydictionary['barycenter']['output_key'] \
                or p == keydictionary['principal-value']['output_key'] \
                or p == keydictionary['contact']['output_key'] \
                or p == keydictionary['history']['output_key'] \
                or p == keydictionary['principal-vector']['output_key'] \
                or p == keydictionary['name-score']['output_key']:
            pass
        else:
            monitoring.to_log_and_console(proc + ": property '" + str(p) + "' not handled yet for writing.")

    #
    # close file
    #
    f.write(")")
    f.write("(nodes ")

    f.close()
