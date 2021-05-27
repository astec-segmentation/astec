import os
import imp
import sys
import copy
import collections
import numpy as np
import random
import math
import operator

import common
import properties as properties

#
#
#
#
#

monitoring = common.Monitoring()

_instrumented_ = False

########################################################################################
#
# classes
# - computation parameters
#
########################################################################################


class NamingParameters(common.PrefixedParameter):

    ############################################################
    #
    # initialisation
    #
    ############################################################

    def __init__(self):

        common.PrefixedParameter.__init__(self)

        if "doc" not in self.__dict__:
            self.doc = {}

        self.inputFiles = []
        self.outputFile = None
        self.referenceFiles = []
        #
        # for test:
        # names will be deleted, and tried to be rebuilt
        self.testFile = None
        #
        #
        #
        self.reference_diagnosis = False
        #
        #
        #
        self.use_symmetric_neighborhood = False
        #
        #
        #
        self.improve_consistency = False

    ############################################################
    #
    # print / write
    #
    ############################################################

    def print_parameters(self):
        print("")
        print('#')
        print('# NamingParameters')
        print('#')
        print("")

        common.PrefixedParameter.print_parameters(self)

        self.varprint('inputFiles', self.inputFiles)
        self.varprint('outputFile', self.outputFile)
        self.varprint('referenceFiles', self.referenceFiles)
        self.varprint('testFile', self.testFile)
        self.varprint('reference_diagnosis', self.reference_diagnosis)
        self.varprint('use_symmetric_neighborhood', self.use_symmetric_neighborhood)
        self.varprint('improve_consistency', self.improve_consistency)

    def write_parameters_in_file(self, logfile):
        logfile.write("\n")
        logfile.write("# \n")
        logfile.write("# PostCorrectionParameters\n")
        logfile.write("# \n")
        logfile.write("\n")

        common.PrefixedParameter.write_parameters_in_file(self, logfile)

        self.varwrite(logfile, 'inputFiles', self.inputFiles)
        self.varwrite(logfile, 'outputFile', self.outputFile)
        self.varwrite(logfile, 'referenceFiles', self.referenceFiles)
        self.varwrite(logfile, 'testFile', self.testFile)
        self.varwrite(logfile, 'reference_diagnosis', self.reference_diagnosis)
        self.varwrite(logfile, 'use_symmetric_neighborhood', self.use_symmetric_neighborhood)
        self.varwrite(logfile, 'improve_consistency', self.improve_consistency)

    def write_parameters(self, log_file_name):
        with open(log_file_name, 'a') as logfile:
            self.write_parameters_in_file(logfile)
        return

    ############################################################
    #
    # update
    #
    ############################################################

    def update_from_parameters(self, parameters):
        self.inputFiles = self.read_parameter(parameters, 'inputFiles', self.inputFiles)
        self.outputFile = self.read_parameter(parameters, 'outputFile', self.outputFile)
        self.referenceFiles = self.read_parameter(parameters, 'referenceFiles', self.referenceFiles)
        self.testFile = self.read_parameter(parameters, 'testFile', self.testFile)
        self.reference_diagnosis = self.read_parameter(parameters, 'reference_diagnosis', self.reference_diagnosis)
        self.use_symmetric_neighborhood = self.read_parameter(parameters, 'use_symmetric_neighborhood',
                                                              self.use_symmetric_neighborhood)
        self.improve_consistency = self.read_parameter(parameters, 'improve_consistency', self.improve_consistency)

    def update_from_parameter_file(self, parameter_file):
        if parameter_file is None:
            return
        if not os.path.isfile(parameter_file):
            print("Error: '" + parameter_file + "' is not a valid file. Exiting.")
            sys.exit(1)

        parameters = imp.load_source('*', parameter_file)
        self.update_from_parameters(parameters)


########################################################################################
#
#
#
########################################################################################

def _get_daughter_names(name):
    #
    # build daughter names from parent name
    #
    # anterior or posterior character 'a' or 'b'
    # stage (round of division)
    # '.'
    # p value (cell index)
    # left or right character '_' or '*'
    #
    abvalue = name.split('.')[0][0]
    stage = name.split('.')[0][1:]
    p = name.split('.')[1][0:4]
    lrvalue = name.split('.')[1][4]
    #
    # build daughter names
    #
    daughters = [abvalue + str(int(stage) + 1) + "." + '{:0{width}d}'.format(2*int(p)-1, width=4) + lrvalue]
    daughters.append(abvalue + str(int(stage) + 1) + "." + '{:0{width}d}'.format(2*int(p), width=4) + lrvalue)
    # print("name = " + str(name) + " -> daughter names = " + str(daughters))
    return daughters


def _get_mother_name(name):
    #
    # build daughter names from parent name
    #
    # anterior or posterior character 'a' or 'b'
    # stage (round of division)
    # '.'
    # p value (cell index)
    # left or right character '_' or '*'
    #
    abvalue = name.split('.')[0][0]
    stage = name.split('.')[0][1:]
    p = name.split('.')[1][0:4]
    lrvalue = name.split('.')[1][4]
    #
    # build parent names
    #
    parent = abvalue + str(int(stage)-1) + "."
    if int(p) % 2 == 1:
        parent += '{:0{width}d}'.format((int(p)+1) // 2, width=4)
    else:
        parent += '{:0{width}d}'.format(int(p) // 2, width=4)
    parent += lrvalue
    # print("name = " + str(name) + " -> parent name = " + str(parent))
    return parent


def _get_sister_name(name):
    sister_names = _get_daughter_names(_get_mother_name(name))
    sister_names.remove(name)
    return sister_names[0]


def _get_symmetric_name(name):
    symname = name[:-1]
    if name[-1] == '*':
        symname += '_'
    elif name[-1] == '_':
        symname += '*'
    else:
        return None
    return symname


def _get_symmetric_neighborhood(neighborhood):
    """
    Changes the names of the cells in the neighborhood to get the symmetric neighborhood
    :param neighborhood:
    :return:
    """
    symneighborhood = {}
    for n in neighborhood:
        if n == 'background':
            sn = 'background'
        else:
            sn = _get_symmetric_name(n)
        symneighborhood[sn] = neighborhood[n]
    return symneighborhood


########################################################################################
#
# diagnosis on naming
# is redundant with the diagnosis made in properties.py
# (except that contact surfaces are assessed too)
#
########################################################################################

def diagnosis(prop, time_digits_for_cell_id=4, verbose=True):
    proc = "diagnosis"
    if 'cell_lineage' not in prop:
        monitoring.to_log_and_console(str(proc) + ": 'cell_lineage' was not in dictionary")
        return None

    if 'cell_contact_surface' not in prop:
        monitoring.to_log_and_console(str(proc) + ": 'cell_contact_surface' was not in dictionary")
        return None

    if 'cell_name' not in prop:
        monitoring.to_log_and_console(str(proc) + ": 'cell_name' was not in dictionary")
        return None

    lineage = prop['cell_lineage']
    name = prop['cell_name']
    contact = prop['cell_contact_surface']

    reverse_lineage = {v: k for k, values in lineage.iteritems() for v in values}

    div = 10 ** time_digits_for_cell_id

    cells = list(set(lineage.keys()).union(set([v for values in lineage.values() for v in values])))
    cells = sorted(cells)

    cells_per_time = {}
    names_per_time = {}
    missing_name = {}
    missing_contact = {}
    error_name = {}
    for c in cells:
        t = int(c) // div
        n = int(c) % div
        #
        # get cells and cell names at each time point
        #
        cells_per_time[t] = cells_per_time.get(t, []) + [n]
        if c in name:
            names_per_time[t] = names_per_time.get(t, []) + [name[c]]

        #
        # check names
        #
        if c not in name:
            missing_name[t] = missing_name.get(t, []) + [n]
        elif c in reverse_lineage:
            mother = reverse_lineage[c]
            if mother not in name:
                if verbose:
                    msg = ": weird, cell " + str(c) + " has a name = " + str(name[c])
                    msg += ", but its mother cell " + str(mother) + " has no name"
                    monitoring.to_log_and_console(str(proc) + msg)
                error_name[t] = error_name.get(t, []) + [n]
            else:
                if len(lineage[mother]) == 1:
                    if name[mother] != name[c]:
                        if verbose:
                            msg = ": weird, cell " + str(c) + " has a name = " + str(name[c])
                            msg += " different than its mother cell " + str(mother) + " name = " + str(name[mother])
                            monitoring.to_log_and_console(str(proc) + msg)
                        error_name[t] = error_name.get(t, []) + [n]
                elif len(lineage[mother]) == 2:
                    daughter_names = _get_daughter_names(name[mother])
                    if name[c] not in daughter_names:
                        if verbose:
                            msg = ": weird, name of cell " + str(c) + " is " + str(name[c])
                            msg += " but should be in " + str(daughter_names)
                            msg += " since its mother cell " + str(mother) + " is named " + str(name[mother])
                            monitoring.to_log_and_console(str(proc) + msg)
                        error_name[t] = error_name.get(t, []) + [n]
                    else:
                        siblings = copy.deepcopy(lineage[mother])
                        siblings.remove(c)
                        daughter_names.remove(name[c])
                        if siblings[0] not in name:
                            if verbose:
                                msg = ": weird, cell " + str(c) + " has no name "
                                msg += ", it should be " + str(daughter_names[0])
                                msg += " since its mother cell " + str(mother) + " is named " + str(name[mother])
                                msg += " and its sibling " + str(c) + " is named " + str(name[c])
                                monitoring.to_log_and_console(str(proc) + msg)
                            error_name[t] = error_name.get(t, []) + [n]
                        elif name[siblings[0]] == name[c]:
                            if verbose:
                                msg = ": weird, name of cell " + str(c) + ", " + str(name[c])
                                msg += ", is the same than its sibling " + str(siblings[0])
                                msg += ", their mother cell " + str(mother) + " is named " + str(name[mother])
                                monitoring.to_log_and_console(str(proc) + msg)
                            error_name[t] = error_name.get(t, []) + [n]
                        elif name[siblings[0]] != daughter_names[0]:
                            if verbose:
                                msg = ": weird, name of cell " + str(siblings[0]) + " is " + str(name[siblings[0]])
                                msg += " but should be " + str(daughter_names[0])
                                msg += " since its mother cell " + str(mother) + " is named " + str(name[mother])
                                msg += " and its sibling " + str(c) + " is named " + str(name[c])
                                monitoring.to_log_and_console(str(proc) + msg)
                            error_name[t] = error_name.get(t, []) + [n]
                else:
                    if verbose:
                        monitoring.to_log_and_console(str(proc) + ": weird, cell " + str(mother) + " has " +
                                                      str(len(lineage[mother])) + " daughter cells")

        #
        # check contact surfaces
        #
        if c not in contact:
            missing_contact[t] = missing_contact.get(t, []) + [n]

    #
    # interval without errors
    #
    first_time = min(cells_per_time.keys())
    last_time = max(cells_per_time.keys())
    if missing_name != {}:
        last_time = min(last_time, min(missing_name.keys())-1)
    if missing_contact != {}:
        last_time = min(last_time, min(missing_contact.keys())-1)
    if error_name != {}:
        last_time = min(last_time, min(error_name.keys())-1)

    #
    # report
    #
    if verbose:
        monitoring.to_log_and_console(str(proc) + ": details")
        monitoring.to_log_and_console("\t - first time in lineage = " + str(min(cells_per_time.keys())))
        monitoring.to_log_and_console("\t   last time in lineage = " + str(max(cells_per_time.keys())))
        monitoring.to_log_and_console("\t - interval without errors = [" + str(first_time) + ", " + str(last_time) + "]")
        monitoring.to_log_and_console("\t - cells in lineage = " + str(len(cells)))
        monitoring.to_log_and_console("\t   #cells at first time = " +
                                      str(len(cells_per_time[min(cells_per_time.keys())])))
        monitoring.to_log_and_console("\t   #cells at last time = " +
                                      str(len(cells_per_time[max(cells_per_time.keys())])))
        # compte le nombre de noms
        monitoring.to_log_and_console("\t - names in lineage = " + str(len(collections.Counter(name.keys()).keys())))

        for t in names_per_time:
            multiples = {k: names_per_time[t].count(k) for k in set(names_per_time[t]) if names_per_time[t].count(k) > 1}
            if multiples != {}:
                monitoring.to_log_and_console("\t - there are " + str(len(multiples)) + " repeated names at time " + str(t))
                for n, p in multiples.iteritems():
                    monitoring.to_log_and_console("\t   " + str(n) + " is repeated " + str(p) + " times ")

        if error_name != {}:
            monitoring.to_log_and_console("\t - first time with cells with name inconsistency = " +
                                          str(min(error_name.keys())))
            for t in sorted(error_name.keys()):
                monitoring.to_log_and_console(
                    "\t   cells with name inconsistency at time " + str(t) + " = " + str(error_name[t]))

        if missing_name != {}:
            monitoring.to_log_and_console("\t - first time with cells without name = " + str(min(missing_name.keys())))
            for t in sorted(missing_name.keys()):
                monitoring.to_log_and_console("\t   cells without name at time " + str(t) + " = " + str(missing_name[t]))

        if missing_contact != {}:
            monitoring.to_log_and_console("\t - first time with cells without contact surfaces = "
                                          + str(min(missing_contact.keys())))
            for t in sorted(missing_contact.keys()):
                monitoring.to_log_and_console("\t   cells without contact surfaces at time " + str(t) + " = " +
                                              str(missing_contact[t]))
        monitoring.to_log_and_console("")

    return [first_time, last_time]


########################################################################################
#
# procedures for test
#
########################################################################################

def _build_test_set(prop, time_digits_for_cell_id=4, ncells=64):
    #
    # from an already named embryo, delete names except at one time point
    #
    proc = "_build_test_set"

    #
    # copy dictionary
    #
    returned_prop = {}
    returned_prop['cell_lineage'] = copy.deepcopy(prop['cell_lineage'])
    returned_prop['cell_name'] = copy.deepcopy(prop['cell_name'])
    returned_prop['cell_contact_surface'] = copy.deepcopy(prop['cell_contact_surface'])

    #
    # get cells to count cells per time point
    #
    lineage = returned_prop['cell_lineage']
    cells = list(set(lineage.keys()).union(set([v for values in lineage.values() for v in values])))
    cells = sorted(cells)

    div = 10 ** time_digits_for_cell_id
    ncells_per_time = np.zeros((max(cells) // div) + 1)
    for c in cells:
        t = int(c) // div
        ncells_per_time[t] += 1

    #
    # draw one time point of the desired number of cells
    #
    indices = np.where(np.array(ncells_per_time == ncells))[0]
    if len(indices) == 0:
        monitoring.to_log_and_console(str(proc) + ": there is no time point with #cells=" + str(ncells))
        indices = np.where(np.array(ncells_per_time > ncells))[0]
        if len(indices) == 0:
            monitoring.to_log_and_console(str(proc) + ": there is no time point with #cells>" + str(ncells))
            for t in range(len(ncells_per_time)):
                if ncells_per_time[t] == 0:
                    continue
                msg = "\t time " + str(t) + ": " + str(ncells_per_time[t]) + " cells"
                monitoring.to_log_and_console(msg)
            return None
        else:
            draw = indices[0]
    else:
        draw = random.choice(indices)

    monitoring.to_log_and_console(str(proc) + ": rename from time point " + str(draw))

    cells = returned_prop['cell_name'].keys()
    for c in cells:
        if int(c) // div != draw:
            del returned_prop['cell_name'][c]

    return returned_prop


def _test_naming(prop, reference_prop, discrepancies):
    proc = "_test_naming"

    monitoring.to_log_and_console("")
    monitoring.to_log_and_console("_test_naming")
    monitoring.to_log_and_console("------------")

    #
    # get the cell names with error
    #
    sisters_errors = {}
    other_errors = {}
    division_errors = {}
    key_list = sorted(prop['cell_name'].keys())
    for k in key_list:
        if k not in reference_prop['cell_name']:
            monitoring.to_log_and_console("\t weird, key " + str(k) + " is not in reference properties")
        elif prop['cell_name'][k] != reference_prop['cell_name'][k]:
            if reference_prop['cell_name'][k] == _get_sister_name(prop['cell_name'][k]):
                if prop['cell_name'][k] not in sisters_errors:
                    sisters_errors[prop['cell_name'][k]] = 1
                    msg = "cell_name[" + str(k) + "]=" + str(prop['cell_name'][k]) + " is named as its sister "
                    msg += str(reference_prop['cell_name'][k])
                    if _get_mother_name(prop['cell_name'][k]) in discrepancies:
                        msg += ", division is incoherent among references"
                    monitoring.to_log_and_console("\t " + msg)
                else:
                    sisters_errors[prop['cell_name'][k]] += 1
                indexed_name = _get_mother_name(prop['cell_name'][k])
                division_errors[indexed_name] = division_errors.get(indexed_name, 0) + 1
            else:
                if prop['cell_name'][k] not in other_errors:
                    other_errors[prop['cell_name'][k]] = 1
                    msg = "cell_name[" + str(k) + "]=" + str(prop['cell_name'][k]) + " is not equal to reference name "
                    msg += str(reference_prop['cell_name'][k])
                    msg += " for cell " + str(k)
                    if _get_mother_name(prop['cell_name'][k]) in discrepancies:
                        msg += ", division is incoherent among references"
                    monitoring.to_log_and_console("\t " + msg)
                else:
                    other_errors[prop['cell_name'][k]] += 1

    #
    # if an error occur, a bad choice has been made at the mother division (first_errors)
    # or at some ancestor (second_errors)
    #
    # first_errors = {}
    # second_errors = {}
    # first_error_mothers = []
    # names = division_errors.keys()
    # for n in names:
    #     if _get_mother_name(n) in names or _get_mother_name(n) in first_error_mothers:
    #         second_errors[n] = division_errors[n]
    #     else:
    #         first_errors[n] = division_errors[n]
    #         if _get_mother_name(n) not in first_error_mothers:
    #             first_error_mothers.append(_get_mother_name(n))

    name_missing = {}
    reference_name = {}
    key_list = sorted(reference_prop['cell_name'].keys())
    for k in key_list:
        indexed_name = reference_prop['cell_name'][k]
        reference_name[indexed_name] = reference_name.get(indexed_name, 0) + 1
        if k not in prop['cell_name']:
            if reference_prop['cell_name'][k] not in name_missing:
                name_missing[reference_prop['cell_name'][k]] = 1
                msg = "reference cell_name[" + str(k) + "]=" + str(reference_prop['cell_name'][k]) + " is not found"
                monitoring.to_log_and_console("\t " + msg)
            else:
                name_missing[reference_prop['cell_name'][k]] += 1

    division_count = 0
    for k in division_errors:
        division_count += division_errors[k]
    other_count = 0
    for k in other_errors:
        other_count += other_errors[k]
    missing_count = 0
    for k in name_missing:
        missing_count += name_missing[k]

    msg = "ground-truth cells = " + str(len(reference_prop['cell_name'])) + " --- "
    msg += "ground-truth names = " + str(len(reference_name)) + " --- "
    msg += "tested cells = " + str(len(prop['cell_name'])) + " --- "
    msg += "retrieved names = " + str(len(reference_name) - len(name_missing)) + "\n"
    msg += "\t missing cell names = " + str(missing_count) + " for " + str(len(name_missing)) + " names --- \n"
    msg += "\t division errors in lineage = " + str(division_count) + " for " + str(len(division_errors)) + " names \n"
    if len(division_errors) > 0:
        msg += "\t    division errors = " + str(sorted(division_errors.keys())) + "\n"
    msg += "\t other errors in lineage = " + str(other_count) + " for " + str(len(other_errors)) + " names --- \n"
    if len(other_errors) > 0:
        msg += "\t    other errors = " + str(sorted(other_errors.keys())) + "\n"
    monitoring.to_log_and_console("summary" + ": " + msg)

    return


########################################################################################
#
#
#
########################################################################################

def _print_neighborhood(neighborhood, title=None):
    msg = ""
    if title is not None and isinstance(title, str):
        msg += title + " = "
    msg += "{"
    key_list = sorted(neighborhood.keys())
    for k in key_list:
        msg += str(k) + ": " + str(neighborhood[k])
        if k != key_list[-1]:
            msg += ",\n\t "
        else:
            msg += "}"
    monitoring.to_log_and_console(msg)


def _print_common_neighborhoods(neighborhood0, neighborhood1, title=None):
    #
    # used by _get_score(), to display neighborhoods after being put in a common frame
    #
    msg = ""
    if title is not None and isinstance(title, str):
        msg += title + " = "
    msg += "{"
    key_list = sorted(neighborhood0.keys())
    for k in key_list:
        msg += str(k) + ": " + str(neighborhood0[k]) + " <-> " + str(neighborhood1[k])
        if k != key_list[-1]:
            msg += ",\n\t "
        else:
            msg += "}"
    monitoring.to_log_and_console(msg)


########################################################################################
#
# how to build the reference neighborhoods
#
########################################################################################

def _add_neighborhoods(previous_neighborhoods, prop, reference_name, time_digits_for_cell_id=4,
                       add_symmetric_neighborhood=False):
    proc = "_add_neighborhoods"

    #
    # build a nested dictionary of neighborhood, where the keys are
    # ['cell name']['reference name']
    # where 'reference name' is the name
    # of the reference lineage, and neighborhood a dictionary of contact surfaces indexed by cell names
    # only consider the first time point after the division
    #

    if 'cell_lineage' not in prop:
        monitoring.to_log_and_console(str(proc) + ": 'cell_lineage' was not in dictionary")
        return

    if 'cell_contact_surface' not in prop:
        monitoring.to_log_and_console(str(proc) + ": 'cell_contact_surface' was not in dictionary")
        return

    if 'cell_name' not in prop:
        monitoring.to_log_and_console(str(proc) + ": 'cell_name' was not in dictionary")
        return

    #
    # remove empty names
    # leading or trailing spaces
    #
    cells = prop['cell_name'].keys()
    for c in cells:
        if prop['cell_name'][c] == '':
            del prop['cell_name'][c]
            continue
        prop['cell_name'][c] = prop['cell_name'][c].strip()

    #
    lineage = prop['cell_lineage']
    name = prop['cell_name']
    contact = prop['cell_contact_surface']

    div = 10 ** time_digits_for_cell_id

    #
    # get the daughter cells just after division
    #
    reverse_lineage = {v: k for k, values in lineage.iteritems() for v in values}
    daughters = [lineage[c][0] for c in lineage if len(lineage[c]) == 2]
    daughters += [lineage[c][1] for c in lineage if len(lineage[c]) == 2]

    missing_name = []
    missing_contact = []
    missing_neighbors = []

    for d in daughters:
        if reverse_lineage[d] not in name:
            continue
        #
        # check whether the cell is in dictionaries
        #
        if d not in name:
            if d not in missing_name:
                missing_name.append(d)
                monitoring.to_log_and_console(str(proc) + ": daughter cell #" + str(d)
                                              + " was not found in 'cell_name' dictionary. Skip it", 6)
            continue

        if d not in contact:
            if d not in missing_contact:
                missing_contact.append(d)
                monitoring.to_log_and_console(str(proc) + ": cell #" + str(d)
                                              + " was not found in 'cell_contact_surface' dictionary. Skip it")
            continue

        #
        # build the neighborhood
        #

        neighbor = {}
        neighbor_is_complete = True
        for c in contact[d]:
            n = int(c) % div
            if n == 1 or n == 0:
                neighbor['background'] = neighbor.get('background', 0) + contact[d][c]
            elif c in name:
                neighbor[prop['cell_name'][c]] = contact[d][c]
            else:
                neighbor_is_complete = False
                if c not in missing_neighbors:
                    missing_neighbors.append(c)
                    msg = "\t cell #" + str(c) + " was not found in 'cell_name' dictionary."
                    monitoring.to_log_and_console(msg)
                continue

        if not neighbor_is_complete:
            msg = ": neighborhood of " + str(prop['cell_name'][d]) + " is not complete. Skip it"
            monitoring.to_log_and_console(str(proc) + msg)

        if neighbor_is_complete:
            if prop['cell_name'][d] not in previous_neighborhoods:
                previous_neighborhoods[prop['cell_name'][d]] = {}
            if reference_name in previous_neighborhoods[prop['cell_name'][d]]:
                msg = "weird, " + str(reference_name) + " was already indexed for cell " + str(prop['cell_name'][d])
                monitoring.to_log_and_console(str(proc) + ": " + msg)
            previous_neighborhoods[prop['cell_name'][d]][reference_name] = neighbor
            #
            # add symmetric neighborhood if asked
            #
            if add_symmetric_neighborhood:
                sname = _get_symmetric_name(prop['cell_name'][d])
                sreference = 'sym-' + reference_name
                sneighbor = _get_symmetric_neighborhood(neighbor)
                if sname not in previous_neighborhoods:
                    previous_neighborhoods[sname] = {}
                if sreference in previous_neighborhoods[sname]:
                    msg = "weird, " + str(sreference) + " was already indexed for cell " + str(sname)
                    monitoring.to_log_and_console(str(proc) + ": " + msg)
                previous_neighborhoods[sname][sreference] = sneighbor

    if len(missing_name) > 0:
        monitoring.to_log_and_console(str(proc) + ": daughter cells without names = " + str(len(missing_name)) + "/" +
                                      str(len(daughters)))

    if len(missing_contact) > 0:
        monitoring.to_log_and_console(str(proc) + ": daughter cells without contact surfaces  = " +
                                      str(len(missing_contact)) + "/" + str(len(daughters)))

    if len(missing_neighbors) > 0:
        monitoring.to_log_and_console(str(proc) + ": neighboring cells without names  = " + str(len(missing_neighbors)))

    monitoring.to_log_and_console("")

    return previous_neighborhoods


def build_neighborhoods(referenceFiles, time_digits_for_cell_id, add_symmetric_neighborhood=False,
                        reference_diagnosis=False):
    neighborhoods = {}
    if isinstance(referenceFiles, str):
        prop = properties.read_dictionary(referenceFiles, inputpropertiesdict={})
        if reference_diagnosis:
            diagnosis(prop, time_digits_for_cell_id=time_digits_for_cell_id)
        name = referenceFiles.split(os.path.sep)[-1]
        if name.endswith(".xml") or name.endswith(".pkl"):
            name = name[:-4]
        neighborhoods = _add_neighborhoods(neighborhoods, prop, reference_name=name,
                                           time_digits_for_cell_id=time_digits_for_cell_id,
                                           add_symmetric_neighborhood=add_symmetric_neighborhood)
        del prop
    elif isinstance(referenceFiles, list):
        for f in referenceFiles:
            prop = properties.read_dictionary(f, inputpropertiesdict={})
            if reference_diagnosis:
                diagnosis(prop, time_digits_for_cell_id=time_digits_for_cell_id)
            name = f.split(os.path.sep)[-1]
            if name.endswith(".xml") or name.endswith(".pkl"):
                name = name[:-4]
            neighborhoods = _add_neighborhoods(neighborhoods, prop, reference_name=name,
                                               time_digits_for_cell_id=time_digits_for_cell_id,
                                               add_symmetric_neighborhood=add_symmetric_neighborhood)
            del prop
    return neighborhoods


########################################################################################
#
# compute scores as a scalar product
#
########################################################################################

#
# build a common representation of two neighborhood
#
def _build_common_contact_surfaces(tmp0, tmp1, vect0, vect1):
    if tmp1 == {}:
        return tmp0, tmp1, vect0, vect1
    names1 = tmp1.keys()
    for n1 in names1:
        # the name could have been deleted at case 2
        if n1 not in tmp1:
            continue
        # case 1: the same name in both neighborhood
        if n1 in tmp0:
            vect0[n1] = tmp0[n1]
            vect1[n1] = tmp1[n1]
            del tmp0[n1]
            del tmp1[n1]
            continue
        # subcase 1: it's the background and it is not in tmp0
        if n1 == 'background':
            vect0[n1] = 0.0
            vect1[n1] = tmp1[n1]
            del tmp1[n1]
            continue
        # case 2: daughter name in tmp1, and mother name in tmp0
        # add daughter surfaces
        pname = _get_mother_name(n1)
        if pname in tmp0:
            vect0[pname] = tmp0[pname]
            vect1[pname] = 0.0
            for n in _get_daughter_names(pname):
                if n in tmp1:
                    vect1[pname] += tmp1[n]
                    del tmp1[n]
            del tmp0[pname]
            continue
        # case 3: mother name in tmp1, and daughter name(s) in tmp0
        dname = _get_daughter_names(n1)
        if dname[0] in tmp0 or dname[1] in tmp0:
            vect0[n1] = 0.0
            vect1[n1] = tmp1[n1]
            for n in dname:
                if n in tmp0:
                    vect0[n1] += tmp0[n]
                    del tmp0[n]
            del tmp1[n1]
            continue
        # case 4: no association in tmp0
        vect0[n1] = 0.0
        vect1[n1] = tmp1[n1]
        del tmp1[n1]
        continue
    return tmp0, tmp1, vect0, vect1


#
#
#
def _scalar_product(vect0, vect1):
    n0 = 0.0
    n1 = 0.0
    ps = 0.0
    for k in vect0:
        ps += vect0[k] * vect1[k]
        n0 += vect0[k] * vect0[k]
        n1 += vect1[k] * vect1[k]
    return ps / (math.sqrt(n0 * n1))


#
#
#
def _get_score(neigh0, neigh1, title=None, debug=False):
    tmp0 = copy.deepcopy(neigh0)
    tmp1 = copy.deepcopy(neigh1)
    # 1. parse names of tmp1
    tmp0, tmp1, vect0, vect1 = _build_common_contact_surfaces(tmp0, tmp1, {}, {})
    # 2. parse remaining names of tmp0
    tmp1, tmp0, vect1, vect0 = _build_common_contact_surfaces(tmp1, tmp0, vect1, vect0)
    score = _scalar_product(vect0, vect1)
    if title is not None:
        _print_common_neighborhoods(vect0, vect1, title=title)
        monitoring.to_log_and_console("\t score = " + str(score) + "\n")
    # compute score as a scalar product
    return score


########################################################################################
#
#
#
########################################################################################

def _ci_global_score(neighbors, references, debug=False):
    score = 0
    for r1 in references:
        for r2 in references:
            if r2 <= r1:
                continue
            if debug:
                print("   - score[" + str(r1) + ", " + str(r2) + "] = " + str(_get_score(neighbors[0][r1], neighbors[0][r2])))
                print("   - score[" + str(r1) + ", " + str(r2) + "] = " + str(_get_score(neighbors[1][r1], neighbors[1][r2])))
            score += _get_score(neighbors[0][r1], neighbors[0][r2])
            score += _get_score(neighbors[1][r1], neighbors[1][r2])
    return score


def _ci_switch_daughters(neighbors, reference, daughters):
    for c in neighbors[0][reference]:
        if c == daughters[1]:
            neighbors[0][reference][daughters[0]] = neighbors[0][reference][daughters[1]]
            del neighbors[0][reference][daughters[1]]
    for c in neighbors[1][reference]:
        if c == daughters[0]:
            neighbors[1][reference][daughters[1]] = neighbors[1][reference][daughters[0]]
            del neighbors[1][reference][daughters[0]]
    tmp = copy.deepcopy(neighbors[0][reference])
    neighbors[0][reference] = neighbors[1][reference]
    neighbors[1][reference] = tmp
    return neighbors


def _ci_one_division(neighborhoods, mother):
    daughters = _get_daughter_names(mother)
    neighbors = {}
    neighbors[0] = copy.deepcopy(neighborhoods[daughters[0]])
    neighbors[1] = copy.deepcopy(neighborhoods[daughters[1]])

    references = set.intersection(set(neighbors[0].keys()), set(neighbors[1].keys()))
    if len(references) <= 1:
        return {}

    score = _ci_global_score(neighbors, references)

    corrections = {}
    i = 1
    while True:
        newscore = {}
        for r in references:
            tmp = copy.deepcopy(neighbors)
            _ci_switch_daughters(tmp, r, daughters)
            newscore[r] = _ci_global_score(tmp, references)
            if newscore[r] < score:
                del newscore[r]
        ref = None
        if len(newscore) == 0:
            return corrections
        elif len(newscore) == 1:
            ref = newscore.keys()[0]
        else:
            ref = max(newscore, key=lambda key: newscore[key])
        corrections[i] = (ref, newscore[ref]-score)
        i += 1
        tmp = copy.deepcopy(neighbors)
        _ci_switch_daughters(tmp, ref, daughters)
        neighbors[0] = copy.deepcopy(tmp[0])
        neighbors[1] = copy.deepcopy(tmp[1])
        score = newscore[ref]


def _consistency_improvement(neighborhoods, debug=False):
    proc = "_consistency_improvement"

    # neighborhoods is a dictionary of dictionaries
    # ['cell name']['reference name']
    # first key is a cell name (daughter cell)
    # second key is the reference from which the neighborhood has been extracted

    corrections = {}

    processed_mothers = {}
    sorted_cells = sorted(neighborhoods.keys())
    for d in sorted_cells:
        mother = _get_mother_name(d)
        if mother in processed_mothers:
            continue
        sister = _get_sister_name(d)
        if sister not in neighborhoods:
            msg = "weird, cell " + str(d) + " is in the reference neighborhoods, while its sister "
            msg += str(sister) + " is not "
            monitoring.to_log_and_console(str(proc) + ": " + msg)
            continue
        correction = _ci_one_division(neighborhoods, mother)
        if len(correction) > 0:
            corrections[mother] = correction

    tmp = {}
    for c in corrections:
        for i in corrections[c]:
            tmp[corrections[c][i][0]] = tmp.get(corrections[c][i][0], []) + [c]

    print("====== consistency improvement =====")
    print(str(corrections))
    refs = sorted(tmp.keys())
    for r in refs:
        msg = str(r) + ": "
        cells = sorted(tmp[r])
        for i in range(len(cells)):
            msg += str(cells[i])
            if i < len(cells)-1:
                msg += ", "
        print(msg)
    print("====================================")

########################################################################################
#
#
#
########################################################################################

def _get_neighborhood_consistency(neighborhoods, debug=False):
    proc = "_get_neighborhood_consistency"
    has_written_something = False

    #
    # list of daughter cells
    #
    cell_names = sorted(list(neighborhoods.keys()))

    #
    # get the list of references per division
    #
    references = {}
    for cell_name in cell_names:
        mother_name = _get_mother_name(cell_name)
        references[mother_name] = references.get(mother_name, set()).union(set(neighborhoods[cell_name].keys()))

    #
    # get discrepancies
    #

    discrepancy = {}
    tested_couples = {}

    debug_choice = {}
    debug_synthese = {}

    for cell_name in cell_names:
        sum_same_choice = 0
        sum_sister_choice = 0
        #
        # get cell name and sister name
        #
        sister_name = _get_sister_name(cell_name)
        mother_name = _get_mother_name(cell_name)
        if sister_name not in neighborhoods:
            msg = "weird, cell " + str(cell_name) + " is in the reference neighborhoods, while its sister "
            msg += str(sister_name) + " is not "
            monitoring.to_log_and_console(str(proc) + ": " + msg)
            has_written_something = True
            # cell_names.remove(cell_name)
            continue

        if debug and mother_name == 'a7.0002*':
            msg = str(proc) + ":cell_name = " + str(cell_name)
            msg += " - sister_name = " + str(sister_name)
            print(msg)

        #
        # only one neighborhood, nothing to test
        #
        if len(neighborhoods[cell_name]) == 1:
            # cell_names.remove(cell_name)
            # cell_names.remove(sister_name)
            continue
        #
        # get two reference embryos
        #
        warnings = []
        for ref1 in neighborhoods[cell_name]:
            # print("ref1 = " + str(ref1) + " - " + str(type(ref1)) + " - " + str(ref1 == ))
            for ref2 in neighborhoods[cell_name]:
                if ref2 == ref1:
                    continue
                if ref2 not in neighborhoods[sister_name]:
                    if ref2 in warnings:
                        continue
                    msg = "weird, reference " + str(ref2) + " is in the neighborhoods of cell "
                    msg += str(cell_name) + " but not of its sister " + str(sister_name)
                    monitoring.to_log_and_console(str(proc) + ": " + msg)
                    has_written_something = True
                    warnings.append(ref2)
                    continue
                title = None
                if debug and mother_name == 'a7.0002*' and (ref1 in ['Astec-pm7', 'Astec-pm9'] and ref2 in ['Astec-pm7', 'Astec-pm9']):
                    title = "(" + str(cell_name) + ", " + str(ref1) + ")"
                    title += "(" + str(cell_name) + ", " + str(ref2) + ")"
                same_choice = _get_score(neighborhoods[cell_name][ref1], neighborhoods[cell_name][ref2], title=title)
                if debug and mother_name == 'a7.0002*' and (ref1 in ['Astec-pm7', 'Astec-pm9'] and ref2 in ['Astec-pm7', 'Astec-pm9']):
                    title = "(" + str(cell_name) + ", " + str(ref1) + ")"
                    title += "(" + str(sister_name) + ", " + str(ref2) + ")"
                sister_choice = _get_score(neighborhoods[cell_name][ref1], neighborhoods[sister_name][ref2], title=title)

                if debug and mother_name == 'a7.0002*':
                    if ref1 not in debug_choice:
                        debug_choice[ref1] = {}
                    if ref2 not in debug_choice[ref1]:
                        debug_choice[ref1][ref2] = [0, 0]
                    if ref1 not in debug_synthese:
                        debug_synthese[ref1] = [0, 0]
                    if ref2 not in debug_synthese:
                        debug_synthese[ref2] = [0, 0]
                    if same_choice > sister_choice:
                        debug_choice[ref1][ref2][0] += 1
                        debug_synthese[ref1][0] += 1
                        debug_synthese[ref2][0] += 1
                    else:
                        debug_choice[ref1][ref2][1] += 1
                        debug_synthese[ref1][1] += 1
                        debug_synthese[ref2][1] += 1

                mother_name = _get_mother_name(cell_name)
                if debug and mother_name == 'a7.0002*':
                    if same_choice > sister_choice:
                        sum_same_choice += 1
                    else:
                        sum_sister_choice += 1
                    if ref1 == 'Astec-pm7' or ref2 == 'Astec-pm7':
                        if same_choice > sister_choice:
                            msg = '   - same_choice   = ' + str(same_choice)
                            msg += ' > sister_choice = ' + str(sister_choice)
                            msg += ' ref1 = ' + str(ref1) + ' ref2 = ' + str(ref2)
                        else:
                            msg = '   - sister_choice = ' + str(same_choice)
                            msg += ' > same_choice   = ' + str(sister_choice)
                            msg += ' ref1 = ' + str(ref1) + ' ref2 = ' + str(ref2)
                        print(msg)

                if mother_name not in tested_couples:
                    tested_couples[mother_name] = 1
                else:
                    tested_couples[mother_name] += 1
                if same_choice > sister_choice:
                    continue
                if mother_name not in discrepancy:
                    discrepancy[mother_name] = []
                discrepancy[mother_name].append((ref1, ref2))

        if debug and mother_name == 'a7.0002*':
            print("- sum_same_choice = " + str(sum_same_choice) + " - sum_sister_choice = " + str(sum_sister_choice))
            print("- discrepancy = " + str(len(discrepancy[mother_name])) + " - tested_couples = " + str(tested_couples[mother_name]))
            for r1 in debug_choice:
                for r2 in debug_choice[r1]:
                    print("  [" + str(r1) + ", " + str(r2) + "] = " + str(debug_choice[r1][r2]))
            for r1 in debug_synthese:
                print("  [" + str(r1) + "] = " + str(debug_synthese[r1]))
    if has_written_something:
        monitoring.to_log_and_console("")

    #
    # for each division/mother cell
    # - reference is a dictionary that gives the set of references
    # - tested_couples is a dictionary that gives the number of tests done
    # - discrepancy is a dictionary that gives the list of couple of non-agreement
    #
    return references, tested_couples, discrepancy


def _get_neighborhood_database_discrepancy(neighborhoods):
    references, tested_couples, discrepancy = _get_neighborhood_consistency(neighborhoods)
    if len(discrepancy) == 0:
        return 0

    total_discrepancy = 0
    for n in discrepancy:
        total_discrepancy += discrepancy[n]
    return total_discrepancy


def _networkx_neighborhood_consistency(neighborhoods, min_discrepancy=0):
    references, tested_couples, discrepancy = _get_neighborhood_consistency(neighborhoods, debug=True)
    if len(discrepancy) == 0:
        return

    first_discrepancy = {}
    first_percents = []
    second_discrepancy = {}
    second_percents = []

    mother_names = sorted(discrepancy.keys())
    for n in mother_names:
        if _get_mother_name(n) in mother_names or _get_mother_name(n) in first_discrepancy:
            second_discrepancy[n] = discrepancy[n]
            second_percents.append(100.0 * float(len(discrepancy[n])) / float(tested_couples[n]))
        else:
            first_discrepancy[n] = discrepancy[n]
            first_percents.append(100.0 * float(len(discrepancy[n])) / float(tested_couples[n]))

    percents = []
    for mother_name in mother_names:
        percents.append(100.0 * float(len(discrepancy[mother_name])) / float(tested_couples[mother_name]))
    [sorted_percents, sorted_mothers] = zip(*sorted(zip(percents, mother_names), reverse=True))

    f = open("networkx_figure" + '.py', "w")

    f.write("import networkx as nx\n")
    f.write("import numpy as np\n")
    f.write("import matplotlib.pyplot as plt\n")

    cellidentifierlist = []
    for mother_name in sorted_mothers:
        percent = 100.0 * float(len(discrepancy[mother_name])) / float(tested_couples[mother_name])
        if percent < min_discrepancy:
            break

        # identifier for mother cell
        cellidentifier = mother_name[0:2]+mother_name[3:7]
        if mother_name[7] == '_':
            cellidentifier += 'U'
        elif mother_name[7] == '*':
            cellidentifier += 'S'
        else:
            cellidentifier += 'S'
        if mother_name in first_discrepancy:
            cellidentifier += "_1"
        elif mother_name in second_discrepancy:
            cellidentifier += "_2"
        fileidentifier = 'D{:03d}_'.format(int(percent)) + cellidentifier
        cellidentifier += '_D{:03d}'.format(int(percent))

        cellidentifierlist.append(cellidentifier)

        edges = {}
        for n1 in references[mother_name]:
            edges[n1] = {}
            for n2 in references[mother_name]:
                if n2 <= n1:
                    continue
                edges[n1][n2] = 4
        for d in discrepancy[mother_name]:
            if d[0] < d[1]:
                edges[d[0]][d[1]] -= 1
            elif d[1] < d[0]:
                edges[d[1]][d[0]] -= 1

        f.write("\n")
        f.write("\n")
        f.write("def draw_" + cellidentifier + "(savefig=False, cutnodes=True):\n")

        nodes = list(references[mother_name])
        nodes.sort()

        f.write("\n")
        f.write("    G_" + cellidentifier + " = nx.Graph()\n")
        f.write("    G_" + cellidentifier + ".add_nodes_from(" + str(nodes) + ")\n")
        f.write("\n")

        edgelist = {}
        for n1 in nodes:
            for n2 in nodes:
                if n2 <= n1:
                    continue
                edgelist[edges[n1][n2]] = edgelist.get(edges[n1][n2], []) + [(n1, n2)]

        for i in range(1, 5):
            if i not in edgelist:
                continue
            if i == 2:
                f.write("    G_" + cellidentifier + ".add_edges_from(" + str(edgelist[i]) + ", weight=0)\n")
            else:
                f.write("    G_" + cellidentifier + ".add_edges_from(" + str(edgelist[i]) + ", weight=" + str(i) +
                        ")\n")
        f.write("\n")

        f.write("    snodes = nx.spectral_ordering(G_" + cellidentifier + ")\n")
        f.write("    scutmin, pnodes = nx.minimum_cut(G_" + cellidentifier + ", snodes[0], snodes[-1]," +
                " capacity='weight')\n")

        f.write("\n")
        f.write("    node_labels = {}\n")
        for n in nodes:
            f.write("    node_labels['" + str(n) + "'] = '" + str(n) + "'\n")
        f.write("\n")

        f.write("    fig, ax = plt.subplots(figsize=(7.5, 7.5))\n")
        f.write("    pos = nx.circular_layout(G_" + cellidentifier + ", scale=1.8)\n")
        f.write("    if cutnodes:\n")
        f.write("        nx.draw_networkx_nodes(G_" + cellidentifier + ", nodelist=pnodes[0] " +
                ", node_color='#b41f78', pos=pos, ax=ax, node_size=1000)\n")
        f.write("        nx.draw_networkx_nodes(G_" + cellidentifier + ", nodelist=pnodes[1] " +
                ", node_color='#1f78b4', pos=pos, ax=ax, node_size=1000)\n")
        f.write("    else:\n")
        f.write("        nx.draw_networkx_nodes(G_" + cellidentifier + ", pos=pos, ax=ax, node_size=1000)\n")
        f.write("    nx.draw_networkx_labels(G_" + cellidentifier + ", ax=ax, pos=pos, labels=node_labels," +
                " font_weight='bold')\n")
        f.write("    ax.set_xlim([-2.2, 2.2])\n")
        f.write("    ax.set_ylim([-2.1, 2.1])\n")
        for i in range(1, 5):
            if i not in edgelist:
                continue
            f.write("    nx.draw_networkx_edges(G_" + cellidentifier + ", pos=pos, ax=ax, edgelist=" + str(edgelist[i]))
            f.write(", edge_color='#1f78b4'")
            f.write(", width=" + str(i))
            if i == 1:
                f.write(", style='dotted'")
            elif i == 2:
                f.write(", style='dashed'")
            f.write(")\n")
        title = "division of " + str(mother_name)
        title += ", discrepancy = " + '{:.2f}'.format(percent) + "%"
        if mother_name in first_discrepancy:
            title += ", 1st order"
        elif mother_name in second_discrepancy:
            title += ", 2nd order"
        title += ", cut = {:2d}"
        f.write("    cstr = \"C{:02d}\".format(scutmin)\n")

        f.write("    ax.set_title(\"" + str(title) + "\".format(scutmin))\n")
        f.write("    if savefig:\n")
        f.write("        plt.savefig('" + str(fileidentifier) + "' + '_'" + " + str(cstr)" + " + '.png')\n")
        f.write("        plt.savefig('" + str(cellidentifier) + "' + '_'" + " + str(cstr)" + " + '.png')\n")
        f.write("        plt.savefig(str(cstr)" + " + '_" + str(cellidentifier) + ".png')\n")
        f.write("    else:\n")
        f.write("        plt.show()\n")

    f.write("\n")
    f.write("\n")
    f.write("def draw_all(savefig=False, cutnodes=True):\n")
    for cellidentifier in cellidentifierlist:
        f.write("    draw_" + cellidentifier + "(savefig=savefig, cutnodes=cutnodes)\n")
        f.write("    plt.close()\n")

    f.write("\n")
    f.write("\n")
    for cellidentifier in cellidentifierlist:
        f.write("if False:\n")
        f.write("    draw_" + cellidentifier + "(savefig=False, cutnodes=True)\n")
    f.write("\n")
    f.write("if True:\n")
    f.write("    draw_all(savefig=True, cutnodes=True)\n")
    f.write("\n")

    f.write("if True:\n")
    f.write("    first_percents = " + str(first_percents) + "\n")
    f.write("    second_percents = " + str(second_percents) + "\n")
    f.write("    fig, ax = plt.subplots(figsize=(5, 5))\n")
    f.write("    bins = range(0,100,5)\n")
    f.write("    label = ['1st order', '2nd order']\n")
    f.write("    ax.hist([first_percents, second_percents], bins, histtype='bar', label=label)\n")
    f.write("    ax.legend(prop={'size': 10})\n")
    f.write("    ax.set_title('neighborhood discrepancies')\n")
    f.write("    fig.tight_layout()\n")
    f.write("    if True:\n")
    f.write("        plt.savefig('neighborhood_consistency_histogram.png')\n")
    f.write("    else:\n")
    f.write("        plt.show()\n")
    f.write("\n")
    f.close()


def _write_neighborhood_consistency(txt, mother_names, references, tested_couples, discrepancy):
    #
    # get discrepancy/inconsistency percentage per division
    #
    percents = []
    for mother_name in mother_names:
        percents.append(100.0 * float(len(discrepancy[mother_name])) / float(tested_couples[mother_name]))
    [sorted_percents, sorted_mothers] = zip(*sorted(zip(percents, mother_names), reverse=True))

    msg = "\n*** " + str(txt) + " = " + str(len(mother_names))
    monitoring.to_log_and_console(msg)

    for mother_name in sorted_mothers:
        msg = " - " + str(mother_name) + " cell division into "
        msg += str(_get_daughter_names(mother_name)) + " has " + str(len(discrepancy[mother_name]))
        if len(discrepancy[mother_name]) > 1:
            msg += " discrepancies"
        else:
            msg += " discrepancy"
        percent = 100.0 * float(len(discrepancy[mother_name])) / float(tested_couples[mother_name])
        msg += " (" + "{:2.2f}%".format(percent) + ") "
        msg += " over " + str(tested_couples[mother_name]) + " tested configurations "
        monitoring.to_log_and_console(msg)
        msg = "\t over " + str(len(references[mother_name]))
        msg += " references: " + str(references[mother_name])
        monitoring.to_log_and_console(msg)
        msg = "\t " + str(discrepancy[mother_name])
        monitoring.to_log_and_console(msg, 3)


def _check_neighborhood_consistency(neighborhoods):
    proc = "_check_neighborhood_consistency"

    references, tested_couples, discrepancy = _get_neighborhood_consistency(neighborhoods)

    monitoring.to_log_and_console("")
    monitoring.to_log_and_console(str(proc))
    monitoring.to_log_and_console("-------------------------------------------")
    monitoring.to_log_and_console("--- reference neighborhoods consistency ---")

    #
    # if some divisions have some discrepancies, the following ones in the lineage
    # are likely to exhibit discrepancies also
    #
    mother_names = sorted(discrepancy.keys())
    if len(mother_names) > 0:
        _write_neighborhood_consistency("neighborhood discrepancies", mother_names, references, tested_couples,
                                        discrepancy)

    msg = "tested divisions = " + str(len(tested_couples))
    monitoring.to_log_and_console(str(proc) + ": " + msg)
    msg = "divisions with discrepancies =  " + str(len(discrepancy))
    monitoring.to_log_and_console("\t " + msg)

    monitoring.to_log_and_console("-------------------------------------------")
    monitoring.to_log_and_console("")


def check_leftright_consistency(neighborhoods):
    monitoring.to_log_and_console("")
    monitoring.to_log_and_console("-------------------------------------------")
    monitoring.to_log_and_console("--- left/right neighborhood consistency ---")

    #
    # list of daughter cells
    #
    cell_names = sorted(list(neighborhoods.keys()))

    #
    # get the list of references per division
    #
    references = {}
    for cell_name in cell_names:
        mother_name = _get_mother_name(cell_name)
        references[mother_name] = references.get(mother_name, set()).union(set(neighborhoods[cell_name].keys()))

    #
    #
    #
    mother_names = sorted(list(references.keys()))
    processed_mothers = []
    discrepancy = {}
    tested_cells = {}
    messages = {}
    for mother in mother_names:
        if mother in processed_mothers:
            continue
        symmother = _get_symmetric_name(mother)
        processed_mothers.append(mother)

        if symmother not in references:
            continue
        daughters = _get_daughter_names(mother)

        for reference in references[mother]:
            if reference not in references[symmother]:
                continue

            if reference not in tested_cells:
                tested_cells[reference] = 1
            else:
                tested_cells[reference] += 1

            for daughter in daughters:
                if daughter not in neighborhoods:
                    continue
                if reference not in neighborhoods[daughter]:
                    continue
                symdaughter = _get_symmetric_name(daughter)
                symsister = _get_sister_name(symdaughter)
                if symdaughter not in neighborhoods or symsister not in neighborhoods:
                    continue
                if reference not in neighborhoods[symdaughter] or reference not in neighborhoods[symsister]:
                    continue
                #
                #
                #
                symsameneigh = _get_symmetric_neighborhood(neighborhoods[symdaughter][reference])
                symsisterneigh = _get_symmetric_neighborhood(neighborhoods[symsister][reference])

                same_choice = _get_score(neighborhoods[daughter][reference], symsameneigh)
                sister_choice = _get_score(neighborhoods[daughter][reference], symsisterneigh)

                if same_choice > sister_choice:
                    continue

                if reference not in discrepancy:
                    discrepancy[reference] = {}
                discrepancy[reference][mother] = discrepancy[reference].get(mother, []) + [(daughter, symdaughter)]

                messages[mother] = messages.get(mother, []) + [(reference, daughter, symsister, symdaughter)]
                # msg = "   - '" + str(reference) + "': " + str(daughter) + " neighborhood is closest to "
                # msg += str(symsister) + " neighborhood than to " + str(symdaughter) + " one"
                # monitoring.to_log_and_console(msg, 3)

    mothers = sorted(messages.keys())
    for mother in mothers:
        msg = "   - '" + str(mother) + "' division"
        monitoring.to_log_and_console(msg, 3)
        for (reference, daughter, symsister, symdaughter) in messages[mother]:
            msg = "      - '" + str(reference) + "': " + str(daughter) + " neighborhood is closest to "
            msg += str(symsister) + " neighborhood than to " + str(symdaughter) + " one"
            monitoring.to_log_and_console(msg, 3)

    for reference in discrepancy:
        msg = "- '" + str(reference) + "' tested divisions = " + str(tested_cells[reference])
        monitoring.to_log_and_console(msg)

        #
        # get the mother cell for each discrepancy value
        #
        mother_by_discrepancy = {}
        processed_mothers = []
        for mother in discrepancy[reference]:
            if mother in processed_mothers:
                continue
            symmother = _get_symmetric_name(mother)
            processed_mothers += [mother, symmother]

            d = len(discrepancy[reference][mother])
            if symmother in discrepancy[reference]:
                d += len(discrepancy[reference][symmother])
            mother_by_discrepancy[d] = mother_by_discrepancy.get(d, []) + [mother]

        divisions_with_discrepancy = 0
        for d in mother_by_discrepancy:
            divisions_with_discrepancy += len(mother_by_discrepancy[d])
        msg = "\t divisions with left/right discrepancies =  " + str(divisions_with_discrepancy)
        monitoring.to_log_and_console(msg)

        for d in sorted(mother_by_discrepancy.keys()):
            if d == 1:
                msg = "\t - divisions with left/right " + str(d) + " discrepancy = "
            else:
                msg = "\t - divisions with left/right " + str(d) + " discrepancies = "
            msg += str(len(mother_by_discrepancy[d]))
            monitoring.to_log_and_console(msg)
            processed_mothers = []
            for mother in mother_by_discrepancy[d]:
                if mother in processed_mothers:
                    continue
                symmother = _get_symmetric_name(mother)
                processed_mothers += [mother, symmother]
                msg = "(" + str(mother) + ", " + str(symmother) + ") divisions: " + str(d)
                if d == 1:
                    msg += " discrepancy"
                else:
                    msg += " discrepancies"
                monitoring.to_log_and_console("\t   " + msg)

    monitoring.to_log_and_console("-------------------------------------------")
    monitoring.to_log_and_console("")
    return discrepancy


def neighborhood_diagnosis(neighborhoods):
    check_leftright_consistency(neighborhoods)
    _check_neighborhood_consistency(neighborhoods)


def add_leftright_discrepancy_selection(d, discrepancy):
    #
    # a selection for all discrepancies
    #
    d['selection_leftright_discrepancy'] = {}
    allmother = []
    alldaughter0 = []
    alldaughter1 = []
    alldiscrepancies = {}
    for reference in discrepancy:
        for mother in discrepancy[reference]:
            daughters = _get_daughter_names(mother)
            symmother = _get_symmetric_name(mother)
            symdaughters = [_get_symmetric_name(daughters[0]), _get_symmetric_name(daughters[1])]

            allmother.append(mother)
            allmother.append(symmother)
            alldaughter0.append(daughters[0])
            alldaughter0.append(symdaughters[0])
            alldaughter1.append(daughters[1])
            alldaughter1.append(symdaughters[1])

            nd = len(discrepancy[reference][mother])
            if symmother in discrepancy[reference]:
                nd += len(discrepancy[reference][symmother])

            alldiscrepancies[mother] = nd
            alldiscrepancies[symmother] = nd
            alldiscrepancies[daughters[0]] = nd
            alldiscrepancies[symdaughters[0]] = nd
            alldiscrepancies[daughters[1]] = nd
            alldiscrepancies[symdaughters[1]] = nd

    for c in d['cell_name']:
        if d['cell_name'][c] in allmother:
            d['selection_leftright_discrepancy'][c] = alldiscrepancies[d['cell_name'][c]] * 10
        elif d['cell_name'][c] in alldaughter0:
            d['selection_leftright_discrepancy'][c] = alldiscrepancies[d['cell_name'][c]] * 10 + 1
        elif d['cell_name'][c] in alldaughter1:
            d['selection_leftright_discrepancy'][c] = alldiscrepancies[d['cell_name'][c]] * 10 + 2

    for reference in discrepancy:
        for mother in discrepancy[reference]:
            daughters = _get_daughter_names(mother)
            symmother = _get_symmetric_name(mother)
            allmother = [mother, symmother]
            alldaughter0 = [daughters[0], _get_symmetric_name(daughters[0])]
            alldaughter1 = [daughters[0], _get_symmetric_name(daughters[1])]

            nd = len(discrepancy[reference][mother])
            if symmother in discrepancy[reference]:
                nd += len(discrepancy[reference][symmother])
            key = 'selection_leftright_' + str(nd) + '_' + mother[:-1]
            d[key] = {}

            for c in d['cell_name']:
                if d['cell_name'][c] in allmother:
                    d[key][c] = alldiscrepancies[d['cell_name'][c]] * 10
                elif d['cell_name'][c] in alldaughter0:
                    d[key][c] = alldiscrepancies[d['cell_name'][c]] * 10 + 1
                elif d['cell_name'][c] in alldaughter1:
                    d[key][c] = alldiscrepancies[d['cell_name'][c]] * 10 + 2

    return d


########################################################################################
#
#
#
########################################################################################

def _build_scores(mother, daughters, ancestor_name, prop, neighborhoods, time_digits_for_cell_id=4, debug=False):
    proc = "_build_scores"

    #
    # are daughter names indexed?
    #
    daughter_names = _get_daughter_names(prop['cell_name'][mother])
    for name in daughter_names:
        #
        # no reference for this name
        #
        if name not in neighborhoods:
            msg = ": no reference neighborhoods for name " + str(name)
            msg += ". Can not name cells " + str(daughters)
            msg += " from mother cell " + str(mother)
            msg += " named " + str(prop['cell_name'][mother])
            monitoring.to_log_and_console(str(proc) + msg, 4)
            return None

    div = 10 ** time_digits_for_cell_id

    score = {}

    #
    # daughters is an array of 2 cell ids
    #
    for d in daughters:
        score[d] = {}

        #
        # build contact surface as a dictionary of names
        # 1. background
        # 2. cell already named
        # 3. sister of d
        #    give prop['cell_contact_surface'][d][c] to the untested name
        # 4. daughter cell not named, then named after its mother
        #    there might be two cells with this name
        #
        contact = {}
        sister = None
        for c in prop['cell_contact_surface'][d]:
            if int(c) % div == 1 or int(c) % div == 0:
                contact['background'] = contact.get('background', 0) + prop['cell_contact_surface'][d][c]
            elif c in prop['cell_name']:
                contact[prop['cell_name'][c]] = prop['cell_contact_surface'][d][c]
            elif c in daughters:
                if c != d:
                    sister = c
            elif c in ancestor_name:
                contact[ancestor_name[c]] = contact.get(ancestor_name[c], 0) + prop['cell_contact_surface'][d][c]
            else:
                monitoring.to_log_and_console("\t cell  " + str(c) + " was not found in 'cell_name' dictionary")
                monitoring.to_log_and_console(str(proc) + ": neighborhood of cell " + str(d)
                                              + " is not complete. Skip it")
                return None

        #
        # compute score for each candidate name by comparison with reference neighborhood
        #
        for name in daughter_names:
            #
            # get sister name
            #
            sister_name = None
            if name == daughter_names[0]:
                sister_name = daughter_names[1]
            else:
                sister_name = daughter_names[0]
            score[d][name] = {}
            #
            # add contact for the sister
            #
            if sister is not None:
                contact[sister_name] = prop['cell_contact_surface'][d][sister]
            for reference_name in neighborhoods[name]:
                title = None
                if debug and reference_name == 'Astec-pm9':
                    title = "(" + str(d) + ", " + str(name) + ", " + str(reference_name) + ")"
                score[d][name][reference_name] = _get_score(contact, neighborhoods[name][reference_name], title=title)
            if sister is not None:
                del contact[sister_name]
    return score


def _old_analyse_scores(scores):
    proc = "_analyse_scores"
    #
    # scores is a dictionary of dictionary
    # scores[d in daughters][n in daughter_names(mother)] is an dictionary of scalar product
    # the length of the array is the occurrence of [n in daughter_names(mother)] in the
    # neighborhood dictionary
    #
    name = {}
    name_certainty = {}

    # cell ids
    # cell name candidates
    # reference names
    ids = scores.keys()
    candidates = scores[ids[0]].keys()
    references = set(scores[ids[0]][candidates[0]].keys()).intersection(set(scores[ids[0]][candidates[1]].keys()),
                                                                        set(scores[ids[1]][candidates[0]].keys()),
                                                                        set(scores[ids[1]][candidates[1]].keys()))
    if references != set(scores[ids[0]][candidates[0]].keys()) or \
        references != set(scores[ids[0]][candidates[1]].keys()) or \
        references != set(scores[ids[1]][candidates[0]].keys()) or \
        references != set(scores[ids[1]][candidates[1]].keys()):
        msg = "weird, the set of references is different for each score"
        monitoring.to_log_and_console(str(proc) + ": " + msg)

    new_scores = {}
    agreement00 = 0
    agreement01 = 0
    disagreement = 0
    for r in references:
        new_scores[r] = [scores[ids[0]][candidates[0]][r], scores[ids[0]][candidates[1]][r],
                         scores[ids[1]][candidates[0]][r], scores[ids[1]][candidates[1]][r]]
        # 00 > 01 and 11 > 10
        if new_scores[r][0] > new_scores[r][1] and new_scores[r][3] > new_scores[r][2]:
            agreement00 += 1
        # 01 > 00 and 10 > 11
        elif new_scores[r][1] > new_scores[r][0] and new_scores[r][2] > new_scores[r][3]:
            agreement01 += 1
        else:
            disagreement += 1

    if agreement00 > agreement01 and agreement00 > disagreement:
        name[ids[0]] = candidates[0]
        name[ids[1]] = candidates[1]
        name_certainty[ids[0]] = agreement00 / (agreement00 + agreement01 + disagreement)
        name_certainty[ids[1]] = name_certainty[ids[0]]
    elif agreement01 > agreement00 and agreement01 > disagreement:
        name[ids[0]] = candidates[1]
        name[ids[1]] = candidates[0]
        name_certainty[ids[0]] = agreement01 / (agreement00 + agreement01 + disagreement)
        name_certainty[ids[1]] = name_certainty[ids[0]]
    else:
        msg = "there is a disagreement for cells " + str(candidates)
        monitoring.to_log_and_console(str(proc) + ": " + msg)
        score00 = 0
        score01 = 0
        for r in references:
            score00 += (new_scores[r][0] + new_scores[r][3]) / 2.0
            score01 += (new_scores[r][1] + new_scores[r][2]) / 2.0
        if score00 > score01:
            name[ids[0]] = candidates[0]
            name[ids[1]] = candidates[1]
            name_certainty[ids[0]] = 0.0
            name_certainty[ids[1]] = 0.0
        else:
            name[ids[0]] = candidates[1]
            name[ids[1]] = candidates[0]
            name_certainty[ids[0]] = 0.0
            name_certainty[ids[1]] = 0.0

    return name, name_certainty


def _old_analyse_scores_2021_05_18(scores, debug=False):
    proc = "_analyse_scores"
    #
    # scores is a dictionary of dictionary of dictionary
    # scores[cell id][name][reference] is the scalar product obtained when
    # associating the cell 'cell id' with 'name' for 'reference' neighborhood
    #
    name = {}
    name_certainty = {}

    # cell ids
    # cell name candidates
    # reference names
    ids = scores.keys()
    candidates = scores[ids[0]].keys()
    #
    # selection des references qui ont les 2 voisinages
    #
    references = set(scores[ids[0]][candidates[0]].keys()).intersection(set(scores[ids[0]][candidates[1]].keys()),
                                                                        set(scores[ids[1]][candidates[0]].keys()),
                                                                        set(scores[ids[1]][candidates[1]].keys()))
    if references != set(scores[ids[0]][candidates[0]].keys()) or \
        references != set(scores[ids[0]][candidates[1]].keys()) or \
        references != set(scores[ids[1]][candidates[0]].keys()) or \
        references != set(scores[ids[1]][candidates[1]].keys()):
        msg = "weird, the set of references is different for each score for cells " + str(candidates)
        monitoring.to_log_and_console(str(proc) + ": " + msg)

    new_scores = {}

    agreement00 = 0
    agreement01 = 0
    disagreement00 = 0
    disagreement01 = 0
    disagreement = 0
    sum_agreement00 = 0.0
    sum_agreement01 = 0.0
    sum_disagreement00 = 0.0
    sum_disagreement01 = 0.0

    for r in references:
        new_scores[r] = [scores[ids[0]][candidates[0]][r], scores[ids[0]][candidates[1]][r],
                         scores[ids[1]][candidates[0]][r], scores[ids[1]][candidates[1]][r]]
        # 00 > 01 and 11 > 10
        if new_scores[r][0] > new_scores[r][1] and new_scores[r][3] > new_scores[r][2]:
            sum_agreement00 += (new_scores[r][0] + new_scores[r][3]) - (new_scores[r][1] + new_scores[r][2])
            agreement00 += 1
            if debug:
                print(str(r) + ": total agreement for 00-11")
        # 01 > 00 and 10 > 11
        elif new_scores[r][0] < new_scores[r][1] and new_scores[r][3] < new_scores[r][2]:
            sum_agreement01 += (new_scores[r][1] + new_scores[r][2]) - (new_scores[r][0] + new_scores[r][3])
            agreement01 += 1
            if debug:
                print(str(r) + ": total agreement for 01-10")
        elif new_scores[r][0] + new_scores[r][3] > new_scores[r][1] + new_scores[r][2]:
            sum_disagreement00 += (new_scores[r][0] + new_scores[r][3]) - (new_scores[r][1] + new_scores[r][2])
            disagreement00 += 1
            if debug:
                print(str(r) + ": mild agreement for 00-11")
        elif new_scores[r][0] + new_scores[r][3] < new_scores[r][1] + new_scores[r][2]:
            sum_disagreement01 += (new_scores[r][1] + new_scores[r][2]) - (new_scores[r][0] + new_scores[r][3])
            disagreement01 += 1
            if debug:
                print(str(r) + ": mild agreement for 01-10")
        else:
            disagreement += 1
            if debug:
                print(str(r) + ": no agreement")

    somme = agreement00 + agreement01 + disagreement00 + disagreement01 + disagreement
    score00 = int(100.0 * (agreement00 + 0.5 * disagreement00) / somme)
    score01 = int(100.0 * (agreement01 + 0.5 * disagreement01) / somme)

    if score00 > score01:
        name[ids[0]] = candidates[0]
        name[ids[1]] = candidates[1]
        name_certainty[ids[0]] = score00
        name_certainty[ids[1]] = score00
        if debug:
            print("\t concludes to 00-11")
    elif score00 < score01:
        name[ids[0]] = candidates[1]
        name[ids[1]] = candidates[0]
        name_certainty[ids[0]] = score01
        name_certainty[ids[1]] = score01
        if debug:
            print("\t concludes to 01-10")
    else:
        msg = "there is no clear agreement for cells " + str(candidates)
        monitoring.to_log_and_console(str(proc) + ": " + msg)
        if sum_agreement00 + 0.5 * sum_disagreement00 > sum_agreement01 + 0.5 * sum_disagreement01:
            name[ids[0]] = candidates[0]
            name[ids[1]] = candidates[1]
            name_certainty[ids[0]] = score00
            name_certainty[ids[1]] = score00
        elif sum_agreement00 + 0.5 * sum_disagreement00 < sum_agreement01 + 0.5 * sum_disagreement01:
            name[ids[0]] = candidates[1]
            name[ids[1]] = candidates[0]
            name_certainty[ids[0]] = score01
            name_certainty[ids[1]] = score01
        else:
            msg = "there is no agreement at all for cells " + str(candidates)
            monitoring.to_log_and_console(str(proc) + ": " + msg)
            name[ids[0]] = None
            name[ids[1]] = None
            name_certainty[ids[0]] = 0
            name_certainty[ids[1]] = 0

    return name, name_certainty


def _analyse_scores_2021_05_25(scores, debug=False):
    proc = "_analyse_scores"
    #
    # scores is a dictionary of dictionary of dictionary
    # scores[cell id][name][reference] is the scalar product obtained when
    # associating the cell 'cell id' with 'name' for 'reference' neighborhood
    #
    name = {}
    name_certainty = {}

    # cell ids
    # cell name candidates
    # reference names
    ids = scores.keys()
    candidates = scores[ids[0]].keys()
    #
    # selection des references qui ont les 2 voisinages
    #
    references = set(scores[ids[0]][candidates[0]].keys()).intersection(set(scores[ids[0]][candidates[1]].keys()),
                                                                        set(scores[ids[1]][candidates[0]].keys()),
                                                                        set(scores[ids[1]][candidates[1]].keys()))
    if references != set(scores[ids[0]][candidates[0]].keys()) or \
        references != set(scores[ids[0]][candidates[1]].keys()) or \
        references != set(scores[ids[1]][candidates[0]].keys()) or \
        references != set(scores[ids[1]][candidates[1]].keys()):
        msg = "weird, the set of references is different for each score for cells " + str(candidates)
        monitoring.to_log_and_console(str(proc) + ": " + msg)

    new_scores = {}

    agreement00 = 0
    agreement01 = 0
    sum_agreement00 = 0.0
    sum_agreement01 = 0.0

    if debug:
        print("scores = " + str(scores))

    for r in references:
        if debug:
            print("- reference " + str(r))
        new_scores[r] = [scores[ids[0]][candidates[0]][r], scores[ids[0]][candidates[1]][r],
                         scores[ids[1]][candidates[0]][r], scores[ids[1]][candidates[1]][r]]
        if debug:
            print("- reference " + str(r) + " --- scores = " + str(new_scores[r]))
        # 00 > 01
        if new_scores[r][0] > new_scores[r][1]:
            agreement00 += 1
            sum_agreement00 += new_scores[r][0] - new_scores[r][1]
            if debug:
                print("   00 > 01")
        elif new_scores[r][0] < new_scores[r][1]:
            agreement01 += 1
            sum_agreement01 += new_scores[r][1] - new_scores[r][0]
            if debug:
                print("   00 < 01")

        # 00 > 10
        if new_scores[r][0] > new_scores[r][2]:
            agreement00 += 1
            sum_agreement00 += new_scores[r][0] - new_scores[r][2]
            if debug:
                print("   00 > 10")
        elif new_scores[r][0] < new_scores[r][2]:
            agreement01 += 1
            sum_agreement01 += new_scores[r][2] - new_scores[r][0]
            if debug:
                print("   00 < 10")

        # 11 > 01
        if new_scores[r][3] > new_scores[r][1]:
            agreement00 += 1
            sum_agreement00 += new_scores[r][3] - new_scores[r][1]
            if debug:
                print("   11 > 01")
        elif new_scores[r][3] < new_scores[r][1]:
            agreement01 += 1
            sum_agreement01 += new_scores[r][1] - new_scores[r][3]
            if debug:
                print("   11 < 01")

        # 11 > 10
        if new_scores[r][3] > new_scores[r][2]:
            agreement00 += 1
            sum_agreement00 += new_scores[r][3] - new_scores[r][2]
            if debug:
                print("   11 > 10")
        elif new_scores[r][3] < new_scores[r][2]:
            agreement01 += 1
            sum_agreement01 += new_scores[r][2] - new_scores[r][3]
            if debug:
                print("   11 < 10")

    if debug:
        print("agreement00 = " + str(agreement00))
        print("agreement01 = " + str(agreement01))
        print("sum_agreement00 = " + str(sum_agreement00))
        print("sum_agreement01 = " + str(sum_agreement01))
    if agreement00 > agreement01:
        name[ids[0]] = candidates[0]
        name[ids[1]] = candidates[1]
        name_certainty[ids[0]] = int(100.0 * agreement00 / (agreement00 + agreement01))
        name_certainty[ids[1]] = int(100.0 * agreement00 / (agreement00 + agreement01))
    elif agreement01 > agreement00:
        name[ids[0]] = candidates[1]
        name[ids[1]] = candidates[0]
        name_certainty[ids[0]] = int(100.0 * agreement01 / (agreement00 + agreement01))
        name_certainty[ids[1]] = int(100.0 * agreement01 / (agreement00 + agreement01))
    else:
        msg = "there is no clear agreement for cells " + str(candidates)
        monitoring.to_log_and_console(str(proc) + ": " + msg)
        if sum_agreement00 > sum_agreement01:
            name[ids[0]] = candidates[0]
            name[ids[1]] = candidates[1]
            name_certainty[ids[0]] = int(100.0 * sum_agreement00 / (sum_agreement00 + sum_agreement01))
            name_certainty[ids[1]] = int(100.0 * sum_agreement00 / (sum_agreement00 + sum_agreement01))
        elif sum_agreement01 > sum_agreement00:
            name[ids[0]] = candidates[1]
            name[ids[1]] = candidates[0]
            name_certainty[ids[0]] = int(100.0 * sum_agreement01 / (sum_agreement00 + sum_agreement01))
            name_certainty[ids[1]] = int(100.0 * sum_agreement01 / (sum_agreement00 + sum_agreement01))
        else:
            msg = "there is no agreement at all for cells " + str(candidates)
            monitoring.to_log_and_console(str(proc) + ": " + msg)
            name[ids[0]] = None
            name[ids[1]] = None
            name_certainty[ids[0]] = 0
            name_certainty[ids[1]] = 0

    return name, name_certainty


def _analyse_scores(scores, debug=False):
    proc = "_analyse_scores"
    #
    # scores is a dictionary of dictionary of dictionary
    # scores[cell id][name][reference] is the scalar product obtained when
    # associating the cell 'cell id' with 'name' for 'reference' neighborhood
    #
    name = {}
    name_certainty = {}

    # cell ids
    # cell name candidates
    # reference names
    ids = scores.keys()
    candidates = scores[ids[0]].keys()
    #
    # selection des references qui ont les 2 voisinages
    #
    references = set(scores[ids[0]][candidates[0]].keys()).intersection(set(scores[ids[0]][candidates[1]].keys()),
                                                                        set(scores[ids[1]][candidates[0]].keys()),
                                                                        set(scores[ids[1]][candidates[1]].keys()))
    if references != set(scores[ids[0]][candidates[0]].keys()) or \
        references != set(scores[ids[0]][candidates[1]].keys()) or \
        references != set(scores[ids[1]][candidates[0]].keys()) or \
        references != set(scores[ids[1]][candidates[1]].keys()):
        msg = "weird, the set of references is different for each score for cells " + str(candidates)
        monitoring.to_log_and_console(str(proc) + ": " + msg)

    new_scores = {}

    sum_agreement00 = 0.0
    sum_agreement01 = 0.0

    if debug:
        print("scores = " + str(scores))

    for r in references:
        if debug:
            print("- reference " + str(r))
        new_scores[r] = [scores[ids[0]][candidates[0]][r], scores[ids[0]][candidates[1]][r],
                         scores[ids[1]][candidates[0]][r], scores[ids[1]][candidates[1]][r]]
        if debug:
            print("- reference " + str(r) + " --- scores = " + str(new_scores[r]))
        sum_agreement00 += new_scores[r][0] + new_scores[r][3]
        sum_agreement01 += new_scores[r][1] + new_scores[r][2]


    if debug:
        print("sum_agreement00 = " + str(sum_agreement00))
        print("sum_agreement01 = " + str(sum_agreement01))

    if sum_agreement00 > sum_agreement01:
        name[ids[0]] = candidates[0]
        name[ids[1]] = candidates[1]
        name_certainty[ids[0]] = int(100.0 * sum_agreement00 / (sum_agreement00 + sum_agreement01))
        name_certainty[ids[1]] = int(100.0 * sum_agreement00 / (sum_agreement00 + sum_agreement01))
    elif sum_agreement01 > sum_agreement00:
        name[ids[0]] = candidates[1]
        name[ids[1]] = candidates[0]
        name_certainty[ids[0]] = int(100.0 * sum_agreement01 / (sum_agreement00 + sum_agreement01))
        name_certainty[ids[1]] = int(100.0 * sum_agreement01 / (sum_agreement00 + sum_agreement01))
    else:
        msg = "there is no agreement at all for cells " + str(candidates)
        monitoring.to_log_and_console(str(proc) + ": " + msg)
        name[ids[0]] = None
        name[ids[1]] = None
        name_certainty[ids[0]] = 0
        name_certainty[ids[1]] = 0

    return name, name_certainty


########################################################################################
#
# naming procedure
#
########################################################################################

def _debug_print_neighborhoods(mother, neighborhoods):
    proc = "_debug_print_neighborhoods"

    daughters = _get_daughter_names(mother)

    print("mother = " + str(mother))
    print("daughters = " + str(daughters))

    neighbors = set()
    for d in daughters:
        for r in neighborhoods[d]:
            neighbors = neighbors.union(set(neighborhoods[d][r].keys()))
    neighbors = sorted(list(neighbors))

    title_is_print = False
    for d in daughters:
        refs = sorted(neighborhoods[d].keys())
        for r in refs:
            names = ""
            surfs = ""
            for n in neighbors:
                names += " " + str(n)
                if n in neighborhoods[d][r]:
                    surfs += " " + "{:.2f}".format(neighborhoods[d][r][n])
                else:
                    surfs += " " + "0"
            if not title_is_print:
                print(str(d) + " " + str(r) + names)
                title_is_print = True
            print(str(d) + " " + str(r) + surfs)

    return


########################################################################################
#
# naming procedure
#
########################################################################################

def _print_neighborhoods(mother, ancestor_name, prop, neighborhoods, time_digits_for_cell_id=4):
    daughters = _get_daughter_names(mother)
    div = 10 ** time_digits_for_cell_id
    for d in daughters:
        #
        # build contact surface as a dictionary of names
        # 1. background
        # 2. cell already named
        # 3. daughter cell not named, then named after its mother
        #    there might be two cells with this name
        #
        contact = {}
        for c in prop['cell_contact_surface'][d]:
            if int(c) % div == 1 or int(c) % div == 0:
                contact['background'] = contact.get('background', []) + [prop['cell_contact_surface'][d][c]]
            elif c in prop['cell_name']:
                contact[prop['cell_name'][c]] = prop['cell_contact_surface'][d][c]
            elif c in ancestor_name:
                contact[ancestor_name[c]] = contact.get(ancestor_name[c], []) + [prop['cell_contact_surface'][d][c]]
            else:
                return None

        if False:
            names = "neighbors"
            surfs = "contacts"
            for n in sorted(contact.keys()):
                names += " " + str(n)
                surfs += " " + str(contact[n])
            print(names)
            print(surfs)


def _propagate_naming(prop, neighborhoods, time_digits_for_cell_id=4):
    proc = "_propagate_naming"

    if 'cell_lineage' not in prop:
        monitoring.to_log_and_console(str(proc) + ": 'cell_lineage' was not in dictionary")
        return None

    if 'cell_contact_surface' not in prop:
        monitoring.to_log_and_console(str(proc) + ": 'cell_contact_surface' was not in dictionary")
        return None

    if 'cell_name' not in prop:
        monitoring.to_log_and_console(str(proc) + ": 'cell_name' was not in dictionary")
        return None

    #
    #
    #
    lineage = prop['cell_lineage']

    #
    # remove empty names
    # leading or trailing spaces
    #
    cells = prop['cell_name'].keys()
    for c in cells:
        if prop['cell_name'][c] == '':
            del prop['cell_name'][c]
            continue
        prop['cell_name'][c] = prop['cell_name'][c].strip()

    #
    # initialize 'selection_name_choice_certainty'
    #
    prop['selection_name_choice_certainty'] = {}
    for k in prop['cell_name']:
        prop['selection_name_choice_certainty'][k] = 100

    #
    #
    #
    reverse_lineage = {v: k for k, values in lineage.iteritems() for v in values}
    cells = list(set(lineage.keys()).union(set([v for values in lineage.values() for v in values])))

    #
    #
    #
    debug_cells = ['a7.0002*']
    if False:
        for d in debug_cells:
            _debug_print_neighborhoods(d, neighborhoods)

    #
    # backward propagation
    #
    monitoring.to_log_and_console(str(proc) + ": backward propagation")
    cells = sorted(cells, reverse=True)
    for c in cells:
        if c not in prop['cell_name']:
            continue
        if c not in reverse_lineage:
            continue
        mother = reverse_lineage[c]
        if len(lineage[mother]) == 1:
            if mother in prop['cell_name']:
                if prop['cell_name'][mother] != prop['cell_name'][c]:
                    prop['selection_name_choice_certainty'][mother] = 0
                    msg = ": weird, cell " + str(mother) + " is named " + str(prop['cell_name'][mother])
                    msg += ", but should be named " + str(prop['cell_name'][c])
                    msg += " as its single daughter"
                    monitoring.to_log_and_console(str(proc) + msg)
            else:
                prop['cell_name'][mother] = prop['cell_name'][c]
                prop['selection_name_choice_certainty'][mother] = 100
        elif len(lineage[mother]) == 2:
            ancestor_name = _get_mother_name(prop['cell_name'][c])
            if mother in prop['cell_name']:
                if prop['cell_name'][mother] != ancestor_name:
                    prop['selection_name_choice_certainty'][mother] = 0
                    msg = ": weird, cell " + str(mother) + " is named " + str(prop['cell_name'][mother])
                    msg += ", but should be named " + str(ancestor_name)
                    msg += " since one of its daughter is named " + str(prop['cell_name'][c])
                    monitoring.to_log_and_console(str(proc) + msg)
            else:
                prop['cell_name'][mother] = ancestor_name
                prop['selection_name_choice_certainty'][mother] = 100
        else:
            msg = ": weird, cell " + str(mother) + " has " + str(len(lineage[mother])) + "daughter(s)"
            monitoring.to_log_and_console(str(proc) + msg)

    #
    # forward propagation
    #
    monitoring.to_log_and_console(str(proc) + ": forward propagation")

    cells = sorted(cells)
    div = 10 ** time_digits_for_cell_id
    cells_per_time = {}
    missing_name = {}
    for c in cells:
        t = int(c) // div
        #
        # get cells and cell names at each time point
        #
        if t not in cells_per_time:
            cells_per_time[t] = [c]
        else:
            cells_per_time[t].append(c)
        if c not in prop['cell_name']:
            if t not in missing_name:
                missing_name[t] = [c]
            else:
                missing_name[t].append(c)

    timepoints = sorted(missing_name.keys())
    ancestor_name = {}
    cell_not_named = []

    for t in timepoints:
        division_to_be_named = {}
        for c in missing_name[t]:

            # already named
            if c in prop['cell_name']:
                continue
            # root of a tree, can not be named
            if c not in reverse_lineage:
                if c not in cell_not_named:
                    cell_not_named.append(c)
                monitoring.to_log_and_console(str(proc) + ": weird, cell " + str(c) + " is root of a subtree")
                continue

            # get its mother
            mother = reverse_lineage[c]
            # mother not in lineage
            # to account for lineage errors
            if mother not in lineage:
                monitoring.to_log_and_console(str(proc) + ": weird, cell " + str(mother) + " is not in lineage")
                continue
            # mother not named, can name the cell either
            if mother in cell_not_named:
                if c not in cell_not_named:
                    cell_not_named.append(c)
                continue
            # mother not named, can name the cell either
            if mother not in prop['cell_name']:
                if mother not in cell_not_named:
                    cell_not_named.append(mother)
                if c not in cell_not_named:
                    cell_not_named.append(c)
                msg = "mother cell " + str(mother) + " is not named."
                msg += " Can not name cell " + str(c) + " either."
                monitoring.to_log_and_console(str(proc) + ": " + msg, 5)
                if mother in ancestor_name:
                    ancestor_name[c] = ancestor_name[mother]
                else:
                    msg = "weird, cell " + str(mother) + " is not named and have no ancestor"
                    monitoring.to_log_and_console(str(proc) + ": " + msg, 5)
                continue
            #
            # easy case
            # give name to cells that are only daughter
            #
            if len(lineage[mother]) == 1:
                prop['cell_name'][c] = prop['cell_name'][mother]
                prop['selection_name_choice_certainty'][c] = prop['selection_name_choice_certainty'][mother]
            #
            # in case of division:
            # 1. give name if the sister cell is named
            # 2. keep divisions to be solved
            #
            elif len(lineage[mother]) == 2:
                daughters = copy.deepcopy(lineage[mother])
                daughters.remove(c)
                daughter_names = _get_daughter_names(prop['cell_name'][mother])
                #
                # daughter cell is already named
                #
                if daughters[0] in prop['cell_name']:
                    if prop['cell_name'][daughters[0]] in daughter_names:
                        daughter_names.remove(prop['cell_name'][daughters[0]])
                        prop['cell_name'][c] = daughter_names[0]
                        prop['selection_name_choice_certainty'][c] = prop['selection_name_choice_certainty'][daughters[0]]
                    else:
                        msg = ": weird, cell " + str(daughters[0]) + " is named " + str(prop['cell_name'][daughters[0]])
                        msg += ", but should be named in " + str(daughter_names) + " since its mother cell "
                        msg += str(mother) + " is named " + str(prop['cell_name'][mother])
                        monitoring.to_log_and_console(str(proc) + msg)
                    continue
                #
                # both daughters are not named: ancestor_name keep trace of their mother name
                #
                division_to_be_named[mother] = [daughters[0], c]
                ancestor_name[daughters[0]] = prop['cell_name'][mother]
                ancestor_name[c] = prop['cell_name'][mother]
            else:
                if mother not in cell_not_named:
                    cell_not_named.append(mother)
                cell_not_named.append(c)
                msg = ": weird, cell " + str(mother) + " has " + str(len(lineage[mother])) + " daughter(s)."
                msg += " Its offspring will not be named."
                monitoring.to_log_and_console(str(proc) + msg)

        if division_to_be_named == {}:
            continue

        #
        # here we have a dictionary of divisions to be named (key = mother id, value = array of sister ids)
        #
        if True:
            for mother, daughters in division_to_be_named.iteritems():
                debug = prop['cell_name'][mother] in debug_cells
                #
                # scores is a dictionary of dictionary
                # scores[cell id][name][ref name] is an array of scalar product
                # cell id = d in daughters
                # name = name in daughter_names(mother)
                # the length of the array is the occurrence of [n in daughter_names(mother)] in the
                # neighborhood dictionary
                #
                scores = _build_scores(mother, daughters, ancestor_name, prop, neighborhoods,
                                       time_digits_for_cell_id=time_digits_for_cell_id, debug=debug)
                if debug:
                    print("scores = " + str(scores))
                if scores is None:
                    for c in daughters:
                        if c not in cell_not_named:
                            cell_not_named.append(c)
                    # msg = "\t error when building scores for daughters " + str(daughters) + " of cell " + str(mother)
                    # monitoring.to_log_and_console(msg)
                    msg = " Can not name cells " + str(daughters) + " and their offsprings."
                    monitoring.to_log_and_console(str(proc) + ": " + msg)
                    continue

                name, name_certainty = _analyse_scores(scores, debug=debug)

                for c in name:
                    if name[c] is not None:
                        prop['cell_name'][c] = name[c]
                    prop['selection_name_choice_certainty'][c] = name_certainty[c]
                    if c in ancestor_name:
                        del ancestor_name[c]

    return prop


########################################################################################
#
#
#
########################################################################################

def naming_process(experiment, parameters):
    proc = "naming_process"
    #
    # parameter type checking
    #

    if not isinstance(experiment, common.Experiment):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'experiment' variable: "
                                      + str(type(experiment)))
        sys.exit(1)

    if not isinstance(parameters, NamingParameters):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'parameters' variable: "
                                      + str(type(parameters)))
        sys.exit(1)

    time_digits_for_cell_id = experiment.get_time_digits_for_cell_id()

    #
    # should we clean reference here?
    #
    neighborhoods = build_neighborhoods(parameters.referenceFiles, time_digits_for_cell_id=time_digits_for_cell_id,
                                        add_symmetric_neighborhood=parameters.use_symmetric_neighborhood,
                                        reference_diagnosis=parameters.reference_diagnosis)
    if parameters.reference_diagnosis:
        neighborhood_diagnosis(neighborhoods)

    if parameters.improve_consistency:
        _consistency_improvement(neighborhoods)

    if True:
        _networkx_neighborhood_consistency(neighborhoods)

    # for c in neighborhoods:
    #     print("- #" + str(len(neighborhoods[c])) + " : " + str(neighborhoods[c]))

    #
    # read input properties to be named
    #
    prop = {}
    reference_prop = {}
    discrepancies = {}

    if parameters.testFile is not None:
        reference_prop = properties.read_dictionary(parameters.testFile, inputpropertiesdict={})
        prop = _build_test_set(reference_prop, time_digits_for_cell_id=time_digits_for_cell_id, ncells=64)
        if prop is None:
            monitoring.to_log_and_console(str(proc) + ": error when building test set")
            sys.exit(1)
    elif parameters.inputFiles is not None:
        prop = properties.read_dictionary(parameters.inputFiles, inputpropertiesdict={})

    if 'cell_name' not in prop:
        monitoring.to_log_and_console(str(proc) + ": no 'cell_name' in input dictionary")
        sys.exit(1)

    # clean from empty names
    cells = prop['cell_name'].keys()
    for c in cells:
        if prop['cell_name'][c] == '':
            del prop['cell_name'][c]

    #
    # naming propagation
    #
    prop = _propagate_naming(prop, neighborhoods)
    prop = properties.set_fate_from_names(prop, time_digits_for_cell_id=time_digits_for_cell_id)
    prop = properties.set_color_from_fate(prop)
    #
    #
    #
    if parameters.testFile is not None:
        _test_naming(prop, reference_prop, discrepancies)

    if isinstance(parameters.outputFile, str):
        properties.write_dictionary(parameters.outputFile, prop)
