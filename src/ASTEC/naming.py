import os
import imp
import sys
import copy
import collections
import numpy as np
import random
import math
import cPickle as pkl

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
        self.varprint('reference_diagnosis', self.testFile)

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
# is redundant with the diagnosis may in properties.py
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
        if t not in cells_per_time:
            cells_per_time[t] = [n]
        else:
            cells_per_time[t].append(n)
        if c in name:
            if t not in names_per_time:
                names_per_time[t] = [name[c]]
            else:
                names_per_time[t].append(name[c])
        #
        # check names
        #
        if c not in name:
            if t not in missing_name:
                missing_name[t] = [n]
            else:
                missing_name[t].append(n)
        elif c in reverse_lineage:
            mother = reverse_lineage[c]
            if mother not in name:
                if verbose:
                    msg = ": weird, cell " + str(c) + " has a name = " + str(name[c])
                    msg += ", but its mother cell " + str(mother) + " has no name"
                    monitoring.to_log_and_console(str(proc) + msg)
                if t not in error_name:
                    error_name[t] = [n]
                else:
                    names_per_time[t].append(n)
            else:
                if len(lineage[mother]) == 1:
                    if name[mother] != name[c]:
                        if verbose:
                            msg = ": weird, cell " + str(c) + " has a name = " + str(name[c])
                            msg += " different than its mother cell " + str(mother) + " name = " + str(name[mother])
                            monitoring.to_log_and_console(str(proc) + msg)
                        if t not in error_name:
                            error_name[t] = [n]
                        else:
                            error_name[t].append(n)
                elif len(lineage[mother]) == 2:
                    daughter_names = _get_daughter_names(name[mother])
                    if name[c] not in daughter_names:
                        if verbose:
                            msg = ": weird, name of cell " + str(c) + " is " + str(name[c])
                            msg += " but should be in " + str(daughter_names)
                            msg += " since its mother cell " + str(mother) + " is named " + str(name[mother])
                            monitoring.to_log_and_console(str(proc) + msg)
                        if t not in error_name:
                            error_name[t] = [n]
                        else:
                            error_name[t].append(n)
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
                            if t not in error_name:
                                error_name[t] = [n]
                            else:
                                error_name[t].append(n)
                        elif name[siblings[0]] == name[c]:
                            if verbose:
                                msg = ": weird, name of cell " + str(c) + ", " + str(name[c])
                                msg += ", is the same than its sibling " + str(siblings[0])
                                msg += ", their mother cell " + str(mother) + " is named " + str(name[mother])
                                monitoring.to_log_and_console(str(proc) + msg)
                            if t not in error_name:
                                error_name[t] = [n]
                            else:
                                error_name[t].append(n)
                        elif name[siblings[0]] != daughter_names[0]:
                            if verbose:
                                msg = ": weird, name of cell " + str(siblings[0]) + " is " + str(name[siblings[0]])
                                msg += " but should be " + str(daughter_names[0])
                                msg += " since its mother cell " + str(mother) + " is named " + str(name[mother])
                                msg += " and its sibling " + str(c) + " is named " + str(name[c])
                                monitoring.to_log_and_console(str(proc) + msg)
                            if t not in error_name:
                                error_name[t] = [n]
                            else:
                                error_name[t].append(n)
                else:
                    if verbose:
                        monitoring.to_log_and_console(str(proc) + ": weird, cell " + str(mother) + " has " +
                                                      str(len(lineage[mother])) + " daughter cells")

        #
        # check contact surfaces
        #
        if c not in contact:
            if t not in missing_contact:
                missing_contact[t] = [n]
            else:
                missing_contact[t].append(n)

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
        monitoring.to_log_and_console("\n")

    return [first_time, last_time]


########################################################################################
#
# obsolete: basic correction on pre-existing naming
# was developed for correction/evaluation of reference embryos
#
########################################################################################

def correct_reference(prop):
    proc = "correct_reference"
    #
    # correction of prop['cell_name']
    #
    # this is a very simple and ad-hoc correction scheme of errors easy to be corrected
    # - the daughter cell has not the same name than its mother cell
    # - in case of division
    #   - there is an error/typo in either [a,b] or [_,*]
    #   - the cell is mis-named but its sibling is correctly named
    #
    if 'cell_lineage' not in prop:
        monitoring.to_log_and_console(str(proc) + ": 'cell_lineage' was not in dictionary")
        return

    if 'cell_name' not in prop:
        monitoring.to_log_and_console(str(proc) + ": 'cell_name' was not in dictionary")
        return

    lineage = prop['cell_lineage']

    reverse_lineage = {v: k for k, values in lineage.iteritems() for v in values}

    cells = list(set(lineage.keys()).union(set([v for values in lineage.values() for v in values])))
    cells = sorted(cells)

    for c in cells:
        if c not in prop['cell_name']:
            continue
        if prop['cell_name'][c] == '':
            msg = ": cell " + str(c) + " has a weird name = '" + str(prop['cell_name'][c]) + "', remove it"
            monitoring.to_log_and_console(str(proc) + msg)
            del prop['cell_name'][c]
            continue
        if c not in reverse_lineage:
            continue
        #
        # c has a name and c is not the first cell of branch
        #
        mother = reverse_lineage[c]
        if mother not in prop['cell_name']:
            msg = ": weird, cell " + str(c) + " has a name = " + str(prop['cell_name'][c])
            msg += ",but its mother cell " + str(mother) + " has no name"
            monitoring.to_log_and_console(str(proc) + msg)
            continue
        #
        # verify whether the cell has the same name than its mother cell
        #
        if len(lineage[mother]) == 1:
            if prop['cell_name'][mother] != prop['cell_name'][c]:
                msg = "\t (1) correct name of cell " + str(c) + " from " + str(prop['cell_name'][c]) + " to "
                msg += str(prop['cell_name'][mother])
                monitoring.to_log_and_console(msg)
                prop['cell_name'][c] = prop['cell_name'][mother]
            continue
        #
        # verify whether the cell has one of the two names deriving from the mother
        # if no:
        # - check whether it is a mistake on either [a,b] or [_,*]
        # - check whether the sibling is named
        #
        if len(lineage[mother]) == 2:
            daughter_names = _get_daughter_names(prop['cell_name'][mother])
            if prop['cell_name'][c] in daughter_names:
                continue
            if prop['cell_name'][c][0] == 'a' and 'b' + prop['cell_name'][c][1:] in daughter_names:
                msg = "\t (2) correct name of cell " + str(c) + " from " + str(prop['cell_name'][c]) + " to "
                msg += 'b' + prop['cell_name'][c][1:]
                monitoring.to_log_and_console(msg)
                prop['cell_name'][c] = 'b' + prop['cell_name'][c][1:]
                continue
            if prop['cell_name'][c][0] == 'b' and 'a' + prop['cell_name'][c][1:] in daughter_names:
                msg = "\t (3) correct name of cell " + str(c) + " from " + str(prop['cell_name'][c]) + " to "
                msg += 'a' + prop['cell_name'][c][1:]
                monitoring.to_log_and_console(msg)
                prop['cell_name'][c] = 'a' + prop['cell_name'][c][1:]
                continue
            if prop['cell_name'][c][-1] == '_' and prop['cell_name'][c][:-1] + '*' in daughter_names:
                msg = "\t (4) correct name of cell " + str(c) + " from " + str(prop['cell_name'][c]) + " to "
                msg += prop['cell_name'][c][:-1] + '*'
                monitoring.to_log_and_console(msg)
                prop['cell_name'][c] = prop['cell_name'][c][:-1] + '*'
                continue
            if prop['cell_name'][c][-1] == '*' and prop['cell_name'][c][:-1] + '_' in daughter_names:
                msg = "\t (5) correct name of cell " + str(c) + " from " + str(prop['cell_name'][c]) + " to "
                msg += prop['cell_name'][c][:-1] + '_'
                monitoring.to_log_and_console(msg)
                prop['cell_name'][c] = prop['cell_name'][c][:-1] + '_'
                continue
            siblings = copy.deepcopy(lineage[mother])
            siblings.remove(c)
            if siblings[0] in prop['cell_name'] and prop['cell_name'][siblings[0]] in daughter_names:
                if prop['cell_name'][c] != prop['cell_name'][siblings[0]]:
                    daughter_names.remove(prop['cell_name'][siblings[0]])
                    msg = "\t (6) correct name of cell " + str(c) + " from " + str(prop['cell_name'][c]) + " to "
                    msg += daughter_names[0]
                    monitoring.to_log_and_console(msg)
                    prop['cell_name'][c] = daughter_names[0]
                    continue
                msg = ": weird, cell " + str(c) + " has the same name " + str(prop['cell_name'][c])
                msg += " than its sibling " + str(siblings[0])
                monitoring.to_log_and_console(str(proc) + msg)
        if len(lineage[mother]) > 2:
            monitoring.to_log_and_console(str(proc) + ": weird, cell " + str(c) + " has " +
                                          str(len(lineage[mother])) + " daughter cells")
        msg = "\t can not correct cell " + str(c) + " with name " + str(prop['cell_name'][c])
        monitoring.to_log_and_console(msg)

    return prop


########################################################################################
#
# obsolete: do some cleaning (keep time interval without errors)
# was developed for correction/evaluation of reference embryos
# and assessment of the developed method
#
########################################################################################

def clean_reference(prop, time_digits_for_cell_id=4):
    proc = "clean_reference"

    #
    # keep only a time interval without incoherencies
    #
    returned_prop = {}
    if 'cell_lineage' not in prop:
        monitoring.to_log_and_console(str(proc) + ": 'cell_lineage' was not in dictionary")
        return None

    if 'cell_contact_surface' not in prop:
        monitoring.to_log_and_console(str(proc) + ": 'cell_contact_surface' was not in dictionary")
        return None

    if 'cell_name' not in prop:
        monitoring.to_log_and_console(str(proc) + ": 'cell_name' was not in dictionary")
        return None

    returned_prop['cell_lineage'] = copy.deepcopy(prop['cell_lineage'])
    returned_prop['cell_name'] = copy.deepcopy(prop['cell_name'])
    returned_prop['cell_contact_surface'] = copy.deepcopy(prop['cell_contact_surface'])

    time_interval = diagnosis(prop, time_digits_for_cell_id=4, verbose=False)
    if time_interval is None:
        monitoring.to_log_and_console(str(proc) + ": unable to get valid time interval")
        return None
    monitoring.to_log_and_console(str(proc) + ": valid time interval = " + str(time_interval))

    first_time = time_interval[0]
    last_time = time_interval[1]
    div = 10 ** time_digits_for_cell_id

    for c in prop['cell_lineage']:
        t = int(c) // div
        if t < first_time or last_time <= t:
            del returned_prop['cell_lineage'][c]

    for c in prop['cell_name']:
        t = int(c) // div
        if t < first_time or last_time < t:
            del returned_prop['cell_name'][c]

    for c in prop['cell_contact_surface']:
        t = int(c) // div
        if t < first_time or last_time < t:
            del returned_prop['cell_contact_surface'][c]

    return returned_prop


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
    name_errors = {}
    key_list = sorted(prop['cell_name'].keys())
    for k in key_list:
        if k not in reference_prop['cell_name']:
            monitoring.to_log_and_console("\t weird, key " + str(k) + " is not in reference properties")
        elif prop['cell_name'][k] != reference_prop['cell_name'][k]:
            if prop['cell_name'][k] not in name_errors:
                name_errors[prop['cell_name'][k]] = 1
                msg = "cell_name[" + str(k) + "]=" + str(prop['cell_name'][k]) + " is not equal to reference name "
                msg += str(reference_prop['cell_name'][k])
                if _get_mother_name(prop['cell_name'][k]) in discrepancies:
                    msg += ", division is incoherent among references"
                monitoring.to_log_and_console("\t " + msg)
            else:
                name_errors[prop['cell_name'][k]] += 1

    #
    # if an error occur, a bad choice has been made at the mother division (first_errors)
    # or at some ancestor (second_errors)
    #
    first_errors = {}
    second_errors = {}
    first_error_mothers = []
    names = name_errors.keys()
    for n in names:
        if _get_mother_name(n) in names or _get_mother_name(n) in first_error_mothers:
            second_errors[n] = name_errors[n]
        else:
            first_errors[n] = name_errors[n]
            if _get_mother_name(n) not in first_error_mothers:
                first_error_mothers.append(_get_mother_name(n))

    name_missing = {}
    reference_name = {}
    key_list = sorted(reference_prop['cell_name'].keys())
    for k in key_list:
        if reference_prop['cell_name'][k] not in reference_name:
            reference_name[reference_prop['cell_name'][k]] = 1
        else:
            reference_name[reference_prop['cell_name'][k]] += 1
        if k not in prop['cell_name']:
            if reference_prop['cell_name'][k] not in name_missing:
                name_missing[reference_prop['cell_name'][k]] = 1
                msg = "reference cell_name[" + str(k) + "]=" + str(reference_prop['cell_name'][k]) + " is not found"
                monitoring.to_log_and_console("\t " + msg)
            else:
                name_missing[reference_prop['cell_name'][k]] += 1

    error_count = 0
    for k in name_errors:
        error_count += name_errors[k]
    first_error_count = 0
    for k in first_errors:
        first_error_count += first_errors[k]
    second_error_count = 0
    for k in second_errors:
        second_error_count += second_errors[k]
    missing_count = 0
    for k in name_missing:
        missing_count += name_missing[k]

    msg = "ground-truth cells = " + str(len(reference_prop['cell_name'])) + " --- "
    msg += "ground-truth names = " + str(len(reference_name)) + " --- "
    msg += "tested cells = " + str(len(prop['cell_name'])) + " --- "
    msg += "retrieved names = " + str(len(reference_name) - len(name_missing)) + "\n"
    msg += "\t missing cell names = " + str(missing_count) + " for " + str(len(name_missing)) + " names --- "
    msg += "total errors on cell name = " + str(error_count) + " for " + str(len(name_errors)) + " names \n"
    msg += "\t first errors in lineage = " + str(first_error_count) + " for " + str(len(first_errors)) + " names --- "
    msg += "induced errors = " + str(second_error_count) + " for " + str(len(second_errors)) + " names \n"
    msg += "\t division of first errors = " + str(sorted(first_error_mothers))
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


def _print_neighborhoods(neighborhood0, neighborhood1, title=None):
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
#
#
########################################################################################

def _add_neighborhoods(previous_neighborhoods, prop, reference_name, time_digits_for_cell_id=4):
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
    daughters = [lineage[c][0] for c in lineage if len(lineage[c]) == 2]
    daughters += [lineage[c][1] for c in lineage if len(lineage[c]) == 2]

    missing_name = 0
    missing_contact = 0

    for d in daughters:
        #
        # check whether the cell is in dictionaries
        #
        if d not in name:
            missing_name += 1
            monitoring.to_log_and_console(str(proc) + ": daughter cell #" + str(d)
                                          + " was not found in 'cell_name' dictionary. Skip it", 4)
            continue

        if d not in contact:
            missing_contact += 1
            monitoring.to_log_and_console(str(proc) + ": cell #" + str(d)
                                          + " was not found in 'cell_contact_surface' dictionary. Skip it")
            continue

        #
        # get the list of neighborhoods for the cell d
        # create an empty list if required
        #
        if prop['cell_name'][d] not in previous_neighborhoods:
            previous_neighborhoods[prop['cell_name'][d]] = {}

        neighbor = {}
        neighbor_is_complete = True
        for c in contact[d]:
            n = int(c) % div
            if n == 1 or n == 0:
                if 'background' not in neighbor:
                    neighbor['background'] = contact[d][c]
                else:
                    neighbor['background'] += contact[d][c]
            elif c in name:
                neighbor[prop['cell_name'][c]] = contact[d][c]
            else:
                neighbor_is_complete = False
                missing_name += 1
                msg = ": cell #" + str(c) + " was not found in 'cell_name' dictionary.\n"
                msg += "\t Neighborhood of " + str(prop['cell_name'][d]) + " is not complete. Skip it"
                monitoring.to_log_and_console(str(proc) + msg)
                continue
        if neighbor_is_complete:
            if reference_name in previous_neighborhoods[prop['cell_name'][d]]:
                msg = "weird, " + str(reference_name) + " was already indexed for cell " + str(prop['cell_name'][d])
                monitoring.to_log_and_console(str(proc) + ": " + msg)
            previous_neighborhoods[prop['cell_name'][d]][reference_name] = neighbor

    if missing_name > 0:
        monitoring.to_log_and_console(str(proc) + ": daughter cells without names = " + str(missing_name) + "/" +
                                      str(len(daughters)))
    if missing_contact > 0:
        monitoring.to_log_and_console(str(proc) + ": daughter cells without contact surfaces  = " +
                                      str(missing_name) + "/" + str(len(daughters)))
    return previous_neighborhoods


def build_neighborhoods(referenceFiles, time_digits_for_cell_id, reference_diagnosis=False):
    neighborhoods = {}
    if isinstance(referenceFiles, str):
        prop = properties.read_dictionary(referenceFiles, inputpropertiesdict={})
        if reference_diagnosis:
            diagnosis(prop, time_digits_for_cell_id=time_digits_for_cell_id)
        name = referenceFiles.split(os.path.sep)[-1]
        neighborhoods = _add_neighborhoods(neighborhoods, prop, reference_name=name,
                                           time_digits_for_cell_id=time_digits_for_cell_id)
        del prop
    elif isinstance(referenceFiles, list):
        for f in referenceFiles:
            prop = properties.read_dictionary(f, inputpropertiesdict={})
            if reference_diagnosis:
                diagnosis(prop, time_digits_for_cell_id=time_digits_for_cell_id)
            name = f.split(os.path.sep)[-1]
            neighborhoods = _add_neighborhoods(neighborhoods, prop, reference_name=name,
                                               time_digits_for_cell_id=time_digits_for_cell_id)
            del prop
    return neighborhoods


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
def _get_score(neigh0, neigh1, title=None):
    tmp0 = copy.deepcopy(neigh0)
    tmp1 = copy.deepcopy(neigh1)
    # 1. parse names of tmp1
    tmp0, tmp1, vect0, vect1 = _build_common_contact_surfaces(tmp0, tmp1, {}, {})
    # 2. parse remaining names of tmp0
    tmp1, tmp0, vect1, vect0 = _build_common_contact_surfaces(tmp1, tmp0, vect1, vect0)
    score = _scalar_product(vect0, vect1)
    if title is not None:
        _print_neighborhoods(vect0, vect1, title=title)
        monitoring.to_log_and_console("\t score = " + str(score) + "\n")
    # compute score as a scalar product
    return score


########################################################################################
#
#
#
########################################################################################

def _check_neighborhood_consistency(neighborhoods):
    proc = "_check_neighborhood_consistency"

    monitoring.to_log_and_console("")
    monitoring.to_log_and_console(str(proc))
    monitoring.to_log_and_console("-------------------------------------------")
    monitoring.to_log_and_console("--- reference neighborhoods consistency ---")

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
        if mother_name not in references:
            references[mother_name] = set(neighborhoods[cell_name].keys())
        else:
            references[mother_name].union(set(neighborhoods[cell_name].keys()))

    #
    # get discrepancies
    #
    discrepancy = {}
    tested_couples = {}
    for cell_name in cell_names:
        #
        # get cell name and sister name
        #
        sister_name = _get_sister_name(cell_name)
        if sister_name not in neighborhoods:
            msg = "weird, cell " + str(cell_name) + " is in the reference neighborhoods, while its sister "
            msg += str(sister_name) + " is not "
            monitoring.to_log_and_console(str(proc) + ": " + msg)
            cell_names.remove(cell_name)
            continue
        #
        # only one neighborhood, nothing to test
        #
        if len(neighborhoods[cell_name]) == 1:
            cell_names.remove(cell_name)
            cell_names.remove(sister_name)
            continue
        #
        # get two reference embryos
        #
        for ref1 in neighborhoods[cell_name]:
            for ref2 in neighborhoods[cell_name]:
                if ref2 <= ref1:
                    continue
                if ref2 not in neighborhoods[sister_name]:
                    msg = "weird, reference " + str(ref2) + " is in the neighborhoods of cell "
                    msg += str(cell_name) + " but not of its sister " + str(sister_name)
                    monitoring.to_log_and_console(str(proc) + ": " + msg)
                    continue
                same_choice = _get_score(neighborhoods[cell_name][ref1], neighborhoods[cell_name][ref2])
                sister_choice = _get_score(neighborhoods[cell_name][ref1], neighborhoods[sister_name][ref2])
                mother_name = _get_mother_name(cell_name)
                if mother_name not in tested_couples:
                    tested_couples[mother_name] = 1
                else:
                    tested_couples[mother_name] += 1
                if same_choice > sister_choice:
                    continue
                if mother_name not in discrepancy:
                    discrepancy[mother_name] = []
                discrepancy[mother_name].append([ref1, ref2])

    #
    # if some divisions have some discrepancies, the following ones in the lineage
    # are likely to exhibit discrepancies also
    #
    second_discrepancy = {}
    first_discrepancy = {}
    if len(discrepancy) > 0:
        #
        mother_names = sorted(discrepancy.keys())
        for n in mother_names:
            if _get_mother_name(n) in mother_names or _get_mother_name(n) in first_discrepancy:
                second_discrepancy[n] = discrepancy[n]
            else:
                first_discrepancy[n] = discrepancy[n]
        #

        mother_names = first_discrepancy.keys()
        if len(mother_names) > 0:

            percents = []
            for mother_name in mother_names:
                percents.append(100.0 * float(len(discrepancy[mother_name])) / float(tested_couples[mother_name]))
            [sorted_percents, sorted_mothers] = zip(*sorted(zip(percents, mother_names), reverse=True))

            msg = "\n*** first order discrepancies = " + str(len(first_discrepancy))
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
        #
        mother_names = second_discrepancy.keys()
        if len(mother_names) > 0:

            percents = []
            for mother_name in mother_names:
                percents.append(100.0 * float(len(discrepancy[mother_name])) / float(tested_couples[mother_name]))
            [sorted_percent, sorted_mothers] = zip(*sorted(zip(percents, mother_names), reverse=True))

            msg = "\n*** second order discrepancies = " + str(len(second_discrepancy))
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

    msg = "tested divisions = " + str(len(tested_couples))
    monitoring.to_log_and_console(str(proc) + ": " + msg)
    msg = "divisions with discrepancies =  " + str(len(discrepancy))
    monitoring.to_log_and_console("\t " + msg)

    monitoring.to_log_and_console("-------------------------------------------")
    monitoring.to_log_and_console("")
    return


def check_leftright_consistency(neighborhoods):
    proc = "_check_leftright_consistency"
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
        if mother_name not in references:
            references[mother_name] = set(neighborhoods[cell_name].keys())
        else:
            references[mother_name].union(set(neighborhoods[cell_name].keys()))

    #
    #
    #
    mother_names = sorted(list(references.keys()))
    processed_mothers = []
    discrepancy = {}
    tested_cells = {}
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
                if mother not in discrepancy[reference]:
                    discrepancy[reference][mother] = [(daughter, symdaughter)]
                else:
                    discrepancy[reference][mother].append((daughter, symdaughter))

                msg = "   - '" + str(reference) + "': " + str(daughter) + " neighborhood is closest to "
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
            if d not in mother_by_discrepancy:
                mother_by_discrepancy[d] = [mother]
            else:
                mother_by_discrepancy[d].append(mother)

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

def _build_scores(mother, daughters, ancestor_name, prop, neighborhoods, time_digits_for_cell_id=4):
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
            return None, None, None

    div = 10 ** time_digits_for_cell_id

    score = {}
    ancestor_count = 0
    ancestor_surface = 0

    for d in daughters:
        score[d] = {}

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
                if 'background' not in contact:
                    contact['background'] = prop['cell_contact_surface'][d][c]
                else:
                    contact['background'] += prop['cell_contact_surface'][d][c]
            elif c in prop['cell_name']:
                contact[prop['cell_name'][c]] = prop['cell_contact_surface'][d][c]
            elif c in ancestor_name:
                if ancestor_name[c] not in contact:
                    contact[ancestor_name[c]] = prop['cell_contact_surface'][d][c]
                else:
                    contact[ancestor_name[c]] += prop['cell_contact_surface'][d][c]
                ancestor_count += 1
                ancestor_surface += contact[ancestor_name[c]]
            else:
                monitoring.to_log_and_console(str(proc) + ": there is no name for cell " + str(c))
                return None, None, None
        #
        # compute score for each candidate name by comparison with reference neighborhood
        #
        for name in daughter_names:
            score[d][name] = {}
            for reference_name in neighborhoods[name]:
                score[d][name][reference_name] = _get_score(contact, neighborhoods[name][reference_name])
    return score, ancestor_count, ancestor_surface


def _analyse_scores(scores):
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


########################################################################################
#
# naming procedure
#
########################################################################################

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
    # initialize 'cell_name_certainty'
    #
    prop['cell_name_certainty'] = {}
    for k in prop['cell_name']:
        prop['cell_name_certainty'][k] = 1.0

    #
    #
    #
    reverse_lineage = {v: k for k, values in lineage.iteritems() for v in values}
    cells = list(set(lineage.keys()).union(set([v for values in lineage.values() for v in values])))

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
                    prop['cell_name_certainty'][mother] = 0.0
                    msg = ": weird, cell " + str(mother) + " is named " + str(prop['cell_name'][mother])
                    msg += ", but should be named " + str(prop['cell_name'][c])
                    msg += " as its single daughter"
                    monitoring.to_log_and_console(str(proc) + msg)
            else:
                prop['cell_name'][mother] = prop['cell_name'][c]
                prop['cell_name_certainty'][mother] = 1.0
        elif len(lineage[mother]) == 2:
            ancestor_name = _get_mother_name(prop['cell_name'][c])
            if mother in prop['cell_name']:
                if prop['cell_name'][mother] != ancestor_name:
                    prop['cell_name_certainty'][mother] = 0.0
                    msg = ": weird, cell " + str(mother) + " is named " + str(prop['cell_name'][mother])
                    msg += ", but should be named " + str(ancestor_name)
                    msg += " since one of its daughter is named " + str(prop['cell_name'][c])
                    monitoring.to_log_and_console(str(proc) + msg)
            else:
                prop['cell_name'][mother] = ancestor_name
                prop['cell_name_certainty'][mother] = 1.0
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

    for t in timepoints:
        division_to_be_named = {}
        for c in missing_name[t]:

            if c in prop['cell_name']:
                continue
            if c not in reverse_lineage:
                continue
            mother = reverse_lineage[c]
            if mother not in lineage:
                monitoring.to_log_and_console(str(proc) + ": weird, cell " + str(mother) + " is not in lineage")
                continue
            if mother not in prop['cell_name']:
                monitoring.to_log_and_console(str(proc) + ": cell " + str(mother) + " is not named", 5)
                if mother in ancestor_name:
                    ancestor_name[c] = ancestor_name[mother]
                else:
                    msg = "weird, cell " + str(mother) + " is not named and have no ancestor"
                    monitoring.to_log_and_console(str(proc) + ": " + msg, 4)
                continue
            #
            # give name to cells that are only daughter
            #
            if len(lineage[mother]) == 1:
                prop['cell_name'][c] = prop['cell_name'][mother]
                prop['cell_name_certainty'][c] = prop['cell_name_certainty'][mother]
            #
            # in case of division:
            # 1. give name if the sister cell is named
            # 2. keep divisions to be solved
            #
            elif len(lineage[mother]) == 2:
                daughters = copy.deepcopy(lineage[mother])
                daughters.remove(c)
                daughter_names = _get_daughter_names(prop['cell_name'][mother])
                if daughters[0] in prop['cell_name']:
                    if prop['cell_name'][daughters[0]] in daughter_names:
                        daughter_names.remove(prop['cell_name'][daughters[0]])
                        prop['cell_name'][c] = daughter_names[0]
                        prop['cell_name_certainty'][c] = 1.0
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
                msg = ": weird, cell " + str(mother) + " has " + str(len(lineage[mother])) + " daughter(s)"
                monitoring.to_log_and_console(str(proc) + msg)

        if division_to_be_named == {}:
            continue

        if True:
            for mother, daughters in division_to_be_named.iteritems():
                #
                # scores is a dictionary of dictionary
                # scores[d in daughters][n in daughter_names(mother)] is an array of scalar product
                # the length of the array is the occurrence of [n in daughter_names(mother)] in the
                # neighborhood dictionary
                #
                scores, a_count, a_surface = _build_scores(mother, daughters, ancestor_name, prop, neighborhoods,
                                                           time_digits_for_cell_id=time_digits_for_cell_id)
                # print("scores = " + str(scores))
                if scores is None:
                    continue
                name, name_certainty = _analyse_scores(scores)

                for c in name:
                    prop['cell_name'][c] = name[c]
                    prop['cell_name_certainty'][c] = name_certainty[c]
                    if c in ancestor_name:
                        del ancestor_name[c]
        else:
            while len(division_to_be_named) > 0:
                scores = {}
                ancestor_count = {}
                ancestor_surface = {}
                mothers = division_to_be_named.keys()
                for mother in mothers:
                    s, a_count, a_surface = _build_scores(mother, division_to_be_named[mother], ancestor_name, prop,
                                                          neighborhoods,
                                                          time_digits_for_cell_id=time_digits_for_cell_id)
                    if s is None:
                        del division_to_be_named[mother]
                    scores[mother] = s
                    ancestor_count[mother] = a_count
                    ancestor_surface[mother] = a_surface

                if len(division_to_be_named) == 0:
                    break

                min_surface = None
                min_count = None
                for mother, daughters in division_to_be_named.iteritems():
                    if min_surface is None:
                        min_surface = mother
                        min_count = mother
                        continue
                    if ancestor_surface[min_surface] > ancestor_surface[mother]:
                        min_surface = mother
                    if ancestor_count[min_count] > ancestor_count[mother]:
                        min_count = mother

                chosen_mother = min_surface

                name, name_certainty = _analyse_scores(scores[chosen_mother])
                for c in name:
                    prop['cell_name'][c] = name[c]
                    prop['cell_name_certainty'][c] = name_certainty[c]
                    if c in ancestor_name:
                        del ancestor_name[c]

                del division_to_be_named[chosen_mother]

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
                                        reference_diagnosis=parameters.reference_diagnosis)
    if parameters.reference_diagnosis:
        neighborhood_diagnosis(neighborhoods)


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
        name_errors = _test_naming(prop, reference_prop, discrepancies)

    if isinstance(parameters.outputFile, str):
        properties.write_dictionary(parameters.outputFile, prop)
