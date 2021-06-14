import os
import imp
import sys
import copy
import operator
import shutil
import numpy as np

from scipy.stats.stats import pearsonr

import ASTEC.common as common
import ASTEC.properties as properties
from ASTEC.CommunFunctions.ImageHandling import imread, imsave, SpatialImage
import ASTEC.CommunFunctions.cpp_wrapping as cpp_wrapping

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


#
#
#

class PostCorrectionParameters(common.PrefixedParameter):

    ############################################################
    #
    # initialisation
    #
    ############################################################

    def __init__(self):

        common.PrefixedParameter.__init__(self, prefix=['postcor_', 'postcorrection_'])

        if "doc" not in self.__dict__:
            self.doc = {}

        doc = "\n"
        doc += "Post-correction overview:\n"
        doc += "=========================\n"
        doc += "The post-correction is twofold\n"
        doc += "1. The lineage tree is corrected. This procedure is also\n"
        doc += "    twofold\n"
        doc += "  a. end branch pruning. A lineage branch that ends before\n"
        doc += "     the last time point, or that ends at the last time point\n"
        doc += "     with a cell of too small volume, may be fused with its\n"
        doc += "     sister branch\n"
        doc += "     - if it is too short\n"
        doc += "     - if the volumes of the branch cell are anti-correlated\n"
        doc += "       with the ones of the sister branch\n"
        doc += "     - the division occurs too early: there is a close (in\n"
        doc += "       time) division sooner (in the common mother branch)\n"
        doc += "       or later (in the sister branch)\n"
        doc += "  b. bifurcation postponing (in time). A bifurcation may\n"
        doc += "     be postponed is the volumes of the branch cell are\n"
        doc += "     globally anti-correlated with the ones of the sister\n"
        doc += "     branch. The bifurcation is postponed as long as there\n"
        doc += "     is an anti-correlation of volumes in a short sliding\n"
        doc += "     window.\n"
        doc += "2. The corrections are reported in the segmentation images\n"
        doc += "   to produce the post-processed images\n"
        doc += "\n"
        self.doc['post_correction_overview'] = doc

        ############################################################
        #
        # initialisation
        #
        ############################################################

        doc = "\t Cell volume threshold.\n"
        doc += "\t Branches that ends at the last time point with\n"
        doc += "\t a too small volume for the last cell are candidates\n"
        doc += "\t for branch pruning.\n"
        self.doc['volume_minimal_value'] = doc
        self.volume_minimal_value = 2000

        doc = "\t Branch length threshold.\n"
        doc += "\t Too short branches are candidates for branch pruning.\n"
        self.doc['lifespan_minimal_value'] = doc
        self.lifespan_minimal_value = 25

        doc = "\t Possible values are True or False\n"
        self.doc['test_branch_length'] = doc
        self.test_branch_length = True

        doc = "\t Possible values are True or False\n"
        self.doc['test_early_division'] = doc
        self.test_early_division = True

        doc = "\t Possible values are True or False\n"
        self.doc['test_volume_correlation'] = doc
        self.test_volume_correlation = True

        doc = "\t Anti-correlation threshold for branch pruning\n"
        self.doc['correlation_threshold'] = doc
        self.correlation_threshold = 0.9

        doc = "\t Possible values are True or False\n"
        self.doc['test_postponing_division'] = doc
        self.test_postponing_division = True

        doc = "\t Anti-correlation threshold for bifurcation postponing\n"
        doc += "\t Used to select the branch candidate, and also\n"
        doc += "\t used for the sliding window.\n"
        self.doc['postponing_correlation_threshold'] = doc
        self.postponing_correlation_threshold = 0.8

        doc = "\t Branch length threshold. \n"
        doc += "\t Branch candidates for bifurcation postponing have\n"
        doc += "\t to be long enough\n"
        self.doc['postponing_minimal_length'] = doc
        self.postponing_minimal_length = 8

        doc = "\t Sliding window length for bifurcation postponing\n"
        self.doc['postponing_window_length'] = doc
        self.postponing_window_length = 5

        doc = "\t Possible values are True or False\n"
        doc += "\t If true, mimics historical behavior for bifurcation\n"
        doc += "\t postponing.\n"
        self.doc['mimic_historical_astec'] = doc
        self.mimic_historical_astec = False

        doc = "\t Number of processors for parallelization\n"
        doc += "\t Fragile\n"
        self.doc['processors'] = doc
        self.processors = 1

        # diagnosis
        doc = "\t Possible values are True or False\n"
        doc += "\t If True, print some diagnosis on lineage\n"
        self.doc['lineage_diagnosis'] = doc
        self.lineage_diagnosis = False

    ############################################################
    #
    # print / write
    #
    ############################################################

    def print_parameters(self):
        print("")
        print('#')
        print('# PostCorrectionParameters')
        print('#')
        print("")
        
        common.PrefixedParameter.print_parameters(self)

        for line in self.doc['post_correction_overview'].splitlines():
            print('# ' + line)

        self.varprint('volume_minimal_value', self.volume_minimal_value, self.doc['volume_minimal_value'])
        self.varprint('lifespan_minimal_value', self.lifespan_minimal_value, self.doc['lifespan_minimal_value'])
        self.varprint('test_branch_length', self.test_branch_length, self.doc['test_branch_length'])
        self.varprint('test_early_division', self.test_early_division, self.doc['test_early_division'])
        self.varprint('test_volume_correlation', self.test_volume_correlation, self.doc['test_volume_correlation'])
        self.varprint('correlation_threshold', self.correlation_threshold, self.doc['correlation_threshold'])

        self.varprint('test_postponing_division', self.test_postponing_division, self.doc['test_postponing_division'])
        self.varprint('postponing_correlation_threshold', self.postponing_correlation_threshold,
                      self.doc['postponing_correlation_threshold'])
        self.varprint('postponing_minimal_length', self.postponing_minimal_length,
                      self.doc['postponing_minimal_length'])
        self.varprint('postponing_window_length', self.postponing_window_length, self.doc['postponing_window_length'])

        self.varprint('mimic_historical_astec', self.mimic_historical_astec, self.doc['mimic_historical_astec'])

        self.varprint('processors', self.processors, self.doc['processors'])

        self.varprint('lineage_diagnosis', self.lineage_diagnosis, self.doc['lineage_diagnosis'])
        print("")

    def write_parameters_in_file(self, logfile):
        logfile.write("\n")
        logfile.write("# \n")
        logfile.write("# PostCorrectionParameters\n")
        logfile.write("# \n")
        logfile.write("\n")

        common.PrefixedParameter.write_parameters_in_file(self, logfile)

        for line in self.doc['post_correction_overview'].splitlines():
            logfile.write('# ' + line + '\n')

        self.varwrite(logfile, 'volume_minimal_value', self.volume_minimal_value, self.doc['volume_minimal_value'])
        self.varwrite(logfile, 'lifespan_minimal_value', self.lifespan_minimal_value,
                      self.doc['lifespan_minimal_value'])
        self.varwrite(logfile, 'test_branch_length', self.test_branch_length, self.doc['test_branch_length'])
        self.varwrite(logfile, 'test_early_division', self.test_early_division, self.doc['test_early_division'])
        self.varwrite(logfile, 'test_volume_correlation', self.test_volume_correlation,
                      self.doc['test_volume_correlation'])
        self.varwrite(logfile, 'correlation_threshold', self.correlation_threshold, self.doc['correlation_threshold'])

        self.varwrite(logfile, 'test_postponing_division', self.test_postponing_division,
                      self.doc['test_postponing_division'])
        self.varwrite(logfile, 'postponing_correlation_threshold', self.postponing_correlation_threshold,
                      self.doc['postponing_correlation_threshold'])
        self.varwrite(logfile, 'postponing_minimal_length', self.postponing_minimal_length,
                      self.doc['postponing_minimal_length'])
        self.varwrite(logfile, 'postponing_window_length', self.postponing_window_length,
                      self.doc['postponing_window_length'])

        self.varwrite(logfile, 'mimic_historical_astec', self.mimic_historical_astec,
                      self.doc['mimic_historical_astec'])

        self.varwrite(logfile, 'processors', self.processors, self.doc['processors'])

        self.varwrite(logfile, 'lineage_diagnosis', self.lineage_diagnosis, self.doc['lineage_diagnosis'])

        logfile.write("\n")
        return

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
        self.volume_minimal_value = self.read_parameter(parameters, 'volume_minimal_value', self.volume_minimal_value)
        self.volume_minimal_value = self.read_parameter(parameters, 'Volume_Threshold', self.volume_minimal_value)

        self.lifespan_minimal_value = self.read_parameter(parameters, 'lifespan_minimal_value',
                                                          self.lifespan_minimal_value)

        self.test_branch_length = self.read_parameter(parameters, 'test_branch_length', self.test_branch_length)
        self.test_early_division = self.read_parameter(parameters, 'test_early_division', self.test_early_division)
        self.test_early_division = self.read_parameter(parameters, 'soon', self.test_early_division)
        self.test_early_division = self.read_parameter(parameters, 'Soon', self.test_early_division)

        self.test_volume_correlation = self.read_parameter(parameters, 'test_volume_correlation',
                                                           self.test_volume_correlation)
        self.correlation_threshold = self.read_parameter(parameters, 'correlation_threshold',
                                                         self.correlation_threshold)
        self.correlation_threshold = self.read_parameter(parameters, 'pearson_threshold', self.correlation_threshold)
        self.correlation_threshold = self.read_parameter(parameters, 'PearsonThreshold', self.correlation_threshold)

        self.test_postponing_division = self.read_parameter(parameters, 'test_postponing_division',
                                                            self.test_postponing_division)
        self.postponing_correlation_threshold = self.read_parameter(parameters, 'postponing_correlation_threshold',
                                                                    self.postponing_correlation_threshold)
        self.postponing_minimal_length = self.read_parameter(parameters, 'postponing_minimal_length',
                                                             self.postponing_minimal_length)
        self.postponing_window_length = self.read_parameter(parameters, 'postponing_window_length',
                                                            self.postponing_window_length)

        self.mimic_historical_astec = self.read_parameter(parameters, 'mimic_historical_astec',
                                                          self.mimic_historical_astec)

        self.processors = self.read_parameter(parameters, 'processors', self.processors)

        self.lineage_diagnosis = self.read_parameter(parameters, 'lineage_diagnosis', self.lineage_diagnosis)

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
# fusion procedure
#
########################################################################################

def _map_cell_fusion(astec_image, post_image, current_time, cells_to_be_fused, time_digits_for_cell_id=4):
    #
    # cells_to_be_fused is a dictionary where
    # - keys are time values
    # - values are arrays of tuples (new label, old label)
    #
    labels_to_be_fused = cells_to_be_fused.get(current_time, '')

    #
    # no fusion to be done, just copy the file
    #
    if labels_to_be_fused == '':
        monitoring.to_log_and_console('    no cell fusion to be done for time #' + str(current_time), 2)
        astec_ext = common.get_image_extension(astec_image)
        post_ext = common.get_image_extension(post_image)
        if astec_ext == post_ext:
            shutil.copy2(astec_image, post_image)
        else:
            im = imread(astec_image)
            imsave(post_image, im)
        return

    #
    # build a mapping
    #
    im = imread(astec_image)
    mapping = list(range(np.max(im) + 1))
    for new, old in labels_to_be_fused:
        newlabel = int(new % 10 ** time_digits_for_cell_id)
        oldlabel = int(old % 10 ** time_digits_for_cell_id)
        mapping = [newlabel if i == oldlabel else i for i in mapping]
    #
    # after list comprehensions, mapping is of type list
    # needs to be casted into np.array to be applied on image
    #
    mapping = np.array(mapping, dtype=np.uint16)

    im = SpatialImage(mapping[im], voxelsize=im.voxelsize)
    imsave(post_image, im)
    return


########################################################################################
#
# some internal procedures
#
########################################################################################


def _test_early_division(direct_lineage, reverse_lineage, division_cell, cell, lifespan_minimal_value, first_time_point,
                         last_time_point, time_digits_for_cell_id):
    proc = "_test_early_division"

    #
    # get all the progeny of the mother cell of 'cell'
    #
    sisters = copy.deepcopy(direct_lineage[division_cell])
    if len(sisters) > 2:
        monitoring.to_log_and_console(str(proc) + ": cell #" + str(division_cell)
                                      + "divides in more than 2 branches, skip it")
        return False
    sisters.remove(cell)
    #
    # extract the sister branch from the sister cell if 'cell'
    # until a leaf or a bifurcation
    #
    c = sisters[0]
    sister_branch = [c]
    while len(direct_lineage.get(c, [])) == 1:
        c = direct_lineage[c][0]
        sister_branch.append(c)

    #
    # the sister branch is too short (and ends before the end of the sequence)
    #
    if len(sister_branch) < lifespan_minimal_value \
            and int(sister_branch[-1] / 10 ** time_digits_for_cell_id) != last_time_point:
        if _instrumented_:
            monitoring.to_log_and_console(str(proc) + ": true because of sister length")
        return True
    #
    # extract the mother branch in the reverse order until the first bifurcation
    #
    mother_branch = [division_cell]
    c = division_cell
    while c in reverse_lineage:
        c = reverse_lineage[c]
        #
        # TODO: test to be added
        #
        # if len(direct_lineage.get(c, [])) > 1:
        #     break
        mother_branch.append(c)

    #
    # the mother branch is too short (and begins after the beginning of the sequence)
    #
    if len(mother_branch) < lifespan_minimal_value \
            and int(mother_branch[-1] / 10 ** time_digits_for_cell_id) != first_time_point:
        if _instrumented_:
            monitoring.to_log_and_console(str(proc) + ": true because of mother length")
        return True

    return False


def _get_volumes(lineage, volume, cell):
    """
    Return the list of volumes of cell n and its progeny up to division (not included)
    lineage: lineage tree
    volume: dictionary of volumes
    cell: starting cell
    """
    v = [volume[cell]]
    while len(lineage.get(cell, '')) == 1:
        v.append(volume[cell])
        cell = lineage[cell][0]
    return v


def _test_volume_correlation(direct_lineage, volume, division_cell, cell, correlation_threshold):
    proc = "_test_volume_correlation"

    sisters = copy.deepcopy(direct_lineage[division_cell])
    if len(sisters) > 2:
        monitoring.to_log_and_console(str(proc) + ": cell #" + str(division_cell) +
                                      "divides in more than 2 branches, skip it")
        return False
    sisters.remove(cell)

    branch_volume = _get_volumes(direct_lineage, volume, cell)
    sister_volume = _get_volumes(direct_lineage, volume, sisters[0])
    min_length = min(len(branch_volume), len(sister_volume))

    #
    # branches of constant volume
    #
    if min(branch_volume[:min_length]) == max(branch_volume[:min_length]):
        return False
    if min(sister_volume[:min_length]) == max(sister_volume[:min_length]):
        return False
    pearson_correlation = pearsonr(branch_volume[:min_length], sister_volume[:min_length])

    if pearson_correlation[0] < -correlation_threshold:
        return True

    return False


########################################################################################
#
#
#
########################################################################################


def _get_leaves(direct_lineage, reverse_lineage, volume, experiment, parameters):
    """
    Return a zip of tuple (leaf, branch_length, division_cell, division_time)
    each leaf defines a simple branch (issued from the division_cell) that is candidate
    for deletion, ie
    - either it finishes before the end of the sequence
    - or the volume of the leaf cell is too small
    :param direct_lineage:
    :param reverse_lineage:
    :param volume:
    :param experiment:
    :param parameters:
    :return:
    """

    time_digits_for_cell_id = experiment.get_time_digits_for_cell_id()
    #
    # leaves are nodes without lineage
    # get the largest time point value (last leaves)
    #
    nodes = list(set(direct_lineage.keys()).union(set([v for values in list(direct_lineage.values()) for v in values])))
    leaves = set(nodes) - set(direct_lineage.keys())
    last_time = int(max(leaves) / 10 ** time_digits_for_cell_id)
    first_time = int(min(set(direct_lineage.keys())) / 10 ** time_digits_for_cell_id)

    #
    # a branch (ended) by a leave is candidate for deletion if
    # - it ends before the last time point (so before the end of the series), or
    # - the volume of the leave cell is too small
    #
    candidates_for_deletion = [leaf for leaf in leaves if ((int(leaf / 10 ** time_digits_for_cell_id) < last_time)
                                                           or (volume[leaf] < parameters.volume_minimal_value))]

    lengths = []
    division_cells = []
    division_times = []
    for leaf in candidates_for_deletion:
        #
        # get a branch from leaf 'leaf'
        # get back from the leaf until its parents has more than 1 daughter
        # (so the parent of two branches does not belong to the extracted branch)
        # reverse it
        # the branch begins then with the first point after the bifurcation
        #
        cell = leaf
        branch = [cell]
        while len(direct_lineage.get(reverse_lineage.get(cell, ''), '')) == 1:
            cell = reverse_lineage[cell]
            branch.append(cell)
        lengths.append(len(branch))
        division_cell = reverse_lineage.get(branch[-1], '')
        division_cells.append(division_cell)
        if division_cell is not '':
            division_times.append(int(division_cell / 10 ** time_digits_for_cell_id))
        else:
            branch.reverse()
            if int(branch[0] / 10 ** time_digits_for_cell_id) > experiment.first_time_point:
                if len(branch) <= 4:
                    monitoring.to_log_and_console("        ... found an orphan branch: " + str(branch), 3)
                else:
                    monitoring.to_log_and_console("        ... found an orphan branch: " + str(branch[0:2]) + "..."
                                                  + str(branch[-2:]), 3)
            division_times.append(first_time - 1)

    #
    # branches is an array of tuples (#leaf_id, branch_length, #division_id)
    #
    branches = list(zip(candidates_for_deletion, lengths, division_cells, division_times))
    return branches


def _fuse_branch(lineage, volume, surfaces, labels_to_be_fused, division_cell, branch, experiment):
    proc = "_fuse_branch"
    time_digits_for_cell_id = experiment.get_time_digits_for_cell_id()

    #
    # check division cell progeny
    #
    progeny = copy.deepcopy(lineage[division_cell])

    if len(progeny) > 2:
        monitoring.to_log_and_console(str(proc) + ": cell #" + str(division_cell) +
                                      " divides in more than 2 branches, skip it")
        return
    elif len(progeny) == 1:
        monitoring.to_log_and_console(str(proc) + ": cell #" + str(division_cell)
                                      + " has only one daughter?! this is weird")
        return

    progeny.remove(branch[0])

    #
    # follow the progeny
    #
    ibranch = 0
    iprogeny = 0
    sister_branch = []

    while True:
        cell_time = int(branch[ibranch] / 10 ** time_digits_for_cell_id)
        labels_to_be_fused.setdefault(cell_time, []).append((progeny[iprogeny], branch[ibranch]))
        sister_branch.append(progeny[iprogeny])

        #
        # update volumes
        #
        volume[progeny[iprogeny]] += volume[branch[ibranch]]
        volume.pop(branch[ibranch], None)
        #
        # update surfaces
        # 1. a. update surfaces for common neighbors
        #    b. add new neighbors
        # 2. suppress branch[ibranch] from progeny[iprogeny] neighbors
        # 3. update neighbors's contact surface
        #
        for cell in surfaces[branch[ibranch]]:
            # skip progeny[iprogeny]
            if cell == progeny[iprogeny]:
                continue
            # update contact surfaces of progeny[iprogeny]
            if cell in surfaces[progeny[iprogeny]]:
                surfaces[progeny[iprogeny]][cell] += surfaces[branch[ibranch]][cell]
            else:
                surfaces[progeny[iprogeny]][cell] = surfaces[branch[ibranch]][cell]
            # background case: it is not in surfaces
            # skip it
            if cell not in surfaces:
                continue
            # update contact surfaces of neighboring cells
            if branch[ibranch] in surfaces[cell]:
                if progeny[iprogeny] in surfaces[cell]:
                    surfaces[cell][progeny[iprogeny]] += surfaces[cell][branch[ibranch]]
                else:
                    surfaces[cell][progeny[iprogeny]] = surfaces[cell][branch[ibranch]]
                del (surfaces[cell][branch[ibranch]])

        if branch[ibranch] in surfaces[progeny[iprogeny]]:
            del (surfaces[progeny[iprogeny]][branch[ibranch]])
        surfaces.pop(branch[ibranch], None)

        #
        # remove branch[0] from the lineage of division_cell
        #
        if ibranch == 0:
            lineage[division_cell].remove(branch[ibranch])

        #
        # next cell, has the branch ended?
        # nothing more to do
        #
        if ibranch + 1 >= len(branch):
            break

        #
        # has the sister branch ended?
        # last cell of the sister branch inherits from the remaining of branch
        #
        if lineage.get(progeny[iprogeny], '') == '':
            if lineage.get(branch[ibranch], '') != '':
                lineage[progeny[iprogeny]] = lineage[branch[ibranch]]
                lineage.pop(branch[ibranch], None)
            break

        #
        # update lineage
        #
        lineage.pop(branch[ibranch], None)

        #
        # go to next cells
        #
        ibranch += 1
        progeny = lineage.get(progeny[iprogeny], '')
        iprogeny = -1
        if len(progeny) >= 2:
            for i in range(len(progeny)):
                if progeny[i] in surfaces[branch[ibranch]]:
                    if iprogeny == -1:
                        iprogeny = i
                    elif surfaces[branch[ibranch]][progeny[i]] > surfaces[branch[ibranch]][progeny[iprogeny]]:
                        iprogeny = i
            if iprogeny == -1:
                monitoring.to_log_and_console(str(proc) + ": cell #" + str(branch[ibranch]) + " has no neighbors in "
                                              + str(progeny) + " ?! this is weird")
                break
        else:
            iprogeny = 0

    #
    #
    #
    if len(branch) <= 4:
        strb = str(branch)
    else:
        strb = "[" + str(branch[0]) + ", " + str(branch[1]) + ", ..., " + str(branch[-2]) + ", " + str(branch[-1]) + "]"
    if len(sister_branch) <= 4:
        strs = str(sister_branch)
    else:
        strs = "[" + str(sister_branch[0]) + ", " + str(sister_branch[1]) + ", ..., " + str(sister_branch[-2]) + ", " \
               + str(sister_branch[-1]) + "]"

    monitoring.to_log_and_console("            ... from cell " + str(division_cell) + ": " + strb + " was fused with "
                                  + strs, 3)

    return


def _get_branch(reverse_lineage, leaf, division_cell):
    """
    Given a leaf cell and the division cell, extract the branch (as a list)
    from the first cell after the division cell to the leaf
    :param reverse_lineage:
    :param leaf:
    :param division_cell:
    :return:
    """
    #
    # get the whole branch from the leaf
    #
    branch = [leaf]
    while reverse_lineage.get(leaf, '') != division_cell:
        leaf = reverse_lineage[leaf]
        branch.append(leaf)
    #
    # reverse it, so it begins from the first point after the division cell
    #
    branch.reverse()
    return branch


########################################################################################
#
#
#
########################################################################################


def _prune_lineage_tree(lineage, volume, surfaces, experiment, parameters):
    proc = "_prune_lineage_tree"
    #
    # parameter type checking
    #

    if not isinstance(experiment, common.Experiment):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'experiment' variable: "
                                      + str(type(experiment)))
        sys.exit(1)

    if not isinstance(parameters, PostCorrectionParameters):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'parameters' variable: "
                                      + str(type(parameters)))
        sys.exit(1)

    #
    #
    #

    time_digits_for_cell_id = experiment.get_time_digits_for_cell_id()
    first_time = int(min(set(lineage.keys())) / 10 ** time_digits_for_cell_id)
    last_time = int(max(set([v for values in list(lineage.values()) for v in values])) / 10 ** time_digits_for_cell_id)

    labels_to_be_fused = {}
    undeletable_leaves = set()

    total_n_fusions_length = 0
    total_n_fusions_early_division = 0
    total_n_fusions_volume_correlation = 0

    iteration = 1
    previous_iteration_division_time = last_time
    while True:

        reverse_lineage = {v: k for k, values in lineage.items() for v in values}

        #
        # get branches that end before the last time point or
        # with a leaf cell that has a small volume
        #
        branches = _get_leaves(lineage, reverse_lineage, volume, experiment, parameters)

        #
        # branches that occurs after iteration_division_time
        # are no more candidates for deletion
        #
        for leaf, branch_length, division_cell, division_time in branches:
            if division_time > previous_iteration_division_time:
                undeletable_leaves.add(leaf)

        candidates = [(le, b, dc, dt) for le, b, dc, dt in branches if le not in undeletable_leaves]
        candidates = sorted(candidates, key=operator.itemgetter(1))
        candidates = sorted(candidates, key=operator.itemgetter(3), reverse=True)

        monitoring.to_log_and_console("       iteration #" + str(iteration) + ", found " + str(len(candidates))
                                      + " candidates among " + str(len(branches)) + " end branches", 2)

        # candidates is an array of tuples (#leaf_id, branch_length, #division_cell, division_time)
        # sorted by decreasing division cell id (and then by increasing length)
        n_fusions_length = 0
        n_fusions_early_division = 0
        n_fusions_volume_correlation = 0
        iteration_division_time = first_time
        for leaf, branch_length, division_cell, division_time in candidates:

            if leaf in undeletable_leaves:
                continue

            #
            # do nothing if
            # - mother_cell is the root (branch begins from the very beginning)
            # - mother_cell has less than 2 progeny (one branch has been fused with the other)
            #   => end branches have to be recomputed
            #
            # break (and recompute branches) when
            # - mother has no more lineage (belongs to a fused branch)
            #

            if division_time < iteration_division_time:
                break
            if division_cell == '':
                monitoring.to_log_and_console("         branch ending at " + str(leaf) + " begins at root. Skip it", 4)
                undeletable_leaves.add(leaf)
                continue
            if len(lineage[division_cell]) < 2:
                # this is probably the sister branch of a branch that has been fused with
                # monitoring.to_log_and_console("            mother cell ("+ str(division_cell) + ") of branch ending at "
                #                              + str(leaf) + " has " + str(len(lineage[division_cell]))
                #                              + " progeny. Skip it", 4)
                continue
            if len(lineage[division_cell]) > 2:
                monitoring.to_log_and_console("         cell " + str(division_cell) +
                                              " divides in more than 2 branches. Skip it", 4)
                continue

            #
            # get the whole branch from the leaf
            # useful for the two last test
            #
            # branch is deleted if
            # - it is too short,
            # - its mother branch is too short (the division occurs too early wrt the previous division)
            # - cell volumes are anti-correlated with the ones of its sister branch (up to its first division)
            #
            branch = _get_branch(reverse_lineage, leaf, division_cell)

            if parameters.test_branch_length and branch_length < parameters.lifespan_minimal_value:
                n_fusions_length += 1
            elif parameters.test_early_division and _test_early_division(lineage, reverse_lineage, division_cell,
                                                                         branch[0], parameters.lifespan_minimal_value,
                                                                         first_time, last_time,
                                                                         time_digits_for_cell_id):
                n_fusions_early_division += 1
            elif parameters.test_volume_correlation and _test_volume_correlation(lineage, volume, division_cell,
                                                                                 branch[0],
                                                                                 parameters.correlation_threshold):
                n_fusions_volume_correlation += 1
            else:
                continue

            #
            # we found a branch to be removed
            # update the iteration division time
            #
            if division_time > iteration_division_time:
                iteration_division_time = division_time

            #
            # fuse the branch
            # lineage, volume and surfaces are passed by their ids, so the modifications done
            # by _fuse_branch() are kept
            #
            _fuse_branch(lineage, volume, surfaces, labels_to_be_fused, division_cell, branch, experiment)

        #
        # end of an iteration
        # no deletion? end of the loop
        #
        total_n_fusions_length += n_fusions_length
        total_n_fusions_early_division += n_fusions_early_division
        total_n_fusions_volume_correlation += n_fusions_volume_correlation

        n_fusions = n_fusions_length + n_fusions_early_division + n_fusions_volume_correlation
        if n_fusions == 0:
            break
        monitoring.to_log_and_console("         " + str(n_fusions) + " branches have been fused at division time "
                                      + str(iteration_division_time), 1)
        previous_iteration_division_time = iteration_division_time
        iteration += 1

    monitoring.to_log_and_console("", 2)
    monitoring.to_log_and_console("       - " + str(total_n_fusions_length) + " fusions because of short length", 2)
    monitoring.to_log_and_console("       - " + str(total_n_fusions_early_division)
                                  + " fusions because of early division", 2)
    monitoring.to_log_and_console("       - " + str(total_n_fusions_volume_correlation)
                                  + " fusions because of volume anti-correlation", 2)
    monitoring.to_log_and_console("", 2)
    return lineage, volume, surfaces, labels_to_be_fused

########################################################################################
#
#
#
########################################################################################


def _get_postponing_candidate_divisions(lineage, volume, parameters):
    #
    # get all division
    #
    division_list = [c for c in lineage if len(lineage[c]) == 2]
    if len(division_list) == 0:
        return []
    #
    # get 'valid' divisions:
    # - branch length must be long enough
    # - the Pearson correlation coefficient should be below the threshold
    #
    valid_division_list = []
    for c in division_list:
        vol0 = _get_volumes(lineage, volume, lineage[c][0])
        vol1 = _get_volumes(lineage, volume, lineage[c][1])
        min_length = min(len(vol0), len(vol1))
        #
        # branch should be long enough
        #
        if min_length <= parameters.postponing_minimal_length:
            continue
        #
        # branch volumes should be anti-correlated
        #
        if pearsonr(vol0[:min_length], vol1[:min_length])[0] >= -parameters.postponing_correlation_threshold:
            continue
        valid_division_list.append(c)

    if len(valid_division_list) == 0:
        return []

    monitoring.to_log_and_console("       found " + str(len(valid_division_list)) + " candidates among " +
                                  str(len(division_list)) + " divisions", 3)

    return valid_division_list


def _get_postponing_scores(valid_division_list, lineage, volume, parameters):
    scores_window = {}
    window_length = parameters.postponing_window_length
    for c in valid_division_list:
        vol0 = _get_volumes(lineage, volume, lineage[c][0])
        vol1 = _get_volumes(lineage, volume, lineage[c][1])
        min_length = min(len(vol0), len(vol1))
        scores = []
        for i in range(0, min_length - window_length + 1):
            scores.append(pearsonr(vol0[i:i + window_length], vol1[i:i + window_length])[0])
        scores_window[c] = np.array(scores)
    return scores_window


def _postpone_division(lineage, volume, surfaces, labels_to_be_fused, experiment, parameters):
    proc = "_postpone_division"
    #
    # parameter type checking
    #

    if not isinstance(experiment, common.Experiment):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'experiment' variable: "
                                      + str(type(experiment)))
        sys.exit(1)

    if not isinstance(parameters, PostCorrectionParameters):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'parameters' variable: "
                                      + str(type(parameters)))
        sys.exit(1)

    #
    # get candidate divisions
    #
    valid_division_list = _get_postponing_candidate_divisions(lineage, volume, parameters)
    if len(valid_division_list) == 0:
        return lineage, volume, surfaces, labels_to_be_fused

    #
    # compute scores for a sliding window
    #
    scores_window = _get_postponing_scores(valid_division_list, lineage, volume, parameters)

    #
    # analyze scores
    #
    time_digits_for_cell_id = experiment.get_time_digits_for_cell_id()
    postponed_divisions = 0

    for c, scores in scores_window.items():
        if (np.array(scores) < -parameters.postponing_correlation_threshold).all():
            first_size = len(scores) - 1
        else:
            #
            # historical astec
            #
            if parameters.mimic_historical_astec:
                #
                # out = score[i+1] - score[i]
                # length(out) = length(scores) - 1
                #
                out = scores[1:] - scores[:-1]
                #
                # sizes are indices of out values sorted in decreasing order
                #
                sizes = np.argsort(out)[::-1]
                #
                # sizes are indices of out values where score[i] < - threshold,
                # out values being sorted in decreasing order
                #
                sizes = np.array([s for s in sizes if scores[s] < -parameters.postponing_correlation_threshold])
                #
                # only indices from the first half of the common part are kept
                # and we chose the first one
                #
                # first_size is then an indice i, in the first half of the common part of the branches,
                # where score[i] < - threshold and where (score[i+1] - score[i]) is maximal
                #
                first_size = sizes[sizes < len(scores) / 2]
                if len(first_size) == 0:
                    continue
                first_size = first_size[0]
            #
            # new scheme
            #
            else:
                first_size = -1
                for s in range(len(scores)):
                    if scores[s] < -parameters.postponing_correlation_threshold:
                        first_size = s
                    else:
                        break
                if first_size == -1:
                    continue

        postponed_divisions += 1
        #
        # cell fusion
        # this is similar to what is done in _fuse_branch()
        # some factorization may be desirable
        #
        c0 = lineage[c][0]
        c1 = lineage[c][1]
        sister0_branch = []
        sister1_branch = []
        for i in range(first_size + 1):
            #
            # fuse c0 and c1 -> c0 disappears
            #
            cell_time = int(c0 / 10 ** time_digits_for_cell_id)
            labels_to_be_fused.setdefault(cell_time, []).append((c1, c0))
            sister0_branch.append(c0)
            sister1_branch.append(c1)

            #
            # update volumes
            #
            volume[c1] += volume[c0]
            volume.pop(c0, None)

            #
            # update surfaces
            # 1. a. update surfaces for common neighbors
            #    b. add new neighbors
            # 2. suppress c1 from c2 neighbors
            # 3. update neighbors's contact surface
            #
            for cell in surfaces[c0]:
                if cell in surfaces[c1]:
                    surfaces[c1][cell] += surfaces[c0][cell]
                else:
                    surfaces[c1][cell] = surfaces[c0][cell]
            if c0 in surfaces[c1]:
                del (surfaces[c1][c0])
            for cell in surfaces[c1]:
                if cell in surfaces:
                    surfaces[cell][c1] = surfaces[c1][cell]
                    if c0 in surfaces[cell]:
                        del (surfaces[cell][c0])
            surfaces.pop(c0, None)

            next_c0 = lineage[c0][0]
            next_c1 = lineage[c1][0]
            #
            # update lineage
            # - first fusion: remove c0 from the lineage of division_cell
            # - last fusion: append c0 lineage to the one of c1
            # - remove c0 from lineage
            #
            if i == 0:
                lineage[c].remove(c0)
            if i == first_size:
                lineage[c1].append(lineage[c0][0])
            lineage.pop(c0, None)
            #
            #
            #
            c0 = next_c0
            c1 = next_c1

        if len(sister0_branch) <= 4:
            str0 = str(sister0_branch)
        else:
            str0 = "[" + str(sister0_branch[0]) + ", " + str(sister0_branch[1]) + ", ..., " + str(sister0_branch[-2]) \
                   + ", " + str(sister0_branch[-1]) + "]"
        if len(sister1_branch) <= 4:
            str1 = str(sister1_branch)
        else:
            str1 = "[" + str(sister1_branch[0]) + ", " + str(sister1_branch[1]) + ", ..., " + str(sister1_branch[-2]) \
                   + ", " + str(sister1_branch[-1]) + "]"

        monitoring.to_log_and_console("            ... from cell " + str(c) + ": " + str0 + " was fused with " + str1,
                                      3)

    monitoring.to_log_and_console("       - " + str(postponed_divisions) + " divisions were postponed", 2)

    return lineage, volume, surfaces, labels_to_be_fused


########################################################################################
#
#
#
########################################################################################


def contact_surface_computation(experiment, parameters):
    """

    :param experiment:
    :param parameters:
    :return:
    """

    proc = 'contact_surface_computation'

    #
    # get directory name where to find co-registered images of the sequence
    # as well as the common image suffix
    #
    astec_dir = experiment.astec_dir.get_directory()

    #
    # is there a post-segmentation directory in the intra-registration directory ?
    #
    if not os.path.isdir(astec_dir):
        monitoring.to_log(proc + ": '" + str(astec_dir) + "' does not exist")
        return None

    monitoring.to_log_and_console("       will compute contact surface properties from '" + str(astec_dir) + "'", 0)

    #
    # build name format for (post-corrected) segmentation images
    #
    name_format = experiment.astec_dir.get_file_prefix() + experiment.astec_dir.get_file_suffix() + \
                  experiment.astec_dir.get_time_prefix() + experiment.get_time_format()

    suffix = common.get_file_suffix(experiment, astec_dir, name_format, flag_time=experiment.get_time_format())
    if suffix is None:
        monitoring.to_log_and_console(proc + ": no consistent naming was found in '"
                                      + str(astec_dir) + "'", 1)
        monitoring.to_log_and_console("Exiting.", 0)
        sys.exit(1)
    name_format += "." + str(suffix)
    template_format = os.path.join(astec_dir, name_format)

    #
    #
    #
    output_name = experiment.astec_dir.get_file_prefix() + experiment.astec_dir.get_file_suffix() + "_surfaces"
    output_name = os.path.join(astec_dir, output_name)

    if os.path.isfile(output_name + ".xml"):
        if not monitoring.forceResultsToBeBuilt:
            monitoring.to_log_and_console('       xml file already existing', 2)
            return output_name + ".xml"
        else:
            monitoring.to_log_and_console('       xml file already existing, but forced', 2)

    monitoring.to_log_and_console("       ... it could be long", 0)

    first_time_point = experiment.first_time_point + experiment.delay_time_point
    last_time_point = experiment.last_time_point + experiment.delay_time_point

    options = "-feature contact -surface-estimation 6-neighbors"
    cpp_wrapping.cell_properties(template_format, output_name + ".xml", first_time_point, last_time_point,
                                 diagnosis_file=output_name + ".txt", n_processors=parameters.processors,
                                 other_options=options, monitoring=monitoring)

    return output_name + ".xml"


def _check_volume_image(volume_from_lineage, experiment):
    proc = "_check_volume_image"
    time_digits_for_cell_id = experiment.get_time_digits_for_cell_id()
    first_time = experiment.first_time_point + experiment.delay_time_point
    last_time = experiment.last_time_point + experiment.delay_time_point
    monitoring.to_log_and_console("  === volume/image cross-checking === ", 1)
    for current_time in range(first_time, last_time + 1, experiment.delta_time_point):

        acquisition_time = experiment.get_time_index(current_time)
        monitoring.to_log_and_console("  * test time point " + str(acquisition_time), 1)

        #
        # nothing to do if the post-segmentation image exists
        #
        post_dir = experiment.post_dir.get_directory()
        post_name = experiment.post_dir.get_image_name(current_time)
        post_image = common.find_file(post_dir, post_name, file_type='image', callfrom=proc, local_monitoring=None,
                                      verbose=False)
        if post_image is None:
            monitoring.to_log_and_console('    post-segmentation image not found', 1)
            continue
        post_image = os.path.join(post_dir, post_image)

        properties.check_volume_image(volume_from_lineage, post_image, current_time,
                                      time_digits_for_cell_id=time_digits_for_cell_id)
    return


def postcorrection_process(experiment, parameters):
    proc = "postcorrection_process"
    #
    # parameter type checking
    #

    if not isinstance(experiment, common.Experiment):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'experiment' variable: "
                                      + str(type(experiment)))
        sys.exit(1)

    if not isinstance(parameters, PostCorrectionParameters):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'parameters' variable: "
                                      + str(type(parameters)))
        sys.exit(1)

    #
    # read the lineage file, if any
    #
    segmentation_dir = experiment.astec_dir.get_directory()
    lineage_tree_file = common.find_file(segmentation_dir, experiment.astec_dir.get_file_name("_lineage"),
                                         file_type='lineage', callfrom=proc, verbose=False)

    if lineage_tree_file is None:
        monitoring.to_log_and_console(str(proc) + ": unable to find lineage file in " + str(segmentation_dir))
        sys.exit(1)
    elif os.path.isfile(os.path.join(segmentation_dir, lineage_tree_file)):
        lineage_tree_path = os.path.join(segmentation_dir, lineage_tree_file)
        lineage_tree_information = properties.read_dictionary(lineage_tree_path)
    else:
        monitoring.to_log_and_console(str(proc) + ": " + str(lineage_tree_file) + " is not a file?")
        sys.exit(1)

    time_digits_for_cell_id = experiment.get_time_digits_for_cell_id()

    #
    # test lineage
    #
    if parameters.lineage_diagnosis:
        monitoring.to_log_and_console("   ... test lineage (before cell merging)", 1)
        properties.check_volume_lineage(lineage_tree_information, time_digits_for_cell_id=time_digits_for_cell_id)

    #
    # get lineage and volume dictionaries
    #
    dict_lineage = properties.get_dictionary_entry(lineage_tree_information, 'lineage')
    if dict_lineage == {}:
        monitoring.to_log_and_console(str(proc) + ": empty lineage information in " + str(lineage_tree_file))
        sys.exit(1)

    dict_volume = properties.get_dictionary_entry(lineage_tree_information, 'volume')
    if dict_volume == {}:
        monitoring.to_log_and_console(str(proc) + ": empty volume information in " + str(lineage_tree_file))
        sys.exit(1)

    #
    #
    #
    monitoring.to_log_and_console("   ... compute contact surfaces", 1)
    surface_file = contact_surface_computation(experiment, parameters)
    surface_information = properties.read_dictionary(surface_file)
    dict_surface = properties.get_dictionary_entry(surface_information, 'contact')
    if dict_surface == {}:
        monitoring.to_log_and_console(str(proc) + ": empty surface information in " + str(surface_file))
        sys.exit(1)

    #
    #
    #
    monitoring.to_log_and_console("   ... lineage pruning", 1)
    lineage, volume, surfaces, cells_to_be_fused = _prune_lineage_tree(dict_lineage, dict_volume, dict_surface,
                                                                       experiment, parameters)

    #
    #
    #
    if parameters.test_postponing_division:
        monitoring.to_log_and_console("   ... division postponing", 1)
        lineage, volume, surfaces, cells_to_be_fused = _postpone_division(lineage, volume, surfaces, cells_to_be_fused,
                                                                          experiment, parameters)

    #
    # save lineage tree
    #
    new_lineage_tree_information = {properties.keydictionary['lineage']['output_key']: lineage,
                        properties.keydictionary['volume']['output_key']: volume}
    segmentation_dir = experiment.post_dir.get_directory()
    lineage_tree_path = os.path.join(segmentation_dir, experiment.post_dir.get_file_name("_lineage") + "."
                                     + experiment.result_lineage_suffix)
    properties.write_dictionary(lineage_tree_path, new_lineage_tree_information)

    #
    # test lineage
    #
    if parameters.lineage_diagnosis:
        monitoring.to_log_and_console("   ... test lineage (after cell merging)", 1)
        properties.check_volume_lineage(new_lineage_tree_information, time_digits_for_cell_id=time_digits_for_cell_id)

    #
    # apply cell fusion
    #
    first_time = experiment.first_time_point + experiment.delay_time_point
    last_time = experiment.last_time_point + experiment.delay_time_point
    for current_time in range(first_time, last_time + 1, experiment.delta_time_point):

        acquisition_time = experiment.get_time_index(current_time)
        monitoring.to_log_and_console("   ... cell fusion of time #" + acquisition_time, 1)

        #
        # nothing to do if the post-segmentation image exists
        #
        post_dir = experiment.post_dir.get_directory()
        post_name = experiment.post_dir.get_image_name(current_time)
        post_image = common.find_file(post_dir, post_name, file_type='image', callfrom=proc, local_monitoring=None,
                                      verbose=False)

        if post_image is not None:
            if monitoring.forceResultsToBeBuilt is False:
                monitoring.to_log_and_console('       post-segmentation image already existing', 2)
                continue
            else:
                monitoring.to_log_and_console('       post-segmentation image already existing, but forced', 2)
        post_image = os.path.join(post_dir, post_name + "." + experiment.result_image_suffix)

        astec_dir = experiment.astec_dir.get_directory()
        astec_name = experiment.astec_dir.get_image_name(current_time)
        astec_image = common.find_file(astec_dir, astec_name, file_type='image', callfrom=proc, local_monitoring=None,
                                       verbose=False)
        if astec_image is None:
            monitoring.to_log_and_console("    .. " + proc + ": no segmentation image was found for time "
                                          + str(current_time), 2)
            return
        astec_image = os.path.join(astec_dir, astec_image)
        #
        #
        #

        # print(str(cells_to_be_fused))
        # print("")
        # print("")
        _map_cell_fusion(astec_image, post_image, current_time, cells_to_be_fused,
                         time_digits_for_cell_id=time_digits_for_cell_id)

    #
    # this test is pretty long, it aims at detecting discrepancies between the volume information
    # of the lineage and the images after label fusion
    #
    if parameters.lineage_diagnosis and False:
        monitoring.to_log_and_console("   ... test lineage (after cell merging in image)", 1)
        dict_volume = properties.get_dictionary_entry(new_lineage_tree_information, 'volume')
        _check_volume_image(dict_volume, experiment)

    return
