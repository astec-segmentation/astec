
import os
import imp
import sys
import time
import multiprocessing
import numpy as np
from scipy import ndimage as nd
import copy
import cPickle as pkl

import ace
import mars
import common
import reconstruction
import EMBRYOPROPERTIES as properties
import CommunFunctions.cpp_wrapping as cpp_wrapping

from CommunFunctions.ImageHandling import imread, imsave, SpatialImage

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


#
#
#
#
#

class AstecParameters(mars.WatershedParameters, reconstruction.ReconstructionParameters):

    ############################################################
    #
    # initialisation
    #
    ############################################################

    def __init__(self):

        ############################################################
        #
        # initialisation
        #
        ############################################################
        mars.WatershedParameters.__init__(self)
        reconstruction.ReconstructionParameters.__init__(self)
        self.intensity_transformation = 'Normalization_to_u8'

        #
        #
        #
        self.propagation_strategy = None

        #
        # erosion of cell from previous segmentation
        #
        # previous_seg_erosion_cell_iterations: maximum number of erosion iteration for cells
        #   if the cell disappears, less iterations are done
        # previous_seg_erosion_cell_min_size: minimal size of a cell to perform erosion
        #

        self.previous_seg_erosion_cell_iterations = 10
        self.previous_seg_erosion_background_iterations = 25
        self.previous_seg_erosion_cell_min_size = 1000

        #
        # astec-dedicated watershed parameters
        #
        self.watershed_seed_hmin_min_value = 4
        self.watershed_seed_hmin_max_value = 18
        self.watershed_seed_hmin_delta_value = 2

        #
        # to decide whether there will be division
        #
        self.seed_selection_tau = 25

        #
        # threshold
        # cells deformed from previous timepoint that does not have any seed
        # and whose volume (in voxels) is below this threshold are discarded
        # they will correspond to dead-end in the lineage
        #
        self.minimum_volume_unseeded_cell = 100

        #
        # magic values for the volume checking
        # - volume_minimal_value is in voxel units
        #
        self.volume_ratio_tolerance = 0.1
        self.volume_ratio_threshold = 0.5
        self.volume_minimal_value = 1000

    ############################################################
    #
    # print / write
    #
    ############################################################

    def print_parameters(self):
        print("")
        print('AstecParameters')

        print('- propagation_strategy = ' + str(self.propagation_strategy))

        print('- previous_seg_erosion_cell_iterations = ' + str(self.previous_seg_erosion_cell_iterations))
        print('- previous_seg_erosion_background_iterations = ' + str(self.previous_seg_erosion_background_iterations))
        print('- previous_seg_erosion_cell_min_size = ' + str(self.previous_seg_erosion_cell_min_size))

        print('- watershed_seed_hmin_min_value = ' + str(self.watershed_seed_hmin_min_value))
        print('- watershed_seed_hmin_max_value = ' + str(self.watershed_seed_hmin_max_value))
        print('- watershed_seed_hmin_delta_value = ' + str(self.watershed_seed_hmin_delta_value))

        print('- seed_selection_tau = ' + str(self.seed_selection_tau))

        print('- minimum_volume_unseeded_cell = ' + str(self.minimum_volume_unseeded_cell))

        print('- volume_ratio_tolerance = ' + str(self.volume_ratio_tolerance))
        print('- volume_ratio_threshold = ' + str(self.volume_ratio_threshold))
        print('- volume_minimal_value = ' + str(self.volume_minimal_value))

        mars.WatershedParameters.print_parameters(self)
        reconstruction.ReconstructionParameters.print_parameters(self)
        print("")

    def write_parameters(self, log_file_name):
        with open(log_file_name, 'a') as logfile:
            logfile.write("\n")
            logfile.write('AstecParameters\n')

            logfile.write('- propagation_strategy = ' + str(self.propagation_strategy) + '\n')

            logfile.write('- previous_seg_erosion_cell_iterations = ' + str(self.previous_seg_erosion_cell_iterations)
                          + '\n')
            logfile.write('- previous_seg_erosion_background_iterations = '
                          + str(self.previous_seg_erosion_background_iterations) + '\n')
            logfile.write('- previous_seg_erosion_cell_min_size = ' + str(self.previous_seg_erosion_cell_min_size)
                          + '\n')

            logfile.write('- watershed_seed_hmin_min_value = ' + str(self.watershed_seed_hmin_min_value) + '\n')
            logfile.write('- watershed_seed_hmin_max_value = ' + str(self.watershed_seed_hmin_max_value) + '\n')
            logfile.write('- watershed_seed_hmin_delta_value = ' + str(self.watershed_seed_hmin_delta_value) + '\n')

            logfile.write('- seed_selection_tau = ' + str(self.seed_selection_tau) + '\n')

            logfile.write('- minimum_volume_unseeded_cell = ' + str(self.minimum_volume_unseeded_cell) + '\n')

            logfile.write('- volume_ratio_tolerance = ' + str(self.volume_ratio_tolerance) + '\n')
            logfile.write('- volume_ratio_threshold = ' + str(self.volume_ratio_threshold) + '\n')
            logfile.write('- volume_minimal_value = ' + str(self.volume_minimal_value) + '\n')

            mars.WatershedParameters.write_parameters(self, log_file_name)
            reconstruction.ReconstructionParameters.write_parameters(self, log_file_name)
            print("")
        return

    ############################################################
    #
    # update
    #
    ############################################################

    def update_from_parameters(self, parameter_file):
        if parameter_file is None:
            return
        if not os.path.isfile(parameter_file):
            print("Error: '" + parameter_file + "' is not a valid file. Exiting.")
            sys.exit(1)

        parameters = imp.load_source('*', parameter_file)

        if hasattr(parameters, 'propagation_strategy'):
            if parameters.propagation_strategy is not None:
                self.propagation_strategy = parameters.propagation_strategy
        if hasattr(parameters, 'astec_propagation_strategy'):
            if parameters.astec_propagation_strategy is not None:
                self.propagation_strategy = parameters.astec_propagation_strategy

        #
        #
        #

        if hasattr(parameters, 'previous_seg_erosion_cell_iterations'):
            if parameters.previous_seg_erosion_cell_iterations is not None:
                self.previous_seg_erosion_cell_iterations = parameters.previous_seg_erosion_cell_iterations
        if hasattr(parameters, 'previous_seg_erosion_background_iterations'):
            if parameters.previous_seg_erosion_background_iterations is not None:
                self.previous_seg_erosion_background_iterations = parameters.previous_seg_erosion_background_iterations
        if hasattr(parameters, 'previous_seg_erosion_cell_min_size'):
            if parameters.previous_seg_erosion_cell_min_size is not None:
                self.previous_seg_erosion_cell_min_size = parameters.previous_seg_erosion_cell_min_size

        #
        # watershed
        #

        if hasattr(parameters, 'watershed_seed_hmin_min_value'):
            if parameters.watershed_seed_hmin_min_value is not None:
                self.watershed_seed_hmin_min_value = parameters.watershed_seed_hmin_min_value
        if hasattr(parameters, 'astec_h_min_min'):
            if parameters.astec_h_min_min is not None:
                self.watershed_seed_hmin_min_value = parameters.astec_h_min_min

        if hasattr(parameters, 'watershed_seed_hmin_max_value'):
            if parameters.watershed_seed_hmin_max_value is not None:
                self.watershed_seed_hmin_max_value = parameters.watershed_seed_hmin_max_value
        if hasattr(parameters, 'astec_h_min_max'):
            if parameters.astec_h_min_max is not None:
                self.watershed_seed_hmin_max_value = parameters.astec_h_min_max

        if hasattr(parameters, 'watershed_seed_hmin_delta_value'):
            if parameters.watershed_seed_hmin_delta_value is not None:
                self.watershed_seed_hmin_delta_value = parameters.watershed_seed_hmin_delta_value

        mars.WatershedParameters.update_from_parameters(self, parameter_file)
        reconstruction.ReconstructionParameters.update_from_parameters(self, parameter_file)


########################################################################################
#
# some internal procedures
#
########################################################################################

#
# create seeds from previous segmentation
# cells are eroded either with a maximum number of iterations (10 for 'true' cells,
# 25 for the background) or with less iterations if the object to be eroded
# disappears
# Note (GM 15/07/2018): it should be more efficient to use distance maps,
# and to threshold them
#

def _erode_cell(parameters):
    """

    :param parameters:
    :return:
    """
    #
    # Erodes the label i in the label image
    # tmp : binary SpatialImage
    # max_size_cell : size max allow for a cell (here put at np.inf)
    # size_cell : size of the cell to erode
    # iterations : maximum number of iterations for normal cells
    # out_iterations : maximum number of iterations for exterior
    # bb : bounding box if tmp in the global image (necessary when computing in parallel)
    # i : label of the cell to erode
    #

    proc = '_erode_cell'

    tmp, iterations, bb, i = parameters

    nb_iter = iterations

    eroded = nd.binary_erosion(tmp, iterations=nb_iter)
    while len(nd.find_objects(eroded)) != 1 and nb_iter >= 0:
        nb_iter -= 1
        eroded = nd.binary_erosion(tmp, iterations=nb_iter)

    return eroded, i, bb


def _build_seeds_from_previous_segmentation(label_image, output_image, parameters, nprocessors=26):
    """
    Erodes all the labels in the segmented image seg
    :param label_image: image whose cells are to be eroded
    :param output_image:
    :param parameters:
    :param nprocessors: number maximum of processors allowed to be used
    :return:
    """

    proc = '_build_seeds_from_previous_segmentation'

    #
    # parameter type checking
    #

    if not isinstance(parameters, AstecParameters):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'parameters' variable: "
                                      + str(type(parameters)))
        sys.exit(1)

    #
    #
    #

    seg = imread(label_image)

    bboxes = nd.find_objects(seg)
    a = np.unique(seg)

    pool = multiprocessing.Pool(processes=nprocessors)
    mapping = []

    for i in a:
        tmp = seg[bboxes[i - 1]] == i
        size_cell = np.sum(tmp)
        if size_cell > parameters.previous_seg_erosion_cell_min_size:
            if i is 1:
                mapping.append((tmp, parameters.previous_seg_erosion_background_iterations, bboxes[i - 1], i))
            else:
                mapping.append((tmp, parameters.previous_seg_erosion_cell_iterations, bboxes[i - 1], i))
        else:
            monitoring.to_log_and_console('     .. skip cell ' + str(i) + ', size (' + str(size_cell) + ') <= '
                                          + str(parameters.previous_seg_erosion_cell_min_size), 2)

    outputs = pool.map(_erode_cell, mapping)
    pool.close()
    pool.terminate()

    seeds = np.zeros_like(seg)
    for eroded, i, bb in outputs:
        seeds[bb][eroded] = i

    seeds._set_resolution(seg._get_resolution())
    imsave(output_image, seeds)

    return


def _build_lineage_for_seeds_from_previous_segmentation(volumes, previous_time, current_time, time_digits=4):
    proc = "_build_lineage_for_seeds_from_previous_segmentation"
    tmp = {}
    # if the volume exists, it means that this cell has a mother cell
    for key, value in volumes.iteritems():
        newkey = previous_time * 10**time_digits + int(key)
        tmp[newkey] = current_time * 10**time_digits + int(key)
    return tmp


########################################################################################
#  newvalue = [current_time * 10**time_digits + i for i in value]
#
#
########################################################################################

#
# compute the seeds for a range of 'h' values
#

def _extract_seeds_in_cell(parameters):
    """
    Return the seeds in seeds_sub_image stricly included in cell c in cell_segmentation
    """
    #
    # cell_segmentation is a sub-image (extracted from the propagated segmentation at t-1) with 'c' for the cell
    # and 1 for the background
    # seeds_sub_image is a sub-image of the extracted seeds
    # c is the cell label
    #
    cell_segmentation, seeds_sub_image, c = parameters

    #
    # check whether the cell_segmentation has only two labels
    #
    if len(np.unique(cell_segmentation)) != 2:
        monitoring.to_log_and_console('    sub-image of cell ' + str(c) + ' contains '
                                      + str(len(np.unique(cell_segmentation))) + ' labels', 2)
        return

    #
    # get the seeds that intersect the cell 'c'
    #
    labels = list(np.unique(seeds_sub_image[cell_segmentation == c]))

    #
    # remove 0 label (correspond to non-minima regions)
    # Note: check whether 0 is inside labels list (?)
    #
    labels.remove(0)

    nb = len(labels)

    return nb, labels, c


def _cell_based_h_minima(first_segmentation, cells, bounding_boxes, membrane_image, experiment, parameters,
                         nprocessors=26):
    """
    Computes the seeds (h-minima) for a range of h values
    Seeds are labeled, and only seeds entirely contained in one single cell are kept
    (seeds that superimposed two cells, or more, are rejected).

    :param first_segmentation: watershed based segmentation where the seeds are the cells from the previous,
        eroded and then deformed
    :param cells:
    :param bounding_boxes:
    :param membrane_image:
    :param experiment:
    :param parameters:
    :param nprocessors:
    :return:
    """

    proc = '_cell_based_h_minima'

    #
    # parameter type checking
    #

    if not isinstance(experiment, common.Experiment):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'experiment' variable: "
                                      + str(type(experiment)))
        sys.exit(1)

    if not isinstance(parameters, AstecParameters):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'parameters' variable: "
                                      + str(type(parameters)))
        sys.exit(1)

    #
    # h-minima extraction with h = max value
    # the difference image is kept for further computation
    #
    h_max = parameters.watershed_seed_hmin_max_value
    wparam = mars.WatershedParameters(parameters)
    wparam.watershed_seed_hmin = h_max
    h_min = h_max

    input_image = membrane_image
    seed_image = common.add_suffix(membrane_image, "_seed_h" + str('{:03d}'.format(h_min)),
                                   new_dirname=experiment.astec_dir.get_tmp_directory(),
                                   new_extension=experiment.default_image_suffix)
    difference_image = common.add_suffix(membrane_image, "_seed_diff_h" + str('{:03d}'.format(h_min)),
                                         new_dirname=experiment.astec_dir.get_tmp_directory(),
                                         new_extension=experiment.default_image_suffix)

    if not os.path.isfile(seed_image) or not os.path.isfile(difference_image) \
            or monitoring.forceResultsToBeBuilt is True:
        #
        # computation of labeled regional minima
        # -> keeping the 'difference' image allows to speed up the further computation
        #    for smaller values of h
        #
        mars.build_seeds(input_image, difference_image, seed_image, experiment, wparam)
        #
        # select only the 'seeds' that are totally included in cells
        #
        cpp_wrapping.mc_mask_seeds(seed_image, first_segmentation, seed_image)

    #
    # collect the number of seeds found for each cell
    #
    #

    n_seeds = {}
    parameter_seeds = {}

    checking = True

    while checking:

        #
        # for each cell,
        # 2. build a sub-image (corresponding to the bounding box) from the propagated segmentation from t-1
        #    with the cell labeled at 'c' and the rest at '1'
        # 3. build a sub-image from the seeds extracted at h
        #

        im_segmentation = imread(first_segmentation)
        im_seed = imread(seed_image)

        mapping = []

        for c in cells:
            cell_segmentation = np.ones_like(im_segmentation[bounding_boxes[c]])
            cell_segmentation[im_segmentation[bounding_boxes[c]] == c] = c
            mapping.append((cell_segmentation, im_seed[bounding_boxes[c]], c))

        del im_seed
        del im_segmentation

        pool = multiprocessing.Pool(processes=nprocessors)
        outputs = pool.map(_extract_seeds_in_cell, mapping)
        pool.close()
        pool.terminate()

        #
        # outputs are
        # - nb: the number of labels/seeds that are totally inside cell 'c'
        # - labels: the list of these labels
        # - c: the id of the cell
        #
        returned_n_seeds = []
        for nb, labels, c in outputs:
            returned_n_seeds.append(nb)
            n_seeds.setdefault(c, []).append(nb)
            parameter_seeds.setdefault(c, []).append([h_min, parameters.watershed_seed_sigma])

        #
        # next h value
        # since we compute the maxima from the previous difference image
        # there is no need for smoothing -> sigma = 0.0
        #
        h_min -= parameters.watershed_seed_hmin_delta_value

        #
        # still compute while
        # - h has not reach the minimum value
        # and
        # - there is at least one cell with a number of seeds in [1, 2]
        # it stops then if all cells have more than 2 seeds
        #
        # Note: I did not catch the utility of 'or returned_n_seeds == []'
        #
        checking = (h_min >= parameters.watershed_seed_hmin_min_value) and \
                   (((np.array(returned_n_seeds) <= 2) & (np.array(returned_n_seeds) != 0)).any()
                    or returned_n_seeds == [])

        if checking:

            #
            # compute seeds fot this new value of h
            # seeds are computed on the previous 'difference' image
            # - they are now local maxima
            # - smoothing has already been done (to get the first difference image)
            #   and is no more required -> sigma = 0.0
            #
            wparam.watershed_seed_hmin = h_min
            wparam.watershed_seed_sigma = 0.0

            input_image = difference_image
            seed_image = common.add_suffix(membrane_image, "_seed_h" + str('{:03d}'.format(h_min)),
                                           new_dirname=experiment.astec_dir.get_tmp_directory(),
                                           new_extension=experiment.default_image_suffix)
            difference_image = common.add_suffix(membrane_image, "_seed_diff_h" + str('{:03d}'.format(h_min)),
                                                 new_dirname=experiment.astec_dir.get_tmp_directory(),
                                                 new_extension=experiment.default_image_suffix)

            if not os.path.isfile(seed_image) or not os.path.isfile(difference_image) \
                    or monitoring.forceResultsToBeBuilt is True:
                mars.build_seeds(input_image, difference_image, seed_image, experiment, wparam, operation_type='max')
                cpp_wrapping.mc_mask_seeds(seed_image, first_segmentation, seed_image)

            if not os.path.isfile(seed_image) or not os.path.isfile(difference_image):
                monitoring.to_log_and_console("       " + proc + ": computation failed at h = " + str(h_min), 2)
                monitoring.to_log_and_console("\t Exiting.")
                sys.exit(1)

    return n_seeds, parameter_seeds


########################################################################################
#
#
#
########################################################################################

#
#
#

def _select_seed_parameters(n_seeds, parameter_seeds, tau=25):
    """
    Return the correct h-minima value for each cell
    :param n_seeds: { cell: [#seeds, ] }: dict, key: cell, values: list of #seeds
    :param parameter_seeds: { cell: [[h_min, sigma], ]}: dict matching nb_cells, key: cell, values: list of parameters
    :param tau: magic threshold (see page 72 of L. Guignard PhD thesis)
    :return:
    """

    selected_parameter_seeds = {}
    unseeded_cells = []

    #
    # the selection whether a cell should divide or not is based on the length
    # of the plateau of h values that yield a division (see section 2.3.3.5, pages 70-72
    # of L. Guignard PhD thesis)
    # nb_2 is $N_2(c)$, nb_3 is $N_{2^{+}}(c)$
    #
    # it can also divided into 2 if there is no h value that gives one seed
    #
    # np.sum(np.array(s) == 2) is equal to s.count(2)
    # 

    for c, s in n_seeds.iteritems():
        nb_2 = np.sum(np.array(s) == 2)
        nb_3 = np.sum(np.array(s) >= 2)
        score = nb_2*nb_3
        if (s.count(1) or s.count(2)) != 0:
            if score >= tau:
                #
                # obviously s.count(2) != 0
                # the largest h that gives 2 seeds is kept
                #
                h, sigma = parameter_seeds[c][np.where(np.array(s) == 2)[0][0]]
                nb_final = 2
            elif s.count(1) != 0:
                #
                # score < tau and s.count(1) != 0
                # the largest h that gives 1 seeds is kept
                #
                h, sigma = parameter_seeds[c][np.where(np.array(s) == 1)[0][0]]
                nb_final = 1
            else:
                #
                # score < tau and s.count(1) == 0 then obviously s.count(2)) != 0
                # the largest h that gives 1 seeds is kept
                #
                h, sigma = parameter_seeds[c][np.where(np.array(s) == 2)[0][0]]
                nb_final = 2
            selected_parameter_seeds[c] = [h, sigma, nb_final]
        #
        # s.count(1) == 0 and  s.count(2) == 0
        #
        elif s.count(3) != 0:
            h, sigma = parameter_seeds[c][s.index(3)]
            selected_parameter_seeds[c] = [h, sigma, 3]
        else:
            unseeded_cells.append(c)
            selected_parameter_seeds[c] = [0, 0, 0]
    return selected_parameter_seeds, unseeded_cells


########################################################################################
#
#
#
########################################################################################

#
# this one is similar to _extract_seeds_in_cell()
#

def _extract_seeds(c, cell_segmentation, cell_seeds=None, bb=None, individual_seeds=True, accept_3_seeds=False):
    """
    Return the seeds from cell_seeds stricly included in cell c from cell_segmentation
    (the labels of the seeds go from 1 to 3)
    :param c: cell label
    :param cell_segmentation: sub-image with 'c' for the cell and '1' for the background
    :param cell_seeds: (sub-)image of labeled h-minima
    :param bb: dilated bounding box of the cell
    :param individual_seeds: if False, all seeds are given the same label (1)
    :param accept_3_seeds: if True, 3 seeds can be accepted as a possible choice
    :return:
    """

    proc = "_extract_seeds"

    #
    # sub-image containing the seeds
    #
    if type(cell_seeds) != SpatialImage:
        seeds_image = imread(cell_seeds)
        if bb is not None:
            seeds = seeds_image[bb]
        else:
            seeds = copy.deepcopy(seeds_image)
        del seeds_image
    else:
        if bb is not None:
            seeds = cell_seeds[bb]
        else:
            seeds = copy.deepcopy(cell_seeds)

    #
    # seeds that intersects the cell
    # regional minima/maxima have already been selected so that they are entirely included in cells
    # of
    #
    labels = list(np.unique(seeds[cell_segmentation == c]))
    labels.remove(0)

    #
    # many seeds, but all with the same label
    # useful for background seeds
    #
    if not individual_seeds:
        seeds[seeds>0] = c
        return 1, seeds.astype(np.uint8)

    #
    # returns
    #
    if len(labels) == 1:
        return 1, (seeds == labels[0]).astype(np.uint8)
    elif len(labels) == 2:
        return 2, ((seeds == labels[0]) + 2 * (seeds == labels[1])).astype(np.uint8)
    elif len(labels) == 3 and not accept_3_seeds:
        #
        # weird, return 3 seeds but label two of them
        #
        monitoring.to_log_and_console("       " + proc + ": weird case, there are 3 seeds but only two are labeled", 2)
        return 3, ((seeds == labels[0]) + 2 * (seeds == labels[1])).astype(np.uint8)
    elif len(labels) == 3 and accept_3_seeds:
        return 3, ((seeds == labels[0]) + 2 * (seeds == labels[1]) + 3 * (seeds == labels[2])).astype(np.uint8)
    else:
        return 0, None


#
#
#


def _build_seeds_from_selected_parameters(selected_parameter_seeds,
                                          segmentation_from_previous, seeds_from_previous, selected_seeds,
                                          cells, unseeded_cells, bounding_boxes, membrane_image,
                                          experiment, parameters):
    """

    :param selected_parameter_seeds:
    :param segmentation_from_previous:
    :param seeds_from_previous:
    :param selected_seeds:
    :param cells:
    :param unseeded_cells:
    :param bounding_boxes:
    :param membrane_image:
    :param experiment:
    :param parameters:
    :return:
    """

    proc = '_build_seeds_from_selected_parameters'

    #
    # parameter type checking
    #

    if not isinstance(experiment, common.Experiment):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'experiment' variable: "
                                      + str(type(experiment)))
        sys.exit(1)

    if not isinstance(parameters, AstecParameters):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'parameters' variable: "
                                      + str(type(parameters)))
        sys.exit(1)

    #
    #
    #

    first_segmentation = imread(segmentation_from_previous)

    #
    # temporary dictionary of spatial images
    # (to avoid multiple readings of the same image)
    #
    seed_image_list = {}

    #
    #
    #
    new_seed_image = np.zeros_like(first_segmentation, dtype=np.uint16)

    #
    # correspondences: dictionary containing the correspondences between cells of previous segmentations
    #                  and new seed labels
    # divided_cells: list of siblings
    #
    label_max = 2
    correspondences = {}
    divided_cells = []

    #
    # if one want to keep these informations
    #
    # h_min_information = {}
    # sigma_information = {}

    monitoring.to_log_and_console('      process cell with childrens', 3)

    for c in cells:

        #
        # cells for which no seeds were found for all value of h
        #
        if c in unseeded_cells:
            continue

        #
        # selected_parameter_seeds[c][0] : h_min
        # selected_parameter_seeds[c][1] : sigma
        # selected_parameter_seeds[c][2] : number of cells (2 or 3 means division)
        #
        h_min = selected_parameter_seeds[c][0]

        #
        # add the seed image to list if required
        #

        if h_min not in seed_image_list:
            seed_image = common.add_suffix(membrane_image, "_seed_h" + str('{:03d}'.format(h_min)),
                                           new_dirname=experiment.astec_dir.get_tmp_directory(),
                                           new_extension=experiment.default_image_suffix)
            if not os.path.isfile(seed_image):
                monitoring.to_log_and_console("       " + proc + ": '" + str(seed_image).split(os.path.sep)[-1]
                                              + "' was not found", 2)
                monitoring.to_log_and_console("\t Exiting.")
                sys.exit(1)
            seed_image_list[h_min] = imread(seed_image)

        #
        # get the seeds totally included in the cell
        # that was already done in _cell_based_h_minima()
        #
        # cell_segmentation is a sub-image with 'c' for the cell and 1 for the background
        # cell_seeds is a sub-image (same dimensions) of the h-minima image
        #

        cell_segmentation = np.ones_like(first_segmentation[bounding_boxes[c]])
        cell_segmentation[first_segmentation[bounding_boxes[c]] == c] = c
        cell_seeds = seed_image_list[h_min][bounding_boxes[c]]

        #
        # n_seeds: number of seeds totally included in the cell
        # labeled_seeds: sub-image with seeds numbered from 1
        #
        n_seeds, labeled_seeds = _extract_seeds(c, cell_segmentation, cell_seeds, accept_3_seeds=False)

        #
        # 1 seed
        #
        if n_seeds == 1:
            monitoring.to_log_and_console('      .. process cell ' + str(c) + ' -> ' + str(label_max), 3)
            correspondences[c] = [label_max]
            #
            # if one want to keep h_min and sigma information
            # t designs the previous time, thus t+delta_t is the current time of the image to be segmented
            # h_min_information[(t + delta_t) * 10 ** 4 + label_max] = right_parameters[c][0]
            # sigma_information[(t + delta_t) * 10 ** 4 + label_max] = right_parameters[c][1]
            #
            # here labeled_seeds has only 0 and 1's
            #
            new_seed_image[bounding_boxes[c]][labeled_seeds == 1] = label_max
            label_max += 1
        elif n_seeds == 2 or n_seeds == 3:
            monitoring.to_log_and_console('      .. process cell ' + str(c) + ' -> ' + str(label_max) + ', '
                                          + str(label_max + 1), 3)
            #
            # case n_seeds == 3
            # since _extract_seeds() has been called with 'accept_3_seeds=False'
            # => there are only the two first labeled seeds in 'labeled_seeds'
            #
            correspondences[c] = [label_max, label_max+1]
            divided_cells.append((label_max, label_max+1))
            new_seed_image[bounding_boxes[c]][labeled_seeds == 1] = label_max
            # h_min_information[(t + delta_t) * 10 ** 4 + label_max] = right_parameters[c][0]
            # sigma_information[(t + delta_t) * 10 ** 4 + label_max] = right_parameters[c][1]
            label_max += 1
            new_seed_image[bounding_boxes[c]][labeled_seeds == 2] = label_max
            # h_min_information[(t + delta_t) * 10 ** 4 + label_max] = right_parameters[c][0]
            # sigma_information[(t + delta_t) * 10 ** 4 + label_max] = right_parameters[c][1]
            label_max += 1
        else:
            monitoring.to_log_and_console("       " + proc + ": weird, there were " + str(n_seeds)
                                          + " seeds found for cell " + str(c), 2)

    #
    # create background seed
    # 1. create a background cell
    # 2. get the seeds from the read h-minima image with the smallest h
    # 3. add all the seeds (individual_seeds=False)
    #

    monitoring.to_log_and_console('      process background', 3)

    background_cell = np.ones_like(first_segmentation)
    background_cell[first_segmentation != 1] = 0

    h_min = min(seed_image_list.keys())
    n_seeds, labeled_seeds = _extract_seeds(1, background_cell, seed_image_list[h_min], individual_seeds=False)
    if n_seeds == 0:
        monitoring.to_log_and_console("       " + proc + ": unable to get background seed", 2)
    else:
        new_seed_image[labeled_seeds > 0] = 1
        correspondences[1] = [1]

    #
    # create seeds for cell for no seed found
    #

    monitoring.to_log_and_console('      process cell without childrens', 3)

    if len(unseeded_cells) > 0:
        first_seeds = imread(seeds_from_previous)
        for c in unseeded_cells:
            vol = np.sum(first_segmentation == c)
            if vol <= parameters.minimum_volume_unseeded_cell:
                monitoring.to_log_and_console("       " + proc + ": cell " + str(c)
                                              + " from previous time point will have no lineage", 2)
                monitoring.to_log_and_console("       .. volume = " + str(vol), 2)
            else:
                monitoring.to_log_and_console('      .. process cell ' + str(c) + ' -> ' + str(label_max), 3)
                correspondences[c] = [label_max]
                new_seed_image[first_seeds == c] = label_max
                label_max += 1
        del first_seeds
    #
    #
    #

    imsave(selected_seeds, new_seed_image)

    #
    #
    #
    del new_seed_image

    for i in seed_image_list.keys():
        del seed_image_list[i]

    del first_segmentation

    #
    #
    #
    return label_max, correspondences, divided_cells


########################################################################################
#
#
#
########################################################################################


def _compute_volumes(im):
    """

    :param im:
    :return:
    """
    proc = "_compute_volumes"
    if type(im) is str:
        readim = imread(im)
    elif type(im) is SpatialImage:
        readim = im
    else:
        monitoring.to_log_and_console(str(proc) + ": unhandled type for 'im': " + str(type(im)) + "'")
        return

    labels = np.unique(readim)
    volume = nd.sum(np.ones_like(readim), readim, index=np.int16(labels))
    if type(im) is str:
        del readim
    return dict(zip(labels, volume))


def _update_volume_properties(current_dict, volumes_to_be_added, current_time, time_digits=4):
    proc = "_update_volume_properties"
    tmp = {}
    for key, value in volumes_to_be_added.iteritems():
        newkey = current_time * 10**time_digits + int(key)
        tmp[newkey] = value
    current_dict.update(tmp)
    return current_dict


def _update_lineage_properties(current_dict, lineage_to_be_added, current_time, time_digits=4):
    tmp = {}
    for key, value in lineage_to_be_added.iteritems():
        newkey = current_time * 10**time_digits + int(key)
        tmp[str(newkey)] = value
    current_dict.update(tmp)
    return current_dict


def _volume_checking(previous_segmentation, segmentation_from_selection, deformed_seeds, selected_seeds, membrane_image,
                     correspondences, selected_parameter_seeds, n_seeds, parameter_seeds, bounding_boxes, experiment,
                     parameters):
    """


    :param previous_segmentation: watershed segmentation obtained with the deformed_seeds
    :param segmentation_from_selection:
    :param deformed_seeds: seeds obtained from the segmentation at a previous time and deformed into the current time
    :param selected_seeds:
    :param membrane_image:
    :param correspondences: is a dictionary that gives, for each 'parent' cell (in the segmentation built from previous
    time segmentation) (ie the key), the list of 'children' cells (in the segmentation built from selected seeds)
    :param selected_parameter_seeds:
    :param n_seeds: dictionary, gives, for each parent cell, give the number of seeds for each couple of
    parameters [h-min, sigma]
    :param parameter_seeds: dictionary, for each parent cell, give the list of used parameters [h-min, sigma]
    :param bounding_boxes:
    :param experiment:
    :param parameters:
    :return:
    """

    proc = "_volume_checking"

    #
    # parameter type checking
    #

    if not isinstance(experiment, common.Experiment):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'experiment' variable: "
                                      + str(type(experiment)))
        sys.exit(1)

    if not isinstance(parameters, AstecParameters):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'parameters' variable: "
                                      + str(type(parameters)))
        sys.exit(1)

    #
    # compute volumes
    #
    prev_seg = imread(previous_segmentation)
    curr_seg = imread(segmentation_from_selection)

    prev_volumes = _compute_volumes(prev_seg)
    curr_volumes = _compute_volumes(curr_seg)

    prev_embryo_volume = prev_seg.size - prev_volumes[1]
    curr_embryo_volume = curr_seg.size - curr_volumes[1]

    #
    # kept from Leo, very weird formula
    # volume_ratio is the opposite of the fraction of volume lose for the
    # volume from previous time compared to current time
    # volume_ratio = 1.0 - vol(mother) / SUM volume(childrens)
    #
    # volume_ratio < 0 => previous volume > current volume
    # volume_ratio > 0 => previous volume < current volume
    #

    volume_ratio = 1.0 - prev_embryo_volume/curr_embryo_volume

    #
    # default value of parameters.volume_ratio_tolerance is 0.1
    #

    if -parameters.volume_ratio_tolerance <= volume_ratio <= parameters.volume_ratio_tolerance:
        pass
    else:
        if volume_ratio < 0:
            monitoring.to_log_and_console('    .. embryo volume has strongly diminished', 2)
        else:
            monitoring.to_log_and_console('    .. embryo volume has strongly increased', 2)

    #
    # lists of (parent (cell at t), children (cell(s) at t+dt))
    #
    # large_volume_ratio          : volume(mother)   <  SUM volume(childrens)
    # small_volume_ratio          : volume(mother)   >  SUM volume(childrens)
    # abnormal_large_volume_ratio : volume(mother)   << SUM volume(childrens)
    # abnormal_small_volume_ratio : volume(mother)   >> SUM volume(childrens)
    # small_volume_daughter       : volume(children) <  threshold
    #
    large_volume_ratio = []
    small_volume_ratio = []
    abnormal_large_volume_ratio = []
    abnormal_small_volume_ratio = []
    small_volume_daughter = []

    seed_label_all = []
    for mother_c, daughters_c in correspondences.iteritems():
        #
        # skip background
        #
        if mother_c == 1:
            continue

        seed_label_all.extend(daughters_c)

        #
        # check whether the volumes exist
        #
        if mother_c in prev_volumes is False:
            monitoring.to_log_and_console('    ' + proc + ': no volume for cell ' + str(mother_c)
                                          + ' in previous segmentation', 2)
        for s in daughters_c:
            if s in curr_volumes is False:
                monitoring.to_log_and_console('    ' + proc + ': no volume for cell ' + str(s)
                                              + ' in current segmentation', 2)

        #
        # compute ratios
        #
        # volume_ratio > 0  <=> volume(mother) < SUM volume(childrens)
        # volume_ratio = 0  <=> volume(mother) = SUM volume(childrens)
        # volume_ratio < 0  <=> volume(mother) > SUM volume(childrens)
        #
        volume_ratio = 1.0 - prev_volumes[mother_c] / np.sum([curr_volumes.get(s, 1) for s in daughters_c])

        #
        # admissible ratio, check whether the daughter cell(s) are large enough
        # default value of parameters.volume_ratio_tolerance is 0.1
        # 1+ratio >= volume(mother) / SUM volume(childrens) >= 1 -ratio
        #
        # check whether a daughter cell if too small
        #
        if -parameters.volume_ratio_tolerance <= volume_ratio <= parameters.volume_ratio_tolerance:
            for daughter_c in daughters_c:
                if curr_volumes[daughter_c] < parameters.volume_minimal_value:
                    small_volume_daughter.append([mother_c, daughter_c])
        else:
            #
            # non-admissible ratios
            # default value of parameters.volume_ratio_threshold is 0.5
            #
            if volume_ratio > 0:
                # volume_ratio > 0  <=> volume(mother) < SUM volume(childrens)
                large_volume_ratio.append((mother_c, daughters_c))
                if volume_ratio > parameters.volume_ratio_threshold:
                    abnormal_large_volume_ratio.append(mother_c)
            elif volume_ratio < 0:
                # volume_ratio < 0  <=> volume(mother) > SUM volume(childrens)
                small_volume_ratio.append((mother_c, daughters_c))
                if volume_ratio < -parameters.volume_ratio_threshold:
                    abnormal_small_volume_ratio.append(mother_c)
            else:
                monitoring.to_log_and_console('    ' + proc + ': should not reach this point', 2)
                monitoring.to_log_and_console('    mother cell was ' + str(mother_c), 2)
                monitoring.to_log_and_console('    daughter cell(s) was(ere) ' + str(daughters_c), 2)
    #
    # get the largest used label
    # -> required to attribute new labels
    #
    seed_label_max = max(seed_label_all)

    if len(abnormal_small_volume_ratio) > 0:
        monitoring.to_log_and_console('    .. cell with large decrease of volume: ' + str(abnormal_small_volume_ratio),
                                      2)
    if len(abnormal_large_volume_ratio) > 0:
        monitoring.to_log_and_console('    .. cell with large increase of volume: ' + str(abnormal_large_volume_ratio),
                                      2)

    #
    # here we look at cells that experiment a large decrease of volume
    # ie vol(mother) >> vol(daughter(s))
    # this is the step (1) of section 2.3.3.6 of L. Guignard thesis
    # [corresponds to the list to_look_at in historical astec code]
    #

    selected_seeds_image = imread(selected_seeds)
    deformed_seeds_image = imread(deformed_seeds)
    has_change_happened = False
    labels_to_be_fused = []

    ############################################################
    #
    # BEGIN: cell with large decrease of volume
    # volume_ratio < 0  <=> volume(mother) > SUM volume(childrens)
    # try to add seeds
    #
    ############################################################
    if len(abnormal_small_volume_ratio) > 0:
        monitoring.to_log_and_console('      process cell with large decrease of volume', 2)

    for mother_c in abnormal_small_volume_ratio:

        #
        # this is similar to _select_seed_parameters()
        # however, the smallest h is retained and not the largest one
        # thus seeds should be larger
        # n_seeds[mother_c] gives the number of seeds for each couple of parameters [h-min, sigma]
        #
        s = n_seeds[mother_c]

        #
        # np.sum(np.array(s) == 2) is equivalent to s.count(2)
        # we redo the h selection
        #
        nb_2 = np.sum(np.array(s) == 2)
        nb_3 = np.sum(np.array(s) >= 2)
        score = nb_2 * nb_3

        if s.count(1) > 0 or s.count(2) > 0:
            #
            # parameter_seeds: dictionary, for each parent cell, give the list of used parameters [h-min, sigma]
            # np.where(np.array(s)==2) yields the indices where n_seeds[mother_c] == 2
            # In fact, it gives something like (array([1, 2, 3, 4]),), so
            # np.where(np.array(s)==2)[0] allows to get only the indexes
            # np.where(np.array(s)==2)[0][-1] is then the last index where we have n_seeds[mother_c] == 2
            #
            if score >= parameters.seed_selection_tau:
                #
                # the retained h value is the smallest h value that yields 2 seeds
                #
                # h, sigma = parameter_seeds[mother_c][np.where(np.array(s)==2)[0][-1]]
                # the final h value will be determined afterwards
                #
                nb_final = 2
            elif s.count(1) != 0:
                #
                # score < tau and s.count(1) != 0
                # the retained h value is the smallest h value that yields 1 seeds
                #
                # h, sigma = parameter_seeds[mother_c][np.where(np.array(s) == 1)[0][-1]]
                nb_final = 1
            else:
                #
                # the retained h value is the smallest h value that yields 2 seeds
                #
                # h, sigma = parameter_seeds[mother_c][np.where(np.array(s) == 2)[0][-1]]
                nb_final = 2

            #
            # from Leo PhD thesis (section 2.3.3.6, page 73)
            # The correction then consists in increasing the number of seeds in order to cover
            # more space and avoid matter loss. If the cell was considered not divided by the
            # previous steps, then it is divided into two cells by the correction procedure if
            # possible. If not the cell snapshot is voluntarily over-segmented to maximize the
            # covered surface by the seeds and minimize the possibility of volume loss. The
            # over-segmented cells are then fused.
            #

            if nb_final == 1 and s.count(2) != 0:
                #
                # if no division was chosen, try to increase the number of seeds
                # in order to try to increase the size of the reconstructed cells
                #
                # get the h-min image corresponding to the first case (seeds == 2)
                # recall that seeds have already being masked by the 'previous' segmentation image
                #
                # shouldn't we check whether the other seed is "under" the daughters and labeled
                # the seeds as in the following case (nb_final == 1 or nb_final == 2) and (np.array(s) > 2).any())?
                #
                h_min, sigma = parameter_seeds[mother_c][s.index(2)]
                seed_image_name = common.add_suffix(membrane_image, "_seed_h" + str('{:03d}'.format(h_min)),
                                               new_dirname=experiment.astec_dir.get_tmp_directory(),
                                               new_extension=experiment.default_image_suffix)
                #
                # create a sub-image where the cell 'mother_c' has the 'mother_c' value
                # and a background at '1'
                # extract the corresponding seeds from 'seed_image_name'
                #
                bb = bounding_boxes[mother_c]
                submask_mother_c = np.ones_like(prev_seg[bb])
                submask_mother_c[prev_seg[bb] == mother_c] = mother_c
                n_found_seeds, labeled_found_seeds = _extract_seeds(mother_c, submask_mother_c, seed_image_name, bb)
                if n_found_seeds == 2:
                    new_correspondences = [seed_label_max+1, seed_label_max+2]
                    monitoring.to_log_and_console('        .. cell ' + str(mother_c) + ': '
                                                  + str(correspondences[mother_c] + ' -> ' + str(new_correspondences)),
                                                  3)
                    #
                    # remove previous seed
                    # add new seeds
                    #
                    selected_seeds_image[selected_seeds_image == correspondences[mother_c][0]] = 0
                    selected_seeds_image[bb][labeled_found_seeds == 1] = seed_label_max + 1
                    selected_seeds_image[bb][labeled_found_seeds == 2] = seed_label_max + 2
                    correspondences[mother_c] = new_correspondences
                    selected_parameter_seeds[mother_c] = [h_min, sigma, n_found_seeds]
                    seed_label_max += 2
                    has_change_happened = True
                else:
                    monitoring.to_log_and_console('        .. (1) cell ' + str(mother_c) + ': weird, has found '
                                                  + str(n_found_seeds) + " instead of 2", 2)
            elif (nb_final == 1 or nb_final == 2) and (np.array(s) > 2).any():
                #
                # there is a h that gives more than 2 seeds
                # get the smallest h
                #
                h_min, sigma = parameter_seeds[mother_c][-1]
                seed_image_name = common.add_suffix(membrane_image, "_seed_h" + str('{:03d}'.format(h_min)),
                                                    new_dirname=experiment.astec_dir.get_tmp_directory(),
                                                    new_extension=experiment.default_image_suffix)
                #
                # create a sub-image where cells 'daughter_c' has the 'mother_c' value
                # and a background at '1'
                # create a sub-image where the cell 'mother_c' has the 'mother_c' value
                # and a background at '1'
                #
                # built a sub-image where seeds 'below' daughters have a '1' value
                # and seeds 'below' the projection segmentation have a '2' value
                #
                # do something if there are seeds 'below' the projection segmentation has a '2' value
                # - seeds 'below' daughters (that have a '1' value) will be fused into a 'seed_label_max + 1' cell
                # - seeds 'below' the projection segmentation (that have a '2' value) will be fused into a
                #   'seed_label_max + 2' cell
                #
                bb = bounding_boxes[mother_c]
                submask_daughter_c = np.ones_like(curr_seg[bb])
                for daughter_c in correspondences[mother_c]:
                    submask_daughter_c[curr_seg[bb] == daughter_c] = mother_c
                submask_mother_c = np.ones_like(prev_seg[bb])
                submask_mother_c[prev_seg[bb] == mother_c] = mother_c
                aux_seed_image = imread(seed_image_name)
                seeds_c = np.zeros_like(curr_seg[bb])
                seeds_c[(aux_seed_image[bb] != 0) & (submask_daughter_c == mother_c)] = 1
                seeds_c[(aux_seed_image[bb] != 0) & (submask_daughter_c == 1) & (submask_mother_c == mother_c)] = 2
                del aux_seed_image
                if 2 in seeds_c:
                    new_correspondences = [seed_label_max + 1, seed_label_max + 2]
                    monitoring.to_log_and_console('        .. (2) cell ' + str(mother_c) + ': '
                                          + str(correspondences[mother_c] + ' -> ' + str(new_correspondences)), 3)
                    #
                    # remove previous seed
                    # add new seeds, note that they might be several seeds per label '1' or '2'
                    #
                    for daughter_c in correspondences[mother_c]:
                        selected_seeds_image[selected_seeds_image == daughter_c] = 0
                    selected_seeds_image[bb][seeds_c == 1] = seed_label_max + 1
                    selected_seeds_image[bb][seeds_c == 2] = seed_label_max + 2
                    correspondences[mother_c] = new_correspondences
                    selected_parameter_seeds[mother_c] = [h_min, sigma, 2]
                    seed_label_max += 2
                    has_change_happened = True
                else:
                    monitoring.to_log_and_console('        .. (3) cell ' + str(mother_c) +
                                                  ': does not know what to do', 3)

            elif nb_final == 1:
                #
                # here s.count(2) == 0 and np.array(s) > 2).any() is False
                # replace the computed seed with the seed from the previous segmentation
                #
                monitoring.to_log_and_console('        .. (4) cell ' + str(mother_c) + ': '
                                              + str(correspondences[mother_c]) + ' -> '
                                              + str(correspondences[mother_c]),  3)
                selected_seeds_image[selected_seeds_image == correspondences[mother_c][0]] = 0
                selected_seeds_image[deformed_seeds_image == mother_c] = correspondences[mother_c]
                selected_parameter_seeds[mother_c] = [-1, -1, 1]
                has_change_happened = True
            else:
                monitoring.to_log_and_console('        .. (5) cell ' + str(mother_c) + ': does not know what to do', 3)

        elif s.count(3) != 0:
            #
            # here we have s.count(1) == and  s.count(2) == 0:
            # get the three seeds, and keep them for further fusion
            #
            h_min, sigma = parameter_seeds[mother_c][s.index(3)]
            seed_image_name = common.add_suffix(membrane_image, "_seed_h" + str('{:03d}'.format(h_min)),
                                                new_dirname=experiment.astec_dir.get_tmp_directory(),
                                                new_extension=experiment.default_image_suffix)
            #
            # create a sub-image where the cell 'mother_c' has the 'mother_c' value
            # and a background at '1'
            # extract the corresponding seeds from 'seed_image_name'
            #
            bb = bounding_boxes[mother_c]
            submask_mother_c = np.ones_like(prev_seg[bb])
            submask_mother_c[prev_seg[bb] == mother_c] = mother_c
            n_found_seeds, labeled_found_seeds = _extract_seeds(mother_c, submask_mother_c, seed_image_name, bb,
                                                                accept_3_seeds=True)
            if n_found_seeds == 3:
                new_correspondences = [seed_label_max + 1, seed_label_max + 2, seed_label_max + 3]
                monitoring.to_log_and_console('        .. (6) cell ' + str(mother_c) + ': '
                                              + str(correspondences[mother_c] + ' -> ' + str(new_correspondences)),
                                              3)
                #
                # remove previous seed
                # add new seeds
                #
                for daughter_c in correspondences[mother_c]:
                    selected_seeds_image[selected_seeds_image == daughter_c] = 0
                selected_seeds_image[bb][labeled_found_seeds == 1] = seed_label_max + 1
                selected_seeds_image[bb][labeled_found_seeds == 2] = seed_label_max + 2
                selected_seeds_image[bb][labeled_found_seeds == 3] = seed_label_max + 3
                correspondences[mother_c] = new_correspondences
                selected_parameter_seeds[mother_c] = [h_min, sigma, n_found_seeds]
                seed_label_max += 3
                has_change_happened = True
                labels_to_be_fused.append(new_correspondences)
            else:
                monitoring.to_log_and_console('        .. (7) cell ' + str(mother_c) + ': weird, has found '
                                              + str(n_found_seeds) + " instead of 3", 2)
        else:
            monitoring.to_log_and_console('        .. (8) cell ' + str(mother_c) + ': detected seed numbers wrt h was '
                                          + str(s), 2)

    ############################################################
    #
    # END: cell with large decrease of volume
    #
    ############################################################

    ############################################################
    #
    # BEGIN: too small 'daughter' cells
    # recall: the volume ratio between 'mother' and 'daughters' is ok
    # but some 'daughter' cells are too small
    # - remove the too small daughter cell from the seed image
    # - remove it from the correspondence array
    # - remove the mother cell is it has no more daughter
    ############################################################
    if small_volume_daughter:
        for mother_c, daughter_c in small_volume_daughter:
            selected_seeds_image[selected_seeds_image == daughter_c] = 0
            daughters_c = correspondences[mother_c]
            daughters_c.remove(daughter_c)
            if daughters_c:
                correspondences[mother_c] = daughters_c
            else:
                correspondences.pop(mother_c)
        has_change_happened = True
    ############################################################
    #
    # END: too small 'daughter' cells
    #
    ############################################################

    del prev_seg
    del curr_seg
    del deformed_seeds_image

    #
    # nothing to do
    #

    if not has_change_happened:
        del selected_seeds_image
        return segmentation_from_selection, selected_seeds, correspondences

    #
    # some corrections are to be done
    # 1. save the image of corrected seeds
    # 2. redo a watershed
    #

    corr_selected_seeds = common.add_suffix(membrane_image, '_seeds_from_corrected_selection',
                                            new_dirname=experiment.astec_dir.get_tmp_directory(),
                                            new_extension=experiment.default_image_suffix)
    voxelsize = selected_seeds_image._get_resolution()
    imsave(corr_selected_seeds, SpatialImage(selected_seeds_image, voxelsize=voxelsize).astype(np.uint16))
    del selected_seeds_image

    segmentation_from_corr_selection = common.add_suffix(membrane_image, '_watershed_from_corrected_selection',
                                                         new_dirname=experiment.astec_dir.get_tmp_directory(),
                                                         new_extension=experiment.default_image_suffix)
    mars.watershed(corr_selected_seeds, membrane_image, segmentation_from_corr_selection, experiment, parameters)

    #
    # ... on en est a la lign
    # no labels to be fused
    #

    if not labels_to_be_fused:
        return segmentation_from_corr_selection, corr_selected_seeds, correspondences

    #
    # fused labels (lines 635 - 660)
    #
    # reste a faire aussi lines 587 634 dans une autre fonction
    #

    return


########################################################################################
#
#
#
########################################################################################

#
#
#
#
#

def astec_process(previous_time, current_time, lineage_tree_information, experiment, parameters):
    """

    :param previous_time:
    :param current_time:
    :param lineage_tree_information:
    :param experiment:
    :param parameters:
    :return:
    """

    proc = "astec_process"

    #
    # 1. retrieve the membrane image
    #    it can be the fused image or a calculated image
    # 2. compute the "deformed" segmentation from previous time
    #    a. erode the segmentation from previous time to get seeds
    #    b. deform the seeds
    #    c. segmentation (watershed-based) from the deformed seeds
    # 3. For each cell, compute the number of h-minima for a collection of h
    # 4. For each cell, select a number of h-minima
    #    typically, 1 if no division, or 2 if division
    # 5. Build a seed image from the selected (cell-based) h-minima
    # 6. segmentation (watershed-based) from the built seeds
    #

    #
    # parameter type checking
    #

    if not isinstance(experiment, common.Experiment):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'experiment' variable: "
                                      + str(type(experiment)))
        sys.exit(1)

    if not isinstance(parameters, AstecParameters):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'parameters' variable: "
                                      + str(type(parameters)))
        sys.exit(1)

    #
    # nothing to do if the segmentation image exists
    #
    astec_dir = experiment.astec_dir.get_directory()
    astec_name = experiment.astec_dir.get_image_name(current_time)
    astec_image = common.find_file(astec_dir, astec_name, callfrom=proc, local_monitoring=None, verbose=False)

    if astec_image is not None:
        if monitoring.forceResultsToBeBuilt is False:
            monitoring.to_log_and_console('    astec image already existing', 2)
            return
        else:
            monitoring.to_log_and_console('    astec image already existing, but forced', 2)

    astec_image = os.path.join(astec_dir, astec_name + "." + experiment.result_image_suffix)

    #
    # build or read the membrane image to be segmented
    #

    reconstruction.monitoring.copy(monitoring)

    membrane_image = reconstruction.build_membrane_image(current_time, experiment, parameters,
                                                         previous_time=previous_time)
    if membrane_image is None:
        monitoring.to_log_and_console("    .. " + proc + ": no membrane image was found/built for time "
                                      + str(current_time), 2)
        return False

    #
    # build seeds by eroding previous segmentation and deforming it
    #
    # erosion iterations are set by default in voxel units
    # there is also a volume defined in voxel units
    #
    # Note: it may be worth trying to deform first the previous segmentation and then
    # extract the seeds
    #
    monitoring.to_log_and_console('    build seeds from previous segmentation', 2)

    previous_segmentation = experiment.get_segmentation_image(previous_time)
    if previous_segmentation is None:
        monitoring.to_log_and_console("    .. " + proc + ": no segmentation image was found for time "
                                      + str(previous_time), 2)
        return False

    undeformed_seeds = common.add_suffix(previous_segmentation, '_undeformed_seeds_from_previous',
                                         new_dirname=experiment.astec_dir.get_tmp_directory(),
                                         new_extension=experiment.default_image_suffix)
    if not os.path.isfile(undeformed_seeds):
        _build_seeds_from_previous_segmentation(previous_segmentation, undeformed_seeds, parameters)

    deformed_seeds = common.add_suffix(previous_segmentation, '_deformed_seeds_from_previous',
                                       new_dirname=experiment.astec_dir.get_tmp_directory(),
                                       new_extension=experiment.default_image_suffix)

    if not os.path.isfile(deformed_seeds):
        deformation = reconstruction.get_deformation_from_current_to_previous(current_time, experiment,
                                                                              parameters, previous_time)
        cpp_wrapping.apply_transformation(undeformed_seeds, deformed_seeds, deformation,
                                          interpolation_mode='nearest', monitoring=monitoring)

    #
    # watershed segmentation with seeds extracted from previous segmentation
    # $\tilde{S}_{t+1}$ in Leo's PhD
    #
    monitoring.to_log_and_console('    watershed from previous segmentation', 2)
    segmentation_from_previous = common.add_suffix(membrane_image, '_watershed_from_previous',
                                                   new_dirname=experiment.astec_dir.get_tmp_directory(),
                                                   new_extension=experiment.default_image_suffix)

    if parameters.propagation_strategy is 'seeds_from_previous_segmentation':
        segmentation_from_previous = astec_image

    if not os.path.isfile(segmentation_from_previous):
        mars.watershed(deformed_seeds, membrane_image, segmentation_from_previous, experiment, parameters)

    #
    # if the propagation strategy is only to get seeds from the erosion of the previous cells
    # we're done. Update the properties
    #

    if parameters.propagation_strategy is 'seeds_from_previous_segmentation':
        #
        # update volumes and lineage (cells may disappear if the erosion is too strong)
        #
        time_digits = experiment.get_unique_id_time_digits()

        current_volumes = _compute_volumes(astec_image)
        volume_key = properties.keydictionary['volume']['output_key']
        volume_key = 'volumes_information'
        if not volume_key in lineage_tree_information.keys():
            lineage_tree_information[volume_key] = {}
        _update_volume_properties(lineage_tree_information[volume_key], current_volumes, current_time,
                                  time_digits=time_digits)
        #
        # update lineage. It is somehow just to check since cells are not supposed to disappeared
        # _erode_cell() performs erosions until the maximal number of iterations is reached
        # or juste before the cell disappears
        #
        lineage_key = properties.keydictionary['volume']['output_key']
        lineage_key = 'lin_tree'
        if not lineage_key in lineage_tree_information.keys():
            lineage_tree_information[lineage_key] = {}
        current_lineage = _build_lineage_for_seeds_from_previous_segmentation(current_volumes, previous_time,
                                                                              current_time, time_digits=time_digits)
        lineage_tree_information[lineage_key].update(current_lineage)
        return lineage_tree_information


    #
    # bounding_boxes: bounding boxes for each cell from the watershed segmentation
    # cells: list of cell labels
    #
    first_segmentation = imread(segmentation_from_previous)
    cells = list(np.unique(first_segmentation))
    cells.remove(1)
    bounding_boxes = dict(zip(range(1, max(cells) + 1), nd.find_objects(first_segmentation)))
    del first_segmentation

    #
    # for each cell and a collection of h values,
    # - compute a seed image for each value of h
    #   seeds are masked by the cells of the 'previous' segmentation
    # - get a number of seeds per cell of the previous segmentation
    #   and the parameters [h, sigma] that gives the corresponding seed image
    #
    # n_seeds, parameter_seeds are dictionaries indexed by mother cell index
    # n_seeds[mother_c] is an array of the number of seeds
    # parameter_seeds[mother_c] is an array (of same length) that gives the parameters ([h, sigma]) used for the
    #   computation, h being decreasing
    #
    monitoring.to_log_and_console('    estimation of h-minima for h in ['
                                  + str(parameters.watershed_seed_hmin_min_value) + ','
                                  + str(parameters.watershed_seed_hmin_max_value) + ']', 2)
    n_seeds, parameter_seeds = _cell_based_h_minima(segmentation_from_previous, cells, bounding_boxes, membrane_image,
                                                    experiment, parameters)

    #
    # First selection of seeds
    #
    # from above results (ie, the number of seeds for every value of h),
    # select the parameter (ie h value)
    # Note: sigma (smoothing parameter to extract seeds) is also kept here, meaning that
    #       it also could be used for selection
    #
    # selected_parameter_seeds is a dictionary indexed by mother cell index
    # selected_parameter_seeds[mother_c] is an array [selected_h, sigma, n_seeds]
    # unseeded_cells is a list
    # is a list
    #
    monitoring.to_log_and_console('    parameter selection', 2)
    selected_parameter_seeds, unseeded_cells = _select_seed_parameters(n_seeds, parameter_seeds,
                                                                       tau=parameters.seed_selection_tau)

    #
    # print out the list of cells without seeds and the list of cells that may divide
    #
    if len(unseeded_cells) > 0:
        string = ""
        for i in range(len(unseeded_cells)):
            if string == "":
                string = str(unseeded_cells[i])
            else:
                string += ", " + str(unseeded_cells[i])
        monitoring.to_log_and_console('    .. cells with no seeds: ' + string, 2)
    string = ""
    for c in cells:
        if c in unseeded_cells:
            continue
        if selected_parameter_seeds[c][2] > 1:
            if string == "":
                string = str(c)
            else:
                string += ", " + str(c)
    if string != "":
        monitoring.to_log_and_console('    .. cells that will divide: ' + string, 2)

    #
    # build an image of seeds with selected parameters h
    #
    monitoring.to_log_and_console('    build seed image from all h-minima images', 2)

    selected_seeds = common.add_suffix(membrane_image, '_seeds_from_selection',
                                       new_dirname=experiment.astec_dir.get_tmp_directory(),
                                       new_extension=experiment.default_image_suffix)

    output = _build_seeds_from_selected_parameters(selected_parameter_seeds, segmentation_from_previous, deformed_seeds,
                                                   selected_seeds, cells, unseeded_cells, bounding_boxes,
                                                   membrane_image, experiment, parameters)

    label_max, correspondences, divided_cells = output
    # print("divided_cells: " +str(divided_cells))

    #
    # watershed segmentation with the selected seeds
    #

    monitoring.to_log_and_console('    watershed from selection of seeds', 2)
    segmentation_from_selection = common.add_suffix(membrane_image, '_watershed_from_selection',
                                                    new_dirname=experiment.astec_dir.get_tmp_directory(),
                                                    new_extension=experiment.default_image_suffix)
    if not os.path.isfile(segmentation_from_selection):
        mars.watershed(selected_seeds, membrane_image, segmentation_from_selection, experiment, parameters)

    #
    # Here, we have a first segmentation
    # let correct it if required
    #

    # print(proc + " :correspondences: " + str(correspondences))
    # print(proc + " :selected_parameter_seeds: " + str(selected_parameter_seeds))
    # print("n_seeds: " + str(n_seeds))
    # print("parameter_seeds: " + str(parameter_seeds))

    monitoring.to_log_and_console('    volume checking', 2)
    _volume_checking(previous_segmentation, segmentation_from_selection, deformed_seeds, selected_seeds, membrane_image,
                     correspondences, selected_parameter_seeds, n_seeds, parameter_seeds,
                     bounding_boxes, experiment, parameters)

    sys.exit(1)

    lineage_tree = lineage_tree_information.get('lin_tree', {})


    tmp = lineage_tree_information.get('volumes_information', {})
    volumes_t_1 = {k % 10 ** 4: v for k, v in tmp.iteritems() if k / 10 ** 4 == t}
    h_min_information = {}

    treated = []



    return


#
# check whether a lineage file exists
# loops over the time points
#


def astec_control(experiment, parameters):
    """

    :param experiment:
    :param parameters:
    :return:
    """

    proc = "astec_control"

    #
    # parameter type checking
    #

    if not isinstance(experiment, common.Experiment):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'experiment' variable: "
                                      + str(type(experiment)))
        sys.exit(1)

    if not isinstance(parameters, AstecParameters):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'parameters' variable: "
                                      + str(type(parameters)))
        sys.exit(1)

    #
    # copy monitoring information
    #
    ace.monitoring.copy(monitoring)
    common.monitoring.copy(monitoring)
    mars.monitoring.copy(monitoring)
    reconstruction.monitoring.copy(monitoring)
    ace.monitoring.copy(monitoring)

    #
    # make sure that the result directory exists
    #

    experiment.astec_dir.make_directory()

    #
    # re-read the lineage file, if any
    # and check whether any time point should be re-computed
    #

    first_time_point = experiment.first_time_point + experiment.delay_time_point
    last_time_point = experiment.last_time_point + experiment.delay_time_point

    lineage_tree_name = experiment.get_embryo_name() + "_seg_lineage.pkl"
    lineage_tree_path = os.path.join(experiment.astec_dir.get_directory(), lineage_tree_name)

    if os.path.isfile(lineage_tree_path):
        lineage_tree_information = properties.read_dictionary(lineage_tree_path)
    else:
        lineage_tree_information = {}

    # print environment.path_seg_exp_lineage
    # print lineage_tree_information

    #
    # test a mettre dans une fonction
    # l'idee est de savoir ou repartir :
    # 1. tester depuis la fin si l'image de segmentation existe
    # 2. tester si le lineage existe
    # a partir de ce temps, effacer ce qu'il y a dans l'arbre
    #
    if len(lineage_tree_information) > 0 and 'lin_tree' in lineage_tree_information:
        monitoring.to_log_and_console("    .. test '" + str(lineage_tree_path) + "'", 1)
        cellat = {}
        for y in lineage_tree_information['lin_tree']:
            t = y/10**4
            if t not in cellat:
                cellat[t] = 1
            else:
                cellat[t] += 1

        segmentation_dir = experiment.astec_dir.get_directory()

        restart = -1
        t = first_time_point
        while restart == -1 and t <= last_time_point:
            #
            # possible time point of segmentation, test if ok
            #
            time_value = t + experiment.delta_time_point
            segmentation_name = experiment.astec_dir.get_image_name(time_value)
            segmentation_file = common.find_file(segmentation_dir, segmentation_name)
            if segmentation_file is None or not os.path.isfile(os.path.join(segmentation_dir, segmentation_file)):
                monitoring.to_log_and_console("       image '" + segmentation_file + "' not found", 1)
                restart = t
            else:
                if cellat[t] == 0:
                    monitoring.to_log_and_console("       lineage of image '" + segmentation_file + "' not found", 1)
                    restart = t
                else:
                    try:
                        #
                        # pourquoi on lit l'image ?
                        #
                        segmentation_image = imread(segmentation_file)
                    except IOError:
                        monitoring.to_log_and_console("       error in image '" + segmentation_file + "'", 1)
                        restart = t
            #
            #
            #

            if restart == -1:
                monitoring.to_log_and_console("       time '" + str(t) + "' seems ok", 1)
            t += 1
        first_time_point = restart
        monitoring.to_log_and_console(".. " + proc + ": restart computation at time '"
                                      + str(first_time_point) + "'", 1)
    else:
        monitoring.to_log_and_console(".. " + proc + ": start computation at time '"
                                      + str(first_time_point) + "'", 1)

    #
    #
    #

    for current_time in range(first_time_point + experiment.delay_time_point + experiment.delta_time_point,
                              last_time_point + experiment.delay_time_point + 1, experiment.delta_time_point):

        acquisition_time = experiment.get_time_index(current_time)
        previous_time = current_time - experiment.delta_time_point

        #
        # start processing
        #

        monitoring.to_log_and_console('... astec processing of time ' + acquisition_time, 1)
        start_time = time.time()

        #
        # set and make temporary directory
        #
        experiment.astec_dir.set_tmp_directory(current_time)
        experiment.astec_dir.make_tmp_directory()

        if parameters.keep_reconstruction is False:
            experiment.astec_dir.set_rec_directory_to_tmp()

        #
        # process
        #

        ret = astec_process(previous_time, current_time, lineage_tree_information, experiment, parameters)
        if ret is False:
            monitoring.to_log_and_console('    an error occurs when processing time ' + acquisition_time, 1)
            return False
        else:
            lineage_tree_information = ret

        #
        # cleaning
        #

        if monitoring.keepTemporaryFiles is False:
            experiment.astec_dir.rmtree_tmp_directory()

        #
        # end processing for a time point
        #
        end_time = time.time()

        monitoring.to_log_and_console('    computation time = ' + str(end_time - start_time) + ' s', 1)
        monitoring.to_log_and_console('', 1)

    if lineage_tree_path.endswith("pkl") is True:
        lineagefile = open(lineage_tree_path, 'w')
        pkl.dump(lineage_tree_information, lineagefile)
        lineagefile.close()

    return
