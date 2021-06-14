
import os
import imp
import sys
import shutil
import time
import morphsnakes
import multiprocessing
import numpy as np
from scipy import ndimage as nd
import copy

import ace
import mars
import common
import reconstruction
import properties as properties
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
class MorphoSnakeParameters(common.PrefixedParameter):

    ############################################################
    #
    # initialisation
    #
    ############################################################

    def __init__(self, prefix=None):
            
        common.PrefixedParameter.__init__(self, prefix=prefix)

        if "doc" not in self.__dict__:
            self.doc = {}

        #
        # number of dilation iterations to get the initialization from 'previous' cell
        #
        doc = "\t Defines the initial shape of the the morphosnake.\n"
        doc += "\t It is the previous (deformed) cell dilated by\n"
        doc += "\t  this amount of iterations\n"
        self.doc['dilation_iterations'] = doc
        self.dilation_iterations = 10
        #
        # maximal number of iterations of the morphosnake
        #
        doc = "\t Maximal number of iteration\n"
        self.doc['iterations'] = doc
        self.iterations = 200
        #
        # threshold on the voxel number to break
        #
        doc = "\t Threshold on the voxel count to stop the morphosnake\n"
        doc += "\t evolution (stability is reached)\n"
        self.doc['delta_voxel'] = doc
        self.delta_voxel = 10**3

        doc = "\t Possible values are 'gradient' or 'image'\n"
        doc += "\t 'image' should be more adapted for membrane images\n"
        self.doc['energy'] = doc
        self.energy = 'image'

        doc = "\t Weight for the morphosnake energy\n"
        self.doc['smoothing'] = doc
        self.smoothing = 3

        doc = "\t Weight for the morphosnake energy\n"
        self.doc['balloon'] = doc
        self.balloon = 1

        doc = "\t Number of processors for parallelism\n"
        self.doc['processors'] = doc
        self.processors = 10

        doc = "\t Possible values are True or False\n"
        doc += "\t If true, mimics historical behavior.\n"
        doc += "\t Kept for test purposes\n"
        self.doc['mimic_historical_astec'] = doc
        self.mimic_historical_astec = False

    ############################################################
    #
    # print / write
    #
    ############################################################

    def print_parameters(self):
        print("")
        print('#')
        print('# MorphoSnakeParameters')
        print('#')
        print("")

        common.PrefixedParameter.print_parameters(self)

        self.varprint('dilation_iterations', self.dilation_iterations, self.doc['dilation_iterations'])
        self.varprint('iterations', self.iterations, self.doc['iterations'])
        self.varprint('delta_voxel', self.delta_voxel, self.doc['delta_voxel'])
        self.varprint('energy', self.energy, self.doc['energy'])
        self.varprint('smoothing', self.smoothing, self.doc['smoothing'])
        self.varprint('balloon', self.balloon, self.doc['balloon'])
        self.varprint('processors', self.processors, self.doc['processors'])
        self.varprint('mimic_historical_astec', self.mimic_historical_astec, self.doc['mimic_historical_astec'])
        print("")
        return

    def write_parameters_in_file(self, logfile):
        logfile.write("\n")
        logfile.write("# \n")
        logfile.write("# MorphoSnakeParameters\n")
        logfile.write("# \n")
        logfile.write("\n")

        common.PrefixedParameter.write_parameters_in_file(self, logfile)

        self.varwrite(logfile, 'dilation_iterations', self.dilation_iterations, self.doc['dilation_iterations'])
        self.varwrite(logfile, 'iterations', self.iterations, self.doc['iterations'])
        self.varwrite(logfile, 'delta_voxel', self.delta_voxel, self.doc['delta_voxel'])
        self.varwrite(logfile, 'energy', self.energy, self.doc['energy'])
        self.varwrite(logfile, 'smoothing', self.smoothing, self.doc['smoothing'])
        self.varwrite(logfile, 'balloon', self.balloon, self.doc['balloon'])
        self.varwrite(logfile, 'processors', self.processors, self.doc['processors'])
        self.varwrite(logfile, 'mimic_historical_astec', self.mimic_historical_astec,
                      self.doc['mimic_historical_astec'])
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
        self.dilation_iterations = self.read_parameter(parameters, 'dilation_iterations', self.dilation_iterations)
        self.dilation_iterations = self.read_parameter(parameters, 'astec_MorphosnakeIterations', 
                                                       self.dilation_iterations)
        
        self.iterations = self.read_parameter(parameters, 'iterations', self.iterations)
        self.iterations = self.read_parameter(parameters, 'astec_NIterations', self.iterations)
        
        self.delta_voxel = self.read_parameter(parameters, 'delta_voxel', self.delta_voxel)
        self.delta_voxel = self.read_parameter(parameters, 'astec_DeltaVoxels', self.delta_voxel)
        
        self.energy = self.read_parameter(parameters, 'energy', self.energy)
        self.smoothing = self.read_parameter(parameters, 'smoothing', self.smoothing)
        self.balloon = self.read_parameter(parameters, 'balloon', self.balloon)
        
        self.processors = self.read_parameter(parameters, 'processors', self.processors)
        self.processors = self.read_parameter(parameters, 'astec_nb_proc', self.processors)
        
        self.mimic_historical_astec = self.read_parameter(parameters, 'mimic_historical_astec', 
                                                          self.mimic_historical_astec)
        
    def update_from_parameter_file(self, parameter_file):
        if parameter_file is None:
            return
        if not os.path.isfile(parameter_file):
            print("Error: '" + parameter_file + "' is not a valid file. Exiting.")
            sys.exit(1)

        parameters = imp.load_source('*', parameter_file)
        self.update_from_parameters(parameters)


#
#
#
#
#

class AstecParameters(mars.WatershedParameters, MorphoSnakeParameters):

    ############################################################
    #
    # initialisation
    #
    ############################################################

    def __init__(self, prefix="astec_"):

        if "doc" not in self.__dict__:
            self.doc = {}

        doc = "\n"
        doc += "Astec parameters overview:\n"
        doc += "==========================\n"
        doc += "Astec is a segmentation propagation procedure, where\n"
        doc += "images at time t-1 are used to segment the image at time t.\n"
        doc += "1. A first segmentation is done by a seeded watershed, where\n"
        doc += "   the seeds are extracted from the deformed segmentation\n"
        doc += "   image at t-1\n"
        doc += "2. For each cell of this first segmentation, the number of\n"
        doc += "   h-minima, for a range of h values, is studied, to decide\n"
        doc += "   whether a division will occur or not\n"
        doc += "3. This allows to build a new set of seeds, and then, a second\n"
        doc += "   segmentation image.\n"
        doc += "4. A first round of corrections corrects cells that experience\n"
        doc += "   a decrease of volume\n"
        doc += "5. A second round of corrections corrects cells that lose\n"
        doc += "   with respect to the background (morphosnake corrections)\n"
        doc += "   The morphosnakes may over-correct the cells, so this\n"
        doc += "   correction is corrected afterwards\n"
        doc += "\n"
        self.doc['astec_overview'] = doc

        ############################################################
        #
        # initialisation
        #
        ############################################################
        mars.WatershedParameters.__init__(self, prefix=prefix)

        self.seed_reconstruction = reconstruction.ReconstructionParameters(prefix=[self._prefix, "seed_"])
        self.seed_reconstruction.intensity_sigma = 0.6
        self.membrane_reconstruction = reconstruction.ReconstructionParameters(prefix=[self._prefix, "membrane_"])
        self.membrane_reconstruction.intensity_sigma = 0.15
        self.morphosnake_reconstruction = reconstruction.ReconstructionParameters(prefix=[self._prefix, "morphosnake_"])
        self.morphosnake_reconstruction.intensity_sigma = 0.15

        MorphoSnakeParameters.__init__(self, prefix=prefix)

        #
        #
        #
        doc = "\t Possible values are 'seeds_from_previous_segmentation', \n"
        doc += "\t 'seeds_selection_without_correction', or None\n"
        doc += "\t Allows to stop the propagation before all stages\n"
        doc += "\t are completed\n"
        self.doc['propagation_strategy'] = doc
        self.propagation_strategy = None

        #
        # erosion of cell from previous segmentation
        #
        # previous_seg_erosion_cell_iterations: maximum number of erosion iteration for cells
        #   if the cell disappears, less iterations are done
        # previous_seg_erosion_cell_min_size: minimal size of a cell to perform erosion
        #

        # self.previous_seg_method = "erode_then_deform"
        doc = "\t Possible values are 'deform_then_erode' or 'erode_then_deform'\n"
        doc += "\t \n"
        doc += "\t \n"
        doc += "\t \n"
        doc += "\t \n"
        self.doc['previous_seg_method'] = doc
        self.previous_seg_method = "deform_then_erode"

        doc = "\t Number of erosions to get any cell seed at t\n"
        doc += "\t (except for the background) from the deformed\n"
        doc += "\t segmentation at t\n"
        self.doc['previous_seg_erosion_cell_iterations'] = doc
        self.previous_seg_erosion_cell_iterations = 10

        doc = "\t Number of erosions to get the background seed at t\n"
        doc += "\t from the deformed segmentation at t\n"
        self.doc['previous_seg_erosion_background_iterations'] = doc
        self.previous_seg_erosion_background_iterations = 25

        doc = "\t Minimal size of a cell from segmentation at t-1\n"
        doc += "\t to generate a cell.\n"
        doc += "\t Lineage of too small cell then ends at t-1\n"
        self.doc['previous_seg_erosion_cell_min_size'] = doc
        self.previous_seg_erosion_cell_min_size = 1000

        #
        # astec-dedicated watershed parameters
        #
        doc = "\t Low bound of the h range used to compute h-minima\n"
        self.doc['watershed_seed_hmin_min_value'] = doc
        self.watershed_seed_hmin_min_value = 4

        doc = "\t High bound of the h range used to compute h-minima\n"
        self.doc['watershed_seed_hmin_max_value'] = doc
        self.watershed_seed_hmin_max_value = 18

        doc = "\t Step between two successive h values when computing\n"
        doc += "\t h-minima\n"
        self.doc['watershed_seed_hmin_delta_value'] = doc
        self.watershed_seed_hmin_delta_value = 2

        #
        #
        #
        doc = "\t Possible values are True or False\n"
        doc += "\t Default behavior.\n"
        self.doc['background_seed_from_hmin'] = doc
        self.background_seed_from_hmin = True

        doc = "\t Possible values are True or False\n"
        doc += "\t Kept for test purposes.\n"
        self.doc['background_seed_from_previous'] = doc
        self.background_seed_from_previous = False

        #
        # to decide whether there will be division
        #
        doc = "\t Magic threshold to decide whether a cell division\n"
        doc += "\t will occur or not\n"
        self.doc['seed_selection_tau'] = doc
        self.seed_selection_tau = 25

        #
        # threshold
        # cells deformed from previous timepoint that does not have any seed
        # and whose volume (in voxels) is below this threshold are discarded
        # they will correspond to dead-end in the lineage
        #
        doc = "\t Volume threshold\n"
        doc += "\t Too small 'mother' cell without corresponding\n"
        doc += "\t 'daughter' cells will not have a lineage\n"
        self.doc['minimum_volume_unseeded_cell'] = doc
        self.minimum_volume_unseeded_cell = 100

        #
        # magic values for the volume checking
        # - volume_minimal_value is in voxel units
        #
        doc = "\t Volume ratio tolerance\n"
        doc += "\t The volume ratio is computed by (Leo's)\n"
        doc += "\t 1 - volume(cell at t-1) / sum volume(cells at t)\n"
        doc += "\t Admissible cells verify\n"
        doc += "\t - volume_ratio_tolerance <= ratio <= volume_ratio_tolerance\n"
        self.doc['volume_ratio_tolerance'] = doc
        self.volume_ratio_tolerance = 0.1

        doc = "\t Volume ratio threshold\n"
        doc += "\t To determine whether the volume change is too small\n"
        self.doc['volume_ratio_threshold'] = doc
        self.volume_ratio_threshold = 0.5

        doc = "\t Volume threshold\n"
        doc += "\t To determine whether a 'daughter' is large enough\n"
        self.doc['volume_minimal_value'] = doc
        self.volume_minimal_value = 1000

        #
        # outer morphosnake correction
        #
        doc = "\t Possible values are true or False\n"
        self.doc['morphosnake_correction'] = doc
        self.morphosnake_correction = True

        doc = "\t Opening size for the correction of the morphosnake\n"
        doc += "\t correction.\n"
        self.doc['outer_correction_radius_opening'] = doc
        self.outer_correction_radius_opening = 20

        # diagnosis
        doc = "\t Possible values are True or False\n"
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
        print('# AstecParameters')
        print('#')
        print("")

        common.PrefixedParameter.print_parameters(self)

        for line in self.doc['astec_overview'].splitlines():
            print('# ' + line)

        mars.WatershedParameters.print_parameters(self)
        self.seed_reconstruction.print_parameters()
        self.membrane_reconstruction.print_parameters()
        self.morphosnake_reconstruction.print_parameters()
        MorphoSnakeParameters.print_parameters(self)

        self.varprint('propagation_strategy', self.propagation_strategy, self.doc['propagation_strategy'])

        self.varprint('previous_seg_method', self.previous_seg_method, self.doc['previous_seg_method'])
        self.varprint('previous_seg_erosion_cell_iterations', self.previous_seg_erosion_cell_iterations,
                      self.doc['previous_seg_erosion_cell_iterations'])
        self.varprint('previous_seg_erosion_background_iterations', self.previous_seg_erosion_background_iterations,
                      self.doc['previous_seg_erosion_background_iterations'])
        self.varprint('previous_seg_erosion_cell_min_size', self.previous_seg_erosion_cell_min_size,
                      self.doc['previous_seg_erosion_cell_min_size'])

        self.varprint('watershed_seed_hmin_min_value', self.watershed_seed_hmin_min_value,
                      self.doc['watershed_seed_hmin_min_value'])
        self.varprint('watershed_seed_hmin_max_value', self.watershed_seed_hmin_max_value,
                      self.doc['watershed_seed_hmin_max_value'])
        self.varprint('watershed_seed_hmin_delta_value', self.watershed_seed_hmin_delta_value,
                      self.doc['watershed_seed_hmin_delta_value'])

        self.varprint('background_seed_from_hmin', self.background_seed_from_hmin,
                      self.doc['background_seed_from_hmin'])
        self.varprint('background_seed_from_previous', self.background_seed_from_previous,
                      self.doc['background_seed_from_previous'])

        self.varprint('seed_selection_tau', self.seed_selection_tau, self.doc['seed_selection_tau'])

        self.varprint('minimum_volume_unseeded_cell', self.minimum_volume_unseeded_cell,
                      self.doc['minimum_volume_unseeded_cell'])

        self.varprint('volume_ratio_tolerance', self.volume_ratio_tolerance, self.doc['volume_ratio_tolerance'])
        self.varprint('volume_ratio_threshold', self.volume_ratio_threshold, self.doc['volume_ratio_threshold'])
        self.varprint('volume_minimal_value', self.volume_minimal_value, self.doc['volume_minimal_value'])

        self.varprint('morphosnake_correction', self.morphosnake_correction, self.doc['morphosnake_correction'])
        self.varprint('outer_correction_radius_opening', self.outer_correction_radius_opening,
                      self.doc['outer_correction_radius_opening'])

        self.varprint('lineage_diagnosis', self.lineage_diagnosis, self.doc['lineage_diagnosis'])
        print("")

    def write_parameters_in_file(self, logfile):
        logfile.write("\n")
        logfile.write("# \n")
        logfile.write("# AstecParameters\n")
        logfile.write("# \n")
        logfile.write("\n")

        common.PrefixedParameter.write_parameters_in_file(self, logfile)

        for line in self.doc['astec_overview'].splitlines():
            logfile.write('# ' + line + '\n')

        mars.WatershedParameters.write_parameters_in_file(self, logfile)
        self.seed_reconstruction.write_parameters_in_file(logfile)
        self.membrane_reconstruction.write_parameters_in_file(logfile)
        self.morphosnake_reconstruction.write_parameters_in_file(logfile)
        MorphoSnakeParameters.write_parameters_in_file(self, logfile)

        self.varwrite(logfile, 'propagation_strategy', self.propagation_strategy, self.doc['propagation_strategy'])

        self.varwrite(logfile, 'previous_seg_method', self.previous_seg_method, self.doc['previous_seg_method'])
        self.varwrite(logfile, 'previous_seg_erosion_cell_iterations', self.previous_seg_erosion_cell_iterations,
                      self.doc['previous_seg_erosion_cell_iterations'])
        self.varwrite(logfile, 'previous_seg_erosion_background_iterations',
                      self.previous_seg_erosion_background_iterations,
                      self.doc['previous_seg_erosion_background_iterations'])
        self.varwrite(logfile, 'previous_seg_erosion_cell_min_size', self.previous_seg_erosion_cell_min_size,
                      self.doc['previous_seg_erosion_cell_min_size'])

        self.varwrite(logfile, 'watershed_seed_hmin_min_value', self.watershed_seed_hmin_min_value,
                      self.doc['watershed_seed_hmin_min_value'])
        self.varwrite(logfile, 'watershed_seed_hmin_max_value',
                      self.watershed_seed_hmin_max_value, self.doc['watershed_seed_hmin_max_value'])
        self.varwrite(logfile, 'watershed_seed_hmin_delta_value', self.watershed_seed_hmin_delta_value,
                      self.doc['watershed_seed_hmin_delta_value'])

        self.varwrite(logfile, 'background_seed_from_hmin', self.background_seed_from_hmin,
                      self.doc['background_seed_from_hmin'])
        self.varwrite(logfile, 'background_seed_from_previous', self.background_seed_from_previous,
                      self.doc['background_seed_from_previous'])

        self.varwrite(logfile, 'seed_selection_tau', self.seed_selection_tau, self.doc['seed_selection_tau'])

        self.varwrite(logfile, 'minimum_volume_unseeded_cell', self.minimum_volume_unseeded_cell,
                      self.doc['minimum_volume_unseeded_cell'])

        self.varwrite(logfile, 'volume_ratio_tolerance', self.volume_ratio_tolerance,
                      self.doc['volume_ratio_tolerance'])
        self.varwrite(logfile, 'volume_ratio_threshold', self.volume_ratio_threshold,
                      self.doc['volume_ratio_threshold'])
        self.varwrite(logfile, 'volume_minimal_value', self.volume_minimal_value, self.doc['volume_minimal_value'])

        self.varwrite(logfile, 'morphosnake_correction', self.morphosnake_correction,
                      self.doc['morphosnake_correction'])
        self.varwrite(logfile, 'outer_correction_radius_opening', self.outer_correction_radius_opening,
                      self.doc['outer_correction_radius_opening'])

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
        mars.WatershedParameters.update_from_parameters(self, parameters)
        self.seed_reconstruction.update_from_parameters(parameters)
        self.membrane_reconstruction.update_from_parameters(parameters)
        self.morphosnake_reconstruction.update_from_parameters(parameters)
        MorphoSnakeParameters.update_from_parameters(self, parameters)

        self.propagation_strategy = self.read_parameter(parameters, 'propagation_strategy', self.propagation_strategy)

        #
        #
        #
        self.previous_seg_method = self.read_parameter(parameters, 'previous_seg_method', self.previous_seg_method)
        self.previous_seg_erosion_cell_iterations = self.read_parameter(parameters,
                                                                        'previous_seg_erosion_cell_iterations',
                                                                        self.previous_seg_erosion_cell_iterations)
        self.previous_seg_erosion_background_iterations = self.read_parameter(parameters,
                                                                              'previous_seg_erosion_background_iterations',
                                                                              self.previous_seg_erosion_background_iterations)
        self.previous_seg_erosion_cell_min_size = self.read_parameter(parameters, 'previous_seg_erosion_cell_min_size',
                                                                      self.previous_seg_erosion_cell_min_size)

        #
        # watershed
        #

        self.watershed_seed_hmin_min_value = self.read_parameter(parameters, 'watershed_seed_hmin_min_value',
                                                                 self.watershed_seed_hmin_min_value)
        self.watershed_seed_hmin_min_value = self.read_parameter(parameters, 'h_min_min',
                                                                 self.watershed_seed_hmin_min_value)

        self.watershed_seed_hmin_max_value = self.read_parameter(parameters, 'watershed_seed_hmin_max_value',
                                                                 self.watershed_seed_hmin_max_value)
        self.watershed_seed_hmin_max_value = self.read_parameter(parameters, 'h_min_max',
                                                                 self.watershed_seed_hmin_max_value)

        self.watershed_seed_hmin_delta_value = self.read_parameter(parameters, 'watershed_seed_hmin_delta_value',
                                                                   self.watershed_seed_hmin_delta_value)

        #
        #
        #
        self.background_seed_from_hmin = self.read_parameter(parameters, 'background_seed_from_hmin',
                                                             self.background_seed_from_hmin)
        self.background_seed_from_previous = self.read_parameter(parameters, 'background_seed_from_previous',
                                                                 self.background_seed_from_previous)

        #
        # seed selection
        #

        self.seed_selection_tau = self.read_parameter(parameters, 'seed_selection_tau', self.seed_selection_tau)
        self.seed_selection_tau = self.read_parameter(parameters, 'Thau', self.seed_selection_tau)

        self.minimum_volume_unseeded_cell = self.read_parameter(parameters, 'minimum_volume_unseeded_cell',
                                                                self.minimum_volume_unseeded_cell)

        self.volume_ratio_tolerance = self.read_parameter(parameters, 'volume_ratio_tolerance',
                                                          self.volume_ratio_tolerance)

        self.volume_ratio_tolerance = self.read_parameter(parameters, 'volume_ratio_tolerance',
                                                          self.volume_ratio_tolerance)
        self.volume_ratio_tolerance = self.read_parameter(parameters, 'VolumeRatioSmaller',
                                                          self.volume_ratio_tolerance)

        self.volume_ratio_threshold = self.read_parameter(parameters, 'VolumeRatioBigger', self.volume_ratio_threshold)
        self.volume_minimal_value = self.read_parameter(parameters, 'volume_minimal_value', self.volume_minimal_value)
        self.volume_minimal_value = self.read_parameter(parameters, 'MinVolume', self.volume_minimal_value)

        self.morphosnake_correction = self.read_parameter(parameters, 'morphosnake_correction',
                                                          self.morphosnake_correction)
        self.outer_correction_radius_opening = self.read_parameter(parameters, 'outer_correction_radius_opening',
                                                                   self.outer_correction_radius_opening)
        self.outer_correction_radius_opening = self.read_parameter(parameters, 'RadiusOpening',
                                                                   self.outer_correction_radius_opening)

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
    labels = np.unique(seg)

    pool = multiprocessing.Pool(processes=nprocessors)
    mapping = []

    #
    # since the label_image is obtained through the deformation of the segmentation at previous
    # time point, it may contains 0 values
    #
    for i in labels[labels != 0]:
        tmp = seg[bboxes[i - 1]] == i
        size_cell = np.sum(tmp)
        if size_cell > parameters.previous_seg_erosion_cell_min_size:
            if i == 1:
                mapping.append((tmp, parameters.previous_seg_erosion_background_iterations, bboxes[i - 1], i))
            else:
                mapping.append((tmp, parameters.previous_seg_erosion_cell_iterations, bboxes[i - 1], i))
        else:
            monitoring.to_log_and_console('     .. skip cell ' + str(i) + ', volume (' + str(size_cell) + ') <= '
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
    # it occurs when the cell is a nxmxp cube
    #
    np_unique = np.unique(cell_segmentation)
    if len(np_unique) != 2:
        monitoring.to_log_and_console('       .. weird, sub-image of cell ' + str(c) + ' contains '
                                      + str(len(np_unique)) + ' labels = ' + str(np_unique), 2)

    #
    # get the seeds that intersect the cell 'c'
    #
    labels = list(np.unique(seeds_sub_image[cell_segmentation == c]))

    #
    # remove 0 label (correspond to non-minima regions)
    # Note: check whether 0 is inside labels list (?)
    #
    if 0 in labels:
        labels.remove(0)

    nb = len(labels)

    return nb, labels, c


def _cell_based_h_minima(first_segmentation, cells, bounding_boxes, image_for_seed, experiment, parameters,
                         nprocessors=26):
    """
    Computes the seeds (h-minima) for a range of h values
    Seeds are labeled, and only seeds entirely contained in one single cell are kept
    (seeds that superimposed two cells, or more, are rejected).

    :param first_segmentation: watershed based segmentation where the seeds are the cells from the previous,
        eroded and then deformed
    :param cells:
    :param bounding_boxes:
    :param image_for_seed:
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
    wparam = mars.WatershedParameters(obj=parameters)
    wparam.seed_hmin = h_max
    h_min = h_max

    input_image = image_for_seed
    unmasked_seed_image = common.add_suffix(image_for_seed, "_unmasked_seed_h" + str('{:03d}'.format(h_min)),
                                            new_dirname=experiment.astec_dir.get_tmp_directory(),
                                            new_extension=experiment.default_image_suffix)
    seed_image = common.add_suffix(image_for_seed, "_seed_h" + str('{:03d}'.format(h_min)),
                                   new_dirname=experiment.astec_dir.get_tmp_directory(),
                                   new_extension=experiment.default_image_suffix)
    difference_image = common.add_suffix(image_for_seed, "_seed_diff_h" + str('{:03d}'.format(h_min)),
                                         new_dirname=experiment.astec_dir.get_tmp_directory(),
                                         new_extension=experiment.default_image_suffix)

    if not os.path.isfile(seed_image) or not os.path.isfile(difference_image) \
            or monitoring.forceResultsToBeBuilt is True:
        #
        # computation of labeled regional minima
        # -> keeping the 'difference' image allows to speed up the further computation
        #    for smaller values of h
        #
        mars.build_seeds(input_image, difference_image, unmasked_seed_image, experiment, wparam)
        #
        # select only the 'seeds' that are totally included in cells
        #
        cpp_wrapping.mc_mask_seeds(unmasked_seed_image, first_segmentation, seed_image)

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
            parameter_seeds.setdefault(c, []).append([h_min, parameters.seed_reconstruction.intensity_sigma])

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
            wparam.seed_hmin = h_min

            input_image = difference_image
            unmasked_seed_image = common.add_suffix(image_for_seed, "_unmasked_seed_h" + str('{:03d}'.format(h_min)),
                                                    new_dirname=experiment.astec_dir.get_tmp_directory(),
                                                    new_extension=experiment.default_image_suffix)
            seed_image = common.add_suffix(image_for_seed, "_seed_h" + str('{:03d}'.format(h_min)),
                                           new_dirname=experiment.astec_dir.get_tmp_directory(),
                                           new_extension=experiment.default_image_suffix)
            difference_image = common.add_suffix(image_for_seed, "_seed_diff_h" + str('{:03d}'.format(h_min)),
                                                 new_dirname=experiment.astec_dir.get_tmp_directory(),
                                                 new_extension=experiment.default_image_suffix)

            if not os.path.isfile(seed_image) or not os.path.isfile(difference_image) \
                    or monitoring.forceResultsToBeBuilt is True:
                mars.build_seeds(input_image, difference_image, unmasked_seed_image, experiment, wparam,
                                 operation_type='max')
                cpp_wrapping.mc_mask_seeds(unmasked_seed_image, first_segmentation, seed_image)

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
    # nb_2 is $N_2(c)$, but nb_3 is *not* $N_{2^{+}}(c)$ ?
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
    Return the seeds:
    - number of seeds
    - a sub-image containing the labeled seeds, either 'cell_seeds' itself or 'cell_seeds[bb]' if
      'bb' is not None
    from cell_seeds stricly included in cell c from cell_segmentation
    (the labels of the seeds go from 1 to 3)
    :param c: cell label
    :param cell_segmentation: sub-image with 'c' for the cell and '0' for the background
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
    # many seeds, but all with the same label
    # useful for background seeds
    #
    if not individual_seeds:
        seeds[cell_segmentation == 0] = 0
        seeds[seeds > 0] = c
        return 1, seeds.astype(np.uint8)

    #
    # seeds that intersects the cell
    # regional minima/maxima have already been selected so that they are entirely included in cells
    # of
    #
    labels = list(np.unique(seeds[cell_segmentation == c]))
    labels.remove(0)

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
        monitoring.to_log_and_console("       " + proc + ": too many labels, not handled yet", 2)
        return 0, None


#
#
#


def _build_seeds_from_selected_parameters(selected_parameter_seeds,
                                          segmentation_from_previous, seeds_from_previous, selected_seeds,
                                          cells, unseeded_cells, bounding_boxes, image_for_seed,
                                          experiment, parameters):
    """

    :param selected_parameter_seeds:
    :param segmentation_from_previous:
    :param seeds_from_previous:
    :param selected_seeds:
    :param cells:
    :param unseeded_cells:
    :param bounding_boxes:
    :param image_for_seed:
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
            seed_image = common.add_suffix(image_for_seed, "_seed_h" + str('{:03d}'.format(h_min)),
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

        cell_segmentation = np.zeros_like(first_segmentation[bounding_boxes[c]])
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
            if n_seeds == 3:
                monitoring.to_log_and_console('         Warning: only 2 seeds out of 3 are labelled for cell ' + str(c),
                                              3)
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
        del labeled_seeds

    #
    # create background seed
    # 1. create a background cell
    # 2. get the seeds from the read h-minima image with the smallest h
    # 3. add all the seeds (individual_seeds=False)
    #

    monitoring.to_log_and_console('      process background', 3)

    if parameters.background_seed_from_hmin or not parameters.background_seed_from_previous:
        background_cell = np.zeros_like(first_segmentation)
        background_cell[first_segmentation == 1] = 1

        h_min = min(seed_image_list.keys())
        n_seeds, labeled_seeds = _extract_seeds(1, background_cell, seed_image_list[h_min], individual_seeds=False)
        if n_seeds == 0:
            monitoring.to_log_and_console("       " + proc + ": unable to get background seed", 2)
        else:
            new_seed_image[labeled_seeds > 0] = 1
            correspondences[1] = [1]

        del labeled_seeds

    #
    # create seeds for cell with no seed found
    #

    if len(unseeded_cells) > 0 or parameters.background_seed_from_previous:
        first_seeds = imread(seeds_from_previous)

        if parameters.background_seed_from_previous:
            new_seed_image[first_seeds == 1] = 1

        if len(unseeded_cells) > 0:
            monitoring.to_log_and_console('      process cell without children', 3)
            for c in unseeded_cells:
                #
                # first_segmentation is segmentation_from_previous
                #
                vol = np.sum(first_segmentation == c)
                if vol <= parameters.minimum_volume_unseeded_cell:
                    monitoring.to_log_and_console('      .. process cell ' + str(c) + ': volume (' + str(vol) + ') <= '
                                                  + str(parameters.minimum_volume_unseeded_cell) + ', remove cell', 2)
                else:
                    monitoring.to_log_and_console('      .. process cell ' + str(c) + ' -> ' + str(label_max), 3)
                    correspondences[c] = [label_max]
                    new_seed_image[first_seeds == c] = label_max
                    label_max += 1
        else:
            monitoring.to_log_and_console('      no cell without children to be processed', 3)

        del first_seeds
    #
    #
    #

    imsave(selected_seeds, new_seed_image)

    #
    #
    #
    del new_seed_image

    #
    # "for i in seed_image_list:" can be used if the dictionary is not changed
    # else use
    # - either "for i in seed_image_list.keys():"
    # - or "for i in list(seed_image_list):"
    #
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


def _update_volume_properties(lineage_tree_information, segmented_image, current_time, experiment):

    time_digits = experiment.get_time_digits_for_cell_id()
    volumes = _compute_volumes(segmented_image)
    volume_key = properties.keydictionary['volume']['output_key']
    # uncomment next line to have a lineage file readable by old post-correction version
    # volume_key = 'volumes_information'
    if volume_key not in lineage_tree_information:
        lineage_tree_information[volume_key] = {}

    dtmp = {}
    for key, value in volumes.iteritems():
        newkey = current_time * 10 ** time_digits + int(key)
        dtmp[newkey] = value
    lineage_tree_information[volume_key].update(dtmp)
    return lineage_tree_information


def _build_correspondences_from_segmentation(segmented_image):
    volumes = _compute_volumes(segmented_image)
    tmp = {}
    # if the volume exists, it means that this cell has a mother cell with the same label
    for key, value in volumes.iteritems():
        tmp[key] = [int(key)]
    return tmp


def _update_lineage_properties(lineage_tree_information, correspondences, previous_time, current_time, experiment):

    time_digits = experiment.get_time_digits_for_cell_id()
    lineage_key = properties.keydictionary['lineage']['output_key']
    # uncomment next line to have a lineage file readable by old post-correction version
    # lineage_key = 'lin_tree'
    if lineage_key not in lineage_tree_information:
        lineage_tree_information[lineage_key] = {}

    dtmp = {}
    for key, value in correspondences.iteritems():
        newkey = previous_time * 10**time_digits + int(key)
        vtmp = []
        for i in value:
            vtmp.append(current_time * 10**time_digits + int(i))
        dtmp[newkey] = vtmp

    lineage_tree_information[lineage_key].update(dtmp)
    return lineage_tree_information


########################################################################################
#
#
#
########################################################################################


def _volume_diagnosis(prev_volumes, curr_volumes, correspondences, parameters):
    """

    :param prev_volumes:
    :param curr_volumes:
    :param correspondences:
    :param parameters:
    :return:
    """

    proc = "_volume_diagnosis"

    #
    # lists of (parent (cell at t))
    #
    # large_volume_ratio : volume(mother)   << SUM volume(childrens)
    # small_volume_ratio : volume(mother)   >> SUM volume(childrens)
    #
    # lists of [parent (cell at t), children (cell(s) at t+dt)]
    #
    # small_volume_daughter       : volume(children) <  threshold
    #

    very_large_volume_ratio = []
    very_small_volume_ratio = []
    small_volume_ratio = []
    small_volume_daughter = []
    small_volume = []

    all_daughter_label = []

    for mother_c, daughters_c in correspondences.iteritems():
        #
        # skip background
        #
        if mother_c == 1:
            continue

        all_daughter_label.extend(daughters_c)

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

        if parameters.volume_ratio_tolerance >= parameters.volume_ratio_threshold:
            monitoring.to_log_and_console('    ' + proc + ': weird, volume_ratio_tolerance is larger than'
                                          + 'volume_ratio_threshold. This may yield unexpected behavior.')
        #
        # kept from Leo, very weird formula
        # volume_ratio is the opposite of the fraction of volume lose for the
        # volume from previous time compared to current time
        # volume_ratio = 1.0 - vol(mother) / SUM volume(childrens)
        #
        # volume_ratio < 0 => previous volume > current volume
        # volume_ratio > 0 => previous volume < current volume
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
                    small_volume.append(mother_c)
        else:
            #
            # non-admissible ratios
            # default value of parameters.volume_ratio_threshold is 0.5
            #
            if volume_ratio > 0:
                # volume_ratio > 0  <=> volume(mother) < SUM volume(childrens)
                if volume_ratio > parameters.volume_ratio_threshold:
                    very_large_volume_ratio.append(mother_c)
            elif volume_ratio < 0:
                # volume_ratio < 0  <=> volume(mother) > SUM volume(childrens)
                small_volume_ratio.append(mother_c)
                if volume_ratio < -parameters.volume_ratio_threshold:
                    very_small_volume_ratio.append(mother_c)
            else:
                monitoring.to_log_and_console('    ' + proc + ': should not reach this point', 2)
                monitoring.to_log_and_console('    mother cell was ' + str(mother_c), 2)
                monitoring.to_log_and_console('    daughter cell(s) was(ere) ' + str(daughters_c), 2)

    if len(small_volume_ratio) > 0:
        monitoring.to_log_and_console('    .. (mother) cell(s) with large decrease of volume: '
                                      + str(small_volume_ratio), 2)
    if len(very_small_volume_ratio) > 0:
        monitoring.to_log_and_console('    .. (mother) cell(s) with very large decrease of volume: '
                                      + str(very_small_volume_ratio), 2)
    if len(very_large_volume_ratio) > 0:
        monitoring.to_log_and_console('    .. (mother) cell(s) with very large increase of volume: '
                                      + str(very_large_volume_ratio), 2)
    if len(small_volume) > 0:
        monitoring.to_log_and_console('    .. (mother) cell(s) with small daughter(s) (volume < '
                                      + str(parameters.volume_minimal_value) + '): ' + str(small_volume), 2)

    return very_large_volume_ratio, very_small_volume_ratio, small_volume_ratio, small_volume_daughter, all_daughter_label


def _volume_decrease_correction(astec_name, previous_segmentation, segmentation_from_previous,
                                segmentation_from_selection, deformed_seeds, selected_seeds, membrane_image,
                                image_for_seed, correspondences, selected_parameter_seeds, n_seeds, parameter_seeds,
                                bounding_boxes, experiment, parameters):
    """
    :param astec_name: generic name for image file name construction
    :param previous_segmentation: reference segmentation for volume change computation
    :param segmentation_from_previous:
    :param segmentation_from_selection:
    :param deformed_seeds: seeds obtained from the segmentation at a previous time and deformed into the current time
    :param selected_seeds:
    :param membrane_image:
    :param image_for_seed: seed images are named after 'image_for_seed'
    :param correspondences: is a dictionary that gives, for each 'parent' cell (in the segmentation built from previous
    time segmentation) (ie the key), the list of 'children' cells (in the segmentation built from selected seeds)
    :param selected_parameter_seeds:
    :param n_seeds: dictionary, gives, for each parent cell, give the number of seeds for each couple of
    parameters [h-min, sigma]
    :param parameter_seeds: dictionary, for each parent cell, give the list of used parameters [h-min, sigma]
    :param bounding_boxes: bounding boxes defined on segmentation_from_previous (the segmentation obtained with the
        seeds from the previous time point segmentation image)
    :param experiment:
    :param parameters:
    :return:
    """

    proc = "_volume_decrease_correction"

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
    # 1. segmentation at previous time point
    # 2. segmentation obtained at current time point with seed selection
    # volume checking is done by comparing volume of cell at t-1
    # with volume(s) of corresponding cell(s) at t
    #
    prev_seg = imread(previous_segmentation)
    curr_seg = imread(segmentation_from_selection)

    prev_volumes = _compute_volumes(prev_seg)
    curr_volumes = _compute_volumes(curr_seg)

    #
    # check embryo volume
    #
    prev_embryo_volume = prev_seg.size - prev_volumes[1]
    curr_embryo_volume = curr_seg.size - curr_volumes[1]

    volume_ratio = 1.0 - prev_embryo_volume / curr_embryo_volume

    #
    # h-minima have been extracted with respect to segmentation_from_previous
    # image is re-read to get revised seeds if required
    #
    prev_seg = imread(segmentation_from_previous)

    #
    # kept from Leo, very weird formula
    # volume_ratio is the opposite of the fraction of volume lose for the
    # volume from previous time compared to current time
    #
    # volume_ratio < 0 => previous volume > current volume
    # volume_ratio > 0 => previous volume < current volume
    #

    if -parameters.volume_ratio_tolerance <= volume_ratio <= parameters.volume_ratio_tolerance:
        pass
    else:
        if volume_ratio < 0:
            monitoring.to_log_and_console('      .. warning: embryo volume has strongly diminished', 2)
        else:
            monitoring.to_log_and_console('      .. warning: embryo volume has strongly increased', 2)

    #
    # cell volume diagnosis
    # - large_volume_ratio: list of mother cells such that volume(mother) << SUM volume(childrens)
    # - small_volume_ratio: list of mother cells such that volume(mother) >> SUM volume(childrens)
    # - small_volume_daughter: list of [mother_c, daughter_c] such that one of the daughter has
    #                          a too small volume (with its mother not in one of the two above lists)
    # - all_daughter_label: all labels of daughter cells

    output = _volume_diagnosis(prev_volumes, curr_volumes, correspondences, parameters)
    very_large_volume_ratio, very_small_volume_ratio, small_volume_ratio, small_volume_daughter, all_daughter_label = output

    #
    # get the largest used label
    # -> required to attribute new labels
    #
    seed_label_max = max(all_daughter_label)

    #
    # here we look at cells that experiment a large decrease of volume
    # ie vol(mother) >> vol(daughter(s))
    # this is the step (1) of section 2.3.3.6 of L. Guignard thesis
    # [corresponds to the list to_look_at in historical astec code]
    #

    selected_seeds_image = imread(selected_seeds)
    deformed_seeds_image = imread(deformed_seeds)
    change_in_seeds = 0
    # labels_to_be_fused = []

    ############################################################
    #
    # BEGIN: cell with large decrease of volume
    # volume_ratio < 0  <=> volume(mother) > SUM volume(childrens)
    # try to add seeds
    #
    ############################################################
    if len(very_small_volume_ratio) > 0:
        monitoring.to_log_and_console('        process cell(s) with very large decrease of volume', 2)
    elif len(small_volume_daughter) == 0:
        monitoring.to_log_and_console('        .. no correction to be done', 2)

    for mother_c in very_small_volume_ratio:

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

            change_is_done = False

            if nb_final == 1 and s.count(2) != 0:
                #
                # if no division was chosen, try to increase the number of seeds
                # in order to try to increase the size of the reconstructed cells
                #
                # get the h-min image corresponding to the first case (seeds == 2), ie the largest h with (seeds == 2)
                # recall that seeds have already being masked by the 'previous' segmentation image
                #
                # shouldn't we check whether the other seed is "under" the daughters and labeled
                # the seeds as in the following case (nb_final == 1 or nb_final == 2) and (np.array(s) > 2).any())?
                #
                h_min, sigma = parameter_seeds[mother_c][s.index(2)]
                seed_image_name = common.add_suffix(image_for_seed, "_seed_h" + str('{:03d}'.format(h_min)),
                                                    new_dirname=experiment.astec_dir.get_tmp_directory(),
                                                    new_extension=experiment.default_image_suffix)
                #
                # create a sub-image where the cell 'mother_c' has the 'mother_c' value
                # and a background at '0'
                # extract the corresponding seeds from 'seed_image_name'
                #
                # dilate the bounding box as in original astec
                #
                bb = _slices_dilation(bounding_boxes[mother_c], maximum=curr_seg.shape, iterations=2)
                submask_mother_c = np.zeros_like(prev_seg[bb])
                submask_mother_c[prev_seg[bb] == mother_c] = mother_c
                n_found_seeds, labeled_found_seeds = _extract_seeds(mother_c, submask_mother_c, seed_image_name, bb)
                if n_found_seeds == 2 and (curr_seg[bb][labeled_found_seeds != 0] == 0).any():
                    new_correspondences = [seed_label_max+1, seed_label_max+2]
                    monitoring.to_log_and_console('          .. (1)  cell ' + str(mother_c) + ': '
                                                  + str(correspondences[mother_c]) + ' -> ' + str(new_correspondences),
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
                    change_in_seeds += 1
                    change_is_done = True

            if change_is_done is False:
                if (nb_final == 1 or nb_final == 2) and (np.array(s) > 2).any():
                    #
                    # there is a h that gives more than 2 seeds
                    # get the smallest h
                    #
                    h_min, sigma = parameter_seeds[mother_c][-1]
                    if parameters.mimic_historical_astec:
                        seed_image_name = common.add_suffix(image_for_seed,
                                                            "_unmasked_seed_h" + str('{:03d}'.format(h_min)),
                                                            new_dirname=experiment.astec_dir.get_tmp_directory(),
                                                            new_extension=experiment.default_image_suffix)
                    else:
                        seed_image_name = common.add_suffix(image_for_seed, "_seed_h" + str('{:03d}'.format(h_min)),
                                                            new_dirname=experiment.astec_dir.get_tmp_directory(),
                                                            new_extension=experiment.default_image_suffix)
                    #
                    # create a sub-image where cells 'daughter_c' has the 'mother_c' value
                    # and a background at '0'
                    # create a sub-image where the cell 'mother_c' has the 'mother_c' value
                    # and a background at '0'
                    #
                    # built a sub-image where seeds 'below' daughters have a '1' value
                    # and seeds 'below' the projection segmentation have a '2' value
                    #
                    # do something if there are seeds 'below' the projection segmentation has a '2' value
                    # - seeds 'below' daughters (that have a '1' value) will be fused into a 'seed_label_max + 1' cell
                    # - seeds 'below' the projection segmentation (that have a '2' value) will be fused into a
                    #   'seed_label_max + 2' cell
                    #
                    bb = _slices_dilation(bounding_boxes[mother_c], maximum=curr_seg.shape, iterations=2)
                    submask_daughter_c = np.zeros_like(curr_seg[bb])
                    for daughter_c in correspondences[mother_c]:
                        submask_daughter_c[curr_seg[bb] == daughter_c] = mother_c
                    submask_mother_c = np.zeros_like(prev_seg[bb])
                    submask_mother_c[prev_seg[bb] == mother_c] = mother_c
                    aux_seed_image = imread(seed_image_name)
                    seeds_c = np.zeros_like(curr_seg[bb])
                    seeds_c[(aux_seed_image[bb] != 0) & (submask_daughter_c == mother_c)] = 1
                    seeds_c[(aux_seed_image[bb] != 0) & (submask_daughter_c == 0) & (submask_mother_c == mother_c)] = 2
                    del aux_seed_image
                    if 2 in seeds_c:
                        new_correspondences = [seed_label_max + 1, seed_label_max + 2]
                        monitoring.to_log_and_console('          .. (2)  cell ' + str(mother_c) + ': '
                                                      + str(correspondences[mother_c]) + ' -> '
                                                      + str(new_correspondences), 3)
                        if monitoring.debug > 0:
                            monitoring.to_log_and_console('                  seeds from h=' + str(h_min) + ' image')
                            monitoring.to_log_and_console('                  parameter_seeds[mother_c]='
                                                          + str(parameter_seeds[mother_c]))
                            monitoring.to_log_and_console('                  s='+ str(s))
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
                        change_in_seeds += 1
                    else:
                        monitoring.to_log_and_console('          .. (3)  cell ' + str(mother_c) +
                                                      ': does not know what to do', 3)

                elif nb_final == 1:
                    #
                    # here s.count(2) == 0 and np.array(s) > 2).any() is False
                    # replace the computed seed with the seed from the previous segmentation
                    #
                    monitoring.to_log_and_console('          .. (4)  cell ' + str(mother_c) +
                                                  ': get seed from previous eroded segmentation', 3)
                    selected_seeds_image[selected_seeds_image == correspondences[mother_c][0]] = 0
                    selected_seeds_image[deformed_seeds_image == mother_c] = correspondences[mother_c]
                    selected_parameter_seeds[mother_c] = [-1, -1, 1]
                    change_in_seeds += 1
                else:
                    monitoring.to_log_and_console('          .. (5)  cell ' + str(mother_c) +
                                                  ': does not know what to do', 3)

        elif s.count(3) != 0:
            #
            # here we have s.count(1) == and  s.count(2) == 0:
            # get the three seeds, and keep them for further fusion
            #
            h_min, sigma = parameter_seeds[mother_c][s.index(3)]
            seed_image_name = common.add_suffix(image_for_seed, "_seed_h" + str('{:03d}'.format(h_min)),
                                                new_dirname=experiment.astec_dir.get_tmp_directory(),
                                                new_extension=experiment.default_image_suffix)
            #
            # create a sub-image where the cell 'mother_c' has the 'mother_c' value
            # and a background at '0'
            # extract the corresponding seeds from 'seed_image_name'
            #
            bb = _slices_dilation(bounding_boxes[mother_c], maximum=curr_seg.shape, iterations=2)
            submask_mother_c = np.zeros_like(prev_seg[bb])
            submask_mother_c[prev_seg[bb] == mother_c] = mother_c
            n_found_seeds, labeled_found_seeds = _extract_seeds(mother_c, submask_mother_c, seed_image_name, bb,
                                                                accept_3_seeds=True)
            if n_found_seeds == 3:
                new_correspondences = [seed_label_max + 1, seed_label_max + 2, seed_label_max + 3]
                monitoring.to_log_and_console('          .. (7)  cell ' + str(mother_c) + ': '
                                              + str(correspondences[mother_c]) + ' -> ' + str(new_correspondences), 3)
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
                change_in_seeds += 1
                # labels_to_be_fused.append([mother_c, new_correspondences])
            else:
                monitoring.to_log_and_console('          .. (8)  cell ' + str(mother_c) + ': weird, has found '
                                              + str(n_found_seeds) + " seeds instead of 3", 2)
        else:
            if s[0] is 0 and s[-1] is 0:
                monitoring.to_log_and_console('          .. (9a) cell ' + str(mother_c) +
                                              ': no h-minima found, no correction ', 2)
            elif s[0] > 4 and s[-1] > 4:
                monitoring.to_log_and_console('          .. (9b) cell ' + str(mother_c) +
                                              ': too many h-minima found, no correction ', 2)
            else:
                monitoring.to_log_and_console('          .. (9c) cell ' + str(mother_c) +
                                              ': unexpected case, no correction ', 2)

    # if len(labels_to_be_fused) > 0:
    #    monitoring.to_log_and_console('      fusion(s) to be done: ' + str(labels_to_be_fused), 2)

    ############################################################
    #
    # END: cell with large decrease of volume
    #
    ############################################################

    if len(very_large_volume_ratio) > 0:
        monitoring.to_log_and_console('        cell(s) with very large increase of volume are not processed (yet)', 2)
        monitoring.to_log_and_console('          .. ' + str(very_large_volume_ratio), 2)

    ############################################################
    #
    # BEGIN: too small 'daughter' cells
    # recall: the volume ratio between 'mother' and 'daughters' is ok
    # but some 'daughter' cells are too small
    # - remove the too small daughter cell from the seed image
    # - remove it from the correspondence array
    # - remove the mother cell is it has no more daughter
    #
    ############################################################
    if len(small_volume_daughter) > 0:
        monitoring.to_log_and_console('        process cell(s) with small daughters', 2)
        for mother_c, daughter_c in small_volume_daughter:
            selected_seeds_image[selected_seeds_image == daughter_c] = 0
            daughters_c = correspondences[mother_c]
            daughters_c.remove(daughter_c)
            if daughters_c:
                correspondences[mother_c] = daughters_c
            else:
                correspondences.pop(mother_c)
                monitoring.to_log_and_console('          .. cell ' + str(mother_c) + ' will have no lineage', 2)
            change_in_seeds += 1
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

    if change_in_seeds == 0:
        monitoring.to_log_and_console('    .. no changes in seeds, do not recompute segmentation', 2)
        del selected_seeds_image
        return segmentation_from_selection, selected_seeds, correspondences

    #
    # some corrections are to be done
    # 1. save the image of corrected seeds
    # 2. redo a watershed
    #
    monitoring.to_log_and_console('    .. ' + str(change_in_seeds) + ' changes in seeds, recompute segmentation', 2)

    corr_selected_seeds = common.add_suffix(astec_name, '_seeds_from_corrected_selection',
                                            new_dirname=experiment.astec_dir.get_tmp_directory(),
                                            new_extension=experiment.default_image_suffix)
    voxelsize = selected_seeds_image._get_resolution()
    imsave(corr_selected_seeds, SpatialImage(selected_seeds_image, voxelsize=voxelsize).astype(np.uint16))
    del selected_seeds_image

    segmentation_from_corr_selection = common.add_suffix(astec_name, '_watershed_from_corrected_selection',
                                                         new_dirname=experiment.astec_dir.get_tmp_directory(),
                                                         new_extension=experiment.default_image_suffix)
    mars.watershed(corr_selected_seeds, membrane_image, segmentation_from_corr_selection, experiment, parameters)

    #
    # there are labels to be fused if there is a case where 3 seeds have been generated for a mother cell
    # no labels to be fused
    #
    # if not labels_to_be_fused:
    #    return segmentation_from_corr_selection, corr_selected_seeds, correspondences

    #
    # in original ASTEC, the volume_checking() procedure was threefold
    # - 1st correction of cells with decrease of volume
    # - 2nd correction of cells with decrease of volume (with morphosnakes)
    # - fusion of 3 seeds
    #
    # here, only the first part is implemented
    #

    return segmentation_from_corr_selection, corr_selected_seeds, correspondences


########################################################################################
#
# Morphosnake correction for cell whoch have a volume decrease
# due to background infiltration
#
########################################################################################


#
# MorphoSnakes can be found through the github site
# https://github.com/pmneila/morphsnakes
#
# It seems that the syntax has changed since Leo's original work (branch python2)
# MorphoSnakes can also be found in the scikit-image distribution
#

def _morphosnakes(parameters_for_parallelism):

    mother_c, bb, subimage_name, subsegmentation_name, astec_parameters = parameters_for_parallelism
    proc = "_morphosnakes"
    write_images = False

    #
    # dilation of the mother cell from subsegmentation to get the initial curve
    #
    subsegmentation = imread(subsegmentation_name)
    initialization = nd.binary_dilation(subsegmentation == mother_c, iterations=astec_parameters.dilation_iterations)
    if write_images:
        initialization_name = common.add_suffix(subsegmentation_name, '_initialization')
        imsave(initialization_name, SpatialImage(initialization.astype(np.uint8)))

    energy = astec_parameters.energy
    if energy .lower() == 'gradient':
        pass
    elif energy.lower() == 'image':
        pass
    else:
        monitoring.to_log_and_console(str(proc) + ": unknown energy function, switch to image-based")
        energy = 'image'

    if energy.lower() == 'gradient':
        gradient_name = common.add_suffix(subimage_name, '_gradient')
        cpp_wrapping.gradient_norm(subimage_name, gradient_name)
        energy = imread(gradient_name)
        energy = 1. / np.sqrt(1 + 100 * energy)
    else:
        # energy.lower() == 'image':
        subimage = imread(subimage_name)
        subimage_min = float(subimage.min())
        subimage_max = float(subimage.max())
        energy = 1. / np.sqrt(1 + 100 * (subimage.astype(float) - subimage_min)/(subimage_max - subimage_min))
        del subimage

    if write_images:
        energy_name = common.add_suffix(subsegmentation_name, '_energy')
        imsave(energy_name, SpatialImage(energy.astype(np.float32)))

    macwe = morphsnakes.MorphGAC(energy, smoothing=astec_parameters.smoothing, threshold=1,
                                 balloon=astec_parameters.balloon)
    macwe.levelset = initialization
    before = np.ones_like(initialization)

    step = 1
    for i in xrange(0, astec_parameters.iterations, step):
        bbefore = copy.deepcopy(before)
        before = copy.deepcopy(macwe.levelset)
        macwe.run(step)
        # print(str(i) + " condition 1 " + str(np.sum(before != macwe.levelset)))
        # print(str(i) + " condition 2 " + str(np.sum(bbefore != macwe.levelset)))
        if write_images:
            if i > 0 and i % 10 == 0:
                result_name = common.add_suffix(subsegmentation_name, '_step' + str(i))
                imsave(result_name, SpatialImage(macwe.levelset.astype(np.uint8)))
        if np.sum(before != macwe.levelset) < astec_parameters.delta_voxel \
                or np.sum(bbefore != macwe.levelset) < astec_parameters.delta_voxel:
            break

    cell_out = macwe.levelset
    cell_out = cell_out.astype(bool)
    del energy

    if write_images:
        levelset_name = common.add_suffix(subsegmentation_name, '_levelset')
        imsave(levelset_name, SpatialImage(cell_out.astype(np.uint8)))

    return mother_c, bb, cell_out


def _historical_morphosnakes(parameters_for_parallelism):

    mother_c, bb, subimage_name, subsegmentation_name, astec_parameters = parameters_for_parallelism
    proc = "_historical_morphosnakes"
    write_images = False

    #
    # dilation of the mother cell from subsegmentation to get the initial curve
    #
    subsegmentation = imread(subsegmentation_name)
    initialization = nd.binary_erosion(subsegmentation != mother_c, iterations=astec_parameters.dilation_iterations,
                                       border_value=1)
    if write_images:
        initialization_name = common.add_suffix(subsegmentation_name, '_initialization')
        imsave(initialization_name, SpatialImage(initialization.astype(np.uint8)))

    gradient_name = common.add_suffix(subimage_name, '_gradient')
    # cpp_wrapping.gradient_norm(subimage_name, gradient_name,  other_options=" -cont 0 ")
    cpp_wrapping.obsolete_gradient_norm(subimage_name, gradient_name)
    energy = imread(gradient_name)
    energy = 1. / np.sqrt(1 + 100 * energy)

    if write_images:
        energy_name = common.add_suffix(subsegmentation_name, '_energy')
        imsave(energy_name, SpatialImage(energy.astype(np.float32)))

    macwe = morphsnakes.MorphGAC(energy, smoothing=astec_parameters.smoothing, threshold=1,
                                 balloon=astec_parameters.balloon)
    macwe.levelset = initialization
    before = np.ones_like(initialization)

    step = 1
    for i in xrange(0, astec_parameters.iterations, step):
        bbefore = copy.deepcopy(before)
        before = copy.deepcopy(macwe.levelset)
        macwe.step()
        # print(str(i) + " condition 1 " + str(np.sum(before != macwe.levelset)))
        # print(str(i) + " condition 2 " + str(np.sum(bbefore != macwe.levelset)))
        if write_images:
            if i > 0 and i % 10 == 0:
                result_name = common.add_suffix(subsegmentation_name, '_step' + str(i))
                imsave(result_name, SpatialImage(macwe.levelset.astype(np.uint8)))
        if np.sum(before != macwe.levelset) < astec_parameters.delta_voxel \
                or np.sum(bbefore != macwe.levelset) < astec_parameters.delta_voxel:
            break

    cell_out = macwe.levelset
    cell_tmp = nd.binary_fill_holes(cell_out)
    cell_out = (cell_out.astype(np.bool) ^ cell_tmp)
    del energy

    if write_images:
        levelset_name = common.add_suffix(subsegmentation_name, '_levelset')
        imsave(levelset_name, SpatialImage(cell_out.astype(np.uint8)))

    return mother_c, bb, cell_out


def _slices_dilation_iteration(slices, maximum):
    return tuple([slice(max(0, s.start-1), min(s.stop+1, maximum[i])) for i, s in enumerate(slices)])


def _slices_dilation(slices, maximum, iterations=1):
    for i in range(iterations):
        slices = _slices_dilation_iteration(slices, maximum)
    return slices


def _outer_volume_decrease_correction(astec_name, previous_segmentation, deformed_segmentation,
                                      segmentation_from_selection, membrane_image, correspondences,
                                      experiment, parameters):
    """
    :param astec_name: generic name for image file name construction
    :param previous_segmentation: watershed segmentation obtained with segmentation image at previous timepoint
    :param deformed_segmentation: segmentation image at previous timepoint deformed to superimpose current time point
    :param membrane_image:
    :param correspondences: is a dictionary that gives, for each 'parent' cell (in the segmentation built from previous
    time segmentation) (ie the key), the list of 'children' cells (in the segmentation built from selected seeds)
    :param experiment:
    :param parameters:
    :return:
    """

    proc = "_outer_volume_decrease_correction"

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

    #
    # cell volume diagnosis
    # - large_volume_ratio: list of mother cells such that volume(mother) << SUM volume(childrens)
    # - small_volume_ratio: list of mother cells such that volume(mother) >> SUM volume(childrens)
    # - small_volume_daughter: list of [mother_c, daughter_c] such that one of the daughter has
    #                          a too small volume (with its mother not in one of the two above lists)
    # - all_daughter_label: all labels of daughter cells

    output = _volume_diagnosis(prev_volumes, curr_volumes, correspondences, parameters)
    very_large_volume_ratio, very_small_volume_ratio, small_volume_ratio, small_volume_daughter, all_daughter_label = output

    #
    # here we look at cells that experiment a large decrease of volume
    # ie vol(mother) > vol(daughter(s))
    # this is the step (1) of section 2.3.3.6 of L. Guignard thesis
    # [corresponds to the list to_look_at in historical astec code]
    #

    ############################################################
    #
    # BEGIN: cell with large decrease of volume
    # volume_ratio < 0  <=> volume(mother) > SUM volume(childrens)
    # try to add seeds
    #
    ############################################################
    if len(small_volume_ratio) > 0:
        monitoring.to_log_and_console('        process cell(s) with large decrease of volume (morphosnake)', 2)
    else:
        del prev_seg
        del curr_seg
        monitoring.to_log_and_console('        .. no correction to be done', 2)
        return segmentation_from_selection, correspondences, []

    #
    # find cells with a volume decrease due to the background
    # volume decrease is checked wrt the deformed segmentation from previous time point
    #
    del prev_seg
    prev_seg = imread(deformed_segmentation)
    bounding_boxes = nd.find_objects(prev_seg)

    exterior_correction = []
    for mother_c in small_volume_ratio:
        #
        # create a sub-image where cells 'daughter_c' has the 'True' value
        # and a background at 'False'
        # create a sub-image where the cell 'mother_c' has the 'True' value
        # and a background at 'False'
        #
        # get the labels of the missed part (mother cell - all daughter cells)
        # 'labels' is no more a nd-array, just an array
        #

        bb = bounding_boxes[mother_c-1]
        submask_daughter_c = np.zeros_like(curr_seg[bb])
        for daughter_c in correspondences[mother_c]:
            submask_daughter_c[curr_seg[bb] == daughter_c] = mother_c
        submask_daughter_c = (submask_daughter_c == mother_c)
        submask_mother_c = (prev_seg[bb] == mother_c)

        labels = curr_seg[bb][submask_mother_c & (submask_daughter_c == False)]
        # print("labels are " + str(labels))

        #
        # is the main label 1 (the background label)?
        #
        labels_size = {}
        max_size = 0
        label_max = 0
        for v in np.unique(labels):
            labels_size[v] = np.sum(labels == v)
            if max_size < labels_size[v]:
                max_size = labels_size[v]
                label_max = v
        #
        # the main label is 1, and was also in the bounding box of the 'mother' cell
        # in the deformed segmentation from previous time (historical behavior)
        # [we should have check that the background is adjacent to the mother cell]
        #
        if label_max == 1 and 1 in prev_seg[bb]:
            exterior_correction.append(mother_c)

    if len(exterior_correction) == 0:
        del prev_seg
        del curr_seg
        monitoring.to_log_and_console('        .. no cells with large background part', 2)
        return segmentation_from_selection, correspondences, []

    #
    #
    #
    monitoring.to_log_and_console('        .. (mother) cell(s) to be corrected (morphosnake step): '
                                  + str(exterior_correction), 2)

    #
    # parameters for morphosnake
    #
    greylevel_image = imread(membrane_image)

    mapping = []
    for mother_c in exterior_correction:
        #
        # cell will be dilated by parameters.dilation_iterations to get the initial curve
        # add a margin of 5 voxels
        #
        bb = _slices_dilation(bounding_boxes[mother_c-1], maximum=prev_seg.shape,
                              iterations=parameters.dilation_iterations+5)
        #
        # subimages
        #
        subsegmentation = os.path.join(experiment.astec_dir.get_tmp_directory(), "cellsegment_" + str(mother_c) + "."
                                       + experiment.default_image_suffix)
        subimage = os.path.join(experiment.astec_dir.get_tmp_directory(), "cellimage_" + str(mother_c) + "."
                                + experiment.default_image_suffix)
        imsave(subsegmentation, prev_seg[bb])
        imsave(subimage, greylevel_image[bb])
        parameters_for_parallelism = (mother_c, bb, subimage, subsegmentation, parameters)
        mapping.append(parameters_for_parallelism)

    del greylevel_image
    del prev_seg

    #
    #
    #

    pool = multiprocessing.Pool(processes=parameters.processors)
    if parameters.mimic_historical_astec:
        outputs = pool.map(_historical_morphosnakes, mapping)
    else:
        outputs = pool.map(_morphosnakes, mapping)
    pool.close()
    pool.terminate()

    #
    # do the corrections issued from the morphosnake
    # check whether they are effective
    # is there is a correction to be done (cell has gained over the background),
    # then fuse all "daughter" cell
    #
    # For reproductibility, sort the outputs wrt mother cell id
    #
    outputs.sort()
    effective_exterior_correction = []
    for mother_c, bb, cell_out in outputs:
        daughter_c = correspondences[mother_c][0]
        if np.sum(curr_seg[bb] == 1 & cell_out) > 0:
            effective_exterior_correction.append(daughter_c)
            curr_seg[bb][curr_seg[bb] == 1 & cell_out] = daughter_c
            if len(correspondences[mother_c]) > 1:
                monitoring.to_log_and_console('           cell ' + str(mother_c) + ' will no more divide -> '
                                              + str(daughter_c), 2)
                for d in correspondences[mother_c]:
                    curr_seg[bb][curr_seg[bb] == d] = daughter_c
            correspondences[mother_c] = [daughter_c]

    #
    #
    #
    if len(effective_exterior_correction) == 0:
        del curr_seg
        monitoring.to_log_and_console('        .. no effective correction', 2)
        return segmentation_from_selection, correspondences, []

    #
    # there has been some effective corrections
    # save the image
    #
    monitoring.to_log_and_console('        .. cell(s) that have been corrected (morphosnake step): '
                                  + str(effective_exterior_correction), 2)

    segmentation_after_morphosnakes = common.add_suffix(astec_name, '_morphosnakes',
                                                        new_dirname=experiment.astec_dir.get_tmp_directory(),
                                                        new_extension=experiment.default_image_suffix)
    imsave(segmentation_after_morphosnakes, curr_seg)
    del curr_seg
    #
    #
    #
    return segmentation_after_morphosnakes, correspondences, effective_exterior_correction


def _outer_morphosnake_correction(astec_name, input_segmentation, reference_segmentation, output_segmentation,
                                  effective_exterior_correction, experiment, parameters):
    if len(effective_exterior_correction) == 0:
        return

    #
    #
    #
    opening_name = common.add_suffix(astec_name, '_morphosnakes_opening',
                                     new_dirname=experiment.astec_dir.get_tmp_directory(),
                                     new_extension=experiment.default_image_suffix)
    options = "-opening -R " + str(parameters.outer_correction_radius_opening)
    cpp_wrapping.mathematical_morphology(input_segmentation, opening_name, other_options=options, monitoring=monitoring)

    #
    # remove part of cell that are outside the opening,
    # that was already ouside in the reference segmentation (to prevent over-correction)
    #
    segmentation = imread(input_segmentation)
    reference = imread(reference_segmentation)
    opening = imread(opening_name)
    for daughter_c in effective_exterior_correction:
        segmentation[((segmentation == daughter_c) & (reference == 1) & (opening == 1))] = 1

    del opening
    del reference
    imsave(output_segmentation, segmentation)
    del segmentation
    return


########################################################################################
#
# 3-seeds fusion
#
########################################################################################


def _multiple_label_fusion(input_segmentation, output_segmentation, correspondences, labels_to_be_fused):
    #
    # compute bounding boxes with the latest segmentation
    #
    segmentation = imread(input_segmentation)
    cells = list(np.unique(segmentation))
    cells.remove(1)
    bounding_boxes = dict(zip(range(1, max(cells) + 1), nd.find_objects(segmentation)))

    for mother, daughters in labels_to_be_fused:
        if len(daughters) != 3:
            monitoring.to_log_and_console('      .. weird, fusion of ' + str(len(daughters)) + ' labels: '
                                          + str(daughters), 2)
        else:
            monitoring.to_log_and_console('      .. fusion of labels: ' + str(daughters), 2)
        #
        # the bounding boxes have been re-defined with daughter cells
        # get the union of the bounding
        #
        bb = bounding_boxes[daughters[0]]
        for d in daughters:
            bb = tuple([slice(min(a[0].start, a[1].start), max(a[0].stop, a[1].stop)) for a in zip(bb, bounding_boxes[d])])
        #
        # get a sub-image with the labels to be fused (and 0 elsewhere)
        #
        seg = np.zeros_like(segmentation[bb])
        for d in daughters:
            seg[segmentation[bb] == d] = d
        #
        # count the voxels for each label
        #
        volumes = []
        for d in daughters:
            volumes.append(np.sum(seg == d))
        min_label = daughters[np.argmin(volumes)]
        dil_min_label = nd.binary_dilation(seg == min_label)
        #
        # estimate the volume of the common border with i_min_label
        # for the other labels
        #
        shared_volumes = []
        shared_labels = []
        for i in range(1, len(daughters)):
            ilabel = daughters[np.argsort(volumes)[i]]
            shared_labels.append(ilabel)
            tmp = np.zeros_like(seg)
            tmp[seg == ilabel] = ilabel
            tmp[dil_min_label == False] = 0
            shared_volumes.append(np.sum(tmp == ilabel))
        #
        # get the label with the maximum volume of common border
        # and relabel the label with minimal volume
        #
        share_label = shared_labels[np.argmax(shared_volumes)]
        monitoring.to_log_and_console('         merge ' + str(min_label) + ' with ' + str(share_label), 2)
        segmentation[segmentation == min_label] = share_label
        correspondences[mother].remove(min_label)

    imsave(output_segmentation, segmentation)
    return correspondences


########################################################################################
#
#
#
########################################################################################


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
    astec_image = common.find_file(astec_dir, astec_name, file_type='image', callfrom=proc, local_monitoring=None,
                                   verbose=False)

    if astec_image is not None:
        if monitoring.forceResultsToBeBuilt is False:
            monitoring.to_log_and_console('    astec image already existing', 2)
            return
        else:
            monitoring.to_log_and_console('    astec image already existing, but forced', 2)

    astec_image = os.path.join(astec_dir, astec_name + "." + experiment.result_image_suffix)

    #
    # build or read the membrane image to be segmented
    # this is the intensity (elevation) image used for the watershed
    #

    reconstruction.monitoring.copy(monitoring)

    membrane_image = reconstruction.build_reconstructed_image(current_time, experiment,
                                                              parameters.membrane_reconstruction,
                                                              suffix="_membrane", previous_time=previous_time)
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
    monitoring.to_log_and_console('    build seeds from previous segmentation', 2)

    previous_segmentation = experiment.get_segmentation_image(previous_time)
    if previous_segmentation is None:
        monitoring.to_log_and_console("    .. " + proc + ": no segmentation image was found for time "
                                      + str(previous_time), 2)
        return False

    #
    # two ways of getting the seeds from the previous segmentation
    # 1. transform the previous segmentation image then erode the cells
    # 2. erode the cells of the previous segmentation image then transform it
    #    this is the original version
    #
    deformed_seeds = common.add_suffix(astec_name, '_deformed_seeds_from_previous',
                                       new_dirname=experiment.astec_dir.get_tmp_directory(),
                                       new_extension=experiment.default_image_suffix)
    #
    # deformed_segmentation will be needed for morphosnake correction
    # may also be used in reconstruction.py
    #
    deformed_segmentation = common.add_suffix(astec_name, '_deformed_segmentation_from_previous',
                                              new_dirname=experiment.astec_dir.get_tmp_directory(),
                                              new_extension=experiment.default_image_suffix)

    if not os.path.isfile(deformed_segmentation) or monitoring.forceResultsToBeBuilt is True:
        deformation = reconstruction.get_deformation_from_current_to_previous(current_time, experiment,
                                                                              parameters.membrane_reconstruction,
                                                                              previous_time)
        cpp_wrapping.apply_transformation(previous_segmentation, deformed_segmentation, deformation,
                                          interpolation_mode='nearest', monitoring=monitoring)

    if parameters.previous_seg_method.lower() == "deform_then_erode":
        monitoring.to_log_and_console('      transform previous segmentation then erode it', 2)

        if not os.path.isfile(deformed_seeds) or monitoring.forceResultsToBeBuilt is True:
            _build_seeds_from_previous_segmentation(deformed_segmentation, deformed_seeds, parameters)

    elif parameters.previous_seg_method.lower() == "erode_then_deform":
        monitoring.to_log_and_console('      erode previous segmentation then transform it', 2)
        eroded_seeds = common.add_suffix(astec_name, '_eroded_seeds_from_previous',
                                         new_dirname=experiment.astec_dir.get_tmp_directory(),
                                         new_extension=experiment.default_image_suffix)
        if not os.path.isfile(eroded_seeds) or monitoring.forceResultsToBeBuilt is True:
            _build_seeds_from_previous_segmentation(previous_segmentation, eroded_seeds, parameters)

        if not os.path.isfile(deformed_seeds) or monitoring.forceResultsToBeBuilt is True:
            deformation = reconstruction.get_deformation_from_current_to_previous(current_time, experiment,
                                                                                  parameters.membrane_reconstruction,
                                                                                  previous_time)
            cpp_wrapping.apply_transformation(eroded_seeds, deformed_seeds, deformation,
                                              interpolation_mode='nearest', monitoring=monitoring)
    else:
        monitoring.to_log_and_console("    .. " + proc + ": unknown method '" + str(parameters.previous_seg_method)
                                      + "' to build seeds from previous segmentation")
        return False

    #
    # watershed segmentation with seeds extracted from previous segmentation
    # $\tilde{S}_{t+1}$ in Leo's PhD
    #
    monitoring.to_log_and_console('    watershed from previous segmentation', 2)

    if parameters.propagation_strategy is 'seeds_from_previous_segmentation':
        segmentation_from_previous = astec_image
    else:
        segmentation_from_previous = common.add_suffix(astec_name, '_watershed_from_previous',
                                                       new_dirname=experiment.astec_dir.get_tmp_directory(),
                                                       new_extension=experiment.default_image_suffix)

    #
    # original astec: there is no smoothing of the membrane image
    #
    if not os.path.isfile(segmentation_from_previous) or monitoring.forceResultsToBeBuilt is True:
        mars.watershed(deformed_seeds, membrane_image, segmentation_from_previous, experiment, parameters)

    #
    # if the propagation strategy is only to get seeds from the erosion of the previous cells
    # we're done. Update the properties
    #

    if parameters.propagation_strategy is 'seeds_from_previous_segmentation':
        #
        # update volumes and lineage
        #
        lineage_tree_information = _update_volume_properties(lineage_tree_information, astec_image, current_time,
                                                             experiment)

        #
        # update lineage. It is somehow just to check since cells are not supposed to disappeared
        # _erode_cell() performs erosions until the maximal number of iterations is reached
        # or juste before the cell disappears
        #
        correspondences = _build_correspondences_from_segmentation(astec_image)
        lineage_tree_information = _update_lineage_properties(lineage_tree_information, correspondences, previous_time,
                                                              current_time, experiment)
        return lineage_tree_information

    #
    # here parameters.propagation_strategy is not 'seeds_from_previous_segmentation'
    # we continue
    #

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

    #
    # In the original astec version, seeds are computed from the original fusion image
    # thus a 2-bytes encoded image
    #
    if parameters.seed_reconstruction.is_equal(parameters.membrane_reconstruction):
        monitoring.to_log_and_console("    .. seed image is identical to membrane image", 2)
        image_for_seed = membrane_image
    else:
        suffix_glace = None
        if ace.AceParameters.is_equal(parameters.seed_reconstruction, parameters.membrane_reconstruction) is True:
            suffix_glace = "_membrane"
        image_for_seed = reconstruction.build_reconstructed_image(current_time, experiment,
                                                                  parameters.seed_reconstruction, suffix="_seed",
                                                                  suffix_glace=suffix_glace,
                                                                  previous_time=previous_time)
    # seed_image = experiment.fusion_dir.get_image_name(current_time)
    # seed_image = common.find_file(experiment.fusion_dir.get_directory(), seed_image, file_type='image',
    # callfrom=proc, local_monitoring=None, verbose=False)
    if image_for_seed is None:
        monitoring.to_log_and_console("    .. " + proc + " no fused image was found for time " + str(current_time), 2)
        return None

    #
    # seed images will be named after 'image_for_seed'
    # seeds are masked by 'segmentation_from_previous'
    # ie the segmentation obtained with the seeds from the previous time point
    #
    monitoring.to_log_and_console('    estimation of h-minima for h in ['
                                  + str(parameters.watershed_seed_hmin_min_value) + ','
                                  + str(parameters.watershed_seed_hmin_max_value) + ']', 2)
    n_seeds, parameter_seeds = _cell_based_h_minima(segmentation_from_previous, cells, bounding_boxes, image_for_seed,
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
        monitoring.to_log_and_console('    .. cells at time ' + str(previous_time) + ' with no seeds: ' + string, 2)
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
        monitoring.to_log_and_console('    .. cells at time ' + str(previous_time) + ' that will divide: ' + string, 2)

    #
    # rebuild an image of seeds with selected parameters h
    #
    monitoring.to_log_and_console('    build seed image from all h-minima images', 2)

    selected_seeds = common.add_suffix(astec_name, '_seeds_from_selection',
                                       new_dirname=experiment.astec_dir.get_tmp_directory(),
                                       new_extension=experiment.default_image_suffix)

    output = _build_seeds_from_selected_parameters(selected_parameter_seeds, segmentation_from_previous, deformed_seeds,
                                                   selected_seeds, cells, unseeded_cells, bounding_boxes,
                                                   image_for_seed, experiment, parameters)

    label_max, correspondences, divided_cells = output

    #
    # second watershed segmentation (with the selected seeds)
    #

    monitoring.to_log_and_console('    watershed from selection of seeds', 2)

    if parameters.propagation_strategy is 'seeds_selection_without_correction':
        segmentation_from_selection = astec_image
    else:
        segmentation_from_selection = common.add_suffix(astec_name, '_watershed_from_selection',
                                                        new_dirname=experiment.astec_dir.get_tmp_directory(),
                                                        new_extension=experiment.default_image_suffix)

    if not os.path.isfile(segmentation_from_selection) or monitoring.forceResultsToBeBuilt is True:
        mars.watershed(selected_seeds, membrane_image, segmentation_from_selection, experiment, parameters)

    #
    # if the propagation strategy is to get this segmentation without corrections
    # we're done. Update the properties
    #

    if parameters.propagation_strategy is 'seeds_selection_without_correction':
        #
        # update volumes and lineage
        #
        lineage_tree_information = _update_volume_properties(lineage_tree_information, astec_image, current_time,
                                                             experiment)

        #
        # update lineage.
        #
        lineage_tree_information = _update_lineage_properties(lineage_tree_information, correspondences, previous_time,
                                                              current_time, experiment)
        return lineage_tree_information

    #
    # Here, we have a first segmentation
    # let correct it if required
    #

    # print(proc + " :correspondences: " + str(correspondences))
    # print(proc + " :selected_parameter_seeds: " + str(selected_parameter_seeds))
    # print("n_seeds: " + str(n_seeds))
    # print("parameter_seeds: " + str(parameter_seeds))

    monitoring.to_log_and_console('    volume decrease correction', 2)
    #
    # reference segmentation for volume decrease checking is
    # the segmentation at the previous time point
    #
    output = _volume_decrease_correction(astec_name, previous_segmentation, segmentation_from_previous,
                                         segmentation_from_selection, deformed_seeds, selected_seeds, membrane_image,
                                         image_for_seed, correspondences, selected_parameter_seeds, n_seeds,
                                         parameter_seeds, bounding_boxes, experiment, parameters)
    segmentation_from_selection, selected_seeds, correspondences = output

    #
    # - segmentation_from_selection: new segmentation image (if any correction)
    # - selected_seeds: new seeds image (if any correction)
    # - correspondences: new lineage correspondences (if any correction)
    #

    input_segmentation = segmentation_from_selection

    #
    # multiple label processing
    #

    labels_to_be_fused = []
    for key, value in correspondences.iteritems():
        if len(value) >= 3:
            labels_to_be_fused.append([key, value])
    if len(labels_to_be_fused) > 0:
        monitoring.to_log_and_console('    3-seeds fusion', 2)
        monitoring.to_log_and_console('      seeds to be fused: ' + str(labels_to_be_fused), 2)
        #
        # bounding boxes have been defined with the watershed obtained with seeds issued
        # from the previous segmentation
        # borders may have changed, so recompute the bounding boxes
        #
        output_segmentation = common.add_suffix(astec_name, '_watershed_after_seeds_fusion',
                                                new_dirname=experiment.astec_dir.get_tmp_directory(),
                                                new_extension=experiment.default_image_suffix)
        correspondences = _multiple_label_fusion(input_segmentation, output_segmentation, correspondences,
                                                 labels_to_be_fused)
        input_segmentation = output_segmentation

    #
    # Morphosnakes
    #
    # keep the current segmentation as a reference
    # - morphosnakes are supposed to gain volume over the background
    # -> morphosnakes have to be corrected if there is a gain
    # -> the correction should not result in a loss
    #
    reference_segmentation = input_segmentation

    if parameters.morphosnake_correction is True:
        monitoring.to_log_and_console('    outer volume decrease correction (morphosnakes)', 2)
        #
        # reference segmentation for morphosnakes is
        # the segmentation computed from previous time point seeds
        # segmentation_from_previous
        #
        if parameters.morphosnake_reconstruction.is_equal(parameters.membrane_reconstruction):
            monitoring.to_log_and_console("    .. morphosnake image is identical to membrane image", 2)
            image_for_morphosnake = membrane_image
        elif parameters.morphosnake_reconstruction.is_equal(parameters.seed_reconstruction):
            monitoring.to_log_and_console("    .. morphosnake image is identical to seed image", 2)
            image_for_morphosnake = image_for_seed
        else:
            suffix_glace = None
            if ace.AceParameters.is_equal(parameters.morphosnake_reconstruction, parameters.membrane_reconstruction) is True:
                suffix_glace = "_membrane"
            elif ace.AceParameters.is_equal(parameters.morphosnake_reconstruction, parameters.seed_reconstruction) is True:
                suffix_glace = "_seed"
            image_for_morphosnake = reconstruction.build_reconstructed_image(current_time, experiment,
                                                                             parameters.morphosnake_reconstruction,
                                                                             suffix="_morphosnake",
                                                                             suffix_glace=suffix_glace,
                                                                             previous_time=previous_time)

        output = _outer_volume_decrease_correction(astec_name, previous_segmentation, deformed_segmentation,
                                                   input_segmentation, image_for_morphosnake, correspondences,
                                                   experiment, parameters)
        output_segmentation, correspondences, effective_exterior_correction = output
        input_segmentation = output_segmentation
    else:
        effective_exterior_correction = []

    #
    # Outer correction (to correct morphosnake)
    #
    if len(effective_exterior_correction) > 0:
        monitoring.to_log_and_console('    outer morphosnake correction', 2)
        output_segmentation = common.add_suffix(astec_name, '_morphosnakes_correction',
                                                new_dirname=experiment.astec_dir.get_tmp_directory(),
                                                new_extension=experiment.default_image_suffix)
        _outer_morphosnake_correction(astec_name, input_segmentation, reference_segmentation, output_segmentation,
                                      effective_exterior_correction, experiment, parameters)
        input_segmentation = output_segmentation

    #
    # copy the last segmentation image (in the auxiliary directory) as the result
    #
    shutil.copyfile(input_segmentation, astec_image)

    #
    # update volumes and lineage
    #
    lineage_tree_information = _update_volume_properties(lineage_tree_information, astec_image,
                                                         current_time, experiment)
    lineage_tree_information = _update_lineage_properties(lineage_tree_information, correspondences, previous_time,
                                                          current_time, experiment)

    return lineage_tree_information


########################################################################################
#
#
#
########################################################################################

#
# check whether a lineage file exists
# loops over the time points
#

def _get_last_time_from_lineage(lineage_tree_information, first_time_point, delta_time_point=1,
                                time_digits_for_cell_id=4):
    if lineage_tree_information == {}:
        return first_time_point
    if len(lineage_tree_information) > 0 \
            and properties.keydictionary['lineage']['output_key'] in lineage_tree_information:
        monitoring.to_log_and_console("    .. test existing lineage tree", 1)
        cellinlineage = {}
        div = 10**time_digits_for_cell_id
        #
        # time points for 'parent cell' are marked into cellinlineage
        #
        for c in lineage_tree_information[properties.keydictionary['lineage']['output_key']]:
            t = int(c)/div
            if t not in cellinlineage:
                cellinlineage[t] = 1
            else:
                cellinlineage[t] += 1
        first_time = first_time_point
        #
        # if a time point is present in cellinlineage, it means that the 'daughter cell' image has been
        # segmented
        #
        while True:
            if first_time in cellinlineage:
                first_time += delta_time_point
            else:
                return first_time

    return first_time_point


def _get_last_time_from_images(experiment, first_time_point, delta_time_point=1):
    current_time = first_time_point + delta_time_point
    while True:
        segmentation_file = experiment.get_segmentation_image(current_time, verbose=False)
        if segmentation_file is None or not os.path.isfile(segmentation_file):
            return current_time - delta_time_point
        current_time += delta_time_point


def _clean_lineage(lineage_tree_information, first_time_point, time_digits_for_cell_id=4):
    #
    # remove information for time points after the first_time_point
    #
    proc = "_clean_lineage"
    if lineage_tree_information == {}:
        return
    mul = 10 ** time_digits_for_cell_id
    for key in lineage_tree_information:
        if key == properties.keydictionary['lineage']['output_key']:
            tmp = copy.deepcopy(lineage_tree_information[properties.keydictionary['lineage']['output_key']])
            for k in tmp:
                if int(k) > first_time_point * mul:
                    del lineage_tree_information[properties.keydictionary['lineage']['output_key']][k]
        elif key == properties.keydictionary['volume']['output_key']:
            tmp = copy.deepcopy(lineage_tree_information[properties.keydictionary['volume']['output_key']])
            for k in tmp:
                if int(k) > (first_time_point + 1) * mul:
                    del lineage_tree_information[properties.keydictionary['volume']['output_key']][k]
        else:
            monitoring.to_log_and_console(str(proc) + ": unhandled key '" + str(key) + "'")
    return


def _clean_images(experiment, first_time_point, last_time_point, delta_time_point=1):
    #
    # rename images after the first_time_point
    #
    for current_time in range(first_time_point, last_time_point + 1, experiment.delta_time_point):
        segmentation_file = experiment.get_segmentation_image(current_time, verbose=False)
        if segmentation_file is not None and os.path.isfile(segmentation_file):
            print("rename " + str(segmentation_file) + " into " + segmentation_file + ".bak")
            shutil.move(segmentation_file, segmentation_file + ".bak")
            current_time += delta_time_point
    return


def _fill_volumes(lineage_tree_information, first_time_point, experiment):
    proc = "_fill_volumes"
    mul = 10 ** experiment.get_time_digits_for_cell_id()
    if lineage_tree_information != {}:
        if properties.keydictionary['volume']['output_key'] in lineage_tree_information:
            tmp = lineage_tree_information[properties.keydictionary['volume']['output_key']]
            for k in tmp:
                if k / mul == first_time_point:
                    monitoring.to_log_and_console("    .. cell volumes found for time #" + str(first_time_point), 1)
                    return lineage_tree_information
    #
    # no key found for first time point
    #
    first_segmentation = experiment.get_segmentation_image(first_time_point)
    if first_segmentation is None:
        monitoring.to_log_and_console(".. " + proc + ": no segmentation image found for time '" + str(first_time_point)
                                      + "'", 1)
        monitoring.to_log_and_console("\t Exiting", 1)
        sys.exit(1)
    monitoring.to_log_and_console("    .. computes cell volumes for time #" + str(first_time_point), 1)
    return _update_volume_properties(lineage_tree_information, first_segmentation, first_time_point, experiment)


def astec_control(experiment, parameters):
    """

    :param experiment:
    :param parameters:
    :return:
    """

    proc = "astec_control"
    _trace_ = True

    #
    # parameter type checking
    #

    monitoring.to_log_and_console("", 1)

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
    properties.monitoring.copy(monitoring)
    reconstruction.monitoring.copy(monitoring)

    #
    # make sure that the result directory exists
    #

    experiment.astec_dir.make_directory()
    segmentation_dir = experiment.astec_dir.get_directory()

    #
    # re-read the lineage file, if any
    # find the restart time point, it is the minimum among
    # - the one issued from the lineage
    # - the one issued from the segmentation images
    # - the one from the parameters
    #
    lineage_tree_file = common.find_file(segmentation_dir, experiment.astec_dir.get_file_name("_lineage"),
                                         file_type='lineage', callfrom=proc, verbose=False)

    if lineage_tree_file is not None and os.path.isfile(os.path.join(segmentation_dir, lineage_tree_file)):
        lineage_tree_path = os.path.join(segmentation_dir, lineage_tree_file)
        lineage_tree_information = properties.read_dictionary(lineage_tree_path)
    else:
        lineage_tree_path = os.path.join(segmentation_dir, experiment.astec_dir.get_file_name("_lineage") + "."
                                         + experiment.result_lineage_suffix)
        lineage_tree_information = {}

    # print(str(lineage_tree_information))

    #
    # check what is the last time point in lineage and in images
    #
    first_time_point = experiment.first_time_point + experiment.delay_time_point
    last_time_point = experiment.last_time_point + experiment.delay_time_point
    time_digits_for_cell_id = experiment.get_time_digits_for_cell_id()

    last_time_lineage = _get_last_time_from_lineage(lineage_tree_information, first_time_point,
                                                    delta_time_point=experiment.delta_time_point,
                                                    time_digits_for_cell_id=time_digits_for_cell_id)
    last_time_images = _get_last_time_from_images(experiment, first_time_point,
                                                  delta_time_point=experiment.delta_time_point)

    #
    # restart is the next time point to be computed
    #
    restart = min(last_time_lineage, last_time_images) + experiment.delta_time_point
    if type(experiment.restart_time_point) == int:
        if experiment.restart_time_point > first_time_point:
            restart = min(restart, experiment.restart_time_point)
    if restart <= last_time_point:
        monitoring.to_log_and_console(".. " + proc + ": start computation at time #" + str(restart), 1)
    else:
        monitoring.to_log_and_console(".. " + proc + ": nothing to do", 1)
    #
    # do some cleaning
    # - the lineage tree: remove lineage information for time points from restart - delta_time_point
    # - the segmentation images
    #
    delta_time_point = experiment.delta_time_point
    if restart <= last_time_point:
        monitoring.to_log_and_console("    .. clean lineage and segmentation directory", 1)
        _clean_lineage(lineage_tree_information, restart - delta_time_point,
                       time_digits_for_cell_id=time_digits_for_cell_id)
        _clean_images(experiment, restart, last_time_point, delta_time_point=delta_time_point)

    #
    # compute volumes for the previous time if required
    # (well, it may exist, if we just restart the segmentation)
    #
    if restart <= last_time_point:
        lineage_tree_information = _fill_volumes(lineage_tree_information, restart - delta_time_point, experiment)

    #
    #
    #
    for current_time in range(restart, last_time_point + 1, experiment.delta_time_point):

        acquisition_time = experiment.get_time_index(current_time)
        previous_time = current_time - experiment.delta_time_point

        #
        # start processing
        #

        monitoring.to_log_and_console('... astec processing of time #' + acquisition_time, 1)
        start_time = time.time()

        #
        # set and make temporary directory
        #
        experiment.astec_dir.set_tmp_directory(current_time)
        experiment.astec_dir.make_tmp_directory()

        if parameters.seed_reconstruction.keep_reconstruction is False \
                and parameters.membrane_reconstruction.keep_reconstruction is False \
                and parameters.morphosnake_reconstruction.keep_reconstruction is False:
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
        # save lineage here
        # thus, we have intermediary pkl in case of future failure
        #

        properties.write_dictionary(lineage_tree_path, lineage_tree_information)

        #
        # end processing for a time point
        #
        end_time = time.time()

        monitoring.to_log_and_console('    computation time = ' + str(end_time - start_time) + ' s', 1)
        monitoring.to_log_and_console('', 1)

    if parameters.lineage_diagnosis:
        monitoring.to_log_and_console("    .. test lineage", 1)
        properties.check_volume_lineage(lineage_tree_information, time_digits_for_cell_id=time_digits_for_cell_id)

    return
