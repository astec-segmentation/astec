
import os
import imp
import sys
import time
import numpy as np
import shutil
from scipy import ndimage as nd

import common

import CommunFunctions.cpp_wrapping as cpp_wrapping
from CommunFunctions.ImageHandling import SpatialImage, imread, imsave


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


class ManualCorrectionParameters(common.PrefixedParameter):

    ############################################################
    #
    # initialisation
    #
    ############################################################

    def __init__(self, prefix=None):

        common.PrefixedParameter.__init__(self, prefix=prefix)

        if "doc" not in self.__dict__:
            self.doc = {}

        doc = "\n"
        doc += "Manual correction overview:\n"
        doc += "===========================\n"
        doc += "Fuses labels from the 'mars' segmentation to correct\n"
        doc += "over-segmentation errors.\n"
        doc += "\n"
        self.doc['manualcorrection_overview'] = doc
        #
        #
        #
        doc = "\t first time point of the series to be processed with\n"
        doc += "\t manual correction (in case of a range of image is to\n"
        doc += "\t be processed).\n"
        doc += "\t Default is that only the time point defined par the 'begin'.\n"
        doc += "\t variable is processed.\n"
        self.doc['first_time_point'] = doc
        self.first_time_point = -1

        doc = "\t last time point of the series to be processed with\n"
        doc += "\t manual correction (in case of a range of image is to\n"
        doc += "\t be processed).\n"
        self.doc['last_time_point'] = doc
        self.last_time_point = -1

        #
        #
        #
        doc = "\t Input image (if both input and output image name is given\n"
        doc += "\t to the command line interface).\n"
        self.doc['input_image'] = doc
        self.input_image = None

        doc = "\t Output image (if both input and output image name is given\n"
        doc += "\t to the command line interface).\n"
        self.doc['output_image'] = doc
        self.output_image = None

        doc = "\t File containing the labels to be fused.\n"
        doc += "\t The syntax of this file is:\n"
        doc += "\t - 1 line per label association, eg\n"
        doc += "\t   '8 7'\n"
        doc += "\t - background label has value 1\n"
        doc += "\t - the character '#' denotes commented lines \n"
        self.doc['mapping_file'] = doc
        self.mapping_file = None

        #
        #
        #
        doc = "\t Number of smallest cells displayed in diagnosis\n"
        self.doc['smallest_cells'] = doc
        self.smallest_cells = 8
        doc = "\t Number of largest cells displayed in diagnosis\n"
        self.doc['largest_cells'] = doc
        self.largest_cells = 8

    ############################################################
    #
    # print / write
    #
    ############################################################

    def print_parameters(self):
        print('')
        print('#')
        print('# ManualCorrectionParameters ')
        print('#')
        print('')

        common.PrefixedParameter.print_parameters(self)

        for line in self.doc['manualcorrection_overview'].splitlines():
            print('# ' + line)

        self.varprint('first_time_point', self.first_time_point, self.doc['first_time_point'])
        self.varprint('last_time_point', self.last_time_point, self.doc['last_time_point'])

        self.varprint('input_image', self.input_image, self.doc['input_image'])
        self.varprint('output_image', self.output_image, self.doc['output_image'])
        self.varprint('mapping_file', self.mapping_file, self.doc['mapping_file'])

        self.varprint('smallest_cells', self.smallest_cells, self.doc['smallest_cells'])
        self.varprint('largest_cells', self.largest_cells, self.doc['largest_cells'])

        print("")

    def write_parameters_in_file(self, logfile):
        logfile.write('\n')
        logfile.write('#' + '\n')
        logfile.write('# ManualCorrectionParameters ' + '\n')
        logfile.write('#' + '\n')
        logfile.write('\n')

        common.PrefixedParameter.write_parameters_in_file(self, logfile)

        for line in self.doc['manualcorrection_overview'].splitlines():
            logfile.write('# ' + line + '\n')

        self.varwrite(logfile, 'first_time_point', self.first_time_point, self.doc['first_time_point'])
        self.varwrite(logfile, 'last_time_point', self.last_time_point, self.doc['last_time_point'])

        self.varwrite(logfile, 'input_image', self.input_image, self.doc['input_image'])
        self.varwrite(logfile, 'output_image', self.output_image, self.doc['output_image'])
        self.varwrite(logfile, 'mapping_file', self.mapping_file, self.doc['mapping_file'])

        self.varwrite(logfile, 'smallest_cells', self.smallest_cells, self.doc['smallest_cells'])
        self.varwrite(logfile, 'largest_cells', self.largest_cells, self.doc['largest_cells'])

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

    def update_from_args(self, args):
        self.input_image = args.input_image
        self.output_image = args.output_image
        self.mapping_file = args.mapping_file
        if int(args.smallest_cells) >= 0:
            self.smallest_cells = args.smallest_cells
        if int(args.largest_cells) >= 0:
            self.largest_cells = args.largest_cells

    def update_from_parameters(self, parameters):

        #
        #
        #
        if hasattr(parameters, 'first_time_point'):
            self.first_time_point = parameters.first_time_point
        if hasattr(parameters, 'mars_begin'):
            self.first_time_point = parameters.mars_begin
        if hasattr(parameters, 'last_time_point'):
            self.last_time_point = parameters.last_time_point
        if hasattr(parameters, 'mars_end'):
            self.last_time_point = parameters.mars_end

        #
        #
        #
        if hasattr(parameters, 'mancor_input_seg_file'):
            if parameters.mancor_input_seg_file is not None and len(str(parameters.mancor_input_seg_file)) > 0:
                self.input_image = parameters.mancor_input_seg_file

        #
        # for back-compatibility
        #
        if hasattr(parameters, 'mancor_seg_file'):
            if parameters.mancor_seg_file is not None and len(str(parameters.mancor_seg_file)) > 0:
                self.input_image = parameters.mancor_seg_file

        if hasattr(parameters, 'mancor_output_seg_file'):
            if parameters.mancor_output_seg_file is not None and len(str(parameters.mancor_output_seg_file)) > 0:
                self.output_image = parameters.mancor_output_seg_file

        if hasattr(parameters, 'mapping_file'):
            if parameters.mapping_file is not None and len(str(parameters.mapping_file)) > 0:
                self.mapping_file = parameters.mapping_file
        if hasattr(parameters, 'mancor_mapping_file'):
            if parameters.mancor_mapping_file is not None and len(str(parameters.mancor_mapping_file)) > 0:
                self.mapping_file = parameters.mancor_mapping_file

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

def correction_process(input_image, output_image, parameters):
    """

    :param input_image:
    :param output_image:
    :param parameters:
    :return:
    """

    proc = "correction_process"

    if monitoring.debug > 2:
        print ""
        print proc + " was called with:"
        print "- input_image = " + str(input_image)
        print "- output_image = " + str(output_image)
        print ""

    #
    # nothing to do if the corrected image exists
    #

    if os.path.isfile(output_image):
        if monitoring.forceResultsToBeBuilt is False:
            monitoring.to_log_and_console('    corrected image already existing', 2)
            return
        else:
            monitoring.to_log_and_console('    corrected image already existing, but forced', 2)

    #
    #
    #
    if parameters.mapping_file is not None and len(str(parameters.mapping_file)) > 0 \
            and os.path.isfile(parameters.mapping_file):

        #
        # corrections to be done
        #
        cpp_wrapping.mc_seed_edit(input_image, output_image, parameters.mapping_file, None)

    else:
        shutil.copy2(input_image, output_image)
        monitoring.to_log_and_console('    no corrections to be done (no valid mapping file)', 2)

    #
    # get
    # - the list of cell label
    # - the list of cell volume
    #
    # build a dictionary and sort it (increasing order) wrt the volume

    im = imread(output_image)
    voxelsize = im._get_resolution()
    vol = voxelsize[0] * voxelsize[1] * voxelsize[2]

    cell_label = np.unique(im)
    cell_volume = nd.sum(np.ones_like(im), im, index=np.int16(cell_label))

    d = dict(zip(cell_label, cell_volume))
    s = sorted(d, key=d.__getitem__)

    monitoring.to_log_and_console('    Number of cells: ' + str(len(cell_label)), 0)
    monitoring.to_log_and_console('    Maximal label: ' + str(np.max(im)), 0)
    # monitoring.to_log_and_console('    Cell ids: ' + str(cell_label), 0)

    monitoring.to_log_and_console('    Sorted cell volumes: ', 0)
    monitoring.to_log_and_console('      Id :    voxels          (um^3)', 0)

    if (int(parameters.smallest_cells) <= 0 and int(parameters.largest_cells) <= 0) \
            or (int(parameters.smallest_cells) + int(parameters.largest_cells) >= len(s)):
        for l in s:
            monitoring.to_log_and_console('    {:>4d} : {:>9d} {:>15s}'.format(l, int(d[l]),
                                                                               '({:.2f})'.format(d[l]*vol)), 0)
    else:
        if int(parameters.smallest_cells) > 0:
            for l in s[:int(parameters.smallest_cells)]:
                monitoring.to_log_and_console('    {:>4d} : {:>9d} {:>15s}'.format(l, int(d[l]),
                                                                                   '({:.2f})'.format(d[l]*vol)), 0)
        monitoring.to_log_and_console('       ...', 0)
        if int(parameters.largest_cells) > 0:
            for l in s[-int(parameters.largest_cells):]:
                monitoring.to_log_and_console('    {:>4d} : {:>9d} {:>15s}'.format(l, int(d[l]),
                                                                                   '({:.2f})'.format(d[l]*vol)), 0)

    return

#
#
#
#
#


def correction_control(experiment, parameters):
    """

    :param experiment:
    :param parameters:
    :return:
    """

    proc = "correction_control"

    #
    # parameter type checking
    #

    if not isinstance(experiment, common.Experiment):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'experiment' variable: "
                                      + str(type(experiment)))
        sys.exit(1)

    if not isinstance(parameters, ManualCorrectionParameters):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'parameters' variable: "
                                      + str(type(parameters)))
        sys.exit(1)

    #
    # make sure that the result directory exists
    #

    experiment.astec_dir.make_directory()
    monitoring.to_log_and_console('', 1)

    #
    # case #1:
    # - input and/or output images are given by the user
    # case #2
    # - input image is the mars image and output image is the corresponding segmentation image
    #

    if (parameters.input_image is not None and len(str(parameters.input_image)) > 0) \
            or (parameters.output_image is not None and len(str(parameters.output_image)) > 0):

        #
        # input image
        #
        if parameters.input_image is not None and len(str(parameters.input_image)) > 0:
            if not os.path.isfile(parameters.input_image):
                monitoring.to_log_and_console("       '" + str(parameters.input_image).split(os.path.sep)[-1]
                                              + "' does not exist", 2)
                monitoring.to_log_and_console("\t Exiting.")
                sys.exit(1)
            input_image = parameters.input_image
        else:
            mars_dir = experiment.mars_dir.get_directory(0)
            mars_name = experiment.mars_dir.get_image_name(experiment.first_time_point + experiment.delay_time_point)
            mars_image = common.find_file(mars_dir, mars_name, file_type='image', callfrom=proc, local_monitoring=None,
                                          verbose=False)
            if mars_image is None:
                monitoring.to_log_and_console("       '" + str(mars_name) + "' does not exist", 2)
                monitoring.to_log_and_console("\t Exiting.")
                sys.exit(1)
            input_image = os.path.join(mars_dir, mars_image)

        #
        # output image
        #
        if parameters.output_image is not None and len(str(parameters.output_image)) > 0:
            output_image = parameters.output_image
            if monitoring.forceResultsToBeBuilt is False:
                monitoring.to_log_and_console("    manual corrected image '" + str(output_image) + "' exists", 2)
                return
        else:
            astec_dir = experiment.astec_dir.get_directory(0)
            seg_name = experiment.astec_dir.get_image_name(experiment.first_time_point + experiment.delay_time_point)
            seg_image = common.find_file(astec_dir, seg_name, file_type='image', callfrom=proc, local_monitoring=None,
                                         verbose=False)
            if seg_image is not None:
                if monitoring.forceResultsToBeBuilt is False:
                    monitoring.to_log_and_console("    manual corrected image '" + str(seg_image) + "' exists", 2)
                    return
                output_image = os.path.join(astec_dir, seg_image)
            else:
                output_image = os.path.join(astec_dir, seg_name + '.' + experiment.result_image_suffix)

        #
        # start processing
        #
        monitoring.to_log_and_console("... correction of '" + str(input_image).split(os.path.sep)[-1] + "'", 1)
        start_time = time.time()

        correction_process(input_image, output_image, parameters)

        #
        # end processing for a time point
        #
        end_time = time.time()
        monitoring.to_log_and_console('    computation time = ' + str(end_time - start_time) + ' s', 1)
        monitoring.to_log_and_console('', 1)

    else:

        if parameters.first_time_point < 0 or parameters.last_time_point < 0:
            monitoring.to_log_and_console("... time interval does not seem to be defined in the parameter file")
            monitoring.to_log_and_console("    set parameters 'begin' and 'end'")
            monitoring.to_log_and_console("\t Exiting")
            sys.exit(1)

        if parameters.first_time_point > parameters.last_time_point:
            monitoring.to_log_and_console("... weird time interval: 'begin' = " + str(parameters.first_time_point)
                                          + ", 'end' = " + str(parameters.last_time_point))

        for time_value in range(parameters.first_time_point + experiment.delay_time_point,
                                parameters.last_time_point + experiment.delay_time_point + 1,
                                experiment.delta_time_point):

            mars_dir = experiment.mars_dir.get_directory(0)
            mars_name = experiment.mars_dir.get_image_name(time_value)
            mars_image = common.find_file(mars_dir, mars_name, file_type='image', callfrom=proc, local_monitoring=None,
                                          verbose=False)

            if mars_image is None:
                monitoring.to_log_and_console("    mars image '" + str(mars_image) + "' not found: skip time "
                                              + str(time_value), 1)
                continue

            input_image = os.path.join(mars_dir, mars_image)

            astec_dir = experiment.astec_dir.get_directory(0)
            seg_name = experiment.astec_dir.get_image_name(experiment.first_time_point + experiment.delay_time_point)
            seg_image = common.find_file(astec_dir, seg_name, file_type='image', callfrom=proc, local_monitoring=None,
                                         verbose=False)

            if seg_image is None or monitoring.forceResultsToBeBuilt is True:
                if seg_image is None:
                    output_image = os.path.join(astec_dir, seg_name + '.' + experiment.result_image_suffix)
                else:
                    output_image = os.path.join(astec_dir, seg_image)

                #
                # start processing
                #
                monitoring.to_log_and_console("... correction of '" + str(input_image).split(os.path.sep)[-1] + "'",
                                                  1)
                start_time = time.time()

                correction_process(input_image, output_image, parameters)

                #
                # end processing for a time point
                #
                end_time = time.time()
                monitoring.to_log_and_console('    computation time = ' + str(end_time - start_time) + ' s', 1)
                monitoring.to_log_and_console('', 1)

            else:
                monitoring.to_log_and_console("    manual corrected image '" + str(seg_image) + "' exists", 2)
