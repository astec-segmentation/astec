
import os
import imp
import sys
import time
import math
import platform
import subprocess
import numpy as np
from scipy import ndimage as nd

from astec.utils import common
from astec.components.spatial_image import SpatialImage
from astec.io.image import imread, imsave
from astec.wrapping import cpp_wrapping

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


class FusionParameters(common.PrefixedParameter):

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
        doc += "Fusion parameter overview:\n"
        doc += "##########################\n"
        doc += "the fusion of the 4 acquisitions follows a number of steps\n"
        doc += "1. Optionally, a slit line correction.\n"
        doc += "   Some Y lines may appear brighter or darker in the acquisition, which\n"
        doc += "   may cause artifacts in the reconstructed (ie fused) image, which, in\n"
        doc += "   turn, may impair further segmentation steps.\n"
        doc += "2. a change of resolution in the X and Y directions only (Z remains unchanged)\n"
        doc += "   it allows to decrease the data volume if the new pixel size is larger\n"
        doc += "   than the acquisition one\n"
        doc += "3. Optionally, a crop of the resampled acquisitions\n"
        doc += "   it allows to decrease the volume of data\n"
        doc += "   the crop is based on the analysis of a MIP view (in the Z direction) of\n"
        doc += "   the volume\n"
        doc += "4. Optionally, a mirroring of the 'right' image\n"
        doc += "5. Linear registration of the 3 last images on the first one (considered as\n"
        doc += "   the reference). The reference image is resampled again, to get an\n"
        doc += "   isotropic voxel (same voxel size in the 3 directions: X, Y, Z) \n"
        doc += "6. Linear combination of images, weighted by an ad-hoc function\n"
        doc += "   The weighting functions are defined by the 'fusion_weighting' variable.\n"
        doc += "7. Crop of the fused image\n"
        doc += "   still based on the analysis of a MIP view (in the Z direction)\n"
        doc += "\n"
        self.doc['fusion_overview'] = doc

        #
        # acquisition parameters
        #

        doc = "\t possible values are 'left' or 'right'.\n"
        doc += "\t - 'right': +90 degrees\n"
        doc += "\t - 'left': -90 degrees\n"
        doc += "\t gives the rotation (wrt to the Y axis) of the left camera frame of\n"
        doc += "\t stack #0 to be aligned with the the left camera frame of stack #1.\n"
        self.doc['acquisition_orientation'] = doc
        self.acquisition_orientation = 'left'

        doc = "\t possible values are True or False\n"
        doc += "\t if False, the right camera images are mirrored to make them similar\n"
        doc += "\t to left camera images\n"
        doc += "\n"
        doc += "\t To determine the configuration (raw_ori,raw_resolution) (ie ('left',False),\n"
        doc += "\t ('left',True), ('right', False), or ('right', True)), it is advised to perform\n"
        doc += "\t the fusion for only one time point (by setting 'begin' and 'end' at the same\n"
        doc += "\t value) with a large 'target_resolution'\n"
        doc += "\n"
        self.doc['acquisition_mirrors'] = doc
        self.acquisition_mirrors = False

        doc = "\t voxel size of acquired images\n"
        doc += "\t example: acquisition_resolution = (.195, .195, 1.)\n"
        self.doc['acquisition_resolution'] = doc
        self.acquisition_resolution = None

        doc = "\t possible values are 'direct' or 'inverse'\n"
        doc += "\t defines where are the high contrasted XZ-sections of the *left* camera\n"
        doc += "\t image of stack0.\n"
        doc += "\t - 'direct': small z are well contrasted (close to the camera), while\n"
        doc += "\t     large z are fuzzy. It is useful for direction-dependent weighting\n"
        doc += "\t     schemes\n"
        doc += "\t - 'inverse': the other way around\n"
        doc += "\t Changing 'direct' to 'inverse' (or the other way) implies to change\n"
        doc += "\t 'acquisition_orientation' as well\n"
        doc += "\t using 'acquisition_leftcamera_z_stacking' will set both\n"
        doc += "\t 'acquisition_stack0_leftcamera_z_stacking' and \n"
        doc += "\t 'acquisition_stack0_leftcamera_z_stacking'\n"
        self.doc['acquisition_stack0_leftcamera_z_stacking'] = doc
        self.acquisition_stack0_leftcamera_z_stacking = 'direct'
        self.doc['acquisition_stack1_leftcamera_z_stacking'] = doc
        self.acquisition_stack1_leftcamera_z_stacking = 'direct'

        #
        # Correction of slit lines
        #
        doc = "\t Possible values are True or False\n"
        doc += "\t Slit lines are Y lines that appear brighter or darker in the acquisition,\n"
        doc += "\t which may cause artifacts in the reconstructed (ie fused) image, which, in\n"
        doc += "\t turn, may impair further segmentation steps.\n"
        self.doc['acquisition_slit_line_correction'] = doc
        self.acquisition_slit_line_correction = False

        #
        # fused image parameters
        #
        doc = "\t Voxel size of the reconstructed image after fusion of the four views.\n"
        doc += "\t Example: target_resolution = 0.3"
        self.doc['target_resolution'] = doc
        self.target_resolution = (0.3, 0.3, 0.3)

        #
        # fusion method
        #
        doc = "\t Possible values are 'direct-fusion' and 'hierarchical-fusion'\n"
        doc += "\t There are two ways to perform the fusion of the 4 acquisitions:\n"
        doc += "\t - 'direct-fusion'\n"
        doc += "\t   each acquisition is linearly co-registered with the first acquisition\n"
        doc += "\t   (stack #0, left camera). Then weights and images are transformed thanks\n"
        doc += "\t   to the computed transformations. Finally a weighted linear combination\n"
        doc += "\t    gives the result.\n"
        doc += "\t - 'hierarchical-fusion'\n"
        doc += "\t   from the couple (left camera, right camera), each stack is reconstructed,\n"
        doc += "\t   following the same scheme than the direct fusion but with only 2 images.\n"
        doc += "\t   Then stack#1 is (non-)linearly co-registered with stack #0. Images and \n"
        doc += "\t   weights associated with stack#1 are then (non-)linearly transformed. \n"
        doc += "\t   Finally a weighted linear combination gives the result.\n"
        doc += "\t fusion_preregistration_* and fusion_registration_* parameters control the\n"
        doc += "\t co-registration of two acquisitions. It is either used in the 'direct-fusion'\n"
        doc += "\t method (to co-register each acquisition onto the first one) or in the\n"
        doc += "\t 'hierarchical-fusion' method (to co-register couple of opposite acquisitions\n"
        doc += "\t to reconstruct stacks).\n"
        doc += "\t fusion_stack_preregistration_* and fusion_stack_registration_* parameters\n"
        doc += "\t control the co-registration of two stacks. It is only used in the \n"
        doc += "\t 'hierarchical-fusion' method (to co-register the reconstructed stacks).\n"
        self.doc['fusion_strategy'] = doc
        self.fusion_strategy = 'direct-fusion'

        #
        # Cropping of acquisition images (before fusion)
        #
        doc = "\t Possible values are True or False\n"
        doc += "\t If True, the acquisition images are cropped along the X and Y directions.\n"
        doc += "\t Maximum Intensity Projection (MIP) images are automatically thresholded\n"
        doc += "\t (Otsu algorithm) to determine the bounding box of the object of interest.\n"
        doc += "\t Margins are then added to the bounding box\n"
        doc += "\t - 'acquisition_cropping_margin' allow to set the four margin values, i.e.\n"
        doc += "\t   'acquisition_cropping_margin_x_0', 'acquisition_cropping_margin_x_1',\n"
        doc += "\t   'acquisition_cropping_margin_y_0', and 'acquisition_cropping_margin_y_1' \n"
        doc += "\t - 'acquisition_cropping_margin_x' allow to set the two margin values along X, i.e.\n"
        doc += "\t   'acquisition_cropping_margin_x_0' and 'acquisition_cropping_margin_x_1',\n"
        doc += "\t - 'acquisition_cropping_margin_y' allow to set the two margin values along Y, i.e.\n"
        doc += "\t   'acquisition_cropping_margin_y_0' and 'acquisition_cropping_margin_y_1',\n"
        self.doc['acquisition_cropping'] = doc
        self.acquisition_cropping = True

        doc = "\t Added margin of the bounding box computed for the cropping of the raw\n"
        doc += "\t acquisition image in 'left' X direction.\n"
        self.doc['acquisition_cropping_margin_x_0'] = doc
        self.acquisition_cropping_margin_x_0 = 40
        doc = "\t Added margin of the bounding box computed for the cropping of the raw\n"
        doc += "\t acquisition image in 'right' X direction.\n"
        self.doc['acquisition_cropping_margin_x_1'] = doc
        self.acquisition_cropping_margin_x_1 = 40
        doc = "\t Added margin of the bounding box computed for the cropping of the raw\n"
        doc += "\t acquisition image in 'left' Y direction.\n"
        self.doc['acquisition_cropping_margin_y_0'] = doc
        self.acquisition_cropping_margin_y_0 = 40
        doc = "\t Added margin of the bounding box computed for the cropping of the raw\n"
        doc += "\t acquisition image in 'right' Y direction.\n"
        self.doc['acquisition_cropping_margin_y_1'] = doc
        self.acquisition_cropping_margin_y_1 = 40

        #
        # Registration parameters
        # 1. acquisition_registration[] are the parameters for the registration
        #    of two acquisitions (ie a camera image)
        # 2. stack_registration are the parameters for the registration
        #    of two stacks (ie reconstruction from two opposite cameras)
        #
        self.acquisition_registration = []

        self.acquisition_registration.append(common.RegistrationParameters(prefix=['fusion_preregistration_']))
        self.acquisition_registration[0].compute_registration = False
        self.acquisition_registration[0].transformation_type = 'translation'

        self.acquisition_registration.append(common.RegistrationParameters(prefix=['fusion_registration_']))

        self.stack_registration = []

        self.stack_registration.append(common.RegistrationParameters(prefix=['fusion_stack_preregistration_']))

        self.stack_registration.append(common.RegistrationParameters(prefix=['fusion_stack_registration_']))
        self.stack_registration[1].transformation_type = 'vectorfield'
        self.stack_registration[1].lts_fraction = 1.0

        #
        #
        #
        doc = "\t Possible values are True or False\n"
        doc += "\t If True, XZ-sections XZ-sections of the co-registered image stacks,\n"
        doc += "\t as well as the weighting function images, are stored in the directory\n"
        doc += "\t <PATH_EMBRYO>/FUSE/FUSE_<EXP_FUSE>/XZSECTION_<xxxx> where <xxxx> is\n"
        doc += "\t the time point index. It provides a direct and efficient means to check\n"
        doc += "\t whether the parameter 'acquisition_leftcamera_z_stacking' is correctly set.\n"
        self.doc['xzsection_extraction'] = doc
        self.xzsection_extraction = False

        #
        # Cropping of fused image (after fusion)
        #
        doc = "\t Possible values are True or False\n"
        doc += "\t If True, the fusion image is cropped along the X and Y directions.\n"
        doc += "\t Maximum Intensity Projection (MIP) images are automatically thresholded\n"
        doc += "\t (Otsu algorithm) to determine the bounding box of the object of interest.\n"
        doc += "\t Margins are then added to the bounding box\n"
        doc += "\t - 'fusion_cropping_margin' allow to set the four margin values, i.e.\n"
        doc += "\t   'fusion_cropping_margin_x_0', 'fusion_cropping_margin_x_1',\n"
        doc += "\t   'fusion_cropping_margin_y_0', and 'fusion_cropping_margin_y_1' \n"
        doc += "\t - 'fusion_cropping_margin_x' allow to set the two margin values along X, i.e.\n"
        doc += "\t   'fusion_cropping_margin_x_0' and 'fusion_cropping_margin_x_1',\n"
        doc += "\t - 'fusion_cropping_margin_y' allow to set the two margin values along Y, i.e.\n"
        doc += "\t   'fusion_cropping_margin_y_0' and 'fusion_cropping_margin_y_1',\n"
        self.doc['fusion_cropping'] = doc
        self.fusion_cropping = True

        doc = "\t Added margin of the bounding box computed for the cropping of the fusion\n"
        doc += "\t image in 'left' X direction.\n"
        self.doc['fusion_cropping_margin_x_0'] = doc
        self.fusion_cropping_margin_x_0 = 40
        doc = "\t Added margin of the bounding box computed for the cropping of the fusion\n"
        doc += "\t image in 'right' X direction.\n"
        self.doc['fusion_cropping_margin_x_1'] = doc
        self.fusion_cropping_margin_x_1 = 40
        doc = "\t Added margin of the bounding box computed for the cropping of the fusion\n"
        doc += "\t image in 'left' Y direction.\n"
        self.doc['fusion_cropping_margin_y_0'] = doc
        self.fusion_cropping_margin_y_0 = 40
        doc = "\t Added margin of the bounding box computed for the cropping of the fusion\n"
        doc += "\t image in 'right' Y direction.\n"
        self.doc['fusion_cropping_margin_y_1'] = doc
        self.fusion_cropping_margin_y_1 = 40

    ############################################################
    #
    # print / write
    #
    ############################################################

    def print_parameters(self):
        print("")
        print('#')
        print('# FusionParameters')
        print('#')
        print("")

        for line in self.doc['fusion_overview'].splitlines():
            print('# ' + line)

        common.PrefixedParameter.print_parameters(self)

        self.varprint('acquisition_orientation', self.acquisition_orientation, self.doc['acquisition_orientation'])
        self.varprint('acquisition_mirrors', self.acquisition_mirrors, self.doc['acquisition_mirrors'])
        self.varprint('acquisition_resolution', self.acquisition_resolution, self.doc['acquisition_resolution'])

        self.varprint('acquisition_stack0_leftcamera_z_stacking', self.acquisition_stack0_leftcamera_z_stacking,
                      self.doc['acquisition_stack0_leftcamera_z_stacking'])
        self.varprint('acquisition_stack1_leftcamera_z_stacking', self.acquisition_stack1_leftcamera_z_stacking,
                      self.doc['acquisition_stack1_leftcamera_z_stacking'])

        self.varprint('acquisition_slit_line_correction', self.acquisition_slit_line_correction,
                      self.doc['acquisition_slit_line_correction'])

        self.varprint('target_resolution', self.target_resolution, self.doc['target_resolution'])

        self.varprint('fusion_strategy', self.fusion_strategy, self.doc['fusion_strategy'])

        self.varprint('acquisition_cropping', self.acquisition_cropping, self.doc['acquisition_cropping'])
        self.varprint('acquisition_cropping_margin_x_0', self.acquisition_cropping_margin_x_0,
                      self.doc['acquisition_cropping_margin_x_0'])
        self.varprint('acquisition_cropping_margin_x_1', self.acquisition_cropping_margin_x_1,
                      self.doc['acquisition_cropping_margin_x_1'])
        self.varprint('acquisition_cropping_margin_y_0', self.acquisition_cropping_margin_y_0,
                      self.doc['acquisition_cropping_margin_y_0'])
        self.varprint('acquisition_cropping_margin_y_1', self.acquisition_cropping_margin_y_1,
                      self.doc['acquisition_cropping_margin_y_1'])

        for p in self.acquisition_registration:
            p.print_parameters()
        for p in self.stack_registration:
            p.print_parameters()

        self.varprint('xzsection_extraction', self.xzsection_extraction, self.doc['xzsection_extraction'])

        self.varprint('fusion_cropping', self.fusion_cropping, self.doc['fusion_cropping'])
        self.varprint('fusion_cropping_margin_x_0', self.fusion_cropping_margin_x_0,
                      self.doc['fusion_cropping_margin_x_0'])
        self.varprint('fusion_cropping_margin_x_1', self.fusion_cropping_margin_x_1,
                      self.doc['fusion_cropping_margin_x_1'])
        self.varprint('fusion_cropping_margin_y_0', self.fusion_cropping_margin_y_0,
                      self.doc['fusion_cropping_margin_y_0'])
        self.varprint('fusion_cropping_margin_y_1', self.fusion_cropping_margin_y_1,
                      self.doc['fusion_cropping_margin_y_1'])
        print("")

    def write_parameters_in_file(self, logfile):
        logfile.write("" + "\n")
        logfile.write('#' + "\n")
        logfile.write('# FusionParameters' + "\n")
        logfile.write('#' + "\n")
        logfile.write("" + "\n")

        for line in self.doc['fusion_overview'].splitlines():
            logfile.write('# ' + line + '\n')

        common.PrefixedParameter.write_parameters_in_file(self, logfile)

        self.varwrite(logfile, 'acquisition_orientation', self.acquisition_orientation,
                      self.doc['acquisition_orientation'])
        self.varwrite(logfile, 'acquisition_mirrors', self.acquisition_mirrors, self.doc['acquisition_mirrors'])
        self.varwrite(logfile, 'acquisition_resolution', self.acquisition_resolution,
                      self.doc['acquisition_resolution'])

        self.varwrite(logfile, 'acquisition_stack0_leftcamera_z_stacking',
                      self.acquisition_stack0_leftcamera_z_stacking,
                      self.doc['acquisition_stack0_leftcamera_z_stacking'])
        self.varwrite(logfile, 'acquisition_stack1_leftcamera_z_stacking',
                      self.acquisition_stack1_leftcamera_z_stacking,
                      self.doc['acquisition_stack1_leftcamera_z_stacking'])

        self.varwrite(logfile, 'acquisition_slit_line_correction', self.acquisition_slit_line_correction,
                      self.doc['acquisition_slit_line_correction'])

        self.varwrite(logfile, 'target_resolution', self.target_resolution, self.doc['target_resolution'])

        self.varwrite(logfile, 'fusion_strategy', self.fusion_strategy, self.doc['fusion_strategy'])

        self.varwrite(logfile, 'acquisition_cropping', self.acquisition_cropping, self.doc['acquisition_cropping'])
        self.varwrite(logfile, 'acquisition_cropping_margin_x_0', self.acquisition_cropping_margin_x_0,
                      self.doc['acquisition_cropping_margin_x_0'])
        self.varwrite(logfile, 'acquisition_cropping_margin_x_1', self.acquisition_cropping_margin_x_1,
                      self.doc['acquisition_cropping_margin_x_1'])
        self.varwrite(logfile, 'acquisition_cropping_margin_y_0', self.acquisition_cropping_margin_y_0,
                      self.doc['acquisition_cropping_margin_y_0'])
        self.varwrite(logfile, 'acquisition_cropping_margin_y_1', self.acquisition_cropping_margin_y_1,
                      self.doc['acquisition_cropping_margin_y_1'])

        for p in self.acquisition_registration:
            p.write_parameters_in_file(logfile)
        for p in self.stack_registration:
            p.write_parameters_in_file(logfile)

        self.varwrite(logfile, 'xzsection_extraction', self.xzsection_extraction, self.doc['xzsection_extraction'])

        self.varwrite(logfile, 'fusion_cropping', self.fusion_cropping, self.doc['fusion_cropping'])
        self.varwrite(logfile, 'fusion_cropping_margin_x_0', self.fusion_cropping_margin_x_0,
                      self.doc['fusion_cropping_margin_x_0'])
        self.varwrite(logfile, 'fusion_cropping_margin_x_1', self.fusion_cropping_margin_x_1,
                      self.doc['fusion_cropping_margin_x_1'])
        self.varwrite(logfile, 'fusion_cropping_margin_y_0', self.fusion_cropping_margin_y_0,
                      self.doc['fusion_cropping_margin_y_0'])
        self.varwrite(logfile, 'fusion_cropping_margin_y_1', self.fusion_cropping_margin_y_1,
                      self.doc['fusion_cropping_margin_y_1'])
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
        #
        # acquisition parameters
        #
        self.acquisition_orientation = self.read_parameter(parameters, 'raw_ori', self.acquisition_orientation)
        self.acquisition_orientation = self.read_parameter(parameters, 'acquisition_orientation',
                                                           self.acquisition_orientation)

        self.acquisition_mirrors = self.read_parameter(parameters, 'raw_mirrors', self.acquisition_mirrors)
        self.acquisition_mirrors = self.read_parameter(parameters, 'acquisition_mirrors', self.acquisition_mirrors)

        self.acquisition_stack0_leftcamera_z_stacking = self.read_parameter(parameters,
                                                                            'raw_leftcamera_z_stacking',
                                                                            self.acquisition_stack0_leftcamera_z_stacking)
        self.acquisition_stack0_leftcamera_z_stacking = self.read_parameter(parameters,
                                                                            'acquisition_leftcamera_z_stacking',
                                                                            self.acquisition_stack0_leftcamera_z_stacking)
        self.acquisition_stack1_leftcamera_z_stacking = self.read_parameter(parameters,
                                                                            'raw_leftcamera_z_stacking',
                                                                            self.acquisition_stack0_leftcamera_z_stacking)
        self.acquisition_stack1_leftcamera_z_stacking = self.read_parameter(parameters,
                                                                            'acquisition_leftcamera_z_stacking',
                                                                            self.acquisition_stack0_leftcamera_z_stacking)

        self.acquisition_stack0_leftcamera_z_stacking = self.read_parameter(parameters,
                                                                            'raw_stack0_leftcamera_z_stacking',
                                                                            self.acquisition_stack0_leftcamera_z_stacking)
        self.acquisition_stack0_leftcamera_z_stacking = self.read_parameter(parameters,
                                                                            'acquisition_stack0_leftcamera_z_stacking',
                                                                            self.acquisition_stack0_leftcamera_z_stacking)
        self.acquisition_stack1_leftcamera_z_stacking = self.read_parameter(parameters,
                                                                            'raw_stack1_leftcamera_z_stacking',
                                                                            self.acquisition_stack0_leftcamera_z_stacking)
        self.acquisition_stack1_leftcamera_z_stacking = self.read_parameter(parameters,
                                                                            'acquisition_stack1_leftcamera_z_stacking',
                                                                            self.acquisition_stack0_leftcamera_z_stacking)

        if hasattr(parameters, 'raw_resolution'):
            if parameters.raw_resolution is not None:
                if type(parameters.raw_resolution) is tuple or type(parameters.raw_resolution) is list:
                    if len(parameters.raw_resolution) == 3:
                        self.acquisition_resolution = parameters.raw_resolution
                    else:
                        print("Error in reading parameters")
                        print("\t 'raw_resolution' has length " + str(len(parameters.raw_resolution)
                              + " instead of 3."))
                        print("\t Exiting.")
                        sys.exit(1)
                else:
                    print("Error in reading parameters")
                    print("\t type of 'raw_resolution' (" + str(type(parameters.raw_resolution)
                          + ") is not handled"))
                    print("\t Exiting.")
                    sys.exit(1)
        elif hasattr(parameters, 'acquisition_resolution'):
            if parameters.acquisition_resolution is not None:
                if type(parameters.acquisition_resolution) is tuple or type(parameters.acquisition_resolution) is list:
                    if len(parameters.acquisition_resolution) == 3:
                        self.acquisition_resolution = parameters.acquisition_resolution
                    else:
                        print("Error in reading parameters")
                        print("\t 'acquisition_resolution' has length " + str(len(parameters.acquisition_resolution)
                              + " instead of 3."))
                        print("\t Exiting.")
                        sys.exit(1)
                else:
                    print("Error in reading parameters")
                    print("\t type of 'acquisition_resolution' (" + str(type(parameters.acquisition_resolution)
                          + ") is not handled"))
                    print("\t Exiting.")
                    sys.exit(1)

        #
        # correction of slit lines
        #
        self.acquisition_slit_line_correction = self.read_parameter(parameters, 'acquisition_slit_line_correction',
                                                                    self.acquisition_slit_line_correction)

        #
        # fused image parameters
        #
        self.target_resolution = self.read_parameter(parameters, 'target_resolution', self.target_resolution)

        #
        # fusion method
        #
        self.fusion_strategy = self.read_parameter(parameters, 'fusion_strategy', self.fusion_strategy)
        self.fusion_strategy = self.read_parameter(parameters, 'fusion_method', self.fusion_strategy)

        #
        # Cropping of acquisition images (before fusion)
        #
        self.acquisition_cropping = self.read_parameter(parameters, 'acquisition_cropping', self.acquisition_cropping)
        self.acquisition_cropping = self.read_parameter(parameters, 'raw_crop', self.acquisition_cropping)

        self.acquisition_cropping_margin_x_0 = self.read_parameter(parameters, 'acquisition_cropping_margin',
                                                                   self.acquisition_cropping_margin_x_0)
        self.acquisition_cropping_margin_x_0 = self.read_parameter(parameters, 'acquisition_cropping_margin_x',
                                                                   self.acquisition_cropping_margin_x_0)
        self.acquisition_cropping_margin_x_0 = self.read_parameter(parameters, 'acquisition_cropping_margin_x_0',
                                                                   self.acquisition_cropping_margin_x_0)
        self.acquisition_cropping_margin_x_0 = self.read_parameter(parameters, 'raw_margin_x_0',
                                                                   self.acquisition_cropping_margin_x_0)

        self.acquisition_cropping_margin_x_1 = self.read_parameter(parameters, 'acquisition_cropping_margin',
                                                                   self.acquisition_cropping_margin_x_1)
        self.acquisition_cropping_margin_x_1 = self.read_parameter(parameters, 'acquisition_cropping_margin_x',
                                                                   self.acquisition_cropping_margin_x_1)
        self.acquisition_cropping_margin_x_1 = self.read_parameter(parameters, 'acquisition_cropping_margin_x_1',
                                                                   self.acquisition_cropping_margin_x_1)
        self.acquisition_cropping_margin_x_1 = self.read_parameter(parameters, 'raw_margin_x_1',
                                                                   self.acquisition_cropping_margin_x_1)

        self.acquisition_cropping_margin_y_0 = self.read_parameter(parameters, 'acquisition_cropping_margin',
                                                                   self.acquisition_cropping_margin_y_0)
        self.acquisition_cropping_margin_y_0 = self.read_parameter(parameters, 'acquisition_cropping_margin_y',
                                                                   self.acquisition_cropping_margin_y_0)
        self.acquisition_cropping_margin_y_0 = self.read_parameter(parameters, 'acquisition_cropping_margin_y_0',
                                                                   self.acquisition_cropping_margin_y_0)
        self.acquisition_cropping_margin_y_0 = self.read_parameter(parameters, 'raw_margin_y_0',
                                                                   self.acquisition_cropping_margin_y_0)

        self.acquisition_cropping_margin_y_1 = self.read_parameter(parameters, 'acquisition_cropping_margin',
                                                                   self.acquisition_cropping_margin_y_1)
        self.acquisition_cropping_margin_y_1 = self.read_parameter(parameters, 'acquisition_cropping_margin_y',
                                                                   self.acquisition_cropping_margin_y_1)
        self.acquisition_cropping_margin_y_1 = self.read_parameter(parameters, 'acquisition_cropping_margin_y_1',
                                                                   self.acquisition_cropping_margin_y_1)
        self.acquisition_cropping_margin_y_1 = self.read_parameter(parameters, 'raw_margin_y_1',
                                                                   self.acquisition_cropping_margin_y_1)

        #
        # registration parameters
        #
        for p in self.acquisition_registration:
            p.update_from_parameters(parameters)
        for p in self.stack_registration:
            p.update_from_parameters(parameters)

        #
        #
        #
        self.xzsection_extraction = self.read_parameter(parameters, 'xzsection_extraction', self.xzsection_extraction)
        self.xzsection_extraction = self.read_parameter(parameters, 'fusion_xzsection_extraction',
                                                        self.xzsection_extraction)

        #
        # Cropping of fused image (after fusion)
        #
        self.fusion_cropping = self.read_parameter(parameters, 'fusion_cropping', self.fusion_cropping)
        self.fusion_cropping = self.read_parameter(parameters, 'fusion_crop', self.fusion_cropping)

        self.fusion_cropping_margin_x_0 = self.read_parameter(parameters, 'fusion_cropping_margin',
                                                              self.fusion_cropping_margin_x_0)
        self.fusion_cropping_margin_x_0 = self.read_parameter(parameters, 'fusion_cropping_margin_x',
                                                              self.fusion_cropping_margin_x_0)
        self.fusion_cropping_margin_x_0 = self.read_parameter(parameters, 'fusion_cropping_margin_x_0',
                                                              self.fusion_cropping_margin_x_0)
        self.fusion_cropping_margin_x_0 = self.read_parameter(parameters, 'fusion_margin_x_0',
                                                              self.fusion_cropping_margin_x_0)

        self.fusion_cropping_margin_x_1 = self.read_parameter(parameters, 'fusion_cropping_margin',
                                                              self.fusion_cropping_margin_x_1)
        self.fusion_cropping_margin_x_1 = self.read_parameter(parameters, 'fusion_cropping_margin_x',
                                                              self.fusion_cropping_margin_x_1)
        self.fusion_cropping_margin_x_1 = self.read_parameter(parameters, 'fusion_cropping_margin_x_1',
                                                              self.fusion_cropping_margin_x_1)
        self.fusion_cropping_margin_x_1 = self.read_parameter(parameters, 'fusion_margin_x_1',
                                                              self.fusion_cropping_margin_x_1)

        self.fusion_cropping_margin_y_0 = self.read_parameter(parameters, 'fusion_cropping_margin',
                                                              self.fusion_cropping_margin_y_0)
        self.fusion_cropping_margin_y_0 = self.read_parameter(parameters, 'fusion_cropping_margin_y',
                                                              self.fusion_cropping_margin_y_0)
        self.fusion_cropping_margin_y_0 = self.read_parameter(parameters, 'fusion_cropping_margin_y_0',
                                                              self.fusion_cropping_margin_y_0)
        self.fusion_cropping_margin_y_0 = self.read_parameter(parameters, 'fusion_margin_y_0',
                                                              self.fusion_cropping_margin_y_0)

        self.fusion_cropping_margin_y_1 = self.read_parameter(parameters, 'fusion_cropping_margin',
                                                              self.fusion_cropping_margin_y_1)
        self.fusion_cropping_margin_y_1 = self.read_parameter(parameters, 'fusion_cropping_margin_y',
                                                              self.fusion_cropping_margin_y_1)
        self.fusion_cropping_margin_y_1 = self.read_parameter(parameters, 'fusion_cropping_margin_y_1',
                                                              self.fusion_cropping_margin_y_1)
        self.fusion_cropping_margin_y_1 = self.read_parameter(parameters, 'fusion_margin_y_1',
                                                              self.fusion_cropping_margin_y_1)

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


__extension_to_be_converted__ = ['.h5', '.tif', '.tiff', '.TIF', '.TIFF']
__extension_with_resolution__ = ['.inr', '.inr.gz', '.mha', '.mha.gz']


def _read_image_name(data_path, temporary_path, file_name, resolution, default_extension='inr'):
    """
    Read an image. Eventually, unzip a compressed file, and convert the image
    to a format known by executables
    :param data_path: path to data directory
    :param temporary_path: directory for temporary file
    :param file_name: image file
    :param resolution: resolution of the result image
            required to write the output image with the right resolution
    :return:
    """

    proc = "_read_image_name"

    #
    # test whether the extension is zip
    #
    f = file_name
    full_name = os.path.join(data_path, f)

    if f[len(f)-4:len(f)] == '.zip':

        prefix = f[0:len(f)-4]

        #
        # maybe the file has already be unzipped
        #
        file_names = []
        for f in os.listdir(temporary_path):
            if len(f) <= len(prefix):
                pass
            if f[0:len(prefix)] == prefix:
                if f[len(prefix):len(f)] in common.recognized_image_extensions:
                    file_names.append(f)

        if len(file_names) > 1:
            monitoring.to_log_and_console(proc + ": already several images with name '"
                                          + str(prefix) + "' were found in '" + str(temporary_path) + "'")
            monitoring.to_log_and_console("\t " + str(file_names))
            monitoring.to_log_and_console("\t Exiting")
            sys.exit(1)

        elif len(file_names) == 0:
            #
            # unzipping
            #
            monitoring.to_log_and_console("    .. unzipping '" + str(f) + "'", 2)
            #
            # there are issues with unzip
            # seems to occur when files are zipped with zip 3.0
            #
            if platform.system() == 'Linux':
                command_line = 'unzip ' + os.path.join(data_path, f) + ' -d ' + str(temporary_path)
            elif platform.system() == 'Darwin':
                command_line = 'tar xvf ' + os.path.join(data_path, f) + ' -C ' + str(temporary_path)
            else:
                command_line = 'unzip ' + os.path.join(data_path, f) + ' -d ' + str(temporary_path)
            if monitoring.verbose >= 3 or monitoring.debug > 0:
                monitoring.to_log("* Launch: " + command_line)
                with open(monitoring.log_filename, 'a') as logfile:
                    subprocess.call(command_line, shell=True, stdout=logfile, stderr=subprocess.STDOUT)
            else:
                subprocess.call(command_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            #
            # parsing again the temporay directory
            #
            file_names = []
            for f in os.listdir(temporary_path):
                if len(f) <= len(prefix):
                    pass
                if f[0:len(prefix)] == prefix:
                    file_names.append(f)

        if len(file_names) == 0:
            monitoring.to_log_and_console(proc + ": no image with name '" + str(prefix)
                                          + "' was found in '" + str(temporary_path) + "'")
            monitoring.to_log_and_console("\t Exiting")
            sys.exit(1)

        if len(file_names) > 1:
            monitoring.to_log_and_console(proc + ": several images with name '"
                                          + str(prefix) + "' were found in '" + str(temporary_path) + "'")
            monitoring.to_log_and_console("\t " + str(file_names))
            monitoring.to_log_and_console("\t Exiting")
            sys.exit(1)
        #
        #
        #
        f = file_names[0]
        full_name = os.path.join(temporary_path, f)

    #
    # test whether the file has to be converted into a more 'readable' format
    # if yes, set the resolution if required
    #

    file_has_been_converted = False
    for extension in __extension_to_be_converted__:
        if f[len(f)-len(extension):len(f)] == extension:
            prefix = f[0:len(f) - len(extension)]

            #
            # new file name
            # check whether it has already been converted
            #
            new_full_name = os.path.join(temporary_path, prefix) + '.' + str(default_extension)
            if not os.path.isfile(new_full_name):
                monitoring.to_log_and_console("    .. converting '" + str(f) + "'", 2)
                image = imread(full_name)
                if type(resolution) is tuple and len(resolution) == 3:
                    image.voxelsize = resolution
                    monitoring.to_log("    * resolution of '" + full_name + "' has been set to "
                                      + str(image.voxelsize))
                elif type(resolution) is list and len(resolution) == 3:
                    image.voxelsize = (resolution(0), resolution(1), resolution(2))
                    monitoring.to_log("    * resolution of '" + full_name + "' has been set to "
                                      + str(image.voxelsize))
                else:
                    monitoring.to_log("    * resolution of '" + full_name + "' is " + str(image.voxelsize)
                                      + "(default/read values)")
                #
                # remove unzipped file to avoid having two files in the directory
                # verify that it is not the input file!
                #
                if os.path.dirname(full_name) == temporary_path:
                    os.remove(full_name)

                #
                # save converted file
                #
                imsave(new_full_name, image)
                file_has_been_converted = True

            full_name = new_full_name
            break

    #
    # test whether the input format is supposed to have the resolution set
    #

    if file_has_been_converted is False:
        for extension in __extension_with_resolution__:
            if f[len(f) - len(extension):len(f)] == extension:
                file_has_been_converted = True
                break

    #
    # if no conversion occurs, the resolution has not been set yet
    #

    if file_has_been_converted is False:
        if (type(resolution) is tuple or type(resolution) is list) and len(resolution) == 3:
            monitoring.to_log_and_console("    .. changing resolution '" + str(f) + "'", 2)
            image = imread(full_name)
            if type(resolution) is tuple and len(resolution) == 3:
                image.voxelsize = resolution
                monitoring.to_log("    * resolution of '" + full_name + "' has been set to " + str(image.voxelsize))
            elif type(resolution) is list and len(resolution) == 3:
                image.voxelsize = (resolution(0), resolution(1), resolution(2))
                monitoring.to_log("    * resolution of '" + full_name + "' has been set to " + str(image.voxelsize))
            imsave(full_name, image)
        else:
            monitoring.to_log("    * resolution of '" + full_name + "' is left unchanged")

    return full_name


def _analyze_data_directory(data_dir):
    """
    Parse a directory containing images
    :param data_dir:
    :return:
    1. the common prefix of image file names
    2. the number of characters used to encoded the variable part
       (time points)
    3. the list of the variable parts
    4. the common suffix of image file names
       may be longer than just the file extension
    """

    proc = "_analyze_data_directory"
    images = []
    extensions = []
    #
    # recognize images and extensions
    #
    for f in os.listdir(data_dir):
        for e in common.recognized_image_extensions:
            if f[len(f)-len(e):len(f)] == e:
                if e not in extensions:
                    extensions.append(e)
                    if len(extensions) > 1:
                        print(proc + ": several image extensions were found in '" + data_dir + "'")
                        print("\t -> " + str(extensions))
                        print("\t Exiting.")
                        sys.exit(1)
                images.append(f)

    if len(images) == 0:
        print(proc + ": no images were found in '" + data_dir + "'")
        print("\t Exiting.")
        sys.exit(1)

    #
    # one image case
    #

    if len(images) == 1:
        suffix = extensions[0]
        time_length = 0
        im = images[0]
        length = len(im) - 1 - len(suffix)
        for i in range(0, 3):
            if '0' <= im[length-i] <= '9':
                time_length += 1
            else:
                break
        prefix = im[0:len(im) - time_length - len(suffix)]
        time_points = im[len(im) - time_length - len(suffix):len(im) - len(suffix)]
        return prefix, time_length, time_points, suffix

    #
    # several images
    # 1. check that image names are of the same length
    # 2. get prefix = common part at beginning
    # 3. get suffix = common part at end
    # 4. get length for time point encoding
    # 5. get list of time points
    #

    for i in range(1, len(images)):
        if len(images[0]) != len(images[i]):
            print(proc + ": image names are of different lengths in '" + data_dir + "'")
            print("\t -> " + images[0] + ", " + images[i])
            print("\t Exiting.")
            sys.exit(1)

    prefix = ''
    for j in range(0, len(images[0])):
        ok = True
        for i in range(1, len(images)):
            if images[0][j] != images[i][j]:
                ok = False
                break
        if ok is True:
            prefix += images[0][j]
        else:
            break

    suffix = ''
    for j in range(len(images[0]) - 1, -1, -1):
        ok = True
        for i in range(1, len(images)):
            if images[0][j] != images[i][j]:
                ok = False
                break
        if ok is True:
            suffix += images[0][j]
        else:
            break
    suffix = suffix[::-1]

    time_length = len(images[0]) - len(prefix) - len(suffix)

    time_points = []
    for i in range(0, len(images)):
        time_points.append(images[i][len(images[i]) - time_length - len(suffix):len(images[i]) - len(suffix)])

    return prefix, time_length, time_points, suffix


########################################################################################
#
# cropping
#
########################################################################################


def _crop_bounding_box(the_image):
    """
    Compute a bounding box to crop an image (ie extract a subimage)
    :param the_image:
    :return:
    """

    #
    # build a 2D binary image from the MIP projection
    #

    the_selection = common.add_suffix(the_image, "_cropselection")
    cpp_wrapping.mip_projection_for_crop(the_image, the_selection, None, monitoring)

    #
    # read input image
    #
    selection = imread(the_selection)

    #
    # the get the connected component (4-connectivity)
    # there should have only two components: the background and the selected component
    #
    cc_image, cc_n = nd.label(selection)

    #
    # compute the volumes of each connected component
    # and create a dictionary of tuples (label, volume)
    #
    labels = np.unique(cc_image)
    volumes = nd.sum(np.ones_like(cc_image), cc_image, index=np.int16(labels))
    dict_volumes = dict(list(zip(labels, volumes)))

    #
    # remove the background
    # then get the label associated to the largest connected component
    dict_volumes.pop(0)
    max_label = list(dict_volumes.keys())[np.argmax(list(dict_volumes.values()))]

    #
    # get the bounding boxes for all objects
    # it is not necessary to searched for all labels
    # seems that there is no bounding box computed for label #0
    #
    # boundingBoxes = nd.find_objects(ccImage, max_label=maxLabel)
    # maxBox = boundingBoxes[int(maxLabel)-1]
    #
    max_box = nd.find_objects(cc_image, max_label=max_label)[int(max_label)-1]

    del cc_image

    return max_box


def _crop_disk_image(the_image, res_image, the_max_box=None,
                     margin_x_0=40, margin_x_1=40, margin_y_0=40, margin_y_1=40):
    """
    Crop an image on disk in XY plane
    :param the_image:
    :param res_image:
    :param the_max_box:
    :param margin_x_0:
    :param margin_x_1:
    :param margin_y_0:
    :param margin_y_1:
    :return:
    """

    #
    # 2D bounding box
    #
    if the_max_box is None:
        max_box = _crop_bounding_box(the_image)
    else:
        max_box = the_max_box

    #
    # 2D bounding box + margin
    #
    image = imread(the_image)

    xmin = max(max_box[0].start - margin_x_0, 0)
    xmax = min(image.shape[0], max_box[0].stop + margin_x_1)
    ymin = max(max_box[1].start - margin_y_0, 0)
    ymax = min(image.shape[1], max_box[1].stop + margin_y_1)

    new_box = (slice(xmin, xmax, None),
               slice(ymin, ymax, None),
               slice(0, image.shape[2]))

    new_image = SpatialImage(image[new_box])
    new_image.voxelsize = image.voxelsize

    imsave(res_image, new_image)

    monitoring.to_log_and_console("       crop from [0," + str(image.shape[0]) + "]x[0,"
                                  + str(image.shape[1]) + "] to [" + str(xmin) + ","
                                  + str(xmax) + "]x[" + str(ymin) + "," + str(ymax) + "]", 2)

    return


########################################################################################
#
# computation of a rotation matrix
#
########################################################################################


def _axis_rotation_matrix(axis, angle, min_space=None, max_space=None):
    """ Return the transformation matrix from the axis and angle necessary
    this is a rigid transformation (rotation) that preserves the center of
    the field of view
    axis : axis of rotation ("X", "Y" or "Z")
    angle : angle of rotation (in degree)
    min_space : coordinates of the bottom point (usually (0, 0, 0))
    max_space : coordinates of the top point (usually im shape)
    """
    i = np.linalg.inv
    d = np.dot
    if axis not in ["X", "Y", "Z"]:
        raise Exception("Unknown axis : " + str(axis))
    rads = math.radians(angle)
    s = math.sin(rads)
    c = math.cos(rads)

    centering = np.identity(4)
    if min_space is None and max_space is not None:
        min_space = np.array([0., 0., 0.])

    if max_space is not None:
        space_center = (max_space-min_space)/2.
        offset = -1.*space_center
        centering[:3, 3] = offset

    rot = np.identity(4)
    if axis == "X":
        rot = np.array([[1., 0., 0., 0.],
                        [0., c, -s, 0.],
                        [0., s, c, 0.],
                        [0., 0., 0., 1.]])
    elif axis == "Y":
        rot = np.array([[c,   0., s,  0.],
                        [0., 1., 0., 0.],
                        [-s, 0., c, 0.],
                        [0., 0., 0., 1.]])

    elif axis == "Z":
        rot = np.array([[c, -s,  0., 0.],
                        [s, c, 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]])

    return d(i(centering), d(rot, centering))


def _init_rotation_matrix(axis, angle, ref_center=None, flo_center=None):

    if axis not in ["X", "Y", "Z"]:
        raise Exception("Unknown axis : " + str(axis))
    rads = math.radians(angle)
    s = math.sin(rads)
    c = math.cos(rads)

    rot = np.identity(3)
    if axis == "X":
        rot = np.array([[1., 0., 0.],
                        [0., c, -s],
                        [0., s, c]])
    elif axis == "Y":
        rot = np.array([[c, 0., s],
                        [0., 1., 0.],
                        [-s, 0., c]])
    elif axis == "Z":
        rot = np.array([[c, -s, 0.],
                        [s, c, 0.],
                        [0., 0., 1.]])

    if ref_center is not None:
        if flo_center is not None:
            trs = flo_center - np.dot(rot, ref_center)
        else:
            trs = ref_center - np.dot(rot, ref_center)
    else:
        if flo_center is not None:
            trs = flo_center - np.dot(rot, flo_center)
        else:
            trs = np.array[0., 0., 0.]

    mat = np.identity(4)
    mat[0:3, 0:3] = rot
    (mat.T)[3:4, 0:3] = trs

    return mat


########################################################################################
#
# function for the ad hoc computation of weights
# for the linear combination of images of the 4 cameras
#
########################################################################################


def _histogram(image, nbins=256):
    """Return histogram of image.

        Unlike `np.histogram`, this function returns the centers of bins and
        does not rebin integer arrays. For integer arrays, each integer value has
        its own bin, which improves speed and intensity-resolution.

        Parameters
        ----------
        image : array
        Input image.
        nbins : int
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.

        Returns
        -------
        hist : array
        The values of the histogram.
        bin_centers : array
        The values at the center of the bins.
        """

    proc = "_histogram"
    if not isinstance(image, SpatialImage):
        print(proc + ": argument image is not an ndarray")
        return

    # For integer types, histogramming with bincount is more efficient.
    if np.issubdtype(image.dtype, np.integer):
        offset = 0
        if np.min(image) < 0:
            offset = np.min(image)
        hist = np.bincount(image.ravel() - offset)
        bin_centers = np.arange(len(hist)) + offset

        # clip histogram to start with a non-zero bin
        idx = np.nonzero(hist)[0][0]
        return hist[idx:], bin_centers[idx:]
    else:
        hist, bin_edges = np.histogram(image.flat, nbins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
        return hist, bin_centers


def _threshold_otsu(image, nbins=256):
    """Return threshold value based on Otsu's method.

        Parameters
        ----------
        image : array
        Input image.
        nbins : int
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.

        Returns
        -------
        threshold : float
        Threshold value.

        References
        ----------
        .. [1] Wikipedia, http://en.wikipedia.org/wiki/Otsu's_Method
        """

    proc = "_threshold_otsu"
    if not isinstance(image, SpatialImage):
        print(proc + ": argument image is not an ndarray")
        return

    hist, bin_centers = _histogram(image, nbins)
    hist = hist.astype(float)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold


def _exp_func(x, length=500, speed=5):
    """ Decay function used to take into account the remotness to the camera
    x : value to compute
    length : lenght of the function
    speed : speed of the function
    """

    return .1+np.exp(-((np.float32(x)*speed)/length))


def _build_guignard_weighting(image, decreasing_weight_with_z):
    """Return the mask on a given image from the decay function
    im : intensity image (SpatialImage)
    direction : if True the camera is in the side of the first slices in Z
    """
    proc = "_build_guignard_weighting"

    if not isinstance(image, SpatialImage):
        print(proc + ": argument image is not an ndarray")
        return

    th = _threshold_otsu(image)
    im_th = np.zeros_like(image, dtype=np.float32)
    im_th[image > th] = 1
    if decreasing_weight_with_z is False:
        im_th = im_th[:, :, -1::-1]
    im_th_sum = np.cumsum(im_th, axis=2)
    if decreasing_weight_with_z is False:
        im_th_sum = im_th_sum[:, :, -1::-1]
    mask = _exp_func(im_th_sum, np.max(im_th_sum))
    return mask


def _build_corner_weighting(image, decreasing_weight_with_z):
    """

    :param image:
    :return:
    """
    proc = "_build_corner_weighting"

    if not isinstance(image, SpatialImage):
        print(proc + ": argument image is not an ndarray")
        return

    mask = np.full_like(image, 0.1, dtype=np.float32)
    dimx = image.shape[0]
    dimz = image.shape[2]
    # build a corner-like weighting image
    # - constant z-slices (during cz slices)
    # - linear decrease along the x dimension until dimz/2 - dz is reached
    # cz : plans entiers de 1 -> [dimz-cz, dimz-1]
    # dz : decalage vers z=0 a partir de mz=dimz/2
    cz = int(dimz/8.0)
    dz = int(dimz/8.0)
    mz = int(dimz/2.0 + 0.5)
    # partie constante
    for z in range(dimz - cz, dimz):
        mask[:, :, z] = 1.0
    # partie variable
    for z in range(mz - dz, dimz - cz):
        dx = int((z - float(dimz - cz)) / float((mz - dz) - (dimz - cz)) * float(dimx / 2.0) + 0.5)
        if dimx - dx > dx:
            mask[dx:dimx-dx, :, z] = 1.0
    if decreasing_weight_with_z is True:
        mask = mask[:, :, -1::-1]
    return mask


def _build_ramp_weighting(image, decreasing_weight_with_z):
    """

    :param image:
    :return:
    """
    proc = "_build_ramp_weighting"

    if not isinstance(image, SpatialImage):
        print(proc + ": argument image is not an ndarray")
        return

    mask = np.zeros_like(image, dtype=np.float32)
    dimz = image.shape[2]
    for z in range(0, dimz):
        mask[:, :, z] = z
    if decreasing_weight_with_z is True:
        mask = mask[:, :, -1::-1]
    return mask


def _build_uniform_weighting(image):
    """

    :param image:
    :return:
    """

    proc = "_build_uniform_weighting"
    if not isinstance(image, SpatialImage):
        print(proc + ": argument image is not an ndarray")
        return

    mask = np.ones_like(image, dtype=np.float32)
    return mask


def _build_unreg_weighting_image(template_image_name, weighting_image_name, decreasing_weight_with_z=True,
                                 fusion_weighting='none'):
    """

    :param template_image_name:
    :param weighting_image_name:
    :param decreasing_weight_with_z:
    :param fusion_weighting:
    :return:
    """
    proc = "_build_unreg_weighting_image"

    im = imread(template_image_name)
    if fusion_weighting.lower() == 'none' or fusion_weighting.lower() == 'uniform' \
            or fusion_weighting.lower() == 'uniform-weighting':
        unreg_weight = _build_uniform_weighting(im)
    elif fusion_weighting.lower() == 'guignard' or fusion_weighting.lower() == 'guignard-weighting':
        unreg_weight = _build_guignard_weighting(im, decreasing_weight_with_z)
    elif fusion_weighting.lower() == 'corner' or fusion_weighting.lower() == 'corner-weighting':
        unreg_weight = _build_corner_weighting(im, decreasing_weight_with_z)
    elif fusion_weighting.lower() == 'ramp' or fusion_weighting.lower() == 'ramp-weighting':
        unreg_weight = _build_ramp_weighting(im, decreasing_weight_with_z)
    else:
        monitoring.to_log_and_console(str(proc) + ": unknown weighting function, switch to uniform")
        unreg_weight = _build_uniform_weighting(im)

    unreg_weight.voxelsize = im.voxelsize
    imsave(weighting_image_name, unreg_weight)

    if fusion_weighting.lower() == 'corner' or fusion_weighting.lower() == 'corner-weighting':
        cpp_wrapping.linear_smoothing(weighting_image_name, weighting_image_name, 5.0, real_scale=False,
                                      filter_type='deriche', border=10, monitoring=monitoring)

    del im
    del unreg_weight
    return


########################################################################################
#
#
#
########################################################################################

def _blockmatching(path_ref, path_flo, path_output, path_output_trsf, path_init_trsf=None,
                   parameters=None):
    """

    :param path_ref:
    :param path_flo:
    :param path_output:
    :param path_output_trsf:
    :param path_init_trsf:
    :param parameters:
    :return:
    """
    if parameters is not None:
        cpp_wrapping.blockmatching(path_ref, path_flo, path_output, path_output_trsf, path_init_trsf=path_init_trsf,
                                   py_hl=parameters.pyramid_highest_level,
                                   py_ll=parameters.pyramid_lowest_level,
                                   transformation_type=parameters.transformation_type,
                                   elastic_sigma=parameters.elastic_sigma,
                                   transformation_estimator=parameters.transformation_estimation_type,
                                   lts_fraction=parameters.lts_fraction,
                                   fluid_sigma=parameters.fluid_sigma,
                                   normalization=parameters.normalization,
                                   monitoring=monitoring)
    else:
        cpp_wrapping.blockmatching(path_ref, path_flo, path_output, path_output_trsf, path_init_trsf=path_init_trsf,
                                   monitoring=monitoring)


def _linear_registration(path_ref, path_flo, path_output, path_output_trsf, path_init_trsf=None,
                         parameters=None):
    if parameters is not None:
        cpp_wrapping.linear_registration(path_ref, path_flo, path_output, path_output_trsf, path_init_trsf,
                                         py_hl=parameters.pyramid_highest_level,
                                         py_ll=parameters.pyramid_lowest_level,
                                         transformation_type=parameters.transformation_type,
                                         transformation_estimator=parameters.transformation_estimation_type,
                                         lts_fraction=parameters.lts_fraction,
                                         normalization=parameters.normalization,
                                         monitoring=monitoring)
    else:
        cpp_wrapping.linear_registration(path_ref, path_flo, path_output, path_output_trsf, path_init_trsf,
                                         monitoring=monitoring)


########################################################################################
#
#
#
########################################################################################

def _get_image_shape(template_image_name):
    im = imread(template_image_name)
    shape = im.shape
    del im
    return shape


def _extract_xzsection(weight_images, res_images, tmp_fused_image, channel_id, experiment):
    """
    Extract XZ sections from registered raw images and weights as well as the fused image
    (before the last crop (if any))
    :param weight_images:
    :param res_images:
    :param tmp_fused_image:
    :param channel_id:
    :param experiment:
    :return:
    """

    d = experiment.fusion_dir.get_xzsection_directory(channel_id)
    shape = _get_image_shape(res_images[0])
    options = "-xz " + str(int(shape[1]/2))

    name = experiment.get_embryo_name() + "_xy" + format(int(shape[1]/2), '0>4')

    xzsection = os.path.join(d, name + "_stack0_lc_reg." + experiment.result_image_suffix)
    cpp_wrapping.ext_image(res_images[0], xzsection, options, monitoring=monitoring)

    xzsection = os.path.join(d, name + "_stack0_rc_reg." + experiment.result_image_suffix)
    cpp_wrapping.ext_image(res_images[1], xzsection, options, monitoring=monitoring)

    if len(res_images) == 4:
        xzsection = os.path.join(d, name + "_stack1_lc_reg." + experiment.result_image_suffix)
        cpp_wrapping.ext_image(res_images[2], xzsection, options, monitoring=monitoring)

        xzsection = os.path.join(d, name + "_stack1_rc_reg." + experiment.result_image_suffix)
        cpp_wrapping.ext_image(res_images[3], xzsection, options, monitoring=monitoring)

    xzsection = os.path.join(d, name + "_stack0_lc_weight." + experiment.result_image_suffix)
    cpp_wrapping.ext_image(weight_images[0], xzsection, options, monitoring=monitoring)

    xzsection = os.path.join(d, name + "_stack0_rc_weight." + experiment.result_image_suffix)
    cpp_wrapping.ext_image(weight_images[1], xzsection, options, monitoring=monitoring)

    if len(weight_images) == 4:
        xzsection = os.path.join(d, name + "_stack1_lc_weight." + experiment.result_image_suffix)
        cpp_wrapping.ext_image(weight_images[2], xzsection, options, monitoring=monitoring)

        xzsection = os.path.join(d, name + "_stack1_rc_weight." + experiment.result_image_suffix)
        cpp_wrapping.ext_image(weight_images[3], xzsection, options, monitoring=monitoring)

    xzsection = os.path.join(d, name + "_fuse." + experiment.result_image_suffix)
    cpp_wrapping.ext_image(tmp_fused_image, xzsection, options, monitoring=monitoring)

    return


########################################################################################
#
#
#
########################################################################################

#
# historical fusion procedure
# each image is co-registered with the left camera acquisition of stack 30
#

def _direct_fusion_process(input_image_list, the_image_list, fused_image, experiment, parameters):
    """

    :param input_image_list:
    :param the_image_list: list of list of preprocessed images (in the temporary directory), ie after
        1. (optional) slit line correction
        2. resolution change (only in X and Y directions)
        3. (optional) 2D crop
        4. mirroring of the right camera images (parameter dependent)

    :param fused_image:
    :param experiment:
    :param parameters:
    :return:
    """

    proc = "_direct_fusion_process"

    n_channels = experiment.rawdata_dir.get_number_channels()

    if monitoring.debug > 1:
        print("")
        print(proc + " was called with:")
        print("- input_image_list = " + str(input_image_list))
        print("- the_image_list = " + str(the_image_list))
        print("- fused_image = " + str(fused_image))
        for c in range(n_channels):
            experiment.rawdata_dir.channel[c].print_parameters('channel #' + str(c))
        print("")

    #
    # parameter type checking
    #

    if not isinstance(experiment, common.Experiment):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'experiment' variable: "
                                      + str(type(experiment)))
        sys.exit(1)

    if not isinstance(parameters, FusionParameters):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'parameters' variable: "
                                      + str(type(parameters)))
        sys.exit(1)

    #
    # list of registered images 
    #

    res_image_list = list()
    for c in range(n_channels):

        the_images = the_image_list[c]
        res_images = []

        for i in range(0, len(the_images)):
            res_images.append(common.add_suffix(input_image_list[c][i], "_reg",
                                                new_dirname=experiment.rawdata_dir.get_tmp_directory(i, c),
                                                new_extension=experiment.default_image_suffix))
        res_image_list.append(res_images)

    #
    # is there something to do ?
    # check whether fused images are missing
    # - if one fusion image is missing, re-process channel #0 to get weights and sum of weights
    # - if the fusion image exists, check whether the registered images exists
    #

    do_something = [False] * len(input_image_list[0])

    for c in range(n_channels):

        the_images = the_image_list[c]
        res_images = res_image_list[c]

        if not os.path.isfile(os.path.join(experiment.fusion_dir.get_directory(c), fused_image)):
            do_something = [True] * len(input_image_list[0])
            break

        for i in range(0, len(the_images)):
            if os.path.isfile(res_images[i]):
                if monitoring.forceResultsToBeBuilt is True:
                    do_something[i] = True
            else:
                do_something[i] = True

    for i in range(1, len(input_image_list[0])):
        if do_something[i] is True:
            do_something[0] = True

    #
    # additional file names for channel #0
    #

    init_trsfs = []
    prereg_trsfs = []
    res_trsfs = []
    unreg_weight_images_list = []
    weight_images_list = []

    #
    # build the file names after the input file names
    #
    the_images = the_image_list[0]
    for i in range(0, len(the_images)):
        init_trsfs.append(common.add_suffix(input_image_list[0][i], "_init",
                                            new_dirname=experiment.rawdata_dir.get_tmp_directory(i, 0),
                                            new_extension="trsf"))
        prereg_trsfs.append(common.add_suffix(input_image_list[0][i], "_prereg",
                                              new_dirname=experiment.rawdata_dir.get_tmp_directory(i, 0),
                                              new_extension="trsf"))
        res_trsfs.append(common.add_suffix(input_image_list[0][i], "_reg",
                                           new_dirname=experiment.rawdata_dir.get_tmp_directory(i, 0),
                                           new_extension="trsf"))

    #
    # the final image is a weighting sum of the transformed acquisition image
    # weighting may be different for all channel
    #
    for c in range(n_channels):
        unreg_weight_images = []
        weight_images = []
        cref = c
        for i in range(0, c):
            if experiment.rawdata_dir.channel[c].fusion_weighting == experiment.rawdata_dir.channel[i].fusion_weighting:
                cref = i
        #
        # check if the weighting mode was used for previous channels (ie weights have already been computed)
        #
        if cref == c:
            for i in range(0, len(the_images)):
                unreg_weight_images.append(common.add_suffix(input_image_list[c][i], "_init_weight_" + str(c),
                                                             new_dirname=experiment.rawdata_dir.get_tmp_directory(i, c),
                                                             new_extension=experiment.default_image_suffix))
                weight_images.append(common.add_suffix(input_image_list[c][i], "_weight_" + str(c),
                                                       new_dirname=experiment.rawdata_dir.get_tmp_directory(i, c),
                                                       new_extension=experiment.default_image_suffix))
        else:
            unreg_weight_images = unreg_weight_images_list[cref]
            weight_images = weight_images_list[cref]

        unreg_weight_images_list.append(unreg_weight_images)
        weight_images_list.append(weight_images)

    #
    # 1. Putting all images in a common reference
    # - resampling of first image in an isotropic grid = reference image
    # - co-registration of other images
    # 2. Compute weights with an ad-hoc method
    #

    #
    # default angle for initial rotation matrix
    #

    if parameters.acquisition_orientation.lower() == 'left':
        default_angle = 270.0
    elif parameters.acquisition_orientation.lower() == 'right':
        default_angle = 90.0
    else:
        monitoring.to_log_and_console(proc + ": unknown acquisition orientation '"
                                      + str(parameters.acquisition_orientation) + "'", 0)
        monitoring.to_log_and_console("Exiting.", 0)
        sys.exit(1)

    #
    # process
    # transformations and weights are computed on channel #0
    # and are used for other channels
    # - first image is just resampled to the destination resolution
    # - other images are co-registered with the first image
    #

    #
    # fusion_box is the croping information computed only on the first channel
    # and then used for all other channels
    #
    fusion_box = None

    for c in range(n_channels):

        if n_channels > 1:
            monitoring.to_log_and_console("    .. process channel #" + str(c), 2)

        the_images = the_image_list[c]
        res_images = res_image_list[c]
        unreg_weight_images = unreg_weight_images_list[c]
        weight_images = weight_images_list[c]

        for i in range(0, len(the_images)):

            monitoring.to_log_and_console("    .. process '"
                                          + the_images[i].split(os.path.sep)[-1] + "' for fusion", 2)

            if do_something[i] is False:
                monitoring.to_log_and_console("       nothing to do", 2)
                continue

            #
            # first image: resampling only
            #
            if i == 0:
                #
                # image center
                #
                im = imread(the_images[i])
                ref_center = np.multiply(im.shape[:3], im.voxelsize) / 2.0
                del im

                #
                # resampling first image
                #
                monitoring.to_log_and_console("       resampling '" + the_images[i].split(os.path.sep)[-1]
                                              + "' at " + str(parameters.target_resolution), 2)
                if not os.path.isfile(res_images[i]) or monitoring.forceResultsToBeBuilt is True:
                    cpp_wrapping.apply_transformation(the_images[i], res_images[i], the_transformation=None,
                                                      template_image=None,
                                                      voxel_size=parameters.target_resolution,
                                                      interpolation_mode='linear',
                                                      monitoring=monitoring)
                else:
                    monitoring.to_log_and_console("       already existing", 2)

            #
            # other images:
            # - channel #0: co-registration
            # - other channels: resampling with transformation of channel #0
            #
            else:
                if c == 0:
                    monitoring.to_log_and_console("       co-registering '" + the_images[i].split(os.path.sep)[-1]
                                                  + "'", 2)
                    #
                    # initial transformation
                    #
                    if not os.path.isfile(init_trsfs[i]) or monitoring.forceResultsToBeBuilt is True:
                        if i == 1:
                            angle = 0.0
                        else:
                            angle = default_angle
                        monitoring.to_log_and_console("       angle used for '" + init_trsfs[i].split(os.path.sep)[-1]
                                                      + "' is " + str(angle), 2)
                        im = imread(the_images[i])
                        flo_center = np.multiply(im.shape[:3], im.voxelsize) / 2.0
                        del im

                        #
                        # the initial transformation was computed with _axis_rotation_matrix(). To compute the
                        # translation, it preserves the center of the field of view of the floating image.
                        # However it seems more coherent to compute a translation that put the center of FOV of the
                        # floating image onto he FOV of the reference one.
                        #
                        # the call to _axis_rotation_matrix() was
                        # rotation_matrix = _axis_rotation_matrix(axis="Y", angle=angle, min_space=(0, 0, 0),
                        #                                         max_space=np.multiply(im.shape[:3], im.voxelsize))
                        # Note: it requires that 'im' is deleted after the call
                        #
                        # it can be mimicked by
                        # _ init_rotation_matrix(axis="Y", angle=angle, ref_center=flo_center, flo_center=flo_center)
                        #

                        rotation_matrix = _init_rotation_matrix(axis="Y", angle=angle, ref_center=ref_center,
                                                                flo_center=flo_center)

                        np.savetxt(init_trsfs[i], rotation_matrix)
                        del rotation_matrix

                    #
                    # a two-fold registration, translation then affine, could be preferable
                    #
                    if not os.path.isfile(res_images[i]) or not os.path.isfile(res_trsfs[i]) \
                            or monitoring.forceResultsToBeBuilt is True:
                        if parameters.acquisition_registration[0].compute_registration is True:
                            _linear_registration(res_images[0], the_images[i], res_images[i],
                                                 prereg_trsfs[i], init_trsfs[i], parameters.acquisition_registration[0])
                            _linear_registration(res_images[0], the_images[i], res_images[i],
                                                 res_trsfs[i], prereg_trsfs[i], parameters.acquisition_registration[1])
                        else:
                            _linear_registration(res_images[0], the_images[i], res_images[i],
                                                 res_trsfs[i], init_trsfs[i], parameters.acquisition_registration[1])
                    else:
                        monitoring.to_log_and_console("       already existing", 2)

                    #
                    # check whether the registration was successful
                    #
                    if not os.path.isfile(res_images[i]) or not os.path.isfile(res_trsfs[i]):
                        monitoring.to_log_and_console(proc + ": error when registering image " + str(i), 0)
                        monitoring.to_log_and_console("   image " + str(res_images[i]) + " or transformation "
                                                      + str(res_trsfs[i]) + " is not existing", 0)
                        monitoring.to_log_and_console("Exiting.", 0)
                        sys.exit(1)

                #
                # other channels
                #
                else:
                    monitoring.to_log_and_console("       resampling '" + the_images[i].split(os.path.sep)[-1] + "'", 2)
                    if not os.path.isfile(res_images[i]) or monitoring.forceResultsToBeBuilt is True:
                        cpp_wrapping.apply_transformation(the_images[i], res_images[i],
                                                          the_transformation=res_trsfs[i], template_image=res_images[0],
                                                          voxel_size=None, interpolation_mode='linear',
                                                          monitoring=monitoring)
                    else:
                        monitoring.to_log_and_console("       already existing", 2)

            #
            # compute weighting masks on every channel
            # - mask is computed on an untransformed image
            #   however, resolution may have changed, or it can be cropped
            #   or it can be mirrored (default behavior is that mask are computed on the '_crop' images
            # - mask are then transformed with the computed transformation
            #

            monitoring.to_log_and_console("       .. computing weights for fusion", 2)

            decreasing_weight_with_z = None

            if i == 0:
                # stack 0, left camera
                decreasing_weight_with_z = True

            elif i == 1:
                # stack 0, right camera
                decreasing_weight_with_z = False

            elif i == 2:
                # stack 1, left camera
                decreasing_weight_with_z = True

            elif i == 3:
                # stack 1, right camera
                decreasing_weight_with_z = False

            if not os.path.isfile(unreg_weight_images[i]) or monitoring.forceResultsToBeBuilt is True:
                #
                #
                #
                _build_unreg_weighting_image(the_images[i], unreg_weight_images[i], decreasing_weight_with_z,
                                             experiment.rawdata_dir.channel[c].fusion_weighting)
            else:
                monitoring.to_log_and_console("          already existing", 2)

            monitoring.to_log_and_console("          resampling '" + unreg_weight_images[i].split(os.path.sep)[-1]
                                          + "'", 2)
            if i == 0:
                if not os.path.isfile(weight_images[i]) or monitoring.forceResultsToBeBuilt is True:
                    cpp_wrapping.apply_transformation(unreg_weight_images[i], weight_images[i],
                                                      the_transformation=None, template_image=None,
                                                      voxel_size=parameters.target_resolution,
                                                      interpolation_mode='linear', monitoring=monitoring)
                else:
                    monitoring.to_log_and_console("          already existing", 2)
            else:
                if not os.path.isfile(weight_images[i]) or monitoring.forceResultsToBeBuilt is True:
                    cpp_wrapping.apply_transformation(unreg_weight_images[i], weight_images[i],
                                                      the_transformation=res_trsfs[i], template_image=res_images[0],
                                                      voxel_size=None, interpolation_mode='linear',
                                                      monitoring=monitoring)
                else:
                    monitoring.to_log_and_console("          already existing", 2)

        #
        # compute fused image as a linear combination of co-registered images
        # the sun of weights have been precomputed to mimic historical behavior
        #
        # do not forget to cast the result on 16 bits
        #

        monitoring.to_log_and_console("    .. combining images", 2)

        if parameters.fusion_cropping is True:
            tmp_fused_image = common.add_suffix(fused_image, "_uncropped_fusion",
                                                new_dirname=experiment.rawdata_dir.get_tmp_directory(4, c),
                                                new_extension=experiment.default_image_suffix)
        else:
            tmp_fused_image = os.path.join(experiment.fusion_dir.get_directory(c), fused_image)

        cpp_wrapping.linear_combination(weight_images, res_images, tmp_fused_image, monitoring=monitoring)

        if not os.path.isfile(tmp_fused_image):
            monitoring.to_log_and_console(proc + ': fused image (channel #' + str(c) + ') has not been generated', 0)
            monitoring.to_log_and_console("Exiting.", 0)
            sys.exit(1)

        #
        #
        #
        if parameters.xzsection_extraction is True:
            _extract_xzsection(weight_images, res_images, tmp_fused_image, c, experiment)

        #
        # last crop
        #
        if parameters.fusion_cropping is True:

            if c == 0:
                fusion_box = _crop_bounding_box(tmp_fused_image)

            monitoring.to_log_and_console("    .. cropping '" + fused_image.split(os.path.sep)[-1], 2)
            _crop_disk_image(tmp_fused_image, os.path.join(experiment.fusion_dir.get_directory(c), fused_image),
                             fusion_box,
                             parameters.fusion_cropping_margin_x_0,
                             parameters.fusion_cropping_margin_x_1,
                             parameters.fusion_cropping_margin_y_0,
                             parameters.fusion_cropping_margin_y_1)
    return


#
# hierarchical fusion procedure
# - each stack is reconstructed
# - stack are co-registered
# - all acquisitions are fused
#

def _hierarchical_fusion_process(input_image_list, the_image_list, fused_image, experiment, parameters):
    """

    :param input_image_list:
    :param the_image_list: list of list of preprocessed images (in the temporary directory), ie after
        1. (optional) slit line correction
        2. resolution change (only in X and Y directions)
        3. (optional) 2D crop
        4. mirroring of the right camera images (parameter dependent)

    :param fused_image:
    :param experiment:
    :param parameters:
    :return:
    """
    
    proc = "_hierarchical_fusion_process"

    n_channels = experiment.rawdata_dir.get_number_channels()

    if monitoring.debug > 1:
        print("")
        print(proc + " was called with:")
        print("- input_image_list = " + str(input_image_list))
        print("- the_image_list = " + str(the_image_list))
        print("- fused_image = " + str(fused_image))
        for c in range(n_channels):
            experiment.rawdata_dir.channel[c].print_parameters('channel #' + str(c))
        print("")

    #
    # parameter type checking
    #

    if not isinstance(experiment, common.Experiment):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'experiment' variable: "
                                      + str(type(experiment)))
        sys.exit(1)

    if not isinstance(parameters, FusionParameters):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'parameters' variable: "
                                      + str(type(parameters)))
        sys.exit(1)
    #
    # list of registered images
    #

    res_image_list = list()

    for c in range(n_channels):
        the_images = the_image_list[c]
        res_images = []
        for i in range(0, len(the_images)):
            res_images.append(common.add_suffix(input_image_list[c][i], "_reg",
                                                new_dirname=experiment.rawdata_dir.get_tmp_directory(i, c),
                                                new_extension=experiment.default_image_suffix))
        res_image_list.append(res_images)

    #
    # is there something to do ?
    # check whether fused images are missing
    # - if one fusion image is missing, re-process channel #0 to get weights and sum of weights
    # - if the fusion image exists, check whether the registered images exists
    #

    do_something = [False] * len(input_image_list[0])

    for c in range(n_channels):

        the_images = the_image_list[c]
        res_images = res_image_list[c]
        if not os.path.isfile(os.path.join(experiment.fusion_dir.get_directory(c), fused_image)):
            do_something = [True] * len(input_image_list[0])
            break

        for i in range(0, len(the_images)):
            if os.path.isfile(res_images[i]):
                if monitoring.forceResultsToBeBuilt is True:
                    do_something[i] = True
            else:
                do_something[i] = True

    for i in range(1, len(input_image_list[0])):
        if do_something[i] is True:
            do_something[0] = True

    #
    # stack reconstruction on channel #0
    #

    the_images = the_image_list[0]
    stack_res_images = []
    res_images = res_image_list[0]
    stack_resample_trsfs = []
    res_trsfs = []

    stack_prereg_trsfs = []
    stack_res_trsfs = []
    unreg_weight_images_list = []
    stack_weight_images = []
    weight_images_list = []

    #
    # additional files only for the first channel
    #

    for i in range(0, len(the_images)):
        if i == 0 or i == 1:
            stack_res_images.append(common.add_suffix(input_image_list[0][i], "_reg",
                                                      new_dirname=experiment.rawdata_dir.get_tmp_directory(i, 0),
                                                      new_extension=experiment.default_image_suffix))
            stack_res_trsfs.append(common.add_suffix(input_image_list[0][i], "_reg",
                                                     new_dirname=experiment.rawdata_dir.get_tmp_directory(i, 0),
                                                     new_extension="trsf"))
        else:
            stack_res_images.append(common.add_suffix(input_image_list[0][i], "_stack_reg",
                                                      new_dirname=experiment.rawdata_dir.get_tmp_directory(i, 0),
                                                      new_extension=experiment.default_image_suffix))
            stack_res_trsfs.append(common.add_suffix(input_image_list[0][i], "_stack_reg",
                                                     new_dirname=experiment.rawdata_dir.get_tmp_directory(i, 0),
                                                     new_extension="trsf"))

        stack_resample_trsfs.append(common.add_suffix(input_image_list[0][i], "_resolutionchange",
                                                      new_dirname=experiment.rawdata_dir.get_tmp_directory(i, 0),
                                                      new_extension="trsf"))

        res_trsfs.append(common.add_suffix(input_image_list[0][i], "_reg",
                                           new_dirname=experiment.rawdata_dir.get_tmp_directory(i, 0),
                                           new_extension="trsf"))

        stack_prereg_trsfs.append(common.add_suffix(input_image_list[0][i], "_stack_prereg",
                                                    new_dirname=experiment.rawdata_dir.get_tmp_directory(i, 0),
                                                    new_extension="trsf"))

        if i == 0 or i == 1:
            stack_weight_images.append(common.add_suffix(input_image_list[0][i], "_weight_0",
                                                         new_dirname=experiment.rawdata_dir.get_tmp_directory(i, 0),
                                                         new_extension=experiment.default_image_suffix))
        else:
            stack_weight_images.append(common.add_suffix(input_image_list[0][i], "_stack_weight_0",
                                                         new_dirname=experiment.rawdata_dir.get_tmp_directory(i, 0),
                                                         new_extension=experiment.default_image_suffix))

    #
    # the final image is a weighting sum of the transformed acquisition image
    # weighting may be different for all channel
    #
    for c in range(n_channels):
        unreg_weight_images = []
        weight_images = []
        cref = c
        for i in range(0, c):
            if experiment.rawdata_dir.channel[c].fusion_weighting == experiment.rawdata_dir.channel[i].fusion_weighting:
                cref = i
        #
        # check if the weighting mode was used for previous channels (ie weights have already been computed)
        #
        if cref == c:
            for i in range(0, len(the_images)):
                unreg_weight_images.append(common.add_suffix(input_image_list[c][i], "_init_weight_" + str(c),
                                                             new_dirname=experiment.rawdata_dir.get_tmp_directory(i, c),
                                                             new_extension=experiment.default_image_suffix))
                weight_images.append(common.add_suffix(input_image_list[c][i], "_weight_" + str(c),
                                                       new_dirname=experiment.rawdata_dir.get_tmp_directory(i, c),
                                                       new_extension=experiment.default_image_suffix))
        else:
            unreg_weight_images = unreg_weight_images_list[cref]
            weight_images = weight_images_list[cref]

        unreg_weight_images_list.append(unreg_weight_images)
        weight_images_list.append(weight_images)

    #
    # there is one temporary path per acquisition (from 0 to 3) and an additional one which is the
    # parent directory (see _fusion_preprocess())
    #
    stack_fused_images = []
    for stack in range(2):
        stack_fused_images.append(common.add_suffix(fused_image, "_stack_" + str(stack),
                                                    new_dirname=experiment.rawdata_dir.get_tmp_directory(4, 0),
                                                    new_extension=experiment.default_image_suffix))

    #
    # stack #0, co-register acquisitions #0 and #1
    # stack #1, co-register acquisitions #2 and #3
    #

    if n_channels > 1:
        monitoring.to_log_and_console("    .. process channel #0", 2)

    unreg_weight_images = unreg_weight_images_list[0]
    weight_images = weight_images_list[0]

    for stack in range(2):

        monitoring.to_log_and_console("    .. reconstruct stack #" + str(stack))

        #
        # resample acquisition 2*stack+0 [0, 2]
        # co-register acquisition 2*stack+1 [1, 3]
        #

        for j in range(2):
            i = 2*stack+j
            r = 2*stack
            monitoring.to_log_and_console("      .. process '" + the_images[i].split(os.path.sep)[-1] + "' for fusion",
                                          2)

            if do_something[i] is False:
                monitoring.to_log_and_console("         nothing to do", 2)
                continue

            #
            # first image: resampling only
            #
            if j == 0:
                #
                # image center
                #
                im = imread(the_images[i])
                ref_center = np.multiply(im.shape[:3], im.voxelsize) / 2.0
                del im

                #
                # resampling first image
                #
                monitoring.to_log_and_console("         resampling '" + the_images[i].split(os.path.sep)[-1]
                                              + "' at " + str(parameters.target_resolution), 2)
                if not os.path.isfile(res_images[i]) or monitoring.forceResultsToBeBuilt is True:
                    cpp_wrapping.apply_transformation(the_images[i], stack_res_images[i],
                                                      the_transformation=None,
                                                      template_image=None,
                                                      res_transformation=stack_resample_trsfs[i],
                                                      voxel_size=parameters.target_resolution,
                                                      interpolation_mode='linear',
                                                      monitoring=monitoring)
                else:
                    monitoring.to_log_and_console("         already existing", 2)

                #
                # other image:
                # - channel #0: co-registration
                # - other channels: resampling with transformation of channel #0
                #
            else:

                monitoring.to_log_and_console("         co-registering '" + the_images[i].split(os.path.sep)[-1]
                                              + "'", 2)

                #
                # a two-fold registration, translation then affine, could be preferable
                #
                if not os.path.isfile(stack_res_images[i]) or not os.path.isfile(stack_res_trsfs[i]) \
                        or monitoring.forceResultsToBeBuilt is True:
                    if parameters.acquisition_registration[0].compute_registration is True:
                        _linear_registration(stack_res_images[r], the_images[i], res_images[i],
                                             stack_prereg_trsfs[i], None, parameters.acquisition_registration[0])
                        _linear_registration(stack_res_images[r], the_images[i], stack_res_images[i],
                                             stack_res_trsfs[i], stack_prereg_trsfs[i],
                                             parameters.acquisition_registration[1])
                    else:
                        _linear_registration(stack_res_images[r], the_images[i], stack_res_images[i],
                                             stack_res_trsfs[i], None, parameters.acquisition_registration[1])
                else:
                    monitoring.to_log_and_console("         already existing", 2)

                #
                # check whether the registration was successful
                #
                if not os.path.isfile(stack_res_images[i]) or not os.path.isfile(stack_res_trsfs[i]):
                    monitoring.to_log_and_console(proc + ": error when registering image " + str(i), 0)
                    monitoring.to_log_and_console("   image " + str(stack_res_images[i]) + " or transformation "
                                                  + str(stack_res_trsfs[i]) + " is not existing", 0)
                    monitoring.to_log_and_console("Exiting.", 0)
                    sys.exit(1)

        #
        # compute weights
        #
        monitoring.to_log_and_console("      .. computing weights for stack fusion of stack #" + str(stack), 2)
        for j in range(2):
            i = 2 * stack + j
            r = 2 * stack
            monitoring.to_log_and_console("         process '"
                                          + the_images[i].split(os.path.sep)[-1] + "' for weight", 2)

            decreasing_weight_with_z = None

            if i == 0:
                # stack 0, left camera
                decreasing_weight_with_z = True

            elif i == 1:
                # stack 0, right camera
                decreasing_weight_with_z = False

            elif i == 2:
                # stack 1, left camera
                decreasing_weight_with_z = True

            elif i == 3:
                # stack 1, right camera
                decreasing_weight_with_z = False

            if not os.path.isfile(unreg_weight_images[i]) or monitoring.forceResultsToBeBuilt is True:
                _build_unreg_weighting_image(the_images[i], unreg_weight_images[i], decreasing_weight_with_z,
                                             experiment.rawdata_dir.channel[0].fusion_weighting)
            else:
                monitoring.to_log_and_console("         already existing", 2)

            monitoring.to_log_and_console("         resampling '" + unreg_weight_images[i].split(os.path.sep)[-1]
                                          + "'", 2)
            if j == 0:
                if not os.path.isfile(stack_weight_images[i]) or monitoring.forceResultsToBeBuilt is True:
                    cpp_wrapping.apply_transformation(unreg_weight_images[i], stack_weight_images[i],
                                                      the_transformation=None, template_image=None,
                                                      voxel_size=parameters.target_resolution,
                                                      interpolation_mode='linear', monitoring=monitoring)
                else:
                    monitoring.to_log_and_console("         already existing", 2)
            else:
                if not os.path.isfile(stack_weight_images[i]) or monitoring.forceResultsToBeBuilt is True:
                    cpp_wrapping.apply_transformation(unreg_weight_images[i], stack_weight_images[i],
                                                      the_transformation=stack_res_trsfs[i],
                                                      template_image=stack_res_images[r],
                                                      voxel_size=None, interpolation_mode='linear',
                                                      monitoring=monitoring)
                else:
                    monitoring.to_log_and_console("         already existing", 2)

        #
        # compute fused image as a linear combination of co-registered images
        # the sun of weights have been precomputed to mimic historical behavior
        #
        # do not forget to cast the result on 16 bits
        #

        monitoring.to_log_and_console("      .. combining images for stack #" + str(stack), 2)

        if not os.path.isfile(stack_fused_images[stack]) or monitoring.forceResultsToBeBuilt is True:
            cpp_wrapping.linear_combination(stack_weight_images[2*stack:2*(stack+1)],
                                            stack_res_images[2*stack:2*(stack+1)],
                                            stack_fused_images[stack], monitoring=monitoring)

    #
    # stacks #0 and #1 have been reconstructed, now we co-registered them
    #

    #
    # compute initial rotation matrix
    #

    monitoring.to_log_and_console("    .. co-registering stack #1 onto stack #0", 2)
    monitoring.to_log_and_console("       initial transformation", 2)

    init_trsfs = common.add_suffix(input_image_list[0][2], "_init", 
                                   new_dirname=experiment.rawdata_dir.get_tmp_directory(2, 0), new_extension="trsf")
    if parameters.acquisition_orientation.lower() == 'left':
        angle = 270.0
    elif parameters.acquisition_orientation.lower() == 'right':
        angle = 90.0
    im = imread(the_images[0])
    ref_center = np.multiply(im.shape[:3], im.voxelsize) / 2.0
    del im
    im = imread(the_images[2])
    flo_center = np.multiply(im.shape[:3], im.voxelsize) / 2.0
    del im
    rotation_matrix = _init_rotation_matrix(axis="Y", angle=angle, ref_center=ref_center, flo_center=flo_center)
    np.savetxt(init_trsfs, rotation_matrix)
    del rotation_matrix

    monitoring.to_log_and_console("       registration", 2)

    reg_stack_image = common.add_suffix(fused_image, "_stack_" + str(stack) + "_reg",
                                        new_dirname=experiment.rawdata_dir.get_tmp_directory(4, 0),
                                        new_extension=experiment.default_image_suffix)
    reg_stack_trsf = common.add_suffix(fused_image, "_stack_" + str(stack) + "_reg",
                                       new_dirname=experiment.rawdata_dir.get_tmp_directory(4, 0),
                                       new_extension="trsf")

    if not os.path.isfile(reg_stack_image) or not os.path.isfile(reg_stack_trsf) \
            or monitoring.forceResultsToBeBuilt is True:
        if parameters.stack_registration[0].compute_registration is True and \
                parameters.stack_registration[1].compute_registration is True:
            monitoring.to_log_and_console("           registration 1/2", 2)
            _blockmatching(stack_fused_images[0], stack_fused_images[1], reg_stack_image,
                           reg_stack_trsf, path_init_trsf=init_trsfs, parameters=parameters.stack_registration[0])
            monitoring.to_log_and_console("           registration 2/2", 2)
            _blockmatching(stack_fused_images[0], stack_fused_images[1], reg_stack_image,
                           reg_stack_trsf, path_init_trsf=init_trsfs, parameters=parameters.stack_registration[1])
        elif parameters.stack_registration[0].compute_registration is True and \
                parameters.stack_registration[1].compute_registration is False:
            _blockmatching(stack_fused_images[0], stack_fused_images[1], reg_stack_image,
                            reg_stack_trsf, path_init_trsf=init_trsfs, parameters=parameters.stack_registration[0])
        elif parameters.stack_registration[0].compute_registration is False and \
                parameters.stack_registration[1].compute_registration is True:
            _blockmatching(stack_fused_images[0], stack_fused_images[1], reg_stack_image,
                           reg_stack_trsf, path_init_trsf=init_trsfs, parameters=parameters.stack_registration[1])
        else:
            monitoring.to_log_and_console(proc + ': no registration to be done ?!', 0)
            monitoring.to_log_and_console("Exiting.", 0)
            sys.exit(1)

    monitoring.to_log_and_console("       transform angle 2 data", 2)

    i = 2
    cpp_wrapping.compose_trsf([stack_resample_trsfs[i], reg_stack_trsf], res_trsfs[i], monitoring=monitoring)
    cpp_wrapping.applyTrsf(the_images[i], res_images[i], the_transformation=res_trsfs[i],
                           template_image=res_images[0], monitoring=monitoring)
    cpp_wrapping.applyTrsf(unreg_weight_images[i], weight_images[i], the_transformation=res_trsfs[i],
                           template_image=res_images[0], monitoring=monitoring)

    monitoring.to_log_and_console("       transform angle 3 data", 2)

    i = 3
    cpp_wrapping.compose_trsf([stack_res_trsfs[i], reg_stack_trsf], res_trsfs[i], monitoring=monitoring)
    cpp_wrapping.applyTrsf(the_images[i], res_images[i], the_transformation=res_trsfs[i],
                           template_image=res_images[0], monitoring=monitoring)
    cpp_wrapping.applyTrsf(unreg_weight_images[i], weight_images[i], the_transformation=res_trsfs[i],
                           template_image=res_images[0], monitoring=monitoring)

    #
    # compute fused image as a linear combination of co-registered images
    # the sun of weights have been precomputed to mimic historical behavior
    #
    # do not forget to cast the result on 16 bits
    #

    monitoring.to_log_and_console("    .. combining images", 2)

    if parameters.fusion_cropping is True:
        tmp_fused_image = common.add_suffix(fused_image, "_uncropped_fusion",
                                            new_dirname=experiment.rawdata_dir.get_tmp_directory(4, 0),
                                            new_extension=experiment.default_image_suffix)
    else:
        tmp_fused_image = os.path.join(experiment.fusion_dir.get_directory(0), fused_image)

    cpp_wrapping.linear_combination(weight_images, res_images, tmp_fused_image, monitoring=monitoring)

    if not os.path.isfile(tmp_fused_image):
        monitoring.to_log_and_console(proc + ': fused image has not been generated', 0)
        monitoring.to_log_and_console("Exiting.", 0)
        sys.exit(1)

    #
    #
    #
    if parameters.xzsection_extraction is True:
        _extract_xzsection(weight_images, res_images, tmp_fused_image, 0, experiment)

    #
    # last crop
    #
    if parameters.fusion_cropping is True:

        fusion_box = _crop_bounding_box(tmp_fused_image)

        monitoring.to_log_and_console("    .. cropping '" + fused_image.split(os.path.sep)[-1], 2)
        _crop_disk_image(tmp_fused_image, os.path.join(experiment.fusion_dir.get_directory(0), fused_image), fusion_box,
                         parameters.fusion_cropping_margin_x_0,
                         parameters.fusion_cropping_margin_x_1,
                         parameters.fusion_cropping_margin_y_0,
                         parameters.fusion_cropping_margin_y_1)

    #
    # other channels
    #
    for c in range(1, n_channels):

        if n_channels > 1:
            monitoring.to_log_and_console("    .. process channel #" + str(c), 2)

        the_images = the_image_list[c]
        res_images = res_image_list[c]
        unreg_weight_images = unreg_weight_images_list[c]
        weight_images = weight_images_list[c]

        for i in range(0, len(the_images)):

            monitoring.to_log_and_console("    .. process '"
                                          + the_images[i].split(os.path.sep)[-1] + "' for fusion", 2)

            if do_something[i] is False:
                monitoring.to_log_and_console("       nothing to do", 2)
                continue

            #
            # first image: resampling only
            #
            if i == 0:
                #
                # image center
                #
                im = imread(the_images[i])
                ref_center = np.multiply(im.shape[:3], im.voxelsize) / 2.0
                del im

                #
                # resampling first image
                #
                monitoring.to_log_and_console("       resampling '" + the_images[i].split(os.path.sep)[-1]
                                              + "' at " + str(parameters.target_resolution), 2)
                if not os.path.isfile(res_images[i]) or monitoring.forceResultsToBeBuilt is True:
                    cpp_wrapping.apply_transformation(the_images[i], res_images[i], the_transformation=None,
                                                      template_image=None,
                                                      voxel_size=parameters.target_resolution,
                                                      interpolation_mode='linear',
                                                      monitoring=monitoring)
                else:
                    monitoring.to_log_and_console("       already existing", 2)

            #
            # other images:
            # - channel #0: co-registration
            # - other channels: resampling with transformation of channel #0
            #
            else:
                monitoring.to_log_and_console("       resampling '" + the_images[i].split(os.path.sep)[-1] + "'", 2)
                if not os.path.isfile(res_images[i]) or monitoring.forceResultsToBeBuilt is True:
                    cpp_wrapping.apply_transformation(the_images[i], res_images[i],
                                                      the_transformation=res_trsfs[i], template_image=res_images[0],
                                                      voxel_size=None, interpolation_mode='linear',
                                                      monitoring=monitoring)
                else:
                    monitoring.to_log_and_console("       already existing", 2)

            #
            # weighting masks on every channel
            #
            monitoring.to_log_and_console("       .. computing weights for fusion", 2)

            if i % 2 == 1:
                direction = False
            else:
                direction = True

            if not os.path.isfile(unreg_weight_images[i]) or monitoring.forceResultsToBeBuilt is True:
                #
                #
                #
                _build_unreg_weighting_image(the_images[i], unreg_weight_images[i], direction,
                                             experiment.rawdata_dir.channel[c].fusion_weighting)
            else:
                monitoring.to_log_and_console("          already existing", 2)

            monitoring.to_log_and_console("          resampling '" + unreg_weight_images[i].split(os.path.sep)[-1]
                                          + "'", 2)
            if i == 0:
                if not os.path.isfile(weight_images[i]) or monitoring.forceResultsToBeBuilt is True:
                    cpp_wrapping.apply_transformation(unreg_weight_images[i], weight_images[i],
                                                      the_transformation=None, template_image=None,
                                                      voxel_size=parameters.target_resolution,
                                                      interpolation_mode='linear', monitoring=monitoring)
                else:
                    monitoring.to_log_and_console("          already existing", 2)
            else:
                if not os.path.isfile(weight_images[i]) or monitoring.forceResultsToBeBuilt is True:
                    cpp_wrapping.apply_transformation(unreg_weight_images[i], weight_images[i],
                                                      the_transformation=res_trsfs[i], template_image=res_images[0],
                                                      voxel_size=None, interpolation_mode='linear',
                                                      monitoring=monitoring)
                else:
                    monitoring.to_log_and_console("          already existing", 2)

        #
        # compute fused image as a linear combination of co-registered images
        # the sun of weights have been precomputed to mimic historical behavior
        #
        # do not forget to cast the result on 16 bits
        #

        monitoring.to_log_and_console("    .. combining images", 2)

        if parameters.fusion_cropping is True:
            tmp_fused_image = common.add_suffix(fused_image, "_uncropped_fusion",
                                                new_dirname=experiment.rawdata_dir.get_tmp_directory(4, c),
                                                new_extension=experiment.default_image_suffix)
        else:
            tmp_fused_image = os.path.join(experiment.fusion_dir.get_directory(c), fused_image)

        cpp_wrapping.linear_combination(weight_images, res_images, tmp_fused_image, monitoring=monitoring)

        if not os.path.isfile(tmp_fused_image):
            monitoring.to_log_and_console(proc + ': fused image (channel #' + str(c) +') has not been generated', 0)
            monitoring.to_log_and_console("Exiting.", 0)
            sys.exit(1)

        #
        #
        #
        if parameters.xzsection_extraction is True:
            _extract_xzsection(weight_images, res_images, tmp_fused_image, c, experiment)

        #
        # last crop
        #
        if parameters.fusion_cropping is True:

            monitoring.to_log_and_console("    .. cropping '" + fused_image.split(os.path.sep)[-1], 2)
            _crop_disk_image(tmp_fused_image, os.path.join(experiment.fusion_dir.get_directory(c), fused_image),
                             fusion_box, parameters.fusion_cropping_margin_x_0, parameters.fusion_cropping_margin_x_1,
                             parameters.fusion_cropping_margin_y_0, parameters.fusion_cropping_margin_y_1)

    return


#
# raw data have been read and eventually converted
# do some pre-processing of each acquisition
# 1. (optional) slit line correction
# 2. resolution change (only in X and Y directions)
# 3. (optional) 2D crop
# 4. mirroring of the right camera images (parameter dependent) wrt the X axis
# 5. mirroring of the all camera images (parameter dependent) wrt the Z axis
# then call a fusion method
#

def _fusion_process(input_image_list, fused_image, experiment, parameters):
    """
    
    :param input_image_list: a list of list of images to be fused. One list per channel
           the list of images to be fused contains successively
           - the left camera of stack #0
           - the right camera of stack #0
           - the left camera of stack #1
           - the right camera of stack #1
    :param fused_image: a generic name for the fusion result
           the same name will be used for each cahnnel
    :param experiment:
    :param parameters:
    :return:
    """

    proc = 'fusion_process'

    n_channels = experiment.rawdata_dir.get_number_channels()

    if monitoring.debug > 1:
        print("")
        print(proc + " was called with:")
        print("- input_image_list = " + str(input_image_list))
        print("- fused_image = " + str(fused_image))
        for c in range(n_channels):
            experiment.rawdata_dir.channel[c].print_parameters('channel #' + str(c))
        print("")

    #
    # parameter type checking
    #

    if not isinstance(experiment, common.Experiment):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'experiment' variable: "
                                      + str(type(experiment)))
        sys.exit(1)

    if not isinstance(parameters, FusionParameters):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'parameters' variable: "
                                      + str(type(parameters)))
        sys.exit(1)

    #
    # nothing to do if the fused image exists
    #

    do_something = False
    for c in range(n_channels):
        if os.path.isfile(os.path.join(experiment.fusion_dir.get_directory(c), fused_image)):
            if monitoring.forceResultsToBeBuilt is False:
                monitoring.to_log_and_console('    fused channel #' + str(c) + ' already existing', 2)
            else:
                monitoring.to_log_and_console('    fused channel #' + str(c) + ' already existing, but forced', 2)
                do_something = True
        else:
            do_something = True

    if do_something is False:
        return

    #
    # how to copy a list:
    # NB: 'res_images = inputImages' acts like pointers
    #

    res_image_list = input_image_list[:]

    #
    # First steps:
    # 1. (optional) slit line correction
    # 2. resolution change (only in X and Y directions)
    # 3. (optional) 2D crop
    # 4. mirroring of the right camera images (parameter dependent) wrt the X axis
    # 5. mirroring of all camera images (parameter dependent) wrt the Z axis
    #

    #
    # 1. slit line correction
    # these corrections are done on original data (without resampling) on channel[0]
    # the compute corrections are then used for the other channels
    # Crop could be done beforehand to reduce the computational burden
    #

    if parameters.acquisition_slit_line_correction is True:

        the_image_list = res_image_list[:]
        res_image_list = list()
        corrections = list()

        #
        # build the file names
        #

        for c in range(n_channels):

            the_images = the_image_list[c]
            res_images = []

            for i in range(0, len(the_images)):
                res_images.append(common.add_suffix(input_image_list[c][i], "_line_corrected",
                                                    new_dirname=experiment.rawdata_dir.get_tmp_directory(i, c),
                                                    new_extension=experiment.default_image_suffix))
                if c == 0:
                    corrections.append(common.add_suffix(input_image_list[c][i], "_line_corrected",
                                                         new_dirname=experiment.rawdata_dir.get_tmp_directory(i, c),
                                                         new_extension='.txt'))
            res_image_list.append(res_images)

        #
        # is there something to do ?
        # check whether one corrected line image is missing
        # for channel #0, check also whether the correction file is missing
        #

        do_something = [False] * len(input_image_list[0])

        for c in range(n_channels):

            the_images = the_image_list[c]
            res_images = res_image_list[c]

            for i in range(0, len(the_images)):
                if os.path.isfile(res_images[i]):
                    if monitoring.forceResultsToBeBuilt is True:
                        do_something[i] = True
                else:
                    do_something[i] = True
                if c == 0:
                    if os.path.isfile(corrections[i]):
                        if monitoring.forceResultsToBeBuilt is True:
                            do_something[i] = True
                    else:
                        do_something[i] = True

        #
        # process
        # corrections are computed on channel #0
        # and are used for other channels
        #

        for c in range(n_channels):

            the_images = the_image_list[c]
            res_images = res_image_list[c]

            for i in range(0, len(the_images)):

                monitoring.to_log_and_console("    .. correcting slit lines of '"
                                              + the_images[i].split(os.path.sep)[-1] + "'", 2)

                if do_something[i] is False:
                    monitoring.to_log_and_console("       nothing to do", 2)
                    continue

                if c == 0:
                    cpp_wrapping.slitline_correction(the_images[i], res_images[i],
                                                     output_corrections=corrections[i],
                                                     monitoring=monitoring)
                else:
                    cpp_wrapping.slitline_correction(the_images[i], res_images[i],
                                                     input_corrections=corrections[i],
                                                     monitoring=monitoring)

    #
    # to do: linear filtering to compensate for resolution change
    # for a change of voxel size from x0 to x1
    # smooth with a Gaussian of sigma = \sqrt(2)^(ln(x0/x1) / ln(2))
    #

    #
    # 2. first change of resolution
    # - for X and Y: target resolution (supposed to be larger than original)
    # - for Z: original resolution (supposed to be larger than target)
    #

    the_image_list = res_image_list[:]
    res_image_list = list()

    for c in range(n_channels):

        the_images = the_image_list[c]
        res_images = []

        #
        # build the file names
        #

        for i in range(0, len(the_images)):
            res_images.append(common.add_suffix(input_image_list[c][i], "_resample",
                                                new_dirname=experiment.rawdata_dir.get_tmp_directory(i, c),
                                                new_extension=experiment.default_image_suffix))
        res_image_list.append(res_images)

        #
        # process
        #

        for i in range(0, len(the_images)):

            im = imread(the_images[i])
            if type(parameters.target_resolution) == int or type(parameters.target_resolution) == float:
                resampling_resolution = [parameters.target_resolution, parameters.target_resolution, im.voxelsize[2]]
            elif (type(parameters.target_resolution) == list or type(parameters.target_resolution) == tuple) \
                    and len(parameters.target_resolution) == 3:
                resampling_resolution = [parameters.target_resolution[0],
                                         parameters.target_resolution[1], im.voxelsize[2]]
            else:
                monitoring.to_log_and_console(proc+': unable to set target resolution for first resampling', 0)
                monitoring.to_log_and_console("\t target resolution was '"+str(parameters.target_resolution)+"'", 0)
                monitoring.to_log_and_console("\t image resolution was '" + str(im.voxelsize) + "'", 0)
                monitoring.to_log_and_console("Exiting.", 0)
                sys.exit(1)
            del im

            monitoring.to_log_and_console("    .. resampling '" + the_images[i].split(os.path.sep)[-1]
                                          + "' at " + str(resampling_resolution), 2)
            if not os.path.isfile(res_images[i]) or monitoring.forceResultsToBeBuilt is True:
                cpp_wrapping.apply_transformation(the_images[i], res_images[i], the_transformation=None,
                                                  template_image=None,
                                                  voxel_size=resampling_resolution, interpolation_mode='linear',
                                                  monitoring=monitoring)
            else:
                monitoring.to_log_and_console("       already existing", 2)

    #
    # 3. 2D crop of resampled acquisition images
    #

    if parameters.acquisition_cropping is True:

        the_image_list = res_image_list[:]
        res_image_list = list()

        #
        # build the file names
        #

        for c in range(n_channels):

            the_images = the_image_list[c]
            res_images = []

            for i in range(0, len(the_images)):
                res_images.append(common.add_suffix(input_image_list[c][i], "_crop",
                                                    new_dirname=experiment.rawdata_dir.get_tmp_directory(i, c),
                                                    new_extension=experiment.default_image_suffix))
            res_image_list.append(res_images)

        #
        # is there something to do ?
        # check whether one cropped image is missing
        #

        do_something = [False] * len(input_image_list[0])

        for c in range(n_channels):

            the_images = the_image_list[c]
            res_images = res_image_list[c]

            for i in range(0, len(the_images)):
                if os.path.isfile(res_images[i]):
                    if monitoring.forceResultsToBeBuilt is True:
                        do_something[i] = True
                else:
                    do_something[i] = True

        #
        # process
        # bounding box are computed on channel #0
        # and are used for other channels
        #

        box_list = list()

        for c in range(n_channels):

            the_images = the_image_list[c]
            res_images = res_image_list[c]

            for i in range(0, len(the_images)):

                monitoring.to_log_and_console("    .. cropping '"
                                              + the_images[i].split(os.path.sep)[-1] + "'", 2)

                if do_something[i] is False:
                    monitoring.to_log_and_console("       nothing to do", 2)
                    continue

                if c == 0:
                    box = _crop_bounding_box(the_images[i])
                    box_list.append(box)
                else:
                    box = box_list[i]

                if not os.path.isfile(res_images[i]) or monitoring.forceResultsToBeBuilt is True:
                    _crop_disk_image(the_images[i], res_images[i], box,
                                     parameters.acquisition_cropping_margin_x_0,
                                     parameters.acquisition_cropping_margin_x_1,
                                     parameters.acquisition_cropping_margin_y_0,
                                     parameters.acquisition_cropping_margin_y_1)
                else:
                    monitoring.to_log_and_console("       already existing", 2)

    #
    # 4. mirroring of the 'right; camera images (parameter dependent) wrt the X axis, if required
    # 5. mirroring of all camera images (parameter dependent) wrt the Z axis, if required
    #
    # Both are done at the same time to avoid reading/writing of images
    #

    if parameters.acquisition_mirrors is False \
        or parameters.acquisition_stack0_leftcamera_z_stacking.lower() == 'inverse' \
        or parameters.acquisition_stack1_leftcamera_z_stacking.lower() == 'inverse':

        the_image_list = res_image_list[:]
        res_image_list = list()

        for c in range(n_channels):

            the_images = the_image_list[c]
            res_images = []

            #
            # build the file names
            #

            for i in range(0, len(the_images)):

                if i == 0:
                    # stack 0, left camera
                    if parameters.acquisition_stack0_leftcamera_z_stacking.lower() == 'inverse':
                        res_images.append(common.add_suffix(input_image_list[c][i], "_mirror",
                                                            new_dirname=experiment.rawdata_dir.get_tmp_directory(i, c),
                                                            new_extension=experiment.default_image_suffix))
                    else:
                        res_images.append(the_images[i])
                elif i == 1:
                    # stack 0, right camera
                    if parameters.acquisition_stack0_leftcamera_z_stacking.lower() == 'inverse' \
                        or parameters.acquisition_mirrors is False:
                        res_images.append(common.add_suffix(input_image_list[c][i], "_mirror",
                                                            new_dirname=experiment.rawdata_dir.get_tmp_directory(i, c),
                                                            new_extension=experiment.default_image_suffix))
                    else:
                        res_images.append(the_images[i])
                elif i == 2:
                    # stack 1, left camera
                    if parameters.acquisition_stack1_leftcamera_z_stacking.lower() == 'inverse':
                        res_images.append(common.add_suffix(input_image_list[c][i], "_mirror",
                                                            new_dirname=experiment.rawdata_dir.get_tmp_directory(i, c),
                                                            new_extension=experiment.default_image_suffix))
                    else:
                        res_images.append(the_images[i])
                elif i == 3:
                    # stack 1, right camera
                    if parameters.acquisition_stack1_leftcamera_z_stacking.lower() == 'inverse' \
                        or parameters.acquisition_mirrors is False:
                        res_images.append(common.add_suffix(input_image_list[c][i], "_mirror",
                                                            new_dirname=experiment.rawdata_dir.get_tmp_directory(i, c),
                                                            new_extension=experiment.default_image_suffix))
                    else:
                        res_images.append(the_images[i])
                else:
                    monitoring.to_log_and_console("       weird index:'"+str(i)+"'", 2)

            res_image_list.append(res_images)

            #
            # process
            #

            for i in range(0, len(the_images)):

                if i == 0:
                    # stack 0, left camera
                    if parameters.acquisition_stack0_leftcamera_z_stacking.lower() == 'inverse':
                        monitoring.to_log_and_console("    .. mirroring  #" + str(i) + " '"
                                                      + the_images[i].split(os.path.sep)[-1], 2)
                        if not os.path.isfile(res_images[i]) or monitoring.forceResultsToBeBuilt is True:
                            the_im = imread(the_images[i])
                            res_im = SpatialImage(the_im.copy())[:, :, -1::-1]
                            res_im.voxelsize = the_im.voxelsize
                            imsave(res_images[i], res_im)
                            del the_im
                            del res_im
                        else:
                            monitoring.to_log_and_console("       already existing", 2)

                elif i == 1:
                    # stack 0, right camera
                    if parameters.acquisition_stack0_leftcamera_z_stacking.lower() == 'inverse' \
                        or parameters.acquisition_mirrors is False:
                        monitoring.to_log_and_console("    .. mirroring  #" + str(i) + " '"
                                                      + the_images[i].split(os.path.sep)[-1], 2)
                        if not os.path.isfile(res_images[i]) or monitoring.forceResultsToBeBuilt is True:
                            the_im = imread(the_images[i])
                            if parameters.acquisition_mirrors is False:
                                if parameters.acquisition_stack0_leftcamera_z_stacking.lower() == 'inverse':
                                    res_im = SpatialImage(the_im.copy())[-1::-1, :, -1::-1]
                                else:
                                    res_im = SpatialImage(the_im.copy())[-1::-1, :, :]
                            else:
                                if parameters.acquisition_stack0_leftcamera_z_stacking.lower() == 'inverse':
                                    res_im = SpatialImage(the_im.copy())[:, :, -1::-1]
                            res_im.voxelsize = the_im.voxelsize
                            imsave(res_images[i], res_im)
                            del the_im
                            del res_im
                        else:
                            monitoring.to_log_and_console("       already existing", 2)

                elif i == 2:
                    # stack 1, left camera
                    if parameters.acquisition_stack1_leftcamera_z_stacking.lower() == 'inverse':
                        monitoring.to_log_and_console("    .. mirroring  #" + str(i) + " '"
                                                      + the_images[i].split(os.path.sep)[-1], 2)
                        if not os.path.isfile(res_images[i]) or monitoring.forceResultsToBeBuilt is True:
                            the_im = imread(the_images[i])
                            res_im = SpatialImage(the_im.copy())[:, :, -1::-1]
                            res_im.voxelsize = the_im.voxelsize
                            imsave(res_images[i], res_im)
                            del the_im
                            del res_im
                        else:
                            monitoring.to_log_and_console("       already existing", 2)

                elif i == 3:
                    # stack 1, right camera
                    if parameters.acquisition_stack1_leftcamera_z_stacking.lower() == 'inverse' \
                        or parameters.acquisition_mirrors is False:
                        monitoring.to_log_and_console("    .. mirroring  #" + str(i) + " '"
                                                      + the_images[i].split(os.path.sep)[-1], 2)
                        if not os.path.isfile(res_images[i]) or monitoring.forceResultsToBeBuilt is True:
                            the_im = imread(the_images[i])
                            if parameters.acquisition_mirrors is False:
                                if parameters.acquisition_stack1_leftcamera_z_stacking.lower() == 'inverse':
                                    res_im = SpatialImage(the_im.copy())[-1::-1, :, -1::-1]
                                else:
                                    res_im = SpatialImage(the_im.copy())[-1::-1, :, :]
                            else:
                                if parameters.acquisition_stack1_leftcamera_z_stacking.lower() == 'inverse':
                                    res_im = SpatialImage(the_im.copy())[:, :, -1::-1]
                            res_im.voxelsize = the_im.voxelsize
                            imsave(res_images[i], res_im)
                            del the_im
                            del res_im
                        else:
                            monitoring.to_log_and_console("       already existing", 2)

    #
    #
    #
    if len(the_images) == 4 and parameters.fusion_strategy.lower() == 'hierarchical-fusion':
        monitoring.to_log_and_console("    .. hierarchical fusion", 2)
        _hierarchical_fusion_process(input_image_list, res_image_list, fused_image, experiment, parameters)
    else:
        #
        # direct fusion
        # each acquisition is co-registered with the left camera of stack #0
        monitoring.to_log_and_console("    .. direct fusion", 2)
        _direct_fusion_process(input_image_list, res_image_list, fused_image, experiment, parameters)

    return


#
#
# read the raw data
#
#


def _fusion_preprocess(input_images, fused_image, time_point, experiment, parameters):
    """

    :param input_images:
    :param fused_image:
    :param time_point:
    :param experiment:
    :param parameters:
    :return:
    """

    proc = "fusion_preprocess"

    if monitoring.debug > 1:

        print(proc + " was called with:")
        print("- input_images = " + str(input_images))
        print("- fused_image = " + str(fused_image))
        print("- time_point = " + str(time_point))
        print("")

    #
    # parameter type checking
    #

    if not isinstance(experiment, common.Experiment):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'experiment' variable: "
                                      + str(type(experiment)))
        sys.exit(1)

    if not isinstance(parameters, FusionParameters):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'parameters' variable: "
                                      + str(type(parameters)))
        sys.exit(1)

    #
    #
    #

    monitoring.to_log_and_console('... fusion of time ' + time_point, 1)
    n_channels = experiment.rawdata_dir.get_number_channels()
    if n_channels > 1:
        monitoring.to_log_and_console('    there are ' + str(n_channels) + ' channels to be fused', 1)

    #
    # check whether there exists some unfused channel
    #

    do_something = False
    for c in range(n_channels):
        if os.path.isfile(os.path.join(experiment.fusion_dir.get_directory(c), fused_image)):
            if not monitoring.forceResultsToBeBuilt:
                monitoring.to_log_and_console('    channel #' + str(c) + ' already existing', 2)
            else:
                monitoring.to_log_and_console('    channel #' + str(c) + ' already existing, but forced', 2)
                do_something = True
        else:
            do_something = True

    if do_something is False:
        monitoring.to_log_and_console('    nothing to do', 2)
        return

    #
    # start processing
    #

    start_time = time.time()

    #
    # directory for auxiliary files
    #
    # ANGLE_0: LC/Stack0000 ; stack_0_channel_0/Cam_Left_*
    # ANGLE_1: RC/Stack0000 ; stack_0_channel_0/Cam_Right_*
    # ANGLE_2: LC/Stack0001 ; stack_1_channel_0/Cam_Left_*
    # ANGLE_3: RC/Stack0001 ; stack_1_channel_0/Cam_Right_*
    #
    # experiment.rawdata_dir.get_tmp_directory(i, channel_id)
    # i=0 experiment.fusion_dir.get_directory(c) / "TEMP_time_value" / "ANGLE_0"
    # i=1 experiment.fusion_dir.get_directory(c) / "TEMP_time_value" / "ANGLE_1"
    # i=2 experiment.fusion_dir.get_directory(c) / "TEMP_time_value" / "ANGLE_2"
    # i=3 experiment.fusion_dir.get_directory(c) / "TEMP_time_value" / "ANGLE_3"
    # i=4 experiment.fusion_dir.get_directory(c) / "TEMP_time_value"
    #
    experiment.set_fusion_tmp_directory(int(time_point))
    experiment.fusion_dir.set_xzsection_directory(int(time_point))

    #
    # get image file names
    # - may involve unzipping and conversion
    #
    monitoring.to_log_and_console('    get original images', 2)

    image_list = list()

    for c in range(experiment.rawdata_dir.get_number_channels()):
        images = list()
        images.append(_read_image_name(experiment.rawdata_dir.channel[c].get_angle_path(0),
                                       experiment.rawdata_dir.get_tmp_directory(0, c),
                                       input_images[0],
                                       parameters.acquisition_resolution, experiment.default_image_suffix))
        images.append(_read_image_name(experiment.rawdata_dir.channel[c].get_angle_path(1),
                                       experiment.rawdata_dir.get_tmp_directory(1, c),
                                       input_images[1],
                                       parameters.acquisition_resolution, experiment.default_image_suffix))
        if len(input_images) == 4:
            images.append(_read_image_name(experiment.rawdata_dir.channel[c].get_angle_path(2),
                                           experiment.rawdata_dir.get_tmp_directory(2, c),
                                           input_images[2],
                                           parameters.acquisition_resolution, experiment.default_image_suffix))
            images.append(_read_image_name(experiment.rawdata_dir.channel[c].get_angle_path(3),
                                           experiment.rawdata_dir.get_tmp_directory(3, c),
                                           input_images[3],
                                           parameters.acquisition_resolution, experiment.default_image_suffix))
        image_list.append(images)

    #
    #
    #

    monitoring.to_log_and_console('    fuse images', 2)
    _fusion_process(image_list, fused_image, experiment, parameters)

    #
    # remove temporary files if required
    #

    if monitoring.keepTemporaryFiles is False:
        experiment.remove_fusion_tmp_directory()

    #
    # end processing
    #

    end_time = time.time()
    monitoring.to_log_and_console('    computation time = ' + str(end_time - start_time) + ' s', 1)
    monitoring.to_log_and_console('', 1)

    return


#
#
# Parse the raw data directories and identify data to be fused for each time point
# - for each time point fusion_preprocess() is then called
#
#


def fusion_control(experiment, parameters):
    """

    :param experiment:
    :param parameters:
    :return:
    """

    proc = 'fusion_control'

    #
    # parameter type checking
    #

    if not isinstance(experiment, common.Experiment):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'experiment' variable: "
                                      + str(type(experiment)))
        sys.exit(1)

    if not isinstance(parameters, FusionParameters):
        monitoring.to_log_and_console(str(proc) + ": unexpected type for 'parameters' variable: "
                                      + str(type(parameters)))
        sys.exit(1)

    #
    # make sure that the result directory exists
    #

    experiment.fusion_dir.make_directory()

    monitoring.to_log_and_console('', 1)

    #
    # make sure that the result directory exists
    # (although it should have been verified in 1-fuse.py)
    #

    # for c in range(0, len(experiment.fusion_dir.get_number_directories())):
    #    if not os.path.isdir(experiment.fusion_dir.get_directory(c)):
    #        os.makedirs(experiment.fusion_dir.get_directory(c))

    # if not os.path.isdir(environment.path_logdir):
    #    os.makedirs(environment.path_logdir)

    #
    # if data directories of the main channel are different, parse them
    # else rely on the given names
    #

    if experiment.rawdata_dir.channel[0].sub_directories_are_different() is True:

        #
        # for each rawdata subdirectory (ie, left/right camera, stack 0/1)
        # get
        # - the common file prefix,
        # - the length of the variable part (ie the time id)
        # - the list of variable parts (ie, all the time ids)
        # - the common file suffix
        #

        path_angle0 = experiment.rawdata_dir.channel[0].get_angle_path(0)
        path_angle1 = experiment.rawdata_dir.channel[0].get_angle_path(1)
        path_angle2 = experiment.rawdata_dir.channel[0].get_angle_path(2)
        path_angle3 = experiment.rawdata_dir.channel[0].get_angle_path(3)

        prefix0, time_length0, time_points0, suffix0 = _analyze_data_directory(path_angle0)
        prefix1, time_length1, time_points1, suffix1 = _analyze_data_directory(path_angle1)
        prefix2, time_length2, time_points2, suffix2 = _analyze_data_directory(path_angle2)
        prefix3, time_length3, time_points3, suffix3 = _analyze_data_directory(path_angle3)

        if monitoring.debug > 0:
            print("")
            print("analysis of '" + str(path_angle0) + "'")
            print("   -> " + prefix0)
            print("   -> " + str(time_length0))
            print("   -> " + str(time_points0))
            print("   -> " + suffix0)
            print("analysis of '" + str(path_angle1) + "'")
            print("   -> " + prefix1)
            print("   -> " + str(time_length1))
            print("   -> " + str(time_points1))
            print("   -> " + suffix1)
            print("analysis of '" + str(path_angle2) + "'")
            print("   -> " + prefix2)
            print("   -> " + str(time_length2))
            print("   -> " + str(time_points2))
            print("   -> " + suffix2)
            print("analysis of '" + str(path_angle3) + "'")
            print("   -> " + prefix3)
            print("   -> " + str(time_length3))
            print("   -> " + str(time_points3))
            print("   -> " + suffix3)
            print("")

        #
        # loop over acquisitions
        # 1. case where all acquisition have to be processed
        #    begin < 0 or end < 0 or begin > end or delta < 0
        # 2. only a few acquisitions have to be processed
        #

        extra_zeros = ''
        if time_length0 < experiment.rawdata_dir.get_time_digits_for_acquisition():
            extra_zeros = (experiment.rawdata_dir.get_time_digits_for_acquisition() - time_length0) * '0'

        #
        # no indication about the time interval to be process
        # -> process all the time ids of the list
        #

        if experiment.first_time_point < 0 or experiment.last_time_point < 0 or experiment.delta_time_point < 0 \
                or experiment.first_time_point > experiment.last_time_point:

            time_points0.sort()
            for time_point in time_points0:

                #
                # fused image name
                #
                fused_image = experiment.fusion_dir.get_image_name(int(time_point)) + "." \
                              + experiment.result_image_suffix

                #
                # input image names
                #

                images = list()

                images.append(prefix0 + time_point + suffix0)
                im = prefix1 + time_point + suffix1
                if time_point not in time_points1:
                    print(proc + ": image '" + im + "' not found in '" + path_angle1 + "'")
                    print("\t Exiting.")
                    sys.exit(1)
                else:
                    images.append(im)
                im = prefix2 + time_point + suffix2
                if time_point not in time_points2:
                    print(proc + ": image '" + im + "' not found in '" + path_angle2 + "'")
                    print("\t Exiting.")
                    sys.exit(1)
                else:
                    images.append(im)
                im = prefix3 + time_point + suffix3
                if time_point not in time_points3:
                    print(proc + ": image '" + im + "' not found in '" + path_angle3 + "'")
                    print("\t Exiting.")
                    sys.exit(1)
                else:
                    images.append(im)

                #
                # process
                #

                _fusion_preprocess(images, fused_image, extra_zeros + time_point, experiment, parameters)

        else:

            if experiment.first_time_point < 0 or experiment.last_time_point < 0:
                monitoring.to_log_and_console("... time interval does not seem to be defined in the parameter file")
                monitoring.to_log_and_console("    set parameters 'begin' and 'end'")
                monitoring.to_log_and_console("\t Exiting")
                sys.exit(1)

            #
            # parse only the required time values
            #

            for time_value in range(experiment.first_time_point, experiment.last_time_point + 1,
                                    experiment.delta_time_point):

                acquisition_time = str('{:0{width}d}'.format(time_value, width=time_length0))

                #
                # fused image name
                #

                fused_image = experiment.fusion_dir.get_image_name(time_value + experiment.delay_time_point) + "." \
                              + experiment.result_image_suffix

                #
                # input image names
                #

                images = list()

                im = prefix0 + acquisition_time + suffix0
                if acquisition_time not in time_points0:
                    monitoring.to_log_and_console("    .. image '" + im + "' not found in '" + path_angle0 + "'", 2)
                    monitoring.to_log_and_console("       skip time " + str(acquisition_time), 2)
                    continue
                else:
                    images.append(im)
                im = prefix1 + acquisition_time + suffix1
                if acquisition_time not in time_points1:
                    monitoring.to_log_and_console("    .. image '" + im + "' not found in '" + path_angle1 + "'", 2)
                    monitoring.to_log_and_console("       skip time " + str(acquisition_time), 2)
                    continue
                else:
                    images.append(im)
                im = prefix2 + acquisition_time + suffix2
                if acquisition_time not in time_points2:
                    monitoring.to_log_and_console("    .. image '" + im + "' not found in '" + path_angle2 + "'", 2)
                    monitoring.to_log_and_console("       skip time " + str(acquisition_time), 2)
                    continue
                else:
                    images.append(im)
                im = prefix3 + acquisition_time + suffix3
                if acquisition_time not in time_points3:
                    monitoring.to_log_and_console("    .. image '" + im + "' not found in '" + path_angle3 + "'", 2)
                    monitoring.to_log_and_console("       skip time " + str(acquisition_time), 2)
                    continue
                else:
                    images.append(im)

                #
                # process
                #

                _fusion_preprocess(images, fused_image, extra_zeros + acquisition_time, experiment, parameters)

    #
    # here data directories are not different, we have to rely on built names
    #

    else:

        if experiment.first_time_point < 0 or experiment.last_time_point < 0:
            monitoring.to_log_and_console("... time interval does not seem to be defined in the parameter file")
            monitoring.to_log_and_console("    set parameters 'begin' and 'end'")
            monitoring.to_log_and_console("\t Exiting")
            sys.exit(1)

        for time_value in range(experiment.first_time_point, experiment.last_time_point+1, experiment.delta_time_point):

            acquisition_time = experiment.get_time_index(time_value)

            #
            # fused image name
            #
            fused_image = experiment.fusion_dir.get_image_name(time_value + experiment.delay_time_point) + "." \
                          + experiment.result_image_suffix

            #
            # input image names
            #

            images = list()

            sname = experiment.rawdata_dir.channel[0].get_image_name(0, time_value)
            sdir = experiment.rawdata_dir.channel[0].get_angle_path(0)
            im = common.find_file(sdir, sname, file_type='image', callfrom=proc, local_monitoring=monitoring)
            if im is None:
                monitoring.to_log_and_console("    .. image '" + sname + "' not found in '" + sdir + "'", 2)
                monitoring.to_log_and_console("       skip time " + str(acquisition_time), 2)
                continue
            else:
                images.append(im)

            sname = experiment.rawdata_dir.channel[0].get_image_name(1, time_value)
            sdir = experiment.rawdata_dir.channel[0].get_angle_path(1)
            im = common.find_file(sdir, sname, file_type='image', callfrom=proc, local_monitoring=monitoring)
            if im is None:
                monitoring.to_log_and_console("    .. image '" + sname + "' not found in '" + sdir + "'", 2)
                monitoring.to_log_and_console("       skip time " + str(acquisition_time), 2)
                continue
            else:
                images.append(im)

            sname = experiment.rawdata_dir.channel[0].get_image_name(2, time_value)
            sdir = experiment.rawdata_dir.channel[0].get_angle_path(2)
            im = common.find_file(sdir, sname, file_type='image', callfrom=proc, local_monitoring=monitoring)

            if im is None:
                monitoring.to_log_and_console("    .. image '" + sname + "' not found in '" + sdir + "'", 2)
                # monitoring.to_log_and_console("       skip time " + str(acquisition_time), 2)
                monitoring.to_log_and_console("       maybe there is only one stack ", 2)
            else:
                images.append(im)
                sname = experiment.rawdata_dir.channel[0].get_image_name(3, time_value)
                sdir = experiment.rawdata_dir.channel[0].get_angle_path(3)
                im = common.find_file(sdir, sname, file_type='image', callfrom=proc, local_monitoring=monitoring)
                if im is None:
                    monitoring.to_log_and_console("    .. image '" + sname + "' not found in '" + sdir + "'", 2)
                    monitoring.to_log_and_console("       skip time " + str(acquisition_time), 2)
                    continue
                else:
                    images.append(im)

            #
            # process
            #

            _fusion_preprocess(images, fused_image, acquisition_time, experiment, parameters)

    return
