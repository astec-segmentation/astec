#!/usr/bin/python2.7

import os
import time
from argparse import ArgumentParser
import sys

#
# local imports
# add ASTEC subdirectory
#


import ASTEC.common as common
import ASTEC.naming as naming
import ASTEC.properties as properties
from ASTEC.CommunFunctions.cpp_wrapping import path_to_vt


#
#
#
#
#


def _set_options(my_parser):
    """

    :param my_parser:
    :return:
    """
    proc = "_set_options"
    if not isinstance(my_parser, ArgumentParser):
        print proc + ": argument is not of type ArgumentParser"
        return
    #
    # common parameters
    #

    my_parser.add_argument('-p', '--parameters',
                           action='store', dest='parameterFile', const=None,
                           help='python file containing parameters definition')
    my_parser.add_argument('-e', '--embryo-rep',
                           action='store', dest='embryo_path', const=None,
                           help='path to the embryo data')

    #
    # other options
    #

    my_parser.add_argument('-i', '--input',
                           action='store', nargs='*', dest='inputFiles', const=None,
                           help='input pkl or xml lineage file(s)')

    my_parser.add_argument('-o', '--output',
                           action='store', nargs='*', dest='outputFiles', const=None,
                           help='output pkl or xml lineage file')

    my_parser.add_argument('-r', '--references',
                           action='store', nargs='*', dest='referenceFiles', const=None,
                           help='reference lineage file(s)')

    #
    # control parameters
    #
    my_parser.add_argument('-diagnosis', '--diagnosis',
                           action='store_const', dest='diagnosis',
                           default=False, const=True,
                           help='make diagnosis on input file(s)')

    my_parser.add_argument('-reference-diagnosis', '--reference-diagnosis',
                           action='store_const', dest='reference_diagnosis',
                           default=False, const=True,
                           help='make diagnosis on reference file(s)')

    my_parser.add_argument('-fate', '--fate',
                           action='store_const', dest='fate',
                           default=False, const=True,
                           help='propagate fates on input file(s), naming being already done')

    my_parser.add_argument('-correction', '--correction',
                           action='store_const', dest='correction',
                           default=False, const=True,
                           help='make name corrections on input file(s)')

    my_parser.add_argument('-clean', '--clean-reference',
                           action='store_const', dest='clean_reference',
                           default=False, const=True,
                           help="build a 'clean' reference property file")

    #
    # control parameters
    #

    my_parser.add_argument('-k', '--keep-temporary-files',
                           action='store_const', dest='keepTemporaryFiles',
                           default=False, const=True,
                           help='keep temporary files')

    my_parser.add_argument('-f', '--force',
                           action='store_const', dest='forceResultsToBeBuilt',
                           default=False, const=True,
                           help='force building of results')

    my_parser.add_argument('-v', '--verbose',
                           action='count', dest='verbose', default=2,
                           help='incrementation of verboseness')
    my_parser.add_argument('-nv', '--no-verbose',
                           action='store_const', dest='verbose', const=0,
                           help='no verbose at all')
    my_parser.add_argument('-d', '--debug',
                           action='count', dest='debug', default=0,
                           help='incrementation of debug level')
    my_parser.add_argument('-nd', '--no-debug',
                           action='store_const', dest='debug', const=0,
                           help='no debug information')

    help = "print the list of parameters (with explanations) in the console and exit. "
    help += "If a parameter file is given, it is taken into account"
    my_parser.add_argument('-pp', '--print-param',
                           action='store_const', dest='printParameters',
                           default=False, const=True, help=help)

    return


#
#
# main function
#
#


def main():

    ############################################################
    #
    # generic part
    #
    ############################################################

    #
    # initialization
    #
    start_time = time.localtime()
    monitoring = common.Monitoring()
    experiment = common.Experiment()

    #
    # reading command line arguments
    # and update from command line arguments
    #
    parser = ArgumentParser(description='Naming')
    _set_options(parser)
    args = parser.parse_args()

    monitoring.update_from_args(args)
    experiment.update_from_args(args)

    if args.printParameters:
        parameters = naming.NamingParameters()
        if args.parameterFile is not None and os.path.isfile(args.parameterFile):
            experiment.update_from_parameter_file(args.parameterFile)
            parameters.update_from_parameter_file(args.parameterFile)
        experiment.print_parameters(directories=['astec', 'post'])
        parameters.print_parameters()
        sys.exit(0)

    #
    # read input file(s) from args, write output file from args
    #

    time_digits_for_cell_id = experiment.get_time_digits_for_cell_id()

    if args.parameterFile is None:
        prop = properties.read_dictionary(args.inputFiles, inputpropertiesdict={})
        if args.clean_reference:
            prop = naming.clean_reference(prop)
        if args.correction:
            prop = naming.correct_reference(prop)
        if args.diagnosis:
            diagnosis = properties.DiagnosisParameters()
            properties.diagnosis(prop, None, diagnosis)
            naming.diagnosis(prop, time_digits_for_cell_id=time_digits_for_cell_id)
        if args.reference_diagnosis:
            neighborhoods = naming.build_neighborhoods(args.inputFiles,
                                                       time_digits_for_cell_id=time_digits_for_cell_id)
            discrepancy = naming.check_leftright_consistency(neighborhoods)
            prop = naming.add_leftright_discrepancy_selection(prop, discrepancy)
        if args.fate:
            prop = properties.set_fate_from_names(prop, time_digits_for_cell_id=time_digits_for_cell_id)
            prop = properties.set_color_from_fate(prop)
        if args.outputFiles is not None:
            properties.write_dictionary(args.outputFiles[0], prop)
        sys.exit(0)

    #
    # reading parameter files
    # and updating parameters
    #
    parameter_file = common.get_parameter_file(args.parameterFile)
    experiment.update_from_parameter_file(parameter_file)

    #
    # set
    # 1. the working directory
    #    that's where the logfile will be written
    # 2. the log file name
    #    it creates the logfile dir, if necessary
    #
    experiment.working_dir = experiment.post_dir
    monitoring.set_log_filename(experiment, __file__, start_time)

    #
    # keep history of command line executions
    # and copy parameter file
    #
    experiment.update_history_at_start(__file__, start_time, parameter_file, path_to_vt())
    # experiment.copy_stamped_file(start_time, parameter_file)

    #
    # copy monitoring information into other "files"
    # so the log filename is known
    #
    common.monitoring.copy(monitoring)

    #
    # write generic information into the log file
    #
    # monitoring.write_configuration()
    # experiment.write_configuration()

    # experiment.write_parameters(monitoring.log_filename)

    ############################################################
    #
    # specific part
    #
    ############################################################

    #
    # copy monitoring information into other "files"
    # so the log filename is known
    #
    naming.monitoring.copy(monitoring)
    properties.monitoring.copy(monitoring)

    #
    # manage parameters
    # 1. initialize
    # 2. update parameters
    # 3. write parameters into the logfile
    #

    parameters = naming.NamingParameters()
    parameters.update_from_parameter_file(parameter_file)
    # parameters.write_parameters(monitoring.log_filename)

    #
    # processing
    #
    if args.inputFiles is not None:
        parameters.inputFiles += args.inputFiles
    if args.referenceFiles is not None:
        parameters.referenceFiles += args.referenceFiles
    if args.parametersreference_diagnosis:
        parameters.reference_diagnosis = True

    time_digits_for_cell_id = experiment.get_time_digits_for_cell_id()

    if args.fate:
        prop = properties.read_dictionary(parameters.inputFiles, inputpropertiesdict={})
        prop = properties.set_fate_from_names(prop, time_digits_for_cell_id=time_digits_for_cell_id)
        prop = properties.set_color_from_fate(prop)
        if isinstance(parameters.outputFile, str) is None and args.outputFiles is not None:
            properties.write_dictionary(args.outputFiles[0], prop)
    else:
        naming.naming_process(experiment, parameters)

    #
    # end of execution
    # write execution time in both log and history file
    #
    end_time = time.localtime()
    monitoring.update_execution_time(start_time, end_time)
    experiment.update_history_execution_time(__file__, start_time, end_time)

    monitoring.to_console('Total computation time = ' + str(time.mktime(end_time) - time.mktime(start_time)) + ' s')


#
#
# main call
#
#


if __name__ == '__main__':
    main()
