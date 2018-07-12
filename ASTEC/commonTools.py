
import os
import sys
import imp
import re
import time
import subprocess
import getpass
import shutil

#
#
#
#
#


class Monitoring(object):

    def __init__(self):
        self.verbose = 1
        self.debug = 0
        self.logfile = None
        self.keepTemporaryFiles = False
        self.forceResultsToBeBuilt = False

    def write_parameters(self, logfilename):
        with open(logfilename, 'a') as logfile:
            logfile.write("\n")
            logfile.write('Monitoring parameters\n')
            logfile.write('- verbose is ' + str(self.verbose)+'\n')
            logfile.write('- debug is ' + str(self.debug)+'\n')
            logfile.write('- logfile is ' + str(self.logfile)+'\n')
            logfile.write('- keepTemporaryFiles is ' + str(self.keepTemporaryFiles)+'\n')
            logfile.write('- forceResultsToBeBuilt is ' + str(self.forceResultsToBeBuilt)+'\n')
            logfile.write("\n")
            return

    def print_parameters(self):
        print("")
        print('Monitoring parameters')
        print('- verbose is ' + str(self.verbose))
        print('- debug is ' + str(self.debug))
        print('- logfile is ' + str(self.logfile))
        print('- keepTemporaryFiles is ' + str(self.keepTemporaryFiles))
        print('- forceResultsToBeBuilt is ' + str(self.forceResultsToBeBuilt))
        print("")

    def update_from_args(self, args):
        self.verbose = args.verbose
        self.debug = args.debug
        self.keepTemporaryFiles = args.keepTemporaryFiles
        self.forceResultsToBeBuilt = args.forceResultsToBeBuilt

    def copy(self, m):
        if m is None:
            return
        self.verbose = m.verbose
        self.debug = m.debug
        self.logfile = m.logfile
        self.keepTemporaryFiles = m.keepTemporaryFiles
        self.forceResultsToBeBuilt = m.forceResultsToBeBuilt

    @staticmethod
    def to_console(text):
        print(text)

    def to_log(self, text):
        if self.logfile is not None:
            with open(self.logfile, 'a') as logfile:
                logfile.write(text+'\n')

    def to_log_and_console(self, text, verboseness=0):
        if self.verbose >= verboseness or self.debug > 0:
            self.to_console(text)
        self.to_log(text)


#
#
#
#
#


class Experiment(object):

    def __init__(self):
        self.embryo_path = None
        self.embryoName = None
        self.first_time_point = -1
        self.last_time_point = -1
        self.delta_time_point = 1
        self.delay_time_point = 0

    #
    #
    #
    def write_parameters(self, logfilename):
        with open(logfilename, 'a') as logfile:
            logfile.write("\n")
            logfile.write('Experiment parameters\n')
            logfile.write('- embryo_path is ' + str(self.embryo_path)+'\n')
            logfile.write('- embryoName is ' + str(self.embryoName)+'\n')
            logfile.write('- first_time_point is ' + str(self.first_time_point)+'\n')
            logfile.write('- last_time_point is ' + str(self.last_time_point)+'\n')
            logfile.write('- delta_time_point is ' + str(self.delta_time_point)+'\n')
            logfile.write('- delay_time_point is ' + str(self.delay_time_point)+'\n')
            logfile.write("\n")
        return

    def print_parameters(self):
        print("")
        print('Experiment parameters')
        print('- embryo_path is ' + str(self.embryo_path))
        print('- embryoName is ' + str(self.embryoName))
        print('- first_time_point is ' + str(self.first_time_point))
        print('- last_time_point is ' + str(self.last_time_point))
        print('- delta_time_point is ' + str(self.delta_time_point))
        print('- delay_time_point is ' + str(self.delay_time_point))
        print("")

    #
    #
    #
    def update_from_args(self, args):
        if args.embryo_path is None:
            return
        if not os.path.isdir(args.embryo_path):
            print ("Experiment.updateFromArgs: '" + args.embryo_path + "' is not a valid directory. Exiting.")
            sys.exit(1)
        self.embryo_path = args.embryo_path
        return

    #
    #
    #
    def update_from_file(self, parameter_file):

        proc = "Experiment.updateFromFile"
        if parameter_file is None:
            return
        if not os.path.isfile(parameter_file):
            print proc + ": '" + parameter_file + "' is not a valid file."
            print "\t Exiting."
            sys.exit(1)

        parameters = imp.load_source('*', parameter_file)

        if hasattr(parameters, 'PATH_EMBRYO'):
            if parameters.PATH_EMBRYO is not None:
                if not os.path.isdir(parameters.PATH_EMBRYO):
                    print ("Experiment.updateFromFile: '" + parameters.PATH_EMBRYO
                           + "' is not a valid directory. Exiting.")
                    sys.exit(1)
                self.embryo_path = parameters.PATH_EMBRYO
            else:
                self.embryo_path = os.getcwd()

        if hasattr(parameters, 'EN'):
            if parameters.EN is not None:
                self.embryoName = parameters.EN
            else:
                if hasattr(parameters, 'PATH_EMBRYO'):
                    embryo_path = parameters.PATH_EMBRYO
                else:
                    embryo_path = os.getcwd()
                self.embryoName = embryo_path.split(os.path.sep)[-1]
        else:
            if hasattr(parameters, 'PATH_EMBRYO'):
                embryo_path = parameters.PATH_EMBRYO
            else:
                embryo_path = os.getcwd()
            self.embryoName = embryo_path.split(os.path.sep)[-1]

        if hasattr(parameters, 'begin'):
            if parameters.begin is not None:
                self.first_time_point = parameters.begin
            else:
                print proc + ": it is mandatory to specify the first time point"
                print "\t Exiting."
                sys.exit(1)

        if hasattr(parameters, 'end'):
            if parameters.end is not None:
                self.last_time_point = parameters.end
            else:
                print proc + ": it is mandatory to specify the last time point"
                print "\t Exiting."
                sys.exit(1)

        if hasattr(parameters, 'delta'):
            if parameters.delta is not None:
                self.delta_time_point = parameters.delta

        if hasattr(parameters, 'raw_delay'):
            if parameters.raw_delay is not None:
                self.delay_time_point = parameters.raw_delay

        return


########################################################################################
#
#
#
########################################################################################

def get_parameter_file(parameter_file):
    """
    check if the given parameter file is valid, otherwise ask for a file name
    :param parameter_file: the parameter file name to be tested
    :return: the parameter file name
    """
    if parameter_file is not None and os.path.isfile(parameter_file):
        return parameter_file
    new_parameter_file = raw_input('   Provide the parameter file: ')
    if os.path.isfile(new_parameter_file) is not False:
        print ("getParameterFile: '"+new_parameter_file+"' is not a valid file. Exiting.")
        sys.exit(1)
    return new_parameter_file


########################################################################################
#
#
#
########################################################################################


def write_history_information(logfile_name,
                              experiment=None,
                              parameter_file=None,
                              start_time=None,
                              path_to_exe=None,
                              path_to_vt=None):
    """
    Write history information
    :param logfile_name: file where to write information
    :param experiment: experiment values (class)
    :param parameter_file: name of the parameter file
    :param start_time: start time
    :param path_to_exe: path to Astec repository, used to get git version
    :param path_to_vt: path to VT build repository, used to get git version
    :return:
    """
    with open(logfile_name, 'a') as logfile:
        logfile.write("\n")
        if start_time is not None:
            logfile.write("# "+time.strftime("%a, %d %b %Y %H:%M:%S", start_time)+"\n")
        if experiment is not None:
            logfile.write("# Embryo path: '"+str(experiment.embryo_path)+"'\n")
            logfile.write("# Embryo name: '"+str(experiment.embryoName)+"'\n")
        if parameter_file is not None:
            logfile.write("# Parameter file: '" + str(parameter_file) + "'\n")
        logfile.write("# Command line: '"+" ".join(sys.argv) + "'\n")
        logfile.write("# Working directory: '"+str(os.getcwd())+"'\n")
        logfile.write("# User: '" + str(getpass.getuser()) + "'\n")
        logfile.write("# Python executable: " + sys.executable + "\n")
        if path_to_exe is not None:
            logfile.write("# ASTEC version: ")
            if not os.path.exists(path_to_exe+os.path.sep+'.git'):
                logfile.write("not found\n")
            else:
                pipe = subprocess.Popen("cd "+path_to_exe+"; git describe; cd "+str(os.getcwd()),
                                        shell=True, stdout=subprocess.PIPE).stdout
                o = pipe.next()
                v = o.split('\n')
                logfile.write(str(v[0]+"\n"))
        if path_to_vt is not None:
            logfile.write("# VT version: ")
            pipe = subprocess.Popen("cd "+path_to_vt+"; git describe; cd "+str(os.getcwd()),
                                    shell=True, stdout=subprocess.PIPE).stdout
            o = pipe.next()
            v = o.split('\n')
            logfile.write(str(v[0]+"\n"))
        logfile.write("# \n")
    return

########################################################################################
#
#
#
########################################################################################


def copy_date_stamped_file(thefile, directory, timestamp):
    """
    Copy a file to the designated directory while adding a time stamp to its name
    :param thefile:
    :param directory:
    :param timestamp:
    :return:
    """
    d = time.strftime("%Y-%m-%d-%H:%M:%S", timestamp)
    if len(thefile.split('.')) > 1:
        ext = thefile.split('.')[-1]
        filename = re.sub(r'(\.*).' + ext, r'\1', thefile.split(os.path.sep)[-1]) + '-' + d + '.' + ext
    else:
        filename = thefile.split(os.path.sep)[-1] + '-' + d
    resfile = os.path.join(directory, filename)
    shutil.copy2(thefile, resfile)


########################################################################################
#
#
#
########################################################################################


def read_lut(filename):
    """
    Return a dictionnary of integer key-to-key correspondances
    :param file:
    :return:
    """
    proc = 'read_lut'
    lut = {}

    if not os.path.isfile(filename):
        monitoring.to_log_and_console(proc + ": file '" + str(filename) + "' does not exists", 0)
        return lut

    f = open(filename)
    for line in f:
        li = line.strip()
        if li.startswith('#'):
            continue
        info = li.split()
        if len(info) == 2:
            # if not lut.has_key(int(info[0])):
            #   lut[int(info[0])] = None
            lut[int(info[0])] = int(info[1])
    f.close()

    return lut


########################################################################################
#
# image file utilities
#
########################################################################################


recognized_extensions = ['.zip', '.h5', '.tif', '.tiff', '.TIF', '.TIFF', '.inr', '.inr.gz', '.mha', '.mha.gz']


def get_extension(filename):
    """ Return the file extension. Must be in the set of recognized extensions.
    :param filename:
    :return: None in case of unrecognized extension,
             else the recognized extension (begins with '.')
    """
    for e in recognized_extensions:
        if len(filename) < len(e):
            continue
        if filename[len(filename)-len(e):len(filename)] == e:
            return e
    return None

#
#
#
#
#


def add_suffix(filename, suffix, new_dirname=None, new_extension=None):
    """
    Add a suffix to a filenename (ie before the extension)
    :param filename:
    :param suffix: suffix to be added
    :param new_dirname: change the directory name of the file
    :param new_extension: change the extension of the file
    :return: the transformed file name
    """
    proc = 'add_suffix'
    b = os.path.basename(filename)
    d = os.path.dirname(filename)
    e = get_extension(b)
    if e is None:
        print(proc + ": file extension of '"+str(filename)+"' was not recognized")
        print("\t Exiting")
#        monitoring.to_log_and_console(proc + ": file extension of '"+str(filename)+"' was not recognized", 0)
#        monitoring.to_log_and_console("\t Exiting", 0)
        sys.exit(1)
    new_basename = b[0:len(b)-len(e)]
    new_basename += suffix
    if new_extension is None:
        new_basename += e
    else:
        new_basename += '.' + new_extension
    if new_dirname is None:
        res_name = os.path.join(d, new_basename)
    else:
        res_name = os.path.join(new_dirname, new_basename)
    return res_name


#
#
#
#
#

def find_file(data_path, file_prefix, monitoring=None, verbose=True):
    """
    find a file in a directory with a given prefix. The suffix is unknowm

    :param data_path:
    :param file_prefix:
    :param monitoring:
    :return:
    """
    proc = "find_file"

    if not os.path.isdir(data_path):
        if monitoring is not None:
            monitoring.to_log_and_console(proc + ": '" + str(data_path) + "' is not a valid directory ?!")
            monitoring.to_log_and_console("\t Exiting.")
        else:
            print(proc + ": '" + str(data_path) + "' is not a valid directory ?!")
            print("\t Exiting.")
        sys.exit(1)

    file_names = []
    for f in os.listdir(data_path):
        if len(f) <= len(file_prefix):
            pass
        if f[0:len(file_prefix)] == file_prefix:
            file_names.append(f)

    if len(file_names) == 0:
        if monitoring is not None:
            monitoring.to_log_and_console(proc + ": no image with name '" + str(file_prefix)
                                          + "' was found in '" + str(data_path) + "'", 4)
        elif verbose is True:
            print(proc + ": no image with name '" + str(file_prefix) + "' was found in '" + str(data_path) + "'")
        return None

    if len(file_names) > 1:
        if monitoring is not None:
            monitoring.to_log_and_console(proc + ": several images with name '"
                                          + str(file_prefix) + "' were found in '" + str(data_path) + "'")
            monitoring.to_log_and_console("\t "+str(file_names))
        else:
            print(proc + ": several images with name '"
                  + str(file_prefix) + "' were found in '" + str(data_path) + "'")
            print("\t "+str(file_names))
        return None

    return file_names[0]
