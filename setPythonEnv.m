function setPythonEnv(pythonLoc)
% setPythonEnv Configures MATLAB to use a specified Python interpreter.
%
% setPythonEnv(pythonLoc) terminates any existing Python environment in
% MATLAB, sets up a new Python environment using the specified Python
% interpreter location, and configures dynamic loading flags for Python
% shared libraries.
%
% % Input
% -------
%   pythonLoc       String specifying the location of the Python 
%                   interpreter.
% Note:
%   This function uses 'OutOfProcess' execution mode for the Python environment.
%   Ensure that the specified Python interpreter and required packages are correctly installed.
%
% % Example: Set up python environment
% ------------------------------------
% % For Python installed using `installUltralytics.m`, it will be placed in
% % the current working directory. In such cases, set up the Python 
% % environment using the following code:
%
% pythonLoc= [pwd,filesep,fullfile('glnxa64','python','bin','python3.11')]
% setPythonEnv(pythonLoc)
%
% Copyright 2024 The MathWorks, Inc.

terminate(pyenv);
pyenv(Version=pythonLoc,ExecutionMode="OutOfProcess")
py.sys.setdlopenflags(int32(bitor(int64(py.os.RTLD_LAZY),int64(py.os.RTLD_DEEPBIND))));
end