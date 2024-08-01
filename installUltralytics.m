arch = computer('arch');

installLocation = string(pwd) + filesep + arch;
mkdir(installLocation)


%%
% Using https://github.com/indygreg/python-build-standalone/releases/tag/20211017

pyVersion = "3.11";
switch arch
    case 'glnxa64'        
        % zstd is not a standard install :(
        pySource = 'https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64_v2-unknown-linux-gnu-install_only.tar.gz';
    case 'win64'
        pySource = "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-pc-windows-msvc-shared-install_only.tar.gz";
    case 'maca64'
        Error('Installation not supported');
end

%% Download Python
disp("Downloading Python: " + pyVersion)
pydlLoc = websave(installLocation+filesep+"pydl", pySource);


%% Extract Python
tic
disp("Extracting Python: " + pydlLoc)
untar(pydlLoc, installLocation);
delete(pydlLoc)
toc
%% Save to settings
if ispc
    pyInterpreter = installLocation+filesep+"python\python";
else
    pyInterpreter = installLocation+filesep+"python/bin/python3.11";
end

if ispc
    pipPath = installLocation+filesep+"python\pip3";
else
    pipPath = installLocation+filesep+"python/bin/pip3";
end

pyModules = installLocation+filesep+"modules";


s = settings;
if ~hasGroup(s, 'python')
    addGroup(s,'python');
end

if hasSetting(s.python,"Python")
    s.python.Python.PersonalValue = pyInterpreter;
else
    addSetting(s.python,"Python","PersonalValue",pyInterpreter);
end


%% Install Ultralytics using pip

disp("Downloading and installing Ultralytics and it's dependencies: Take ~5 mins")

tic
[status, cmdout] = system(pyInterpreter...
    +" -m pip install ultralytics==8.2.66");
if status==0
    disp('Done')
else
    disp(cmdout)
end

[status, cmdout] = system(pyInterpreter...
    +" -m pip install dill==0.3.8");
if status==0
    disp('Done')
else
    disp(cmdout)
end

[status, cmdout] = system(pyInterpreter...
    +" -m pip install onnx==1.16.1");
if status==0
    disp('Done')
else
    disp(cmdout)
end
toc

if ispc
    !win64\python\Scripts\ultralytics.exe settings datasets_dir=datasets
else
    !./glnxa64/python/bin/ultralytics settings datasets_dir=datasets
end

terminate(pyenv);
pyenv(Version=pyInterpreter,ExecutionMode="OutOfProcess")

if isunix
    py.sys.setdlopenflags(int32(bitor(int64(py.os.RTLD_LAZY),int64(py.os.RTLD_DEEPBIND))));
end