function varargout = Experiment_PB_DAQ(varargin)
%EXPERIMENT_PB_DAQ MATLAB code file for Experiment_PB_DAQ.fig
%      EXPERIMENT_PB_DAQ, by itself, creates a new EXPERIMENT_PB_DAQ or raises the existing
%      singleton*.
%
%      H = EXPERIMENT_PB_DAQ returns the handle to a new EXPERIMENT_PB_DAQ or the handle to
%      the existing singleton*.
%
%      EXPERIMENT_PB_DAQ('Property','Value',...) creates a new EXPERIMENT_PB_DAQ using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to Experiment_PB_DAQ_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      EXPERIMENT_PB_DAQ('CALLBACK') and EXPERIMENT_PB_DAQ('CALLBACK',hObject,...) call the
%      local function named CALLBACK in EXPERIMENT_PB_DAQ.M with the given input
%      arguments.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Experiment_PB_DAQ

% Last Modified by GUIDE v2.5 29-Sep-2020 17:35:45
%
% ========================================================================
% EXTERNAL MICROWAVE CONTROL MODE
% ========================================================================
% This code has been modified to support external microwave source control.
% When ExternalMicrowaveControl flag is set to true (in OpeningFcn):
%   - All microwave frequency and power control callbacks are bypassed
%   - fixFreq, fixPow, FixSG2Freq, FixSG2Pow callbacks return immediately
%   - The code assumes microwave sweep/frequency control is handled externally
%
% IMPORTANT: If ExperimentFunctionPool.m contains microwave control code,
% you may need to modify that file as well to check for the
% ExternalMicrowaveControl flag and bypass microwave operations.
%
% To disable external control mode, set handles.ExternalMicrowaveControl = false
% in the Experiment_PB_DAQ_OpeningFcn function.
% ========================================================================

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Experiment_PB_DAQ_OpeningFcn, ...
                   'gui_OutputFcn',  @Experiment_PB_DAQ_OutputFcn, ...
                   'gui_LayoutFcn',  [], ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
   gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before Experiment_PB_DAQ is made visible.
function Experiment_PB_DAQ_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   unrecognized PropertyName/PropertyValue pairs from the
%            command line (see VARARGIN)

% ========================================================================
% EXTERNAL MICROWAVE CONTROL MODE
% ========================================================================
% Set this flag to true if you are controlling the microwave source
% externally. When true, all microwave control operations will be bypassed.
% ========================================================================
handles.ExternalMicrowaveControl = true;  % Set to true for external MW control

% Choose default command line output for Experiment_PB_DAQ
handles.output = hObject;
ExperimentFunctionPool('Initialize',hObject, eventdata, handles);
% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Experiment_PB_DAQ wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Experiment_PB_DAQ_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes during object creation, after setting all properties.
function axes2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes2


% --- Executes during object creation, after setting all properties.
function axes3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes3


% --- Executes on button press in LoadExp.
function LoadExp_Callback(hObject, eventdata, handles)
% hObject    handle to LoadExp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in SaveExp.
function SaveExp_Callback(hObject, eventdata, handles)
% hObject    handle to SaveExp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in SaveAsExp.
function SaveAsExp_Callback(hObject, eventdata, handles)
% hObject    handle to SaveAsExp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function fixPow_Callback(hObject, eventdata, handles)
% hObject    handle to fixPow (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% EXTERNAL MICROWAVE CONTROL: This callback is disabled when using external MW control
if isfield(handles, 'ExternalMicrowaveControl') && handles.ExternalMicrowaveControl
    % Microwave power is controlled externally - no action taken
    return;
end

% Hints: get(hObject,'String') returns contents of fixPow as text
%        str2double(get(hObject,'String')) returns contents of fixPow as a double


% --- Executes during object creation, after setting all properties.
function fixPow_CreateFcn(hObject, eventdata, handles)
% hObject    handle to fixPow (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in LioPB0.
function LioPB0_Callback(hObject, eventdata, handles)
% hObject    handle to LioPB0 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of LioPB0


% --- Executes on button press in LioPB1.
function LioPB1_Callback(hObject, eventdata, handles)
% hObject    handle to LioPB1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of LioPB1


% --- Executes on button press in LioPB2.
function LioPB2_Callback(hObject, eventdata, handles)
% hObject    handle to LioPB2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of LioPB2


% --- Executes on button press in LioPB3.
function LioPB3_Callback(hObject, eventdata, handles)
% hObject    handle to LioPB3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of LioPB3


% --- Executes on button press in LioPB4.
function LioPB4_Callback(hObject, eventdata, handles)
% hObject    handle to LioPB4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of LioPB4


% --- Executes on button press in LioPB5.
function LioPB5_Callback(hObject, eventdata, handles)
% hObject    handle to LioPB5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of LioPB5


% --- Executes on button press in LioPB6.
function LioPB6_Callback(hObject, eventdata, handles)
% hObject    handle to LioPB6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of LioPB6


% --- Executes on button press in LioPB7.
function LioPB7_Callback(hObject, eventdata, handles)
% hObject    handle to LioPB7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of LioPB7


% --- Executes on button press in RunLeaveItOn.
function RunLeaveItOn_Callback(hObject, eventdata, handles)
% hObject    handle to RunLeaveItOn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
ExperimentFunctionPool('PBON',hObject, eventdata, handles);

% --- Executes on button press in StopLeaveItOn.
function StopLeaveItOn_Callback(hObject, eventdata, handles)
% hObject    handle to StopLeaveItOn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
ExperimentFunctionPool('PBOFF',hObject, eventdata, handles);

% --- Executes on button press in RunSequence.
function RunSequence_Callback(hObject, eventdata, handles)
% hObject    handle to RunSequence (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
ExperimentFunctionPool('Run',hObject, eventdata, handles);



function RepeatNTimes_Callback(hObject, eventdata, handles)
% hObject    handle to Repeat (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Repeat as text
%        str2double(get(hObject,'String')) returns contents of Repeat as a double


% --- Executes during object creation, after setting all properties.
function RepeatNTimes_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Repeat (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in bSweepSEQ.
function bSweepSEQ_Callback(hObject, eventdata, handles)
% hObject    handle to bSweepSEQ (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of bSweepSEQ


% --- Executes on button press in StopSeq.
function StopSeq_Callback(hObject, eventdata, handles)
% hObject    handle to StopSeq (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global gmSEQ
gmSEQ.bGo=0;

% --- Executes on button press in bTrack.
function bTrack_Callback(hObject, eventdata, handles)
% hObject    handle to bTrack (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of bTrack


% --- Executes on button press in bAverage.
function bAverage_Callback(hObject, eventdata, handles)
% hObject    handle to bAverage (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of bAverage



function Repeat_Callback(hObject, eventdata, handles)
% hObject    handle to Repeat (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Repeat as text
%        str2double(get(hObject,'String')) returns contents of Repeat as a double
function Average_Callback(hObject, eventdata, handles)
% hObject    handle to Average (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Repeat as text
%        str2double(get(hObject,'String')) returns contents of Repeat as a double


% --- Executes during object creation, after setting all properties.
function Repeat_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Repeat (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in bFit.
function bFit_Callback(hObject, eventdata, handles)
% hObject    handle to bFit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of bFit



function pi2_Callback(hObject, eventdata, handles)
% hObject    handle to pi2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of pi2 as text
%        str2double(get(hObject,'String')) returns contents of pi2 as a double


% --- Executes during object creation, after setting all properties.
function pi2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pi2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function pi_Callback(hObject, eventdata, handles)
% hObject    handle to pi (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of pi as text
%        str2double(get(hObject,'String')) returns contents of pi as a double


% --- Executes during object creation, after setting all properties.
function pi_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pi (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function FROM1_Callback(hObject, eventdata, handles)
% hObject    handle to FROM1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of FROM1 as text
%        str2double(get(hObject,'String')) returns contents of FROM1 as a double


% --- Executes during object creation, after setting all properties.
function FROM1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to FROM1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function TO1_Callback(hObject, eventdata, handles)
% hObject    handle to TO1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of TO1 as text
%        str2double(get(hObject,'String')) returns contents of TO1 as a double


% --- Executes during object creation, after setting all properties.
function TO1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to TO1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function SweepNPoints_Callback(hObject, eventdata, handles)
% hObject    handle to SweepNPoints (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of SweepNPoints as text
%        str2double(get(hObject,'String')) returns contents of SweepNPoints as a double


% --- Executes during object creation, after setting all properties.
function SweepNPoints_CreateFcn(hObject, eventdata, handles)
% hObject    handle to SweepNPoints (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function SweepFile_CreateFcn(hObject, eventdata, handles)
% hObject    handle to SweepFile (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in OpenSweepSeq.
function OpenSweepSeq_Callback(hObject, eventdata, handles)
% hObject    handle to OpenSweepSeq (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in SweepSave.
function SweepSave_Callback(hObject, eventdata, handles)
% hObject    handle to SweepSave (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in SweepSaveAs.
function SweepSaveAs_Callback(hObject, eventdata, handles)
% hObject    handle to SweepSaveAs (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on selection change in PBN.
function PBN_Callback(hObject, eventdata, handles)
% hObject    handle to PBN (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns PBN contents as cell array
%        contents{get(hObject,'Value')} returns selected item from PBN

function PBN_CreateFcn(hObject, eventdata, handles)
% hObject    handle to PBN (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
% --- Executes on selection change in SelectRise2.
function SelectRise2_Callback(hObject, eventdata, handles)
% hObject    handle to SelectRise2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns SelectRise2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from SelectRise2


% --- Executes during object creation, after setting all properties.
function SelectRise2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to SelectRise2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in SelectRise.
function SelectRise_Callback(hObject, eventdata, handles)
% hObject    handle to SelectRise (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns SelectRise contents as cell array
%        contents{get(hObject,'Value')} returns selected item from SelectRise


% --- Executes during object creation, after setting all properties.
function SelectRise_CreateFcn(hObject, eventdata, handles)
% hObject    handle to SelectRise (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in PBN2.
function PBN2_Callback(hObject, eventdata, handles)
% hObject    handle to PBN2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns PBN2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from PBN2


% --- Executes during object creation, after setting all properties.
function PBN2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to PBN2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function axes1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axes1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axes1


% --- Executes on mouse press over axes background.
function axes1_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to axes1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes during object creation, after setting all properties.
function Sequence_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Sequence (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in OpenSequence.
function OpenSequence_Callback(hObject, eventdata, handles)
% hObject    handle to OpenSequence (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over OpenSequence.
function OpenSequence_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to OpenSequence (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in RunSEQ.
function RunSEQ_Callback(hObject, eventdata, handles)
% hObject    handle to RunSEQ (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
sequence_Callback(hObject, eventdata, handles);
ExperimentFunctionPool('RunFirstPoint',hObject, eventdata, handles);

% --- Executes on button press in bShiftCh.
function bShiftCh_Callback(hObject, eventdata, handles)
% hObject    handle to bShiftCh (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of bShiftCh


% --------------------------------------------------------------------
function File_Callback(hObject, eventdata, handles)
% hObject    handle to File (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function experiment_Callback(hObject, eventdata, handles)
% hObject    handle to experiment (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function menuSweeps_Callback(hObject, eventdata, handles)
% hObject    handle to menuSweeps (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function menuTracking_Callback(hObject, eventdata, handles)
% hObject    handle to menuTracking (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function menuSetThreshold_Callback(hObject, eventdata, handles)
% hObject    handle to menuSetThreshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function menuConfigureSweepOther_Callback(hObject, eventdata, handles)
% hObject    handle to menuConfigureSweepOther (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function load_Callback(hObject, eventdata, handles)
% hObject    handle to load (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function save_Callback(hObject, eventdata, handles)
% hObject    handle to save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function saveas_Callback(hObject, eventdata, handles)
% hObject    handle to saveas (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function clone_Callback(hObject, eventdata, handles)
% hObject    handle to clone (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function make_report_Callback(hObject, eventdata, handles)
% hObject    handle to make_report (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function menuDAQ_Callback(hObject, eventdata, handles)
% hObject    handle to menuDAQ (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function menuOpenPHD_Callback(hObject, eventdata, handles)
% hObject    handle to menuOpenPHD (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function quit_Callback(hObject, eventdata, handles)
% hObject    handle to quit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function fixFreq_Callback(hObject, eventdata, handles)
% hObject    handle to fixFreq (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% EXTERNAL MICROWAVE CONTROL: This callback is disabled when using external MW control
if isfield(handles, 'ExternalMicrowaveControl') && handles.ExternalMicrowaveControl
    % Microwave frequency is controlled externally - no action taken
    return;
end

% Hints: get(hObject,'String') returns contents of fixFreq as text
%        str2double(get(hObject,'String')) returns contents of fixFreq as a double


% --- Executes during object creation, after setting all properties.
function fixFreq_CreateFcn(hObject, eventdata, handles)
% hObject    handle to fixFreq (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in bTextMe.
function bTextMe_Callback(hObject, eventdata, handles)
% hObject    handle to bTextMe (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of bTextMe


% --- Executes on button press in bPlotDelays.
function bPlotDelays_Callback(hObject, eventdata, handles)
% hObject    handle to bPlotDelays (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of bPlotDelays


% --- Executes on button press in bPlotExt.
function bPlotExt_Callback(hObject, eventdata, handles)
% hObject    handle to bPlotExt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
ExperimentFunctionPool('PlotExt',hObject, eventdata, handles);


function misc_Callback(hObject, eventdata, handles)
% hObject    handle to misc (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of misc as text
%        str2double(get(hObject,'String')) returns contents of misc as a double


% --- Executes during object creation, after setting all properties.
function misc_CreateFcn(hObject, eventdata, handles)
% hObject    handle to misc (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function readout_Callback(hObject, eventdata, handles)
% hObject    handle to readout (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of readout as text
%        str2double(get(hObject,'String')) returns contents of readout as a double


% --- Executes during object creation, after setting all properties.
function readout_CreateFcn(hObject, eventdata, handles)
% hObject    handle to readout (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in sequence.
function sequence_Callback(hObject, eventdata, handles)
% hObject    handle to sequence (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global gmSEQ
if ~gmSEQ.bGo % don't load this while the experiment is running
    gmSEQ.m=str2double(get(handles.FROM1,'String'));
    ExperimentFunctionPool('LoadSEQ',hObject,eventdata,handles,handles.axes1);
end

% Hints: contents = cellstr(get(hObject,'String')) returns sequence contents as cell array
%        contents{get(hObject,'Value')} returns selected item from sequence


% --- Executes during object creation, after setting all properties.
function sequence_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sequence (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes during object creation, after setting all properties.
function Average_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Average (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
 selection = questdlg('Close Experiment?',...
                     'Close Request Function',...
                     'Yes','No','Yes');
 switch selection
   case 'Yes'
%     unloadlibrary('mypbesr')
     delete(hObject);
   case 'No'
     return
 end



function CtrGateDur_Callback(hObject, eventdata, handles)
% hObject    handle to CtrGateDur (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of CtrGateDur as text
%        str2double(get(hObject,'String')) returns contents of CtrGateDur as a double


% --- Executes during object creation, after setting all properties.
function CtrGateDur_CreateFcn(hObject, eventdata, handles)
% hObject    handle to CtrGateDur (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in StopSeqAfterAvg.
function StopSeqAfterAvg_Callback(hObject, eventdata, handles)
% hObject    handle to StopSeqAfterAvg (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global gmSEQ
gmSEQ.bGoAfterAvg=0;


% --- Executes on button press in bPlotExtRaw.
function bPlotExtRaw_Callback(hObject, eventdata, handles)
% hObject    handle to bPlotExtRaw (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
ExperimentFunctionPool('PlotExtRaw',hObject, eventdata, handles);


% --- Executes on button press in bSaveData.
function bSaveData_Callback(hObject, eventdata, handles)
% hObject    handle to bSaveData (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
ExperimentFunctionPool('SaveData',hObject, eventdata, handles);


% --- Executes on button press in bPlotSequence.
function bPlotSequence_Callback(hObject, eventdata, handles)
% hObject    handle to bPlotSequence (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global gmSEQ
if ~gmSEQ.bGo % don't load this while the experiment is running
    gmSEQ.m=str2double(get(handles.FROM1,'String'));
    figure
    ExperimentFunctionPool('LoadSEQ',hObject,eventdata,handles,cla);
end


% --- Executes on button press in bWarmUpAOM.
function bWarmUpAOM_Callback(hObject, eventdata, handles)
% hObject    handle to bWarmUpAOM (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of bWarmUpAOM


% --- Executes on button press in bAAR.
function bAAR_Callback(hObject, eventdata, handles)
% hObject    handle to bAAR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of bAAR



function FROM2_Callback(hObject, eventdata, handles)
% hObject    handle to FROM2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of FROM2 as text
%        str2double(get(hObject,'String')) returns contents of FROM2 as a double


% --- Executes during object creation, after setting all properties.
function FROM2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to FROM2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function TO2_Callback(hObject, eventdata, handles)
% hObject    handle to TO2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of TO2 as text
%        str2double(get(hObject,'String')) returns contents of TO2 as a double


% --- Executes during object creation, after setting all properties.
function TO2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to TO2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function SweepNPoints2_Callback(hObject, eventdata, handles)
% hObject    handle to SweepNPoints2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of SweepNPoints2 as text
%        str2double(get(hObject,'String')) returns contents of SweepNPoints2 as a double


% --- Executes during object creation, after setting all properties.
function SweepNPoints2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to SweepNPoints2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in bSweep2.
function bSweep2_Callback(hObject, eventdata, handles)
% hObject    handle to bSweep2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of bSweep2



function PulseNum_Callback(hObject, eventdata, handles)
% hObject    handle to PulseNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of PulseNum as text
%        str2double(get(hObject,'String')) returns contents of PulseNum as a double


% --- Executes during object creation, after setting all properties.
function PulseNum_CreateFcn(hObject, eventdata, handles)
% hObject    handle to PulseNum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function tP1Polar_Callback(hObject, eventdata, handles)
% hObject    handle to tP1Polar (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tP1Polar as text
%        str2double(get(hObject,'String')) returns contents of tP1Polar as a double


% --- Executes during object creation, after setting all properties.
function tP1Polar_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tP1Polar (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function tP1PolarStep_Callback(hObject, eventdata, handles)
% hObject    handle to tP1PolarStep (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tP1PolarStep as text
%        str2double(get(hObject,'String')) returns contents of tP1PolarStep as a double


% --- Executes during object creation, after setting all properties.
function tP1PolarStep_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tP1PolarStep (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function tP1AfterPolarWait_Callback(hObject, eventdata, handles)
% hObject    handle to tP1AfterPolarWait (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of tP1AfterPolarWait as text
%        str2double(get(hObject,'String')) returns contents of tP1AfterPolarWait as a double


% --- Executes during object creation, after setting all properties.
function tP1AfterPolarWait_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tP1AfterPolarWait (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function FixSG2Pow_Callback(hObject, eventdata, handles)
% hObject    handle to FixSG2Pow (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% EXTERNAL MICROWAVE CONTROL: This callback is disabled when using external MW control
if isfield(handles, 'ExternalMicrowaveControl') && handles.ExternalMicrowaveControl
    % Signal Generator 2 power is controlled externally - no action taken
    return;
end

% Hints: get(hObject,'String') returns contents of FixSG2Pow as text
%        str2double(get(hObject,'String')) returns contents of FixSG2Pow as a double


% --- Executes during object creation, after setting all properties.
function FixSG2Pow_CreateFcn(hObject, eventdata, handles)
% hObject    handle to FixSG2Pow (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function FixSG2Freq_Callback(hObject, eventdata, handles)
% hObject    handle to FixSG2Freq (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% EXTERNAL MICROWAVE CONTROL: This callback is disabled when using external MW control
if isfield(handles, 'ExternalMicrowaveControl') && handles.ExternalMicrowaveControl
    % Signal Generator 2 frequency is controlled externally - no action taken
    return;
end

% Hints: get(hObject,'String') returns contents of FixSG2Freq as text
%        str2double(get(hObject,'String')) returns contents of FixSG2Freq as a double


% --- Executes during object creation, after setting all properties.
function FixSG2Freq_CreateFcn(hObject, eventdata, handles)
% hObject    handle to FixSG2Freq (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function DEERpi_Callback(hObject, eventdata, handles)
% hObject    handle to DEERpi (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of DEERpi as text
%        str2double(get(hObject,'String')) returns contents of DEERpi as a double


% --- Executes during object creation, after setting all properties.
function DEERpi_CreateFcn(hObject, eventdata, handles)
% hObject    handle to DEERpi (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function DEERt_Callback(hObject, eventdata, handles)
% hObject    handle to DEERt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of DEERt as text
%        str2double(get(hObject,'String')) returns contents of DEERt as a double


% --- Executes during object creation, after setting all properties.
function DEERt_CreateFcn(hObject, eventdata, handles)
% hObject    handle to DEERt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function P1Pulse_Callback(hObject, eventdata, handles)
% hObject    handle to P1Pulse (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of P1Pulse as text
%        str2double(get(hObject,'String')) returns contents of P1Pulse as a double


% --- Executes during object creation, after setting all properties.
function P1Pulse_CreateFcn(hObject, eventdata, handles)
% hObject    handle to P1Pulse (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Repolarize_Callback(hObject, eventdata, handles)
% hObject    handle to Repolarize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Repolarize as text
%        str2double(get(hObject,'String')) returns contents of Repolarize as a double


% --- Executes during object creation, after setting all properties.
function Repolarize_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Repolarize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Diffwait_Callback(hObject, eventdata, handles)
% hObject    handle to Diffwait (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of Diffwait as text
%        str2double(get(hObject,'String')) returns contents of Diffwait as a double


% --- Executes during object creation, after setting all properties.
function Diffwait_CreateFcn(hObject, eventdata, handles)
% hObject    handle to Diffwait (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in bSweep3.
function bSweep3_Callback(hObject, eventdata, handles)
% hObject    handle to bSweep3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of bSweep3



function FROM3_Callback(hObject, eventdata, handles)
% hObject    handle to FROM3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of FROM3 as text
%        str2double(get(hObject,'String')) returns contents of FROM3 as a double


% --- Executes during object creation, after setting all properties.
function FROM3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to FROM3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function TO3_Callback(hObject, eventdata, handles)
% hObject    handle to TO3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of TO3 as text
%        str2double(get(hObject,'String')) returns contents of TO3 as a double


% --- Executes during object creation, after setting all properties.
function TO3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to TO3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function SweepNPoints3_Callback(hObject, eventdata, handles)
% hObject    handle to SweepNPoints3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of SweepNPoints3 as text
%        str2double(get(hObject,'String')) returns contents of SweepNPoints3 as a double


% --- Executes during object creation, after setting all properties.
function SweepNPoints3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to SweepNPoints3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function MREVN_Callback(hObject, eventdata, handles)
% hObject    handle to MREVN (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of MREVN as text
%        str2double(get(hObject,'String')) returns contents of MREVN as a double


% --- Executes during object creation, after setting all properties.
function MREVN_CreateFcn(hObject, eventdata, handles)
% hObject    handle to MREVN (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function MREVtau_Callback(hObject, eventdata, handles)
% hObject    handle to MREVtau (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of MREVtau as text
%        str2double(get(hObject,'String')) returns contents of MREVtau as a double


% --- Executes during object creation, after setting all properties.
function MREVtau_CreateFcn(hObject, eventdata, handles)
% hObject    handle to MREVtau (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in bSweep1log.
function bSweep1log_Callback(hObject, eventdata, handles)
% hObject    handle to bSweep1log (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of bSweep1log


% --- Executes on button press in bSweep2log.
function bSweep2log_Callback(hObject, eventdata, handles)
% hObject    handle to bSweep2log (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of bSweep2log


% --- Executes on button press in bSweep3log.
function bSweep3log_Callback(hObject, eventdata, handles)
% hObject    handle to bSweep3log (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of bSweep3log


% --- Executes on button press in ACmodAWG.
function ACmodAWG_Callback(hObject, eventdata, handles)
% hObject    handle to ACmodAWG (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of ACmodAWG



function AWGFreq_Callback(hObject, eventdata, handles)
% hObject    handle to AWGFreq (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of AWGFreq as text
%        str2double(get(hObject,'String')) returns contents of AWGFreq as a double


% --- Executes during object creation, after setting all properties.
function AWGFreq_CreateFcn(hObject, eventdata, handles)
% hObject    handle to AWGFreq (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function AWGAmp_Callback(hObject, eventdata, handles)
% hObject    handle to AWGAmp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of AWGAmp as text
%        str2double(get(hObject,'String')) returns contents of AWGAmp as a double


% --- Executes during object creation, after setting all properties.
function AWGAmp_CreateFcn(hObject, eventdata, handles)
% hObject    handle to AWGAmp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function TrackPointN_Callback(hObject, eventdata, handles)
% hObject    handle to TrackPointN (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of TrackPointN as text
%        str2double(get(hObject,'String')) returns contents of TrackPointN as a double


% --- Executes during object creation, after setting all properties.
function TrackPointN_CreateFcn(hObject, eventdata, handles)
% hObject    handle to TrackPointN (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
