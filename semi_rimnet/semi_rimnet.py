import logging
import os
import time

import numpy as np
import json
import vtk
import qt
import ctk
import re
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import datetime

try:
    import nibabel
    import pandas as pd
    import subprocess
    import getpass
    from scipy import ndimage
except:
    slicer.util.pip_install('nibabel')
    import nibabel

    slicer.util.pip_install('pandas')
    import pandas as pd

    slicer.util.pip_install('subprocess')
    import subprocess

    slicer.util.pip_install('getpass')
    import getpass

    slicer.util.pip_install('scipy')
    from scipy import ndimage

import sys

# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py

#
# semi_rimnet
#

class semi_rimnet(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Assisted RimNet"  # TODO: make this more human readable by adding spaces
        self.parent.categories = [
            "RimNet"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Joe Najm (MIAL)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#semi_rimnet">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # semi_rimnet1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='semi_rimnet',
        sampleName='semi_rimnet1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'semi_rimnet1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='semi_rimnet1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='semi_rimnet1'
    )

    # semi_rimnet2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='semi_rimnet',
        sampleName='semi_rimnet2',
        thumbnailFileName=os.path.join(iconsPath, 'semi_rimnet2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='semi_rimnet2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='semi_rimnet2'
    )


#
# semi_rimnetWidget
#

class semi_rimnetWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/semi_rimnet.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = semi_rimnetLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        ################ Create widget to choose the type of image ###############################
        now = datetime.datetime.now()
        self.session_name = f"{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}"
        self.loaded_PRLs = False

        self.images = ["Flair", "Phase"]
        self.last_present = [self.images[0], self.images[1]]

        filesCollapsibleButton = ctk.ctkCollapsibleButton()
        filesCollapsibleButton.collapsed = True
        filesCollapsibleButton.text = "Images"
        self.layout.addWidget(filesCollapsibleButton)
        filesFormLayout = qt.QFormLayout(filesCollapsibleButton)

        self.fileorientationBox_one = qt.QGroupBox()
        self.fileorientationBox_one.setLayout(qt.QFormLayout())
        self.fileButtons = {}
        self.files = ("Flair", "Phase", "Magnitude", "Label")
        for file in self.files:
            self.fileButtons[file] = qt.QCheckBox()
            self.fileButtons[file].text = file
            self.fileButtons[file].connect("clicked()",
                                           lambda f=file: self.setFile(
                                               f))  # Select the files of ticked boxes (still need to confirm choice)
            self.fileorientationBox_one.layout().addWidget(
                self.fileButtons[file])

        self.fileButtons["Flair"].setChecked(True)  # Default
        self.fileButtons["Phase"].setChecked(True)  # Default

        filesFormLayout.addRow("Image", self.fileorientationBox_one)

        self.confirmImgButton = qt.QPushButton("Confirm Images")
        filesFormLayout.addRow(self.confirmImgButton)
        self.confirmImgButton.connect("clicked()", self.confirmFiles)

        self.two_files = qt.QLabel("Please choose exactly two files")  # Make sure that chose 2 views
        filesFormLayout.addRow(self.two_files)
        self.two_files.setVisible(False)
        self.available_files = qt.QLabel(
            "One of the chosen files was not loaded")  # Make sure that chose 2 loaded files
        filesFormLayout.addRow(self.available_files)
        self.available_files.setVisible(False)

        # Add the bottom section to switch orientations (Same as choose files)
        self.orientation = "All Views"

        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.collapsed = True
        parametersCollapsibleButton.text = "Orientation"
        self.layout.addWidget(parametersCollapsibleButton)
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        self.orientationBox = qt.QGroupBox()
        self.orientationBox.setLayout(qt.QFormLayout())
        self.orientationButtons = {}
        self.orientations = ("Axial", "Sagittal", "Coronal", "All Views", "Default")
        for orientation in self.orientations:
            self.orientationButtons[orientation] = qt.QRadioButton()
            self.orientationButtons[orientation].text = orientation
            self.orientationButtons[orientation].connect("clicked()",
                                                         lambda o=orientation: self.setOrientation(o))
            self.orientationBox.layout().addWidget(
                self.orientationButtons[orientation])
        parametersFormLayout.addRow("Orientation", self.orientationBox)

        self.orientationButtons["All Views"].setChecked(True)

        self.switchConfigButton = qt.QPushButton("Switch Configuration")
        parametersFormLayout.addRow(self.switchConfigButton)
        self.switchConfigButton.connect("clicked()", self.switchConfig)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

        # Buttons
        self.ui.flairButton.connect('clicked(bool)', self.showFlair)
        self.ui.recentreButton.connect('clicked(bool)', self.recentre)
        self.ui.pickButton.connect('clicked(bool)', self.pickPointPressed)
        self.ui.processLesionButton.connect('clicked(bool)', self.processLesion)
        self.ui.confirmButton.connect('clicked(bool)', self.confirmButtonPressed)
        self.ui.confirmModificationsButton.connect('clicked(bool)', self.confirmModificationsButtonPressed)
        self.ui.saveButton.connect('clicked(bool)', self.saveButtonPressed)
        self.ui.displayArrowsButton.connect('clicked(bool)', self.displayArrows)
        self.ui.runRimNetButton.connect('clicked(bool)', self.runRimNet)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        self.preloaded_centres = []
        self.observations = []  # store the observations so that later can be removed
        self.ras_all = []
        self.number_lesion = 0
        self.coordinates_all = []
        self.doctor_conf_all = []
        self.rim_conf_all = []
        self.doctor_comment_all = []

        self.current_coordinates = []
        self.current_ras = []

        self.placing_pointer = False
        self.within_bounds = False
        self.spacing = []
        self.origin = []

        self.clicked_lesion_index = -1

        self.arrowsDisplayed = False

        # Load the volumes if none are already loaded, otherwise take the volumes that already in slicer
        if (self._parameterNode.GetNodeReference("InputVolume") is None):

            # Opening JSON file
            config_json = open('config.json')
            self.config_params = json.load(config_json)

            flair_path = self.config_params["Flair_path"]
            phase_path = self.config_params["Phase_path"]
            mag_path = self.config_params["Mag_path"]
            label_path = self.config_params["Label_path"]
            run_rimnet_on_preloaded_data = self.config_params["Run_RimNet_On_Preloaded_Data"]
            if run_rimnet_on_preloaded_data == "True":
                self.run_rimnet_on_preloaded_data = True
            else:
                self.run_rimnet_on_preloaded_data = False
            self.filename = self.config_params["Subject_ID"]

            if flair_path != "None":
                self.flair = slicer.util.loadVolume(flair_path, {"center": True})
                self.nibabel_img_flair = nibabel.load(flair_path)
                self.numpy_flair = self.nibabel_img_flair.get_fdata()
            else:
                self.flair = None
                self.fileButtons["Flair"].setEnabled(False)

            if phase_path != "None":
                self.phase = slicer.util.loadVolume(phase_path, {"center": True})
                self.nibabel_img_phase = nibabel.load(phase_path)
                self.numpy_phase = self.nibabel_img_phase.get_fdata()
            else:
                self.phase = None
                self.fileButtons["Phase"].setEnabled(False)

            if mag_path != "None":
                self.magnitude = slicer.util.loadVolume(mag_path, {"center": True})
                self.nibabel_img_mag = nibabel.load(mag_path)
                self.numpy_mag = self.nibabel_img_mag.get_fdata()
            else:
                self.magnitude = None
                self.fileButtons["Magnitude"].setEnabled(False)

            if label_path != "None":
                self.label = slicer.util.loadVolume(label_path, {"center": True})
                self.nibabel_img_label = nibabel.load(label_path)
                self.numpy_label = self.nibabel_img_label.get_fdata()
                if self.run_rimnet_on_preloaded_data:
                    self.compute_com_labelmap()
            else:
                self.label = None
                self.fileButtons["Label"].setEnabled(False)

        else:
            self.flair = slicer.mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode").GetItemAsObject(0)
            self.phase = slicer.mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode").GetItemAsObject(1)

        self.foreground = self.flair
        self.background = self.phase

        self.ui.doctorScore.setVisible(False)
        self.ui.doctorSlider.setVisible(False)
        self.ui.processLesionButton.setVisible(False)
        self.ui.rimConfNb.setVisible(False)
        self.ui.rimConf.setVisible(False)
        self.ui.confirmButton.setVisible(False)
        self.ui.confirmModificationsButton.setVisible(False)
        self.ui.chooseInside.setVisible(False)
        self.ui.doctorComment.setVisible(False)
        if self.run_rimnet_on_preloaded_data:
            self.ui.runRimNetButton.setVisible(True)
            self.ui.saveButton.setVisible(True)
        else:
            self.ui.runRimNetButton.setVisible(False)
            self.ui.saveButton.setVisible(False)
        self.ui.pleaseRunRimNet.setVisible(False)

        self.ui.inputsCollapsibleButton.setVisible(False)
        self.ui.outputsCollapsibleButton.setVisible(False)
        self.ui.advancedCollapsibleButton.setVisible(False)
        self.ui.label_5.setVisible(False)

        self.ui.conf_0.setVisible(False)
        self.ui.conf_1.setVisible(False)
        self.ui.conf_2.setVisible(False)
        self.ui.conf_3.setVisible(False)
        self.ui.conf_4.setVisible(False)
        self.ui.conf_5.setVisible(False)

        self.ui.rimnetDoneLabel.setVisible(False)

        self.ui.flairButton.setVisible(False)

        # Set default layout to 3-3 over each other
        layoutManager = slicer.app.layoutManager()
        layoutManager.layoutLogic().GetLayoutNode().SetViewArrangement(21)

        slicer.util.setSliceViewerLayers(background=self.background, foreground=self.foreground)

        # Display Flair and Phase with correct orientations
        self.assign_opacity("Red", 0.0)
        self.assign_opacity("Yellow", 0.0)
        self.assign_opacity("Green", 0.0)
        self.assign_opacity("Red+", 1.0)
        self.assign_opacity("Yellow+", 1.0)
        self.assign_opacity("Green+", 1.0)

        # Initialise the markup nodes as named "Lesion"
        self.markup_node = slicer.mrmlScene.GetNodeByID(slicer.modules.markups.logic().AddNewFiducialNode('Lesion'))



        # # Make R+ linked with R and rest as well
        for node in slicer.util.getNodesByClass('vtkMRMLSliceCompositeNode'):
            node.SetLinkedControl(1)
            slicer.app.layoutManager().sliceWidget(node.GetLayoutName()).mrmlSliceNode().SetViewGroup(0)
            #     # adjust contrast ctrl and right click
            self.adjustContrast(node.GetLayoutName())

        # Show basic crosshair (small cross) when load screen, weird that doesnt work with big arrow
        crosshairNode = slicer.util.getNode("Crosshair")
        crosshairNode.SetCrosshairMode(slicer.vtkMRMLCrosshairNode.ShowBasic)

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        # Remove observations of click lesions:
        for observedNode, observation in self.observations:
            observedNode.RemoveObserver(observation)

        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # slicer.mrmlScene.Clear()
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())
        #
        # # Select default input nodes if nothing is selected yet to save a few clicks for the user
        # if not self._parameterNode.GetNodeReference("InputVolume"):
        #     firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        #     if firstVolumeNode:
        #         self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())
        #         self._parameterNode.SetNodeReferenceID("OutputVolume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
        self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))

        # Update buttons states and tooltips
        if self._parameterNode.GetNodeReference("InputVolume"):
            # if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("OutputVolume"):
            self.ui.flairButton.enabled = True
        else:
            self.ui.flairButton.enabled = False

        if self._parameterNode.GetNodeReference("InputVolume"):
            self.ui.pickButton.enabled = True
            self.ui.loadFile.text = "File successfully loaded"
        else:
            self.ui.pickButton.enabled = False
            self.ui.loadFile.text = "Please load a Nifti file to proceed"

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.outputSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)

    # Adjust contrast with ctrl right click
    def adjustContrast(self, view_name):
        sliceViewLabel = view_name
        sliceViewWidget = slicer.app.layoutManager().sliceWidget(sliceViewLabel)
        displayableManager = sliceViewWidget.sliceView().displayableManagerByClassName(
            "vtkMRMLScalarBarDisplayableManager")
        w = displayableManager.GetWindowLevelWidget()
        w.SetEventTranslationClickAndDrag(w.WidgetStateIdle,
                                          vtk.vtkCommand.RightButtonPressEvent, vtk.vtkEvent.ControlModifier,
                                          w.WidgetStateAdjustWindowLevel, w.WidgetEventAlwaysOnAdjustWindowLevelStart,
                                          w.WidgetEventAlwaysOnAdjustWindowLevelEnd)

    # Update images list when click box
    def setFile(self, f):
        if (self.fileButtons[f].checkState() == 2):
            self.images.append(f)
        elif (self.fileButtons[f].checkState() == 0):
            self.images.remove(f)

    # Show the selected images
    def confirmFiles(self):
        if (len(self.images) != 2):
            self.two_files.setVisible(True)
        else:
            first_im = ""
            second_im = ""
            self.two_files.setVisible(False)

            isPresent = True
            if "Flair" in self.images:
                if self.flair is None:
                    isPresent = False
            if "Phase" in self.images:
                if self.phase is None:
                    isPresent = False
            if "Magnitude" in self.images:
                if self.magnitude is None:
                    isPresent = False
            if "Label" in self.images:
                if self.label is None:
                    isPresent = False

            if isPresent:
                self.last_present = [self.images[0], self.images[1]]
                if (self.images[0] == "Flair"):
                    self.foreground = self.flair
                    first_im = "flair"
                elif (self.images[0] == "Phase"):
                    self.foreground = self.phase
                    first_im = "phase"
                elif (self.images[0] == "Magnitude"):
                    if hasattr(self, 'magnitude'):
                        self.foreground = self.magnitude
                        first_im = "magnitude"
                    else:
                        self.available_files.setVisible(True)
                elif (self.images[0] == "Label"):
                    if hasattr(self, 'label'):
                        self.foreground = self.label
                        first_im = "Label"
                    else:
                        self.available_files.setVisible(True)

                if (self.images[1] == "Flair"):
                    self.background = self.flair
                    second_im = "flair"
                elif (self.images[1] == "Phase"):
                    self.background = self.phase
                    second_im = "phase"
                elif (self.images[1] == "Magnitude"):
                    if hasattr(self, 'magnitude'):
                        self.background = self.magnitude
                        second_im = "magnitude"
                    else:
                        self.available_files.setVisible(True)
                elif (self.images[1] == "Label"):
                    if hasattr(self, 'label'):
                        self.background = self.label
                        second_im = "Label"
                    else:
                        self.available_files.setVisible(True)

                if (first_im != "" and second_im != ""):
                    self.available_files.setVisible(False)

                slicer.util.setSliceViewerLayers(background=self.background, foreground=self.foreground)
                self.switchConfig()

            else:
                self.available_files.setVisible(True)
                self.images[0] = self.last_present[0]
                self.images[1] = self.last_present[1]


    def setOrientation(self, o):
        self.orientation = o

    # Give name of view and it will assign opacity value
    def assign_opacity(self, name, value):
        view = slicer.app.layoutManager().sliceWidget(name).sliceView()
        sliceNode = view.mrmlSliceNode()
        sliceLogic = slicer.app.applicationLogic().GetSliceLogic(sliceNode)
        compositeNode = sliceLogic.GetSliceCompositeNode()
        compositeNode.SetForegroundOpacity(value)

    def compute_com_labelmap(self):
        if self.label is not None:
            binary_map = np.zeros(self.numpy_label.shape)
            binary_map[self.numpy_label > 0] = 1

            labeled, ncomponents = ndimage.measurements.label(binary_map)

            labels = np.unique(labeled)[1::]

            volumeNode = self._parameterNode.GetNodeReference("InputVolume")
            ijkToRasMatrix = vtk.vtkMatrix4x4()
            volumeNode.GetIJKToRASMatrix(ijkToRasMatrix)

            self.spacing = volumeNode.GetSpacing()
            self.origin = volumeNode.GetOrigin()

            if len(labels):
                self.loaded_PRLs = True

            for idx, label in enumerate(labels):
                mask_bool = np.ma.masked_where(labeled == label, labeled).mask
                binary_mask = np.zeros(labeled.shape)
                binary_mask[mask_bool] = 1

                c = ndimage.center_of_mass(binary_mask)
                c = np.round(c).astype(int)
                self.preloaded_centres.append(c)
                rasPoint = ijkToRasMatrix.MultiplyFloatPoint(np.append(c, 1))
                self.ras_all.append(rasPoint)
                self.addFiducial(rasPoint, idx + 1)
                # ADD DEFAULT CONFIDENCE COMMENTS ...



                self.coordinates_all.append(c)

                preloaded_text = "Preloaded PRL"


                preloaded_conf_doc = 5

                self.doctor_conf_all.append(preloaded_conf_doc)
                self.rim_conf_all.append(preloaded_text)
                self.doctor_comment_all.append("")

                self.ui.doctorSlider.setValue(preloaded_conf_doc)
                self.ui.doctorComment.setText("")
                self.ui.rimConfNb.setText(preloaded_text)

                self.save_patches(idx+1, False)

            fileName = self.filename
            fileName = "temp_" + fileName
            self.write_json(fileName)




                # Display the new lesion once it is created with new name
    def viewLesion(self, node_name):
        def selectNode(caller, eventId):
            markupsDisplayNode = caller
            label = str(markupsDisplayNode.GetMarkupsNode().GetNthFiducialLabel())
            idx = int(re.findall(r'\d+', label)[-1]) - 1
            self.show_data_when_click(idx)

        markupsDisplayNode = slicer.util.getNode(node_name).GetDisplayNode()
        self.observations.append([markupsDisplayNode,
                                  markupsDisplayNode.AddObserver(markupsDisplayNode.ActionEvent, selectNode)])

    # Display all features of Lesion when double click on lesion
    def show_data_when_click(self, idx):
        self.clicked_lesion_index = idx

        self.ui.doctorSlider.value = self.doctor_conf_all[idx]
        self.ui.rimConfNb.text = str(self.rim_conf_all[idx])
        self.ui.doctorComment.setText(self.doctor_comment_all[idx])

        self.ui.saveButton.hide()
        self.ui.pickButton.hide()

        self.ui.doctorScore.show()
        self.ui.doctorSlider.show()
        self.ui.rimConf.show()
        self.ui.rimConfNb.show()
        self.ui.confirmModificationsButton.show()
        self.ui.doctorComment.show()

        self.ui.conf_0.show()
        self.ui.conf_1.show()
        self.ui.conf_2.show()
        self.ui.conf_3.show()
        self.ui.conf_4.show()
        self.ui.conf_5.show()

    # Initialise new lesion when we click on "pick a lesion"
    def initialize_points(self):
        mDisplayNode = slicer.util.getNode('Lesion').GetDisplayNode()
        mDisplayNode.SetGlyphScale(2.0)
        mDisplayNode.SetTextScale(0.0)
        mDisplayNode.SetColor(0, 0, 1)
        mDisplayNode.SetSelectedColor(0, 0, 1)
        slicer.mrmlScene.AddDefaultNode(mDisplayNode)

    # Link the unplaced lesion with movement of mouse
    def onPointPushButton(self, name):
        slicer.modules.markups.logic().SetActiveListID(self.markup_node)
        index = 0
        self.markup_node.UnsetNthControlPointPosition(index)
        self.markup_node.SetControlPointPlacementStartIndex(index)
        self.markup_node.SetNthControlPointLabel(index, name)
        slicer.modules.markups.logic().StartPlaceMode(0)

    # Add the new lesion to the scene when we confirm the lesion
    def addFiducial(self, ras, number):
        id = slicer.modules.markups.logic().AddNewFiducialNode('Lesion_' + str(number))
        self.fiducials = slicer.mrmlScene.GetNodeByID(id)
        slicer.mrmlScene.AddNode(self.fiducials)
        self.fiducials.AddFiducial(ras[0], ras[1], ras[2], "Lesion_" + str(number))
        mDisplayNode = self.fiducials.GetDisplayNode()
        mDisplayNode.SetGlyphScale(1.0)
        mDisplayNode.SetTextScale(2.0)
        slicer.mrmlScene.AddDefaultNode(mDisplayNode)
        self.viewLesion(mDisplayNode.GetMarkupsNode().GetNthFiducialLabel())

    # Switch display config (all, axial ...)
    def switchConfig(self):
        if (self.orientation == "Axial" or self.orientation == "Sagittal" or self.orientation == "Coronal"):
            logic = CompareVolumesLogic()
            volumeNodes = [self.foreground, self.background]
            viewers = logic.viewerPerVolume(
                volumeNodes=volumeNodes,
                orientation=self.orientation,
                background=None,
                label=None,
                layout=[1, 2],
                opacity=1.0,
            )
            for viewName in viewers.keys():
                sliceWidget = slicer.app.layoutManager().sliceWidget(viewName)
                compositeNode = sliceWidget.sliceLogic().GetSliceCompositeNode()
                compositeNode.SetLinkedControl(True)
                compositeNode.SetHotLinkedControl(True)
            self.ui.flairButton.hide()

        elif (self.orientation == "All Views"):
            layoutManager = slicer.app.layoutManager()
            layoutManager.layoutLogic().GetLayoutNode().SetViewArrangement(21)
            self.ui.flairButton.hide()
        elif (self.orientation == "Default"):
            layoutManager = slicer.app.layoutManager()
            layoutManager.layoutLogic().GetLayoutNode().SetViewArrangement(3)
            self.ui.flairButton.show()

    # Switch between flair and phase
    def showFlair(self):
        all_previous = []
        layoutManager = slicer.app.layoutManager()
        for sliceViewName in layoutManager.sliceViewNames():
            view = layoutManager.sliceWidget(sliceViewName).sliceView()
            sliceNode = view.mrmlSliceNode()
            sliceLogic = slicer.app.applicationLogic().GetSliceLogic(sliceNode)
            compositeNode = sliceLogic.GetSliceCompositeNode()
            compositeNode.SetLinkedControl(0)
            previous = compositeNode.GetForegroundOpacity()
            all_previous.append(previous)
        if self.orientation == "All Views":
            if sum(all_previous) == 0.0 or sum(all_previous) == 6.0:
                all_same = True
            else:
                all_same = False
        for sliceViewName in layoutManager.sliceViewNames():
            view = layoutManager.sliceWidget(sliceViewName).sliceView()
            sliceNode = view.mrmlSliceNode()
            sliceLogic = slicer.app.applicationLogic().GetSliceLogic(sliceNode)
            compositeNode = sliceLogic.GetSliceCompositeNode()
            compositeNode.SetLinkedControl(0)
            previous = compositeNode.GetForegroundOpacity()
            new = (np.round(previous) + 1) % 2
            name = compositeNode.GetLayoutName()
            if self.orientation == "All Views":
                if (previous != 0.0 and previous != 1.0) or all_same:
                    if (name == "Red" or name == "Green" or name == "Yellow"):
                        new = 1.0
                    else:
                        new = 0.0
            compositeNode.SetForegroundOpacity(new)
            compositeNode.SetLinkedControl(1)

    # Recenter image
    def recentre(self):
        position_RAS = [0, 0, 0]
        crosshairNode = slicer.mrmlScene.GetSingletonNode("default", "vtkMRMLCrosshairNode")
        crosshairNode.SetCrosshairRAS(position_RAS)
        slicer.vtkMRMLSliceNode.JumpAllSlices(slicer.mrmlScene, *position_RAS,
                                              slicer.vtkMRMLSliceNode.CenteredJumpSlice)
        if (self.ui.displayArrowsButton.text == "Hide Arrows"):
            crosshairNode.SetCrosshairMode(crosshairNode.ShowIntersection)
        elif (self.ui.displayArrowsButton.text == "Show Arrows"):
            crosshairNode.SetCrosshairMode(0)
        slicer.util.setSliceViewerLayers(fit=True)

    def displayArrows(self):
        crosshairNode = slicer.mrmlScene.GetSingletonNode("default", "vtkMRMLCrosshairNode")
        if (self.ui.displayArrowsButton.text == "Hide Arrows"):
            crosshairNode.SetCrosshairMode(0)
            self.ui.displayArrowsButton.text = "Show Arrows"
        elif (self.ui.displayArrowsButton.text == "Show Arrows"):
            crosshairNode.SetCrosshairMode(crosshairNode.ShowIntersection)
            self.ui.displayArrowsButton.text = "Hide Arrows"

    # Click on button "pick a lesion"
    def pickPointPressed(self):
        self.ui.pickButton.hide()
        self.initialize_points()
        self.onPointPushButton("Lesion")
        if (self.placing_pointer == False):
            self.ui.processLesionButton.show()
            self.ui.loadFile.hide()

    # Click on button "process lesion"
    def processLesion(self):
        self.getCoordinates()
        self.number_lesion += 1
        if (self.ui.doctorScore.isHidden() and (self.within_bounds)):
            self.ui.doctorScore.show()
            self.ui.doctorSlider.show()
            self.ui.confirmButton.show()
            self.ui.doctorComment.show()

            self.ui.conf_0.show()
            self.ui.conf_1.show()
            self.ui.conf_2.show()
            self.ui.conf_3.show()
            self.ui.conf_4.show()
            self.ui.conf_5.show()
            self.ui.pleaseRunRimNet.show()

            self.ui.saveButton.hide()

            self.coordinates_all.append(self.current_coordinates)
            self.ras_all.append(self.current_ras)

            self.rim_conf_all.append("None")
            self.ui.runRimNetButton.hide()

    # Call docker on all patches
    def runRimNet(self):

        self.ui.rimnetDoneLabel.show()
        self.ui.rimnetDoneLabel.setText("Running RimNet...")
        print("Running RimNet...")


        # SEND CSV FILE TO DOCKER HERE #
        fileName = self.filename
        docker_data = f"{os.getcwd()}/sessions/{self.session_name}/patches/"
        docker_model = f"{os.getcwd()}/{self.config_params['Docker_model']}"
        docker_code = f"{os.getcwd()}/{self.config_params['Docker_code']}"
        model_to_use = self.config_params["model_to_use"]

        docker_cmd = f"docker run -v {docker_data}:/data -v {docker_model}:/models -v {docker_code}:/code ghcr.io/medical-image-analysis-laboratory/rimnet:latest dataset_test_{fileName}.csv --model {model_to_use} --fold all"
        subprocess.run(docker_cmd, shell=True, stderr=subprocess.PIPE)

        result = pd.read_csv(f"sessions/{self.session_name}/patches/predictions_{model_to_use}_all.csv").sort_values(by=['patch_id'])
        prediction_csv = result[["fold_0", "fold_1", "fold_2", "fold_3"]]

        means = list(prediction_csv.mean(axis=1))

        self.rim_conf_all = list(np.round(np.array(means), 3))
        self.ui.runRimNetButton.hide()
        self.ui.rimnetDoneLabel.show()
        self.ui.saveButton.show()
        self.ui.rimnetDoneLabel.setText("RimNet Done !")

    # Save patches and add them to the csv list
    def save_patches(self, number, fromModif):
        self.ui.pleaseRunRimNet.hide()

        x = self.coordinates_all[number-1][0]
        y = self.coordinates_all[number-1][1]
        z = self.coordinates_all[number-1][2]

        lesion_name = f"Lesion_{number}"

        if not os.path.exists(f"sessions/{self.session_name}/patches"):
            os.makedirs(f"sessions/{self.session_name}/patches")

        if (self._parameterNode.GetNodeReference("InputVolume") is not None):
            fileName = self.filename

            data = {'sub_id': [fileName],
                    'patch_id': [number],
                    'cont_T2STAR_PHASE': [f"{fileName}_{lesion_name}_phase.nii.gz"],
                    'cont_FLAIR': [f"{fileName}_{lesion_name}_flair.nii.gz"],
                    'to_Run': ["yes"],
                    'Coordinates': [f"{str(x), str(y), str(z)}"]}

            # Check if file exists
            if os.path.isfile(f"sessions/{self.session_name}/patches/dataset_test_{fileName}.csv"):
                df = pd.read_csv(f"sessions/{self.session_name}/patches/dataset_test_{fileName}.csv")
                if fromModif:
                    df.drop(df.index[df['patch_id'] == number].tolist()[0], inplace=True)
                df_single_row = pd.DataFrame(data=data)
                df = pd.concat([df.loc[:], df_single_row])
                df.to_csv(f"sessions/{self.session_name}/patches/dataset_test_{fileName}.csv", index=False)
            else:
                df = pd.DataFrame(data=data)
                df.to_csv(f"sessions/{self.session_name}/patches/dataset_test_{fileName}.csv", index=False)

            if self.flair is not None:
                flair_patch = self.numpy_flair[x - 17:x + 17, y - 17:y + 17, z - 17:z + 17]
                flair_patch = 2 * (
                            (flair_patch - np.min(flair_patch)) / (np.max(flair_patch) - np.min(flair_patch))) - 1
                flair_to_save = nibabel.Nifti1Image(flair_patch, affine=self.nibabel_img_flair.affine,
                                                    header=self.nibabel_img_flair.header)
                nibabel.save(flair_to_save, f"sessions/{self.session_name}/patches/{fileName}_{lesion_name}_flair.nii.gz")

            if self.phase is not None:
                phase_patch = self.numpy_phase[x - 17:x + 17, y - 17:y + 17, z - 17:z + 17]
                phase_patch = 2 * (
                            (phase_patch - np.min(phase_patch)) / (np.max(phase_patch) - np.min(phase_patch))) - 1
                phase_to_save = nibabel.Nifti1Image(phase_patch, affine=self.nibabel_img_phase.affine,
                                                header=self.nibabel_img_phase.header)
                nibabel.save(phase_to_save, f"sessions/{self.session_name}/patches/{fileName}_{lesion_name}_phase.nii.gz")

            if self.magnitude is not None:
                mag_patch = self.numpy_mag[x - 17:x + 17, y - 17:y + 17, z - 17:z + 17]
                mag_patch = 2 * (
                            (mag_patch - np.min(mag_patch)) / (np.max(mag_patch) - np.min(mag_patch))) - 1
                mag_to_save = nibabel.Nifti1Image(mag_patch, affine=self.nibabel_img_mag.affine,
                                                  header=self.nibabel_img_mag.header)
                nibabel.save(mag_to_save, f"sessions/{self.session_name}/patches/{fileName}_{lesion_name}_mag.nii.gz")

        prediction = "Run Later"

        idx = self.clicked_lesion_index
        self.rim_conf_all[idx] = prediction
        self.ui.rimConfNb.text = str(prediction)

    def getCoordinates(self):
        markupsNode = slicer.util.getNode('Lesion')
        volumeNode = self._parameterNode.GetNodeReference("InputVolume")

        point_ras = [0, 0, 0]
        markupsNode.GetNthFiducialPosition(0, point_ras)

        # Apply that transform to get volume's RAS coordinates
        transform_ras_to_volume_ras = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, volumeNode.GetParentTransformNode(),
                                                             transform_ras_to_volume_ras)
        point_volume_ras = transform_ras_to_volume_ras.TransformPoint(point_ras[0:3])

        # Get voxel coordinates from physical coordinates
        volume_ras_to_ijk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(volume_ras_to_ijk)
        point_ijk = [0, 0, 0, 1]
        volume_ras_to_ijk.MultiplyPoint(np.append(point_volume_ras, 1.0), point_ijk)
        point_ijk = [int(round(c)) for c in point_ijk[0:3]]

        # Get markup's position
        i, j, k = point_ijk[0], point_ijk[1], point_ijk[2]

        self.current_ras = point_ras
        self.current_coordinates = [i, j, k]
        self.spacing = volumeNode.GetSpacing()
        self.origin = volumeNode.GetOrigin()
        voxelArray = slicer.util.arrayFromVolume(volumeNode)
        k_dim, j_dim, i_dim = voxelArray.shape
        if (i < i_dim and j < j_dim and k < k_dim) and (i >= 0 and j >= 0 and k >= 0):
            self.within_bounds = True
            self.ui.chooseInside.hide()
        else:
            self.within_bounds = False
            self.ui.chooseInside.show()

    # Click on button "confirm modif"
    def confirmModificationsButtonPressed(self):

        # Get coordinates of lesion (if it has been moved) by using the name (ie. Lesion_1, Lesion_2 ...)
        fiducialNodes = slicer.util.getNodes('vtkMRMLMarkupsFiducialNode*')
        idx = self.clicked_lesion_index
        if self.loaded_PRLs:
            lesion_name = list(fiducialNodes.items())[idx][0]
        else:
            lesion_name = list(fiducialNodes.items())[idx + 1][
                0]  # +1 cause the first element is "Lesion" that we initialized in beginning

        point_ras = [0, 0, 0]
        fiducialNodes[lesion_name].GetNthFiducialPosition(0, point_ras)
        volumeNode = self._parameterNode.GetNodeReference("InputVolume")

        # Apply that transform to get volume's RAS coordinates
        transform_ras_to_volume_ras = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, volumeNode.GetParentTransformNode(),
                                                             transform_ras_to_volume_ras)
        point_volume_ras = transform_ras_to_volume_ras.TransformPoint(point_ras[0:3])

        # Get voxel coordinates from physical coordinates
        volume_ras_to_ijk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(volume_ras_to_ijk)
        point_ijk = [0, 0, 0, 1]
        volume_ras_to_ijk.MultiplyPoint(np.append(point_volume_ras, 1.0), point_ijk)
        point_ijk = [int(round(c)) for c in point_ijk[0:3]]

        previous_coordinates = self.coordinates_all[idx]
        # Get markup's position
        i, j, k = point_ijk[0], point_ijk[1], point_ijk[2]


        self.coordinates_all[idx] = [i, j, k]

        expert = self.ui.doctorSlider.value
        self.doctor_conf_all[idx] = expert

        text = self.ui.doctorComment.toPlainText()
        self.doctor_comment_all[idx] = text
        self.ui.doctorComment.setText("")

        self.ui.doctorScore.hide()
        self.ui.doctorSlider.hide()
        self.ui.processLesionButton.hide()
        self.ui.rimConf.hide()
        self.ui.rimConfNb.hide()
        self.ui.confirmModificationsButton.hide()
        self.ui.doctorComment.hide()
        self.ui.conf_0.hide()
        self.ui.conf_1.hide()
        self.ui.conf_2.hide()
        self.ui.conf_3.hide()
        self.ui.conf_4.hide()
        self.ui.conf_5.hide()

        self.ui.saveButton.show()
        self.ui.pickButton.show()

        if (self._parameterNode.GetNodeReference("InputVolume") is not None):
            fileName = self.filename
            fileName = "temp_" + fileName
        self.write_json(fileName)

        if list(self.coordinates_all[idx]) != list(previous_coordinates):
            self.ui.runRimNetButton.show()
            self.ui.rimnetDoneLabel.hide()
            self.save_patches(idx + 1, fromModif=True)

    # Click on button "Confirm"
    def confirmButtonPressed(self):
        #### FORCE USER TO USE RIMNET ####
        self.markup_node.UnsetNthControlPointPosition(0)
        expert = self.ui.doctorSlider.value
        self.doctor_conf_all.append(expert)
        text = self.ui.doctorComment.toPlainText()
        self.doctor_comment_all.append(text)
        self.ui.doctorComment.setText("")

        self.ui.doctorScore.hide()
        self.ui.doctorSlider.hide()
        self.ui.processLesionButton.hide()
        self.ui.rimConf.hide()
        self.ui.rimConfNb.hide()
        self.ui.confirmButton.hide()
        self.ui.doctorComment.hide()

        self.ui.conf_0.hide()
        self.ui.conf_1.hide()
        self.ui.conf_2.hide()
        self.ui.conf_3.hide()
        self.ui.conf_4.hide()
        self.ui.conf_5.hide()

        self.ui.saveButton.show()
        self.ui.pickButton.show()

        self.current_coordinates = []

        number = len(self.ras_all)
        self.addFiducial(self.current_ras, number)

        self.save_patches(number, fromModif=False)

        if (self._parameterNode.GetNodeReference("InputVolume") is not None):
            fileName = self.filename
            fileName = "temp_" + fileName
        self.write_json(fileName)

        self.ui.doctorSlider.value = 0
        self.ui.saveButton.show()
        self.ui.runRimNetButton.show()
        self.ui.rimnetDoneLabel.hide()

    # Click on Save data
    def saveButtonPressed(self):
        if (self._parameterNode.GetNodeReference("InputVolume") is not None):
            fileName = self.filename
            if (os.path.exists(f"sessions/{self.session_name}/temp_{fileName}.json")):
                self.write_json(fileName)
                os.remove(f"sessions/{self.session_name}/temp_{fileName}.json")
            self.ui.saveButton.hide()

    # Write current state to json
    def write_json(self, filename):
        s = self.spacing
        o = self.origin
        img = self.nibabel_img_flair
        affine = img.affine

        # create json file and write
        dictionnary = {
            "spacing": s,
            "origin": o,
            "affine": affine.tolist(),
            "instances": {
            }
        }

        for i in range(len(self.rim_conf_all)):
            dict_element = {
                "Lesion": str(i + 1),
                "Expert's confidence score": self.map_conf_int_to_str(int(self.doctor_conf_all[i])),
                "RimNet's confidence score": self.rim_conf_all[i],
                "Position x": int(np.float(self.coordinates_all[i][0])),
                "Position y": int(np.float(self.coordinates_all[i][1])),
                "Position z": int(np.float(self.coordinates_all[i][2])),
                "Comments": self.doctor_comment_all[i]
            }
            dictionnary["instances"][i + 1] = dict_element
        json_object = json.dumps(dictionnary, indent=4)

        json_path = f"sessions/{self.session_name}/{str(filename)}.json"
        with open(json_path, "w") as outfile:
            outfile.write(json_object)

    # Map an int of score to a string
    def map_conf_int_to_str(self, conf_value):
        if conf_value == 0:
            return "Not a PRL"
        if conf_value == 1:
            return "Probably not a PRL"
        if conf_value == 2:
            return "Uncertain"
        if conf_value == 3:
            return "Maybe a PRL"
        if conf_value == 4:
            return "Probably a PRL"
        if conf_value == 5:
            return "PRL"


#
# semi_rimnetLogic
#

class semi_rimnetLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "-50.0")
        if not parameterNode.GetParameter("Invert"):
            parameterNode.SetParameter("Invert", "false")

    def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            'InputVolume': inputVolume.GetID(),
            'OutputVolume': outputVolume.GetID(),
            'ThresholdValue': imageThreshold,
            'ThresholdType': 'Above' if invert else 'Below'
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True,
                                 update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime - startTime:.2f} seconds')

        # slicer.util.setSliceViewerLayers(background=mrVolume, foreground=ctVolume)


#
# semi_rimnetTest
#

class semi_rimnetTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_semi_rimnet1()

    def test_semi_rimnet1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('semi_rimnet1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = -50

        # Test the module logic

        logic = semi_rimnetLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')


class CompareVolumesLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget
    """

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.sliceViewItemPattern = """
      <item><view class="vtkMRMLSliceNode" singletontag="{viewName}">
        <property name="orientation" action="default">{orientation}</property>
        <property name="viewlabel" action="default">{viewName}</property>
        <property name="viewcolor" action="default">{color}</property>
      </view></item>
     """
        # use a nice set of colors
        self.colors = slicer.util.getNode('GenericColors')
        self.lookupTable = self.colors.GetLookupTable()

    def assignLayoutDescription(self, layoutDescription):
        """assign the xml to the user-defined layout slot"""
        layoutNode = slicer.util.getNode('*LayoutNode*')
        if layoutNode.IsLayoutDescription(layoutNode.SlicerLayoutUserView):
            layoutNode.SetLayoutDescription(layoutNode.SlicerLayoutUserView, layoutDescription)
        else:
            layoutNode.AddLayoutDescription(layoutNode.SlicerLayoutUserView, layoutDescription)
        layoutNode.SetViewArrangement(layoutNode.SlicerLayoutUserView)

    def viewerPerVolume(self, volumeNodes=None, background=None, label=None, viewNames=[], layout=None,
                        orientation='Axial', opacity=0.5):
        """ Load each volume in the scene into its own
        slice viewer and link them all together.
        If background is specified, put it in the background
        of all viewers and make the other volumes be the
        forground.  If label is specified, make it active as
        the label layer of all viewers.
        Return a map of slice nodes indexed by the view name (given or generated).
        Opacity applies only when background is selected.
        """
        if not volumeNodes:
            volumeNodes = list(slicer.util.getNodes('*VolumeNode*').values())

        if len(volumeNodes) == 0:
            return

        if layout:
            rows = layout[0]
            columns = layout[1]

        #
        # construct the XML for the layout
        # - one viewer per volume
        # - default orientation as specified
        #
        actualViewNames = []
        index = 1
        layoutDescription = ''
        layoutDescription += '<layout type="vertical">\n'
        for row in range(int(rows)):
            layoutDescription += ' <item> <layout type="horizontal">\n'
            for column in range(int(columns)):
                try:
                    viewName = viewNames[index - 1]
                except IndexError:
                    viewName = '%d_%d' % (row, column)
                rgb = [int(round(v * 255)) for v in self.lookupTable.GetTableValue(index)[:-1]]
                color = '#%0.2X%0.2X%0.2X' % tuple(rgb)
                layoutDescription += self.sliceViewItemPattern.format(viewName=viewName, orientation=orientation,
                                                                      color=color)
                actualViewNames.append(viewName)
                index += 1
            layoutDescription += '</layout></item>\n'
        layoutDescription += '</layout>'
        self.assignLayoutDescription(layoutDescription)

        # let the widgets all decide how big they should be
        slicer.app.processEvents()

        # put one of the volumes into each view, or none if it should be blank
        sliceNodesByViewName = {}
        layoutManager = slicer.app.layoutManager()
        for index in range(len(actualViewNames)):
            viewName = actualViewNames[index]
            try:
                volumeNodeID = volumeNodes[index].GetID()
            except IndexError:
                volumeNodeID = ""

            sliceWidget = layoutManager.sliceWidget(viewName)
            compositeNode = sliceWidget.mrmlSliceCompositeNode()
            if background:
                compositeNode.SetBackgroundVolumeID(background.GetID())
                compositeNode.SetForegroundVolumeID(volumeNodeID)
                compositeNode.SetForegroundOpacity(opacity)
            else:
                compositeNode.SetBackgroundVolumeID(volumeNodeID)
                compositeNode.SetForegroundVolumeID("")

            if label:
                compositeNode.SetLabelVolumeID(label.GetID())
            else:
                compositeNode.SetLabelVolumeID("")

            sliceNode = sliceWidget.mrmlSliceNode()
            sliceNode.SetOrientation(orientation)
            sliceNodesByViewName[viewName] = sliceNode
        return sliceNodesByViewName
