import ipywidgets as ipyw
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable


class ImageSliceViewer3D:
    
    def __init__(self, im, mas, figsize=(4,4)):
        self.image = im
        self.mask = mas
        self.figsize = figsize
        self.v = [np.min(self.image), np.max(self.image)]
        self.m = [np.min(self.mask), np.max(self.mask)]
        
        x_size, y_size, z_size = self.image.shape
        self.xy_label = f'x-y ({x_size}x{y_size})'
        self.yz_label = f'y-z ({y_size}x{z_size})'
        self.zx_label = f'z-x ({z_size}x{x_size})'
        
        # Call to select slice plane
        ipyw.interact(self.view_selection, view=ipyw.RadioButtons(
            options=[self.xy_label, self.yz_label, self.zx_label], 
            description='Plane:', disabled=False,
            style={'description_width': 'initial'}))
    
    def view_selection(self, view):
        # Transpose the volume to orient according to the slice plane selection
        orient = {self.xy_label:[0,1,2], self.yz_label:[2,0,1], self.zx_label: [1,2,0]}
        label = {self.xy_label: "z", self.yz_label: "x", self.zx_label: "y"}
        self.vol = np.transpose(self.image, orient[view])
        self.mas = np.transpose(self.mask, orient[view])
        maxZ = self.vol.shape[2] - 1
        
        # Call to view a slice within the selected slice plane
        ipyw.interact(self.plot_slice, 
            z=ipyw.IntSlider(value=maxZ//2, min=0, max=maxZ, step=1, continuous_update=False, 
            description=f'{label[view]} (0-{maxZ}):'))
        
    def plot_slice(self, z):
        # Plot slice for the given plane and slice
        self.fig = plt.figure(figsize=self.figsize)
        
        plt.subplot(121, title="Image")
        plt.imshow(self.vol[:,:,z], vmin=self.v[0], vmax=self.v[1], cmap='gray')
        
        plt.subplot(122, title="Mask")
        #plt.imshow(self.mas[:,:,z], vmin=self.m[0], vmax=self.m[1], cmap='gray')
        plt.imshow(self.mas[:,:,z], vmin=0, vmax=2, cmap='gray')
        

        
class ContrastsViewer3D:

    def __init__(self, phase, t2, mask, *, show_mask=True, figsize=(8,8), titles=("Phase", "T2", "Mask")):
        self.titles = titles
        self.phase = phase
        self.figsize = figsize
        self.mask = mask
        self.t2 = t2
        self.show_mask = show_mask
        self.v = [np.min(self.phase), np.max(self.phase)]
        self.m = [np.min(self.mask), np.max(self.mask)]
        self.o = [np.min(self.t2), np.max(self.t2)]
        
        x_size, y_size, z_size = self.phase.shape
        self.xy_label = f'x-y ({x_size}x{y_size})'
        self.yz_label = f'y-z ({y_size}x{z_size})'
        self.zx_label = f'z-x ({z_size}x{x_size})'
        
        # Call to select slice plane
        ipyw.interact(self.view_selection, view=ipyw.RadioButtons(
            options=[self.xy_label, self.yz_label, self.zx_label], 
            description='Plane:', disabled=False,
            style={'description_width': 'initial'}))
    
    def view_selection(self, view):
        # Transpose the volume to orient according to the slice plane selection
        orient = {self.xy_label:[0,1,2], self.yz_label:[2,0,1], self.zx_label: [1,2,0]}
        label = {self.xy_label: "z", self.yz_label: "x", self.zx_label: "y"}
        self.phase = np.transpose(self.phase, orient[view])
        self.mask = np.transpose(self.mask, orient[view])
        self.t2 = np.transpose(self.t2, orient[view])
        maxZ = self.phase.shape[2] - 1
        
        # Call to view a slice within the selected slice plane
        ipyw.interact(self.plot_slice, 
            z=ipyw.IntSlider(value=maxZ//2, min=0, max=maxZ, step=1, continuous_update=False, 
            description=f'{label[view]} (0-{maxZ}):'))
        
    def plot_slice(self, z):
        # Plot slice for the given plane and slice
        self.fig = plt.figure(figsize=self.figsize)
        
        plt.subplot(131, title=self.titles[0])
        plt.imshow(self.phase[:,:,z], vmin=self.v[0], vmax=self.v[1], cmap='gray')
        plt.subplot(132, title=self.titles[1])
        plt.imshow(self.t2[:,:,z], vmin=self.o[0], vmax=self.o[1], cmap='gray')
        
        if self.show_mask:
            plt.subplot(133, title=self.titles[2])
            plt.imshow(self.mask[:,:,z], vmin=self.m[0], vmax=self.m[1], cmap='gray')
            
        
class OcclusionMapViewer3D:

    def __init__(self, im, mask, occl, *, show_mask=False, figsize=(8,8)):
        self.image = im
        self.figsize = figsize
        self.mask = mask
        self.occl = occl
        self.show_mask = show_mask
        self.v = [np.min(self.image), np.max(self.image)]
        self.m = [np.min(self.mask), np.max(self.mask)]
        self.o = [np.min(self.occl), np.max(self.occl)]
        
        x_size, y_size, z_size = self.image.shape
        self.xy_label = f'x-y ({x_size}x{y_size})'
        self.yz_label = f'y-z ({y_size}x{z_size})'
        self.zx_label = f'z-x ({z_size}x{x_size})'
        
        # Call to select slice plane
        ipyw.interact(self.view_selection, view=ipyw.RadioButtons(
            options=[self.xy_label, self.yz_label, self.zx_label], 
            description='Plane:', disabled=False,
            style={'description_width': 'initial'}))
    
    def view_selection(self, view):
        # Transpose the volume to orient according to the slice plane selection
        orient = {self.xy_label:[0,1,2], self.yz_label:[2,0,1], self.zx_label: [1,2,0]}
        label = {self.xy_label: "z", self.yz_label: "x", self.zx_label: "y"}
        self.image = np.transpose(self.image, orient[view])
        self.mask = np.transpose(self.mask, orient[view])
        self.occl = np.transpose(self.occl, orient[view])
        maxZ = self.image.shape[2] - 1
        
        # Call to view a slice within the selected slice plane
        ipyw.interact(self.plot_slice, 
            z=ipyw.IntSlider(value=maxZ//2, min=0, max=maxZ, step=1, continuous_update=False, 
            description=f'{label[view]} (0-{maxZ}):'))
        
    def plot_slice(self, z):
        # Plot slice for the given plane and slice
        self.fig = plt.figure(figsize=self.figsize)
        
        plt.subplot(131, title="Image")
        plt.imshow(self.image[:,:,z], vmin=self.v[0], vmax=self.v[1], cmap='gray')
        
        if self.show_mask:
            plt.subplot(132, title="Mask")
            plt.imshow(self.mask[:,:,z], vmin=self.m[0], vmax=self.m[1], cmap='gray')
            ax = plt.subplot(133, title="Map")
            im = plt.imshow(self.occl[:,:,z], vmin=self.o[0], vmax=self.o[1], cmap='gray')
        else:
            plt.subplot(121, title="Image")
            plt.imshow(self.image[:,:,z], vmin=self.v[0], vmax=self.v[1], cmap='gray')
            ax = plt.subplot(122, title="Map")
            im = plt.imshow(self.occl[:,:,z], vmin=self.o[0], vmax=self.o[1], cmap='gray')
 