# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 01:39:51 2020

@author: gcg
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, io
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from scipy import interpolate, optimize
import cv2
import triaxiality

class Silhouette():
    def __init__(self, filename, plot=False, transpose=False):
        self.filename = filename
        self.transpose = transpose
        self.read_image()
        self.cannyTransform()
        #self.find_contour_points()
        self.find_edges()
        
        #plt.imshow(self.edges)
        #plt.show()
        
        
        self.find_minimum_distance()
        self.curvature_valid = False
        self.find_neck_curvature()    
            
        self.status = True

        #print("triaxiality: ", triaxiality.triax_function(self.diameter, self.notch_radius))

        if plot and self.status == True:
            
            plt.figure(figsize=(8,8))
            ax = plt.subplot(aspect='equal')
            
            
            
            if self.curvature_valid:
                circle1 = plt.Circle(self.circle_center_up, self.notch_radius_up, color='g', fill=False)
                circle2 = plt.Circle(self.circle_center_lo, self.notch_radius_lo, color='r', fill=False)
                ax.add_artist(circle1)
                ax.add_artist(circle2)
                
            #ax.annotate('double-headed arrow', xy=(0.45,0.5), xytext=(0.01,0.5),
            self.edge_bot_pts - self.edge_top_pts
            #ax.annotate(s='', xy=(self.neck_location, self.edge_top_spline(self.neck_location)),
            #             xytext=(self.neck_location,self.edge_bot_spline(self.neck_location)), arrowprops=dict(arrowstyle='<->'))
            
            
            ax.imshow(self.edges, cmap=plt.get_cmap('binary'))    
            #ax.imshow(self.im, cmap=plt.get_cmap('binary'))
            #ax.imshow(self.im)    
            #plt.title("%s \n x=%4.2f dia=%4.2f " % (self.filename, self.neck_bot_x, self.diameter))
            
            #plt.axvline(x=self.neck_location, color='y', label="min. diameter location")
            #plt.plot(self.top_px[:,0], self.top_px[:,1], "gs", fillstyle="none", label="top contour pixels")
            #plt.plot(self.bot_px[:,0], self.bot_px[:,1], "rs", fillstyle="none", label="bot contour pixels")
            #plt.plot(self.x, self.edge_top_pts, 'g--', linewidth=3, label="top spline")
            #plt.plot(self.x, self.edge_bot_pts, 'r--', linewidth=3, label="bot spline")
            
            #plt.legend()
            plt.show()
            
            
    def remove_small_objects(self, img):
        #find all your connected components (white blobs in your image)
        
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        
        # minimum size of particles we want to keep (number of pixels)
        #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        min_size = 200
        
        #your answer image
        img2 = np.zeros((output.shape))
        #for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255
        #cv2.imshow("Biggest component", img2)
        #cv2.waitKey()
        return img2
            
    def cannyTransform(self):
        
        #th, im_bin = cv2.threshold(self.im, 128, 255, cv2.THRESH_OTSU)
        
        
        low = 0.19 * self.im.max()
        high = 0.79 * self.im.max()
        sigma = 5
        
        
        #
        #self.im = self.remove_small_objects(self.im)
        #cv2.imshow("Biggest component", self.im)
        #cv2.waitKey()
        
        
        
        
        self.edges = cv2.Canny(self.im, low, high, sigma)
        #self.edges = self.remove_small_objects(self.edges)
        
            
    def read_image(self):
        im = io.imread(self.filename)
        
        if (len(np.shape(im)) == 3): # this is a color image, convert to gray
            #print ("length of shape:", len(np.shape(im)))
            im = rgb2gray(im)
        
        im = img_as_ubyte(im) # for the opencv canny edge detecter we need an 8-bit image, so convert
            
        if self.transpose:
            print("* TRANSPOSING IMAGE AS REQUESTED IN CONSTRAIN.INI *")
            im = im.T
            
        self.ny, self.nx = im.shape
        self.x = np.arange(self.nx)
        self.y = np.arange(self.ny)
        print("image dtype: ", im.dtype, "image shape: ", im.shape)
        print("image range: min=%g, max=%g" % (im.min(), im.max()))
        
        
        if True:
            # remove small features -- dust
            t1 = 128
            th, im2 = cv2.threshold(im, t1, 255, cv2.THRESH_OTSU)
            im2 = self.remove_small_objects(im2)
            im2 = img_as_ubyte(im2/255.0)
            #print("*** image dtype: ", im.dtype, "image shape: ", im2.shape)
            #print("*** image range: min=%g, max=%g" % (im2.min(), im2.max()))
            im = im2
        
        
        self.im = im
        
        
    def find_notch_radius_with_points_SAVE(self, center, points):
        """ determine curvature around the necking point
        Here: using the Hyperfit function and the raw edge
        locations determined in function find_edges()"""
        import circle_fit
        
        all_x, all_y = points[:,0], points[:,1] # entire edge
        
        stencil = int(0.45*self.diameter)
        
        print("+++ minimum diameter: 6.2f%" % (self.diameter))
        #print("stencil", stencil)
        #print("center: ", center)
        
        #task: find those entries in all_x, which define the stencil
        import sys
        np.set_printoptions(threshold=sys.maxsize)

        xs = (all_x - center).flatten() # shifted coords
        #print("xs:", xs, "stencil:", -stencil, stencil)
        indices = np.where(np.logical_and(xs >= -stencil, xs <= stencil))
        
        #print("indices:", indices)
        indices = np.asarray(indices).flatten()
        #print("indices:", indices)
        if len(indices) < 0.25*stencil: # not enough data, failing
            print("@@@ stencil too small")
            return None, None, None
        else: # we have enough data to carry on
            
            start_index = indices[0]
            stop_index = indices[-1]
            #print("start_index:", start_index)
            #print("stop_index:", stop_index)
            x = all_x[start_index:stop_index]
            y = all_y[start_index:stop_index]
            
            pts = np.column_stack((x,y))
            xc, yc, radius, sigma = circle_fit.hyper_fit(pts)
        
        import math
        if math.isnan(radius) or math.isnan(xc) or math.isnan(yc) or math.isnan(sigma):
            return None, None, None

        return radius, sigma, np.asarray((xc,yc))        

    def find_notch_radius_with_points(self, center, points):
        """ determine curvature around the necking point
        Here: using the Hyperfit function and the raw edge
        locations determined in function find_edges()"""
        import circle_fit
        
        all_x, all_y = points[:,0], points[:,1] # entire edge
        
        stencil = int(0.45*self.diameter)
        
        #print("### diameter", self.diameter)
        #print("stencil", stencil)
        #print("center: ", center)
        lower_cutoff = center - stencil
        upper_cutoff = center + stencil
        #print("lower cutoff: ", lower_cutoff)
        #print("upper cutoff: ", upper_cutoff)

        import sys
        np.set_printoptions(threshold=sys.maxsize)
        #print("all_x", all_x)
        indices = np.where((all_x > lower_cutoff) & (all_x < upper_cutoff))
        #print("indices:", indices)
        x = all_x[indices]
        y = all_y[indices]
        #print("x: ", x)
        
        #sys.exit()
               
        if len(x) < 0.25*stencil: # not enough data, failing
            print("@@@ stencil too small")
            return None, None, None
        else: # we have enough data to carry on
            pts = np.column_stack((x,y))
            xc, yc, radius, sigma = circle_fit.hyper_fit(pts)
        
        import math
        if math.isnan(radius) or math.isnan(xc) or math.isnan(yc) or math.isnan(sigma):
            return None, None, None

        return radius, sigma, np.asarray((xc,yc))        
    
    
    def find_spline_curvature(self, center, spline):
        """ determine curvature around the necking point """
        
        stencil = int(0.2*self.diameter)
        x = np.linspace(center - stencil, center + stencil, 2*stencil+1)
        #plt.plot(self.x, spline(self.x))
        #plt.plot(x, spline(x))
        
        curvature = spline.derivative(n=2)(x) # array of curavture for stencil
        #plt.plot(x, curvature)
        
        curvature_mean, curvature_std = np.mean(curvature), np.std(curvature)
        
        #radii = 1. / curvature
        #radius, sigma = np.mean(radii), np.std(radii)
        sigma = ( 1. / (curvature_mean**2)) * curvature_std
        radius =  1. / curvature_mean
        print("spline radius: %4.2f +- %g" % (radius, sigma))
        #plt.show()
        return radius, sigma
    
    def find_notch_radius_with_spline(self, center, spline):
        """ determine curvature around the necking point
        Here: using the Hyperfit function"""
        import circle_fit
        
        stencil = int(0.25*self.diameter)
        x = np.linspace(center - stencil, center + stencil, 2*stencil+1)
        y = spline(x)
        
        pts = np.column_stack((x,y))
        xc, yc, radius, sigma = circle_fit. hyper_fit(pts)
        
        #print("+++ circle radius: %f +- %f" % (radius, sigma))
        #circle_fit.plot_data_circle(x, y, xc, yc, radius)
        #print(res)
        
        return radius, sigma, np.asarray((xc,yc))
    
    def find_neck_curvature(self):
        """ determine average curvature of upper and lower necks """
        
        # --- compute neck radii with canny-edge points and hyperfit:
        #plt.plot(self.top_px[:,0], self.top_px[:,1])
        #plt.plot(self.bot_px[:,0], self.bot_px[:,1])
        #plt.show()
        self.notch_radius_up, self.sigma_up, self.circle_center_up = \
            self.find_notch_radius_with_points(self.neck_location, self.top_px)
            
        self.notch_radius_lo, self.sigma_lo, self.circle_center_lo = \
            self.find_notch_radius_with_points(self.neck_location, self.bot_px)
            
        # --- compute neck radii based on spline curvature
        #self.notch_radius_up, self.sigma_up = self.find_spline_curvature(self.neck_location, self.edge_top_spline)
        #self.notch_radius_lo, self.sigma_lo = self.find_spline_curvature(self.neck_location, self.edge_bot_spline)
        
        #self.circle_center_up = (self.neck_location, self.edge_top_spline(self.neck_location) + self.notch_radius_up)
        #self.circle_center_lo = (self.neck_location, self.edge_bot_spline(self.neck_location) + self.notch_radius_lo)
        
            
        # check if results are valid
        self.curvature_valid = True
        if self.notch_radius_up == None:
            print("upper notch radius is invalid")
            self.curvature_valid = False
        if self.notch_radius_lo == None: 
            print("lower notch radius is invalid")
            self.curvature_valid = False 
        
        #if abs(self.sigma_up) > 5*abs(self.notch_radius_up) or \
        #    abs(self.sigma_lo) > 5*abs(self.notch_radius_lo):
        #        print("notch uncertainty is too large, disabling notch")
        #        self.curvature_valid = False 
        
        if self.curvature_valid:
            #print("@", self.circle_center_lo[1] , self.bot_px[self.neck_location][0])
            if self.circle_center_lo[1] < self.edge_bot_pts[self.neck_location]:
                print("lower edge: circle center on wrong side")
                print("neck location: ", self.neck_location)
                print("circle center: ", self.circle_center_lo)
                self.curvature_valid = False
            if self.circle_center_up[1] > self.edge_top_pts[self.neck_location]:
                print("upper edge: circle center on wrong side")
                print("neck location: ", self.neck_location)
                print("circle center: ", self.circle_center_up)
                self.curvature_valid = False
        
        if not self.curvature_valid:
            print("******** neck curvature detection failed *********")
            self.notch_radius = None
            self.notch_sigma = None
            return
            
        
        
        
        
        print("upper neck radius: %6.1f +- %6.1f  (%3.1f %%)" % (self.notch_radius_up, self.sigma_up, 100*self.sigma_up/self.notch_radius_up))
        print("lower neck radius: %6.1f +- %6.1f  (%3.1f %%)" % (self.notch_radius_lo, self.sigma_lo, 100*self.sigma_lo/self.notch_radius_lo))
        
        
        wt_up = 1./(self.sigma_up + 1)
        wt_lo = 1./(self.sigma_lo + 1)
        
        self.notch_radius = (wt_up * np.abs(self.notch_radius_up) + wt_lo * np.abs(self.notch_radius_lo)) / (wt_up + wt_lo)
        
        #self.notch_radius = self.notch_radius_up + self.notch_radius_lo)
        
        #self.notch_radius = 0.5 * (self.notch_radius_up + self.notch_radius_lo)
        self.notch_sigma = np.sqrt(self.sigma_up**2 +  self.sigma_lo**2)
            
    
    def find_contour_points(self):
        """ traverse each column of the Canny edge image and report the transition 
        from white to black, starting from below, and the same, starting from the top.
        """
        ny, nx = np.shape(self.edges)
        bot_transitions = np.zeros(nx, dtype=int)
        top_transitions = np.zeros(nx, dtype=int)
        #print("+++ shape of bot_transitions is: ", np.shape(bot_transitions))
        for ix in range(nx): # columns have index 0
            
            column = self.edges[:,ix]
            #print("column:", column)
            white_pixels = np.argwhere(column[:] > 0)
            #print("+++ shape of white_pixels is: ", np.shape(white_pixels))
            #print("white_ixels: ", white_pixels)
            if len(white_pixels >= 2):
                top_transitions[ix] = white_pixels[0]
                bot_transitions[ix] = white_pixels[-1]
            
            # take care of outliers
            if bot_transitions[ix] < 0.5*ny:
                bot_transitions[ix] = ny-1
            
            if top_transitions[ix] > 0.5*ny:
                top_transitions[ix] = 0
            
        self.bot_px = bot_transitions
        self.top_px = top_transitions
            
        # interpolate transition line with smoothing spline
        self.x = np.arange(nx)
        
        # define spline knot vector
        dt = self.nsmooth
        t = np.arange(dt, nx-dt, step=dt)
        #print("this is t:", t)
        
        self.edge_bot_spline = interpolate.LSQUnivariateSpline(self.x, bot_transitions, t, k=self.order)
        self.edge_bot_pts = self.edge_bot_spline(self.x)
        
        self.edge_top_spline = interpolate.LSQUnivariateSpline(self.x, top_transitions, t, k=self.order)
        self.edge_top_pts = self.edge_top_spline(self.x)
        
    
    def find_edges(self):
        """ traverse each column of the Canny edge image and report the transition 
        from white to black, starting from below, and the same, starting from the top.
        """
        
        bot_transitions = []
        top_transitions = []
        centerline = self.ny / 2 # long axis of specimen -- centerline

        for ix in range(self.nx): # columns have index 0
            
            column = self.edges[:,ix]
            white_pixels = np.argwhere(column[:] > 0)
            if len(white_pixels > 0):
                
                # look at only those white pixels which are above centerline
                top_white_pixels = np.select([white_pixels[:] < centerline], [white_pixels], default=-1)
                if top_white_pixels[0] > 0:
                    top_transitions.append((ix, top_white_pixels[0][0]))
                    
                # look at only those white pixels which are above centerline
                bot_white_pixels = np.select([white_pixels[:] > centerline], [white_pixels], default=-1)
                if bot_white_pixels[-1] > 0:
                    bot_transitions.append((ix, bot_white_pixels[-1][0]))
            
        self.bot_px = np.asarray(bot_transitions)
        self.top_px = np.asarray(top_transitions)
        print("*** len top contour = %d, bot contour = %d" % (len(self.bot_px), len(self.top_px)))
            
        s = self.nx/8
        self.edge_bot_spline = interpolate.UnivariateSpline(self.bot_px[:,0], self.bot_px[:,1], k=2, s=s, ext=3)
        self.edge_bot_pts = self.edge_bot_spline(self.x)
        
        self.edge_top_spline = interpolate.UnivariateSpline(self.top_px[:,0], self.top_px[:,1], k=2, s=s, ext=3)
        self.edge_top_pts = self.edge_top_spline(self.x)
            
        
    def find_minimum_distance(self):
        """
        find smallest diameter, i.e. that point where the edges come closest
        """
        #assert (len(x1 == len(x2)))
        
        self.neck_top_x = np.argmax(self.edge_top_pts)
        self.neck_top_y = self.edge_top_pts[self.neck_top_x]
        
        self.neck_bot_x = np.argmin(self.edge_bot_pts)
        self.neck_bot_y = self.edge_bot_pts[self.neck_bot_x]
        
        #self.diameter = np.sqrt((self.neck_top_x - self.neck_bot_x)**2 + (self.neck_top_y - self.neck_bot_y)**2)
        #return
        
        
        distance = self.edge_bot_pts - self.edge_top_pts
        self.neck_top_y
        #print("distance:", distance)
        min_distance_index = np.argmin(distance)
        min_distance_value = distance[min_distance_index]
        print("*** min distance of %g occurs at pixel index %d" % (min_distance_value, min_distance_index))
        self.diameter = min_distance_value
        self.neck_location = min_distance_index
        
        #return min_distance_index, min_distance_value
    

if __name__ == '__main__':
    Silhouette(filename, plot=True, transpose=transpose)
    #import cProfile
    #cProfile.run("SilhouetteDiameter(filename, transpose=False, plot=False)", sort='cumulative')



