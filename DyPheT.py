
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 07:48:48 2022
@author: phykc
"""
import rgbcelltracker2 as rgb
import numpy as np
from math import cos, sin, pi
import cv2 as cv2
import os
import pandas as pd
import csv
import datetime

class OpenVideo(rgb.OpenVideo):
    """Opens the video given of the path. Contains methods for runing the video, processing the frames, and collecting cell information"""

    def __init__(self, file, path, search_angle, autotrack=True):
        super().__init__(file, path, search_angle,autotrack)
        
        if 'RGB' in file:
            file1 = file.replace('RGB', 'Overlay')
        else:
            file1=file
        filename = os.path.join(path, file1)
        self.filename = filename
        self.file1=file1[:-4]
        print(filename)
        self.autotrack=autotrack
        self.no_keypoints=36
        self.previouswrapsize=540
        self.search_angle=search_angle*10
        self.flags = cv2.INTER_AREA  + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print(f'Search angle {self.search_angle}')
        # Set the lists and arrays for storing T and R
        self.T_values = []
        self.R_values = []
        self.M_values = []
        self.rotation = []
        self.x_shift = []
        self.y_shift = []
        self.frameshift = []
        self.rotationerr=[]
        self.hit=[]
        self.frameno = 0
        self.blur_values=(21,21)
        self.cap = cv2.VideoCapture(self.filename)
        self.maxframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.center = (self.width/2, self.height/2)
        self.cx, self.cy =self.center[0],self.center[1]
        self.pcx, self.pcy= self.cx, self.cy
        self.stdangtemp=[]
        self.trackdata=[]
        self.R = 0
        self.Rprevious=0
        self.Rdiff=2
        
        print(f'width {self.width}, height {self.height}')
        
    
    def transrotate(self, anglestep, windowsize, allowedcentreshift,maxrounds, tolerance):
        ''' A function to determine the translation and rotation of the assembliod'''
        self.angle_rot = np.zeros(self.maxframes, dtype=float)
        self.anglestep=anglestep
        self.windowsize=windowsize
        self.maxrounds=maxrounds
        self.tolerance=tolerance
        # Loop through the frames of the movie
        #log the process
        flogname='logger_'+str(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"))+'.csv'
        with open(flogname,'a+') as self.csvfile1:
            self.writer=csv.writer(self.csvfile1)
            self.writer.writerow(['Frame Number','iteration','Angle applide','Angle measured','Spheriod angle','Angle error', 'cx', 'cy', 'dx', 'dy', 'derr'])
            # Move through the frames one by one
            while self.frameno < self.maxframes:
                print(f'Frame {self.frameno+1} of {self.maxframes}')
                
                # Read each frame ret is True if there is a frame
                ret, self.frame = self.cap.read()
                cv2.imwrite(f'frame{self.frameno}.jpg', self.frame)
                # This ends the movie when no frame is present
                if not ret:
                    print('End of frames')
                    cv2.waitKey(1)
                    break
                
                # estimate the organiod centre position using the outline
                self.cx, self.cy=self.find_centre()
                
                    
                # if the centre seems to have changed considerably we assume linear motion instead    
                if (abs(self.cx - self.pcx)>allowedcentreshift or abs(self.cy - self.pcy)>allowedcentreshift) and self.frameno>2:
                
                    self.cx, self.cy=self.predictcxcy()
                # Manual fits are possible if autotrack is set to False
                # Autotrack fine tunes the centre position and determines the rotation.
                
                if self.autotrack and self.frameno>0:
                    self.runautotrack()
                    self.writer.writerows(self.tracking_run)
                    self.trackdata.append(self.tracking_run)
                    
                    
                else:
                    self.T = np.float32([[1, 0, self.center[0]-self.cx], [0, 1, self.center[1]-self.cy]])
 
                    self.manadjustRT()
                
                # Define the Translation matrix to move the COM of the organoid to the centre of the image in preparation for rotation
                # Compute translation offsets to center the organoid
                tx = self.center[0] - self.cx
                ty = self.center[1] - self.cy
                
                # Create a single affine matrix to rotate around self.center and translate
                M = cv2.getRotationMatrix2D(self.center, self.R, 1.0)
                M[0, 2] += tx
                M[1, 2] += ty
                
                # Store for tracking/debug
                self.M_values.append(M.copy())
                self.T_values.append(self.T)
                self.R_values.append(self.R)
                
                # Apply the transformation to the image
                self.rotated = cv2.warpAffine(self.frame, M, (self.width, self.height))
                
                # Apply the same transformation to the contour template
                self.contourtemplate = np.zeros(self.morath.shape, dtype=np.uint8)
                cv2.drawContours(self.contourtemplate, [self.hull], 0, 255, -1)
                self.contourtemp = cv2.warpAffine(self.contourtemplate, M, (self.width, self.height))
                
                # store previous
                if self.frameno == 0:
                    self.firstframe = self.rotated.copy()
                    self.previousframe = self.rotated.copy()
                    self.previousframe1 = self.rotated.copy()
                    self.previousframe=cv2.GaussianBlur(self.previousframe, (11,11),0)
                    self.previouscontemp=self.contourtemp.copy()
                    self.createwarpimgs()  
                    # create warp template
                    previous_result = self.previousframe.copy()
                    # gray_result= cv2.cvtColor(previous_result, cv2.COLOR_BGR2GRAY)
                    # blurred = cv2.GaussianBlur(gray_result, (11, 11), 0)
                    # T, threshInv_result = cv2.threshold(blurred, 1, 255,cv2.THRESH_BINARY)
                    
                    # self.mask_result = cv2.cvtColor(threshInv_result, cv2.COLOR_GRAY2RGB)
                    self.mask_result=self.contourtemp.copy()
                if self.frameno ==1:
                    self.firstframe = self.previousframe.copy()
                    self.previousframe = self.rotated.copy()
                    self.previousframe=cv2.GaussianBlur(self.previousframe, (11,11),0)
                    self.previousframe1=self.previousframe.copy()
                    self.Rprevious=self.R
                    self.previouscontemp=self.contourtemp.copy()
                    self.createwarpimgs()   
                    previous_result = self.previousframe.copy()
                    
                    
                
                   
                    
                elif self.frameno > 1:
                    self.previousframe1=self.previousframe.copy()
                    self.previousframe = self.rotated.copy()
                    self.Rprevious=self.R
                    self.previouscontemp=self.contourtemp.copy()
                    self.createwarpimgs() 
                    previous_result = self.previousframe.copy()
                    
                   
                    
                 
                
                # store the x, y and r shifts for anaylysis
                self.rotation.append(self.R_values[-1])
                self.x_shift.append(self.T[0, 2])
                self.y_shift.append(self.T[1, 2])
                self.frameshift.append(self.frameno)
                self.pcx,self.pcy=self.cx, self.cy
                # move to the next frame
                cv2.destroyWindow(f'Overlayed match update frame {self.frameno+1}')
                # cv2.waitKey(1)
                self.frameno += 1
            
            # Release the video
            self.cap.release()
            
            cv2.waitKey(1)
            # Kill the windows
            cv2.destroyAllWindows()
            cv2.waitKey(1)
    
            Rfilename = self.filename[:-4]+'Rvals.npy'
            Tfilename = self.filename[:-4]+'Tvals.npy'
            indexnumber=0
            if self.autotrack:
                isFile = True
                while isFile:
                    RTfilename = self.filename[:-4]+f'RT_data_auto{indexnumber:03d}.xlsx'
                    isFile = os.path.isfile(RTfilename)
                    indexnumber+=1
                    
            else:
                isFile = True
                while isFile:
                    RTfilename = self.filename[:-4]+f'RT_data_man{indexnumber:03d}.xlsx'
                    isFile = os.path.isfile(RTfilename)
                    indexnumber+=1
        self.csvfile1.close()
        np.save(Rfilename, self.R_values, allow_pickle=True)
        np.save(Tfilename, self.T_values, allow_pickle=True)
        # save the shift data
        dRT = {'R': self.rotation, 'x': self.x_shift, 'y': self.y_shift}
        print(len(self.rotation), len(self.x_shift), len(self.y_shift))
        RTdataframe = pd.DataFrame(dRT)
        RTdataframe.to_excel(RTfilename)
        try:
            print(self.stdangtemp)
        except:
            print('finished')
    
    def find_centre(self, areamin=7000):
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            # Create a grayscale image from the orginal frame
            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            # use a heavy adpative threshold to make a pronounced image of the organiod.
            ath = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 301, 6)
            #  Use morpthological changes to distort the organoid and invert the grayscale
            self.morath = cv2.morphologyEx(ath, cv2.MORPH_CLOSE, kernel, iterations=1)
            #invert
            self.morath = 255-self.morath
            #dilate to join the outline up
            self.morath = cv2.dilate(self.morath, kernel1, iterations=5)
            
            # find the contours in the dilated threshold image
            contours, hist = cv2.findContours(self.morath, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # arbitary parameter areamin limits for the cnt size.  All cnt's bigger than this are concatenated.
            cnts = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > areamin:
                    cnts.append(cnt)
            if len(cnts)>0:
                cnt1 = np.concatenate(cnts)
            else:
                cnt1=contours[0]
            # take the hull of the concatenated contour (the organoid)
            
            self.hull = cv2.convexHull(cnt1)
            # Find the centre of the hull using moments.  weakness assumes com is centre of turning point.
            M = cv2.moments(self.hull)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        except:
        
            cx,cy=self.predictcxcy()
        
        # This attempts to deal with the situation when the organoid overlaps the FOV, it then predicts the centre based on the trend in the motion.  It is not as effective as I hoped.
        if self.doesithit() and self.frameno>2:
            cx,cy=self.predictcxcy()
        return cx,cy
    
    def doesithit(self):
    # test if the hull touches the edge of frame
        xtest,ytest,wtest,htest = cv2.boundingRect(self.hull)
        # determine if the organoid is overlapping boundaries
        if xtest<=0 or xtest+wtest>=self.width or ytest<=0 or ytest+htest>=self.height:
            print('overlaps edge - using previous cx and cy')
            hit=True
        else:
            hit=False
        
        return hit
    
    def predictcxcy(self):
        # Assumes the assembloid is moving in the same direction as previously to estimate the position
        if len(self.x_shift)>3:
            x1=self.center[0]-self.x_shift[-3]
            x2=self.center[0]-self.x_shift[-1]
            diffx=(x2-x1)/2
            newx=x2+diffx
            y1=self.center[1]-self.y_shift[-3]
            y2=self.center[1]-self.y_shift[-1]
            diffy=(y2-y1)/2
            newy=y2+diffy
            print('predicting cx and cy')
        else:
            newx=self.cx
            newy=self.cy
        return newx, newy
    
    def runautotrack(self):
        #Initialise parameters
        rdiff=10
        rounds=0
        dx,dy=1,1
        angerr=1
        if not hasattr(self, 'rotation_conf_all'):
            self.rotation_conf_all = []
        if not hasattr(self, 'confidence_history'):
            self.confidence_history = []
        
        # capture the assembliod without too much background
        cutsize=self.findcutsize()
        self.angle_test_list=[]
        self.tracking_run=[]
        fail=False
        oversmooth=False
        if self.frameno == 1:
            self.calibration_derr = []
            self.calibration_anstde = []
        checklast=self.R_values[-1]
        # The tolerance is that the combination of angle difference and magnitude of shift should be less and a tolerance (0.8)
        if self.frameno == 3 and len(self.rotation_conf_all) >= 10:
            self.rotation_conf_thresh = np.percentile(self.rotation_conf_all, 75)
            print(f"Calibrated confidence threshold: {self.rotation_conf_thresh:.2f}")
        elif not hasattr(self, 'rotation_conf_thresh'):
            self.rotation_conf_thresh = 0.95  # fallback default

        while np.sqrt(dx**2+dy**2+(self.Rdiff)**2)>self.tolerance and rounds<self.maxrounds:
            # Find the adjustements needed in x and y to best match the previous frame
            dx, dy, derr=self.estimate_T(rounds, cutsize)
            # Smoothing factor between 0 and 1 (lower = more smoothing but slower)
            # Fine adjustments to rotation
            anstde, an, ag, conf_list, good_match =self.estimate_R(rounds)
            
            if good_match==False:
                an=ag
                
            if len(conf_list) > 0:
                avg_conf = np.mean(conf_list)
                self.confidence_history.append(avg_conf)
            
                # Keep only last 5 frames
                if len(self.confidence_history) > 5:
                    self.confidence_history.pop(0)
            
                # Dynamic threshold: 10th percentile of recent 5, or fallback
                self.rotation_conf_thresh = max(np.percentile(self.confidence_history, 10), 0.6)
            else:
                self.rotation_conf_thresh = 0.6  # fallback

            
            if self.frameno > 3:
                derr_norm = derr / (self.mean_derr + 1e-3)  # Normalize current error
                anstde_norm = anstde / (self.std_anstde + 1e-3)
    
                alpha1 = np.clip(0.9 - 0.6 *(derr_norm / 4.0), 0.3, 0.95)
                alpha = np.clip(0.9 - 0.6 * (anstde_norm / 4.0), 0.3, 0.9)
            else:
                alpha1 = np.clip(0.9 - 0.4 * (derr / 4.0), 0.4, 0.95)
                alpha = np.clip(0.9 - 0.4 * (anstde / 4.0), 0.4, 0.9)
            
            
            if self.frameno <= 3:
                self.calibration_derr.append(derr)
                self.calibration_anstde.append(anstde)
            if self.frameno <= 3 and len(conf_list) > 0:
                self.rotation_conf_all.extend(conf_list)


            # Detect if angle update has stalled and confidence is still poor
            if rounds > 2 and len(self.angle_test_list) > 3:
                recent_angles = [x[0] for x in self.angle_test_list[-3:]]
                delta_angles = [abs(recent_angles[i+1] - recent_angles[i]) for i in range(2)]
                mean_change = np.mean(delta_angles)
                if mean_change < 0.4 and derr > 1 and not oversmooth:
                    print(" Angle estimate appears stuck — forcing stronger update.")
                    self.Rdiff=1
                    alpha1 = 0.7  # Override smoothing
                    alpha=1.0 # Override smoothing
                    self.maxrounds=30
                    oversmooth=True
            
            
            # Apply  smoothing
            self.cx = (1 - alpha) * self.cx + alpha * (self.cx + dx)
            self.cy = (1 - alpha) * self.cy + alpha * (self.cy + dy)
            
            
            
            if abs(an - checklast) < self.search_angle:
                self.R = (1 - alpha1) * self.R + alpha1 * an
                if rounds > 0 and anstde < self.angle_test_list[-1][1]:
                    checklast = an
            elif rounds==0:
                self.R=self.R_values[-1]
                print('Using the last frame angle')
            
            else:
                print('using the nearest vales to the last frame angle')
                self.R=checklast

            if anstde>4 and rounds>=self.maxrounds:
                self.manadjustRT()
            
            
            if rounds> self.maxrounds/2:
                diverging=self.detect_behavior()
                if diverging:
                    print('Attempt to correcting non-divergent behavior')
                    svals=np.sin(self.angmag[:-3]*pi/180)
                    cvals=np.cos(self.angmag[:-3]*pi/180)
                    self.R=np.arctan2(np.sum(svals),np.sum(cvals))*180/pi
                    self.maxrounds=30
                    
            if rounds>9:
                dx=(0.5*dx)
                dy=(0.5*dy)
            if self.R>180:
                self.R-=360
            if self.R<-180:
                self.R+=360
                
            self.angle_test_list.append([self.R,anstde])
            if dx>100 or dy>100 or derr>100:
                print(f'might be a T problem the error the {derr:.1f}')
            
            # Act on the frame
           
            self.T = np.float32(
                [[1, 0, self.center[0]-self.cx], [0, 1, self.center[1]-self.cy]])
           
            tx = self.center[0] - self.cx
            ty = self.center[1] - self.cy
            
            
            M = cv2.getRotationMatrix2D(self.center, self.R, 1.0)
            M[0, 2] += tx
            M[1, 2] += ty
            
            # Apply the affine warp
            self.rotated = cv2.warpAffine(self.frame, M, (self.width, self.height))
            
            
            self.Rprevious = self.R
            angle=0
            self.tracking_run.append([self.frameno,rounds,self.R,an,angle,anstde, self.cx, self.cy, dx, dy, derr])
           
            print(f'file: {self.file1} ,frame {self.frameno+1}, iteration={rounds+1} , dx = {dx:.1f}, dy = {dy:.1f}, Angle = {self.R:.1f}, derr ={derr:.1f}, alpha={alpha:.2f}')
            
            rounds+=1
            if fail==True:
                self.manadjustRT()
                break
        if self.frameno == 3:
            self.mean_derr = np.mean(self.calibration_derr)
            self.std_anstde = np.mean(self.calibration_anstde)
            
            print(f"Calibration complete: mean derr = {self.mean_derr:.2f}, std anstde = {self.std_anstde:.2f}")

    def findcutsize(self):
        # Use the hull to determine the radius
        circle=cv2.minEnclosingCircle(self.hull)
        R=circle[1]
        cutsize=int((self.width//2)-(0.9*(R)))
        print(f'Frame Cut size is {cutsize} and Radius {int(R)}')
        if cutsize<0:
            cutsize=0
        return cutsize
    
    def estimate_T(self,rounds, cutsize):
        #perform linear translation using the matrix T
        
        tx = self.center[0] - self.cx
        ty = self.center[1] - self.cy
        M_rot = cv2.getRotationMatrix2D(self.center, self.R, 1.0)
        M_rot[0, 2] += tx
        M_rot[1, 2] += ty

        # Test the combined transformation
        self.rotated = cv2.warpAffine(self.frame, M_rot, (self.width, self.height))
        dst2 = cv2.addWeighted(self.previousframe,0.4, self.rotated, 0.6, 0)
        cv2.imshow(f'Overlayed match update frame {self.frameno+1}',dst2)
        cv2.moveWindow(f'Overlayed match update frame {self.frameno+1}',200,0)
        cv2.waitKey(1)
        rot_draw=self.rotated.copy()
        # rot_draw=cv2.GaussianBlur(rot_draw, (11,11),0)
        confidenceth=70
        # do linear matching by template panels
        notlongenough=True
        panels=2
        stdxy=100
        solvefailcount=0
        outsideonly=True
        dxlonglist=[]
        dylonglist=[]
        panth_level=5
        while notlongenough:
            print(f'Matching {panels} x {panels} panels')
            rot_draw, delxlist,delylist ,delconarray ,outcoord=self.linear_shift(cutsize, rot_draw, confidenceth,outsideonly,panels)
            if len(delxlist)>0:
                dxlonglist=dxlonglist+delxlist
                dylonglist=dylonglist+delylist

            if len(dxlonglist)>=panth_level:
                notlongenough=False
            else:
                if solvefailcount>2:
                    confidenceth-=1
                panels+=1
                
                if panels>6:
                    outsideonly=False
               
            if solvefailcount>panth_level:
                notlongenough=False
                print('failed to obtain more fits.')
            
                    
            solvefailcount+=1
            
        if len(dxlonglist)>0:
        
            delxlist=self.removeoutliers(dxlonglist)
            delylist=self.removeoutliers(dylonglist)
            
            delatxtm = np.median(delxlist)
            delatytm = np.median(delylist)*-1
            
            xstd=np.std(delxlist)
            ystd=np.std(delylist)
            stdxy=np.sqrt(xstd**2+ystd**2)/np.sqrt(len(delxlist))
            
            if np.isnan(delatxtm):
                delatxtm=0
            if np.isnan(delatytm):
                delatytm=0
            cv2.circle(rot_draw,(int(self.width//2+delatxtm),int(self.height//2+delatytm)),20,(0,255,0),2)
            print(f'Mean condidence in panel match :{100*np.mean(delconarray):.3f}, {100*len(delxlist)/(panels**2):.1f}')
            dst2 = cv2.addWeighted(self.previousframe,0.4, rot_draw, 0.8, 0)
            cv2.imshow(f'Overlayed match update frame {self.frameno+1}',dst2)
            cv2.moveWindow(f'Overlayed match update frame {self.frameno+1}',200,0)
            cv2.waitKey(1)    
            r_prime =np.sqrt(delatxtm**2+delatytm**2)
            a_prime = 180*np.arctan2(delatytm, delatxtm)/pi
            
            sigma = (self.R-a_prime)*pi/180
            dx1 = r_prime*cos(sigma)
            dy1 = r_prime*sin(sigma)
            
            return dx1,dy1, stdxy
            # return delatxtm,delatytm, stdxy
    
    def linear_shift(self, cutsize, rot_draw, confidenceth, outsideonly, panels=3):
        
        
        testimage=cv2.cvtColor(self.rotated,cv2.COLOR_BGR2GRAY)
        templateimage=cv2.cvtColor(self.previousframe,cv2.COLOR_BGR2GRAY)
        # self.liner_matching(templateimage, testimage)
        
        method= cv2.TM_CCOEFF_NORMED
        
        x_panels =panels
        y_panels =panels
        
        xt_step = (self.width-cutsize*2)//x_panels
        yt_step = (self.height-cutsize*2)//y_panels
        
        i_values = np.linspace(cutsize, self.width-cutsize-xt_step, x_panels, dtype=int)
        k_values = np.linspace(cutsize, self.height-cutsize-yt_step, y_panels, dtype=int)
       
        xt_step=int(i_values[1]-i_values[0])
        yt_step=int(k_values[1]-k_values[0])
        
        delxlist = []
        delylist = []
        delconfidence=[]
        outcoord=[]
    
        max_shift=100
        
        
        for pa,i in enumerate(i_values):
            for pb,k in enumerate(k_values):
                if (pa==0 or pb==0 or pa==x_panels-1 or pb==y_panels-1) or not outsideonly:
                    L_template = templateimage[k:k+yt_step, i:i+xt_step].copy()
                    w, h = L_template.shape
                    
                    mask = self.contourtemp[k:k+yt_step, i:i+xt_step].copy()
                    
                    if np.sum(mask)>0:
                       
                        res = cv2.matchTemplate(testimage, L_template, method,None, mask=mask)
                        search_template=np.zeros(res.shape,dtype=np.uint8)
                        cv2.rectangle(search_template,(i-max_shift,k-max_shift),(i+xt_step+max_shift,k+yt_step+max_shift),255,-1)
                        
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res, mask=search_template)
                       
                        
                        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                            max_val=1-min_val
                            top_left = min_loc
    
                        else:
                            top_left = max_loc
    
                        bottom_right = (top_left[0] + w, top_left[1] + h)
                        top_right = (top_left[0] + w, top_left[1])
                        bottom_left = (top_left[0], top_left[1] + h)
                        
                        
                        confidenceth1=98
                        if max_val*100>=confidenceth1:
                            cl=(0,0,255)
                        else:
                            cl=(255,0,0)
                        if max_val*100>confidenceth and not np.isinf(max_val):
                            delxlist.append(bottom_right[0]-(i+xt_step))
                            delylist.append(bottom_right[1]-(k+yt_step))
                            delconfidence.append(max_val)
                            cv2.rectangle(rot_draw, top_left, bottom_right, cl, 2)
                        else:
                            cv2.rectangle(rot_draw, top_left, bottom_right, cl, 2)
        
                       
            
            delconarray=np.array(delconfidence)
            
            delxlistarray=np.array(delxlist)
            delylistarray=np.array(delylist)
            
            delxlist=list(delxlistarray)
            delylist=list(delylistarray)
            
        cv2.circle(rot_draw,(self.width//2,self.height//2),18,(0,0,255),2)
        
        return rot_draw, delxlist,delylist ,delconarray ,outcoord
    
    def removeoutliers(self, l_list):
        sorted_list=sorted(l_list)
        upper_q=np.percentile(sorted_list,75)
        lower_q=np.percentile(sorted_list,25)
        iqr=(upper_q-lower_q)*1.5
        q_set=(lower_q-iqr, upper_q+iqr)
        result_list=[]
        for anp in sorted_list:
            if anp>=q_set[0] and anp<= q_set[1]:
                result_list.append(anp)
        return result_list
    
    def estimate_R(self, rounds):
        ''' Estmates the rotation using template matching of the polar warped tranformed frame image'''
        self.warp = cv2.warpPolar(self.frame, (500, 3600),(self.cx, self.cy), self.previouswrapsize, self.flags)
        # self.warp=cv2.GaussianBlur(self.warp, self.blur_values,0)
        upperlim=50
        lowerlim=50
        c_t=getattr(self, 'rotation_conf_thresh', 0.7)
        if self.frameno==0:
            self.perviouswarpmask1=None
        whiteblock=np.ones(self.perviouswarpmask1[0:self.windowsize, lowerlim:-upperlim].shape,dtype=np.uint8)
        whiteblock=whiteblock*255
        sumwhite=np.sum(whiteblock)
        method=cv2.TM_CCOEFF_NORMED
        y_shift_list = []
        conf_list=[]
        con_y_shift_list = []
        filtered_list=[]
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 3
        color = (200, 50, 0)
        thickness = 5
        displayimage=cv2.hconcat([self.previouswarp,self.warp])
        # add the angles as text for display
        for o in range(0,360,45):
            cv2.putText(displayimage,str(o), (0,o*10), font,fontScale, color, thickness, cv2.LINE_AA, False)
        testimage=cv2.cvtColor(self.warp,cv2.COLOR_BGR2GRAY)
        templateimage=cv2.cvtColor(self.previouswarp,cv2.COLOR_BGR2GRAY)
        if len(self.templateranges)<5:
            print('changing to uniform sampling')
            self.templateranges=np.linspace(0,3600-self.windowsize//2,num=self.no_keypoints,endpoint=False, dtype=int)
        for n,rang_val in enumerate(self.templateranges):
            top_left1=(lowerlim,rang_val)
            bottom_right1=(500-upperlim,rang_val+self.windowsize)
            
            # cv2.imshow('warp mask1',self.perviouswarpmask1[rang_val:rang_val+self.windowsize, lowerlim:-upperlim])
            
            tempsum=np.sum(self.perviouswarpmask1[rang_val:rang_val+self.windowsize, lowerlim:-upperlim].copy())
            ratio=tempsum/sumwhite
            if ratio<0.8:
                cv2.rectangle(displayimage,top_left1,bottom_right1,(255,255,0),2)
                res = cv2.matchTemplate(testimage, templateimage[rang_val:rang_val+self.windowsize, lowerlim:-upperlim], method, None, mask=self.perviouswarpmask1[rang_val:rang_val+self.windowsize, lowerlim:-upperlim].copy())
                # Create a search template
                search_template=np.zeros(res.shape,dtype=np.uint8)
                
                starty=int(self.R_values[-1]*10+rang_val-self.search_angle)
                endy=int(self.R_values[-1]*10+rang_val+self.search_angle)
                negsplit=False
                possplit=False
                bothneg=False
                bothpos=False
                
                if starty<0 and endy>0:
                    negsplit=True
                if endy>3600 and starty<3600:
                    possplit=True
                if starty<0 and endy<0:
                    bothneg=True
                if endy>3600 and starty>3600:
                    bothpos=True
                
                if negsplit:
                    cv2.rectangle(search_template,(lowerlim,0),(500-upperlim,endy), 255,-1)
                    cv2.rectangle(search_template,(lowerlim,3600+starty),(500-upperlim,3600), 255,-1)
                elif possplit:
                    cv2.rectangle(search_template,(lowerlim,starty),(500-upperlim,3600), 255,-1)
                    cv2.rectangle(search_template,(lowerlim,0),(500-upperlim,endy-3600), 255,-1)
                elif bothneg:
                    starty=starty+3600
                    endy=endy+3600
                    cv2.rectangle(search_template,(lowerlim,starty),(500-upperlim,endy), 255,-1)
                elif bothpos:
                    starty=starty-3600
                    endy=endy-3600
                    cv2.rectangle(search_template,(lowerlim,starty),(500-upperlim,endy), 255,-1)
                else:
                    cv2.rectangle(search_template,(lowerlim,starty),(500-upperlim,endy), 255,-1)
                
                
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res, mask=search_template)
                dimensions, w, h = self.previouswarp[rang_val:rang_val+self.windowsize, lowerlim:-upperlim].shape[::-1]
                if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    top_left = min_loc
                    max_val=1-min_val
                else:
                    top_left = max_loc
    
                bottom_right = (top_left[0] + w, top_left[1] + h)
                
                if not np.isinf(max_val):
                    if max_val<c_t:
                        clr=(0,0,255)
                    else:
                        clr=(255,0,0)
    
                    top_left2=(top_left[0]+500,top_left[1])
                    bottom_right2=(bottom_right[0]+500,bottom_right[1])
                    cv2.rectangle(displayimage, top_left2, bottom_right2, clr, 2)
                    cv2.line(displayimage, bottom_right1, (bottom_right2[0],bottom_right2[1]), (0,255,0), 2)
                    conf_list.append(max_val)
                    con_y_shift_list.append([max_val, top_left[1]-self.templateranges[n],top_left[0]])
            
        
        counting=0
        notlongenough=True
        if len(conf_list)>0:
            confidence_thresh = c_t

            condec=0.01
            
        else:
            self.templateranges=np.linspace(self.windowsize//2,3600-self.windowsize//2,num=self.no_keypoints*3,endpoint=False, dtype=int)
            confidence_thresh=0.8
            condec=0.05
        countinglimit=10    
        while notlongenough and counting<countinglimit:
            counting+=1
            y_shift_list=[x[1] for x in con_y_shift_list if x[0]>=confidence_thresh and abs(x[2]-lowerlim)<35]
            for hsind,ys in enumerate(y_shift_list):
                if ys<-1800:
                    y_shift_list[hsind]=ys+3600
                elif ys>1800:
                    y_shift_list[hsind]=ys-3600

            yshiftarray=np.array(y_shift_list)
            f_a_list=list(yshiftarray)
            
            if len(f_a_list)>0:
                filtered_list=self.removeoutliers(f_a_list)
                if len(filtered_list)>6:
                    notlongenough=False
                confidence_thresh-=condec
            else:
                confidence_thresh-=condec         
        good_match=True
        if counting>=countinglimit:
            filtered_list=[x[1] for x in con_y_shift_list if x[0]>=np.max(conf_list)-(condec/2) and abs(x[2]-lowerlim)<50]
        # Default ag based on previous known value
        if self.frameno == 0:
            ag = 0.0
        else:
            ag = self.R_values[-1]
        
        if len(filtered_list) < 4:
            print("No good rotation match found — reverting to previous angle.")
            an = self.R_values[-1]
            anstde = 10  # High uncertainty
            
            if len(filtered_list) > 0:
                f_l = np.array(filtered_list)
                ag = np.mean(f_l) / 10
            good_match=False
        else:
            an, anstde = self.calculatetanangles(filtered_list)
            self.Rdiff = abs(an - self.Rprevious)
        
            if len(filtered_list) > 0:
                f_l = np.array(filtered_list)
                binx = np.argmin(abs(f_l - self.R_values[-1] * 10))
                ag = f_l[binx] / 10
                if ag > 180:
                    ag -= 360
                if ag < -180:
                    ag += 360
        
                         
                
                cv2.imshow('warp boxes n and n+1', displayimage)
                cv2.waitKey(1)
        
        return anstde, an, ag, conf_list, good_match
    
    def createwarpimgs(self):
        polar_size = (500, 3600)
        previous_result = cv2.bitwise_and(self.previousframe,self.previousframe, mask=self.previouscontemp)
        gray_result= cv2.cvtColor(previous_result, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray_result, (11, 11), 0)
        _, threshInv_result = cv2.threshold(gray_result, 1, 255,cv2.THRESH_BINARY)
        self.mask_res = cv2.cvtColor(threshInv_result, cv2.COLOR_GRAY2RGB)
       
        self.previouswrapsize=self.width//2
        self.previouswarp=cv2.warpPolar(self.previousframe, polar_size,self.center, self.previouswrapsize, self.flags)
        # self.previouswarp=cv2.GaussianBlur(self.previouswarp, self.blur_values,0)
        self.warpmask=cv2.warpPolar(self.mask_res, polar_size,self.center, self.previouswrapsize, self.flags)
        warpmaskcanvas=np.zeros(gray_result.shape,dtype=np.uint8)
        cv2.drawContours(warpmaskcanvas,[self.hull],0,255,-1)
        self.perviouswarpmask1=cv2.warpPolar(warpmaskcanvas, polar_size,(self.cx,self.cy), self.previouswrapsize, self.flags)
        if self.frameno>0:
            testimage=cv2.cvtColor(self.warp,cv2.COLOR_BGR2GRAY)
            ret,thresh2 = cv2.threshold(testimage,1,255,cv2.THRESH_BINARY_INV)
            self.perviouswarpmask1 = cv2.subtract(self.perviouswarpmask1,thresh2)
        self.findpointsofinterest(self.previouswarp, uniform=True)
        
    def findpointsofinterest(self, img, uniform=True):
        # The orb detection seems to be ineffective for videos tested
        self.templateranges = []
        self.halfwindowsize = self.windowsize // 2
    
        self.templateranges = []


        if uniform:
            numberofpositions = self.no_keypoints
            angle_centers = np.linspace(0, 3600, numberofpositions, endpoint=False, dtype=int)
        
            for center in angle_centers:
                start = center - self.halfwindowsize
                # Handle wraparound explicitly
                if start < 0:
                    start += 3600
                self.templateranges.append(start)
        
    
            print(f'Uniform sampling: {len(self.templateranges)} templates')
    
        else:
            numberofpositions = 1
            angle_step_range = 3600 // numberofpositions
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
            for a in range(numberofpositions):
                if a == 0:
                    for start in [3150, 0]:
                        end = start + angle_step_range // 2
                        orb = cv2.ORB_create(nfeatures=self.no_keypoints)
                        kp, _ = orb.detectAndCompute(gray_img[start:end, :], None)
                        self.templateranges += [
                            (int(x.pt[1]) - self.halfwindowsize) + start
                            for x in kp if x.pt[1] >= self.halfwindowsize
                        ]
                else:
                    start = a * angle_step_range - 300
                    end = start + angle_step_range
                    orb = cv2.ORB_create(nfeatures=self.no_keypoints)
                    kp, _ = orb.detectAndCompute(gray_img[start:end, :], None)
                    self.templateranges += [
                        (int(x.pt[1]) - self.halfwindowsize) + start
                        for x in kp if x.pt[1] >= self.halfwindowsize
                    ]
    
            print(f'ORB-based sampling: {len(self.templateranges)} templates')

    
    
    def calculatetanangles(self, fl):
        flarray=(np.array(fl)/10)*pi/180
        sinvals=np.sin(flarray)
        cosvals=np.cos(flarray)
        anglemed=np.arctan2(np.median(sinvals),np.median(cosvals))*180/pi
        stdval=(1/(1+(np.std(sinvals)/np.std(cosvals))**2))/np.sqrt(len(flarray))*180/pi
        print(f'Calculated angle {anglemed:.1f} +/- {stdval:.1f}')
        return  anglemed, stdval
    
    def detect_behavior(self):
        signal=np.array(self.angle_test_list)
        self.angmag=signal[:,0]
        diff=np.diff(self.angmag)
        diffsq=diff**2
        if np.mean(diffsq[-4:])>20**2:
            return True
        else:
            return False
      
    def manadjustRT(self):
        countcheck = 0
        if self.frameno == 0:
            self.R_values.append(0)
            self.R = 0
        newangle = self.R_values[-1]
    
        while self.frameno > 0:
            countcheck += 1
    
            # Apply translation to center the object at self.center
            self.T = np.float32([
                [1, 0, self.center[0] - self.cx],
                [0, 1, self.center[1] - self.cy]
            ])
            translated = cv2.warpAffine(self.frame, self.T, (self.width, self.height))
    
            # Apply rotation around image center
            M = cv2.getRotationMatrix2D(self.center, newangle, 1.0)
            self.rotated = cv2.warpAffine(translated, M, (self.width, self.height))
    
            # Display overlayed images
            dst = cv2.addWeighted(self.previousframe, 0.5, self.rotated, 0.7, 0)
            dst1 = cv2.addWeighted(self.firstframe, 0.5, self.rotated, 0.7, 0)
    
            cv2.imshow('n and n+1', dst)
            cv2.imshow('First frame reference', dst1)
            cv2.moveWindow('First frame reference', 1200, 0)
            cv2.waitKey(1)
    
            # Rotation direction matrix
            ma = pi * (newangle - 90) / 180
            k = cv2.waitKey(100)
    
            if k == ord('y'):
                print(f'Frame {self.frameno} accepted with rotation {newangle:.2f}°')
                self.R = newangle
                break
    
            elif k == ord('w'):  # move up in rotated frame
                self.cy += sin(-ma)
                self.cx -= cos(-ma)
    
            elif k == ord('s'):  # move down
                self.cy -= sin(-ma)
                self.cx += cos(-ma)
    
            elif k == ord('a'):  # move left
                self.cy += sin(-ma + pi / 2)
                self.cx -= cos(-ma + pi / 2)
    
            elif k == ord('d'):  # move right
                self.cy -= sin(-ma + pi / 2)
                self.cx += cos(-ma + pi / 2)
    
            elif k == ord('o'):  # rotate clockwise
                newangle = (newangle + 0.5) % 360
    
            elif k == ord('p'):  # rotate counterclockwise
                newangle = (newangle - 0.5) % 360
    
            elif k == ord('q'):  # quit manual adjustment
                print("Manual adjustment cancelled.")
                break

    def run(self):
        # This function applied the Rotation and Translation to the fluorecence image.
        print('File: ', self.filename)
        print('Size: ', self.width, ' x ', self.height)
        print('Frames: ', self.maxframes)
        print('Runnning...')
        self.framenumber = 0
        # Ensures the correct file is captured.
        if 'verlay' in self.filename:
            filename = self.filename.replace('Overlay', 'RGB')
        elif 'RGB' in self.filename:
            filename = self.filename
        print(filename)
        # define the new cap
        self.cap = cv2.VideoCapture(filename)
        # self.maxframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Loop through the frames.
        
        while self.framenumber < self.maxframes:
            print('frame ', self.framenumber+1, 'of ', self.maxframes)
            # Read each frame ret is True if there is a frame
            ret, self.frame = self.cap.read()
            # This ends the movie when no frame is present or q is pressed
            if not ret:
                print('End of frames')
                cv2.waitKey(1)
                break
            key = cv2.waitKey(200) & 0xFF
            if key == ord('q'):
                cv2.waitKey(1)
                break
            # Apply the rotation and translation, place the img +information as an object (Frame) in a list.
            
            self.frame2 = cv2.warpAffine(self.frame, self.M_values[self.framenumber], (self.width, self.height))
            self.frames.append(Frame(self.file, self.path, self.frame2,
                               f_no=self.framenumber, maskspheriod=False, autoTH=True))
            # Detect cells in the list by applying the analyse method to the last frame object.
            self.frames[-1].analyse()
            self.framenumber += 1
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def loadRT_data(self):
        self.cap = cv2.VideoCapture(self.filename)
        self.maxframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Rfilename = self.filename[:-4]+'Rvals.npy'
        Tfilename = self.filename[:-4]+'Tvals.npy'
        self.R_values = np.load(Rfilename, allow_pickle=True)
        self.T_values = np.load(Tfilename, allow_pickle=True)


class Frame(rgb.FileManager):
    """Class method of saving frame information and detecting cells. Sufficently different method from the RGB 2D method to create a new class and not inhert from rgbcelltracker """

    def __init__(self, file, path, img, f_no, maskspheriod, autoTH=True, thred=10, thgreen=30):
        super().__init__(file, path)
        self.img = img
        self.frameno = f_no
        self.kernelA = np.ones((3, 3), np.uint8)
        self.kernelB = np.ones((6, 6), np.uint8)
        self.thred = thred
        self.thgreen = thgreen
        # introduce a colour filter here to split the colours and make single cell detection easier
        self.boundaries = [
            ([0, 2, 2], [20, 255, 255]),
            ([21, 25, 35], [180, 255, 255])
        ]
        self.maskspheriod = maskspheriod
        self.autoTH = autoTH
        self.cells = []
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]

    def analyse(self):
        # A method to detect cells using colour boundaries to help separate red and green.
        self.RGB = self.img.copy()
        self.RGB2 = self.img.copy()
        self.HSV = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

        # Create a mask for the spheriod (use the largest contour)
        grayframe = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        blurredframe = cv2.GaussianBlur(grayframe, (15, 15), 8)
        ret, threshframe = cv2.threshold(blurredframe, 37, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(
            threshframe, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        check = False
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area > 25000:
                indexforarea = i
                check = True

        if check and self.maskspheriod:
            mask_frame = np.zeros(grayframe.shape, np.uint8)
            cv2.drawContours(mask_frame, [contours[indexforarea]], 0, 255, -1)
            mask_frame = cv2.dilate(mask_frame, self.kernelA, iterations=4)

        else:
            mask_frame = np.zeros(grayframe.shape, np.uint8)
        mask_frame = cv2.bitwise_not(mask_frame)

        for n in range(2):
            lower = np.array(self.boundaries[n][0], dtype="uint8")
            upper = np.array(self.boundaries[n][1], dtype="uint8")
            maskA = cv2.inRange(self.HSV, lower, upper)
            output = cv2.bitwise_and(self.img, self.img, mask=maskA)
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            smoothgray = cv2.blur(gray, (301, 301))
            sub_gray = cv2.subtract(gray, smoothgray)
            fgray = cv2.bilateralFilter(sub_gray, 21, 20, 20)
            fgray = cv2.GaussianBlur(fgray, (11, 11), 5)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            fgray1 = cv2.morphologyEx(fgray, cv2.MORPH_OPEN, kernel)

            # Does an auto threshold on each frame -crude, but works.
            if self.autoTH == True:

                histogram, bin_edges = np.histogram(
                    fgray, bins=256, range=(0, 255))
                gradient = np.gradient(histogram)

                indy = len(gradient)

                val = 0.0
                # Potentially adjust the value to higher to capture more (but risk noise)
                while val > -120.0 and indy > 0:
                    val = gradient[indy-1]
                    indy -= 1
                factor = indy

            else:
                scales = [self.thred, self.thgreen]
                factor = scales[n]
            
            print('Threshold is', factor-1)
            ret, th_a = cv2.threshold(fgray1, factor-1, 255, cv2.THRESH_BINARY)
            
            if check:
                th_a = cv2.bitwise_and(th_a, th_a, mask=mask_frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            openingimg = cv2.morphologyEx(th_a, cv2.MORPH_OPEN, kernel)
            contours, hierarchy = cv2.findContours(
                openingimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 5 and area < 1200 and len(cnt) > 4:
                    perimeter = cv2.arcLength(cnt, True)
                    ellipse = cv2.fitEllipse(cnt)
                    testratio = (4*pi*area/perimeter**2) * \
                        (ellipse[1][0]/ellipse[1][1])
                    testratio = 1
                    if testratio >= 0.55:

                        
                        (x, y), radius = cv2.minEnclosingCircle(cnt)
                        mask = np.zeros(gray.shape, np.uint8)
                        cv2.drawContours(mask, [cnt], 0, 255, -1)
                        mean_val = cv2.mean(self.img, mask=mask)
                        mean_rgb = (mean_val[2]/255,
                                    mean_val[1]/255, mean_val[0]/255)
                        HSV = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
                        mean_HSV = cv2.mean(HSV, mask=mask)

                        x5, y5, w5, h5 = cv2.boundingRect(cnt)
                        cv2.rectangle(self.RGB, (x5, y5),
                                      (x5+w5, y5+h5), (255, 255, 255), 2)

                        # font
                        font = cv2.FONT_HERSHEY_SIMPLEX

                        # org
                        org = (x5+w5, y5+h5)

                        # fontScale
                        fontScale = 0.5

                        #  BGR
                        color = (255, 0, 0)

                        # Line thickness of 2 px
                        thickness = 2
                        offset = int(w5/2)
                        if x5-offset < 0:
                            sx = 0
                        else:
                            sx = x5-offset
                        if y5-offset < 0:
                            sy = 0
                        else:
                            sy = y5-offset
                        if x5+w5+offset > self.width:
                            ex = self.width
                        else:
                            ex = x5+w5+offset
                        if y5+h5+offset > self.height:
                            ey = self.height
                        else:
                            ey = y5+h5+offset

                        cutoutimg = self.img[sy:ey, sx:ex]
                        cutoutfile = 'img' + \
                            str(self.frameno)+'_'+str(len(self.cells))+'.jpg'
                        cutoutpath = self.filename = os.path.join(
                            self.imagepath, cutoutfile)
                        cv2.imwrite(cutoutpath, cutoutimg)
                        ellipse = cv2.fitEllipse(cnt)
                        cell_angle = ellipse[2]
                        aspectratio = ellipse[1][1]/ellipse[1][0]
                        if aspectratio < 1:
                            aspectratio = 1/aspectratio
                        cv2.putText(self.RGB, str(len(self.cells)), org, font,
                                    fontScale, color, thickness, cv2.LINE_AA, False)
                        self.cells.append(Cell(x, y, mean_HSV[0], area, mean_rgb, self.frameno, [
                                          self.frameno, len(self.cells)], cell_angle, aspectratio))
                    elif testratio > 0.1:
                        xbox, ybox, w1, h1 = cv2.boundingRect(cnt)
                        cropimage = th_a[ybox:ybox+h1, xbox:xbox+w1]
                        fgray1test = fgray1[ybox:ybox+h1, xbox:xbox+w1]


                        maxf = np.max(fgray1test)
                        fact = 0.69*maxf
                        ret, th = cv2.threshold(
                            fgray1test, fact, 255, cv2.THRESH_BINARY)
                        contours1, hierarchy = cv2.findContours(
                            th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        for cnt2 in contours1:
                            if len(cnt2) > 4:
                                area1 = cv2.contourArea(cnt2)
                                perimeter1 = cv2.arcLength(cnt2, True)
                                ellipse1 = cv2.fitEllipse(cnt2)
                                testratio1 = (4*pi*area1/perimeter1**2) * \
                                    (ellipse1[1][0]/ellipse1[1][1])
                                x3, y3, w3, h3 = cv2.boundingRect(cnt2)

                                if testratio1 >= 0.14:

                                    (x1, y1), radius = cv2.minEnclosingCircle(cnt2)

                                    mask1 = np.zeros(cropimage.shape, np.uint8)
                                    cv2.drawContours(mask1, [cnt2], 0, 255, -1)
                                    mask1 = cv2.dilate(mask1, kernel)
                                    mean_val = cv2.mean(
                                        self.img[ybox:ybox+h1, xbox:xbox+w1], mask=mask1)
                                    mean_rgb = (
                                        mean_val[2]/255, mean_val[1]/255, mean_val[0]/255)
                                    HSV1 = cv2.cvtColor(
                                        self.img[ybox:ybox+h1, xbox:xbox+w1], cv2.COLOR_BGR2HSV)
                                    mean_HSV = cv2.mean(HSV1, mask=mask1)
                                    offset = int(w3/2)
                                    if x3+xbox-offset < 0:
                                        sx = 0
                                    else:
                                        sx = x3+xbox-offset
                                    if y3-offset+ybox < 0:
                                        sy = 0
                                    else:
                                        sy = y3+ybox-offset
                                    if x3+w3+offset+xbox > self.width:
                                        ex = self.width
                                    else:
                                        ex = x3+w3+offset+xbox
                                    if y3+h3+offset+ybox > self.height:
                                        ey = self.height
                                    else:
                                        ey = y3+h3+offset+ybox

                                    cutoutimg = self.img[sy:ey, sx:ex]
                                    cutoutfile = 'img' + \
                                        str(self.frameno)+'_' + \
                                        str(len(self.cells))+'.jpg'
                                    cutoutpath = self.filename = os.path.join(
                                        self.imagepath, cutoutfile)
                                    cv2.imwrite(cutoutpath, cutoutimg)
                                    ellipse = cv2.fitEllipse(cnt2)
                                    cell_angle = ellipse[2]
                                    aspectratio = ellipse[1][1]/ellipse[1][0]
                                    if aspectratio < 1:
                                        aspectratio = 1/aspectratio
                                    self.cells.append(Cell(x1+xbox, y1+ybox, mean_HSV[0], area, mean_rgb, self.frameno, [
                                                      self.frameno, len(self.cells)], cell_angle, aspectratio))
                                    cv2.rectangle(
                                        self.RGB, ((x3+xbox), (y3+ybox)), ((x3+w3+xbox), (y3+ybox+h3)), (255, 255, 255), 2)

                                    font = cv2.FONT_HERSHEY_SIMPLEX

                                    # org
                                    org = (x3+xbox+w3, y3+ybox+h3)

                                    # fontScale
                                    fontScale = 0.5

                                    #  BGR
                                    color = (255, 0, 0)

                                    # Line thickness of 2 px
                                    thickness = 2
                                    cv2.putText(self.RGB, str(
                                        len(self.cells)-1), org, font, fontScale, color, thickness, cv2.LINE_AA, False)

        cv2.imshow('colour'+str(self.frameno), self.RGB)
        cv2.waitKey(1)
        
        self.frameno += 1


class Cell(rgb.Cell):
    def __init__(self, xpos, ypos, hue, area, rgb, frame_no, cell_number, cell_angle, aspectratio):
        super().__init__(xpos, ypos, hue, area, rgb,
                         frame_no, cell_number, cell_angle, aspectratio)


class Process_cells(rgb.Process_cells, OpenVideo):
    def __init__(self, file, path, scale, tpf, maxdistance, addnumbers,search_angle,dpi,autotrack, w1, w2):
        super().__init__(file, path, scale, tpf, maxdistance, addnumbers,search_angle,dpi,autotrack, w1, w2)
        

class ThreeDvideoAnalysis(Process_cells):
    def __init__(self, file, path, scale, tpf, maxdistance, addnumbers,search_angle,dpi,autotrack,w1,w2):
        super().__init__(file, path, scale, tpf, maxdistance, addnumbers, search_angle,dpi,autotrack, w1, w2)

    def start(self, anglestep, windowsize, allowedcentreshift, maxrounds, tolerance):
        
        self.transrotate(anglestep, windowsize, allowedcentreshift, maxrounds, tolerance) # adjusts the position of the frames to centre and orientate the assembloid
        self.run()
        self.calculate_traj()
        self.analyse_traj()
        self.vector_analysis()
        print('Complete and csv files saved')
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def calculate_traj(self):
        self.create_data_lists()
        self.pairup()
        self.listup()
        self.cellpathways()
        self.createdatapack()
        self.plotalltracks()
        self.applyscales()

    def analyse_traj(self):
        self.MSDanalysis()
        self.distance_histogram()
        self.plot_area_covered()
        self.plot_hull_area_covered()
        self.write_csv()

    def vector_analysis(self):
        self.vectorplot(colour=False, subset='none')
        self.velocityvectorplot(colour=False, subset='none')
        self.vectorplot(colour=True, subset='red')
        self.velocityvectorplot(colour=True, subset='red')
        self.vectorplot(colour=True, subset='green')
        self.velocityvectorplot(colour=True, subset='green')
        self.vectorplot(colour=True, subset='none')
        self.velocityvectorplot(colour=True, subset='none')


if __name__ == '__main__':
    # these are the filenames of the movie files - you need a pair (overlay and RGB using the same name just replace overlay with RGB)
    files=['C3_Overlay_80_120.wmv']
    
    
    for file in files:
        # path = '/Users/phykc/Documents/Work/organiod'
        path='/Users/phykc/Library/CloudStorage/OneDrive-UniversityofLeeds/organiod/DyPheT/rdgtracker/data'
        #  scale is in units of µm per pixel
        # tpf is time per frame 3600s is 60 minutes
        # Maxdistance limits the distance in pixels that a cell in frame n can be associated to a cell in frame n+1.
        #  Search angle (in degrees limits the +/-)
        # Autotrack = True automates the tracking of rotation and translation.  False give manual control.
        # Most of the time angle shifts are a few degrees but in one case there is ~90 degree shift between frames hence below.
        # This means that if the video only shows small angle shifts you can reduce the search angle (delta angle) to something like 15 or 20.
        rgb_analysis = ThreeDvideoAnalysis(file, path, scale=1.0, tpf=3600, maxdistance=100, addnumbers=True, search_angle=45, autotrack=True, dpi=1200, w1=1,w2=2)
        rgb_analysis.start(anglestep=30, windowsize=500, allowedcentreshift=100, maxrounds=25, tolerance=0.6)
