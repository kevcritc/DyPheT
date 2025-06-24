
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
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar


class OpenVideo(rgb.OpenVideo):
    """Opens the video given of the path. Contains methods for runing the video, processing the frames, and collecting cell information"""

    def __init__(self, file, path, search_angle, keypoints, autotrack=True):
        super().__init__(file, path, search_angle,keypoints, autotrack)
        
        if 'RGB' in file:
            file1 = file.replace('RGB', 'Overlay')
        else:
            file1=file
        filename = os.path.join(path, file1)
        self.filename = filename
        self.file1=file1[:-4]
        print(filename)
        self.autotrack=autotrack
        self.keypoints=keypoints
        self.checkpreviousfit_x_frame=50
        self.previouswrapsize=540
        self.search_angle=search_angle*10
        
        self.flags=cv2.INTER_LANCZOS4  | cv2.WARP_FILL_OUTLIERS | cv2.WARP_POLAR_LINEAR
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
        self.rotation_lock = 0
        self.Rprevious=0
        self.Rdiff=2
        self.rotatedframes=[]
        self.ref_frames=[]
        self.ref_contour=[]
        self.ref_warp=[]
        self.warp_radius=700
        
        print(f'width {self.width}, height {self.height}')
        
    
    def transrotate(self, anglestep, windowsize, allowedcentreshift,maxrounds, tolerance,panel_thresh):
        ''' A function to determine the translation and rotation of the assembliod'''
        self.angle_rot = np.zeros(self.maxframes, dtype=float)
        self.anglestep=anglestep
        self.windowsize=windowsize
        self.maxrounds=maxrounds
        self.tolerance=tolerance
        self.linear_conf_thresh =panel_thresh
        list_of_min_confidence=[]
        # Loop through the frames of the movie
        #log the process

        flogname=self.datapath('logger_'+str(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"))+'.csv')
        
        with open(flogname,'a+') as self.csvfile1:
            self.writer=csv.writer(self.csvfile1)
            self.writer.writerow(['Frame Number','iteration','Angle applide','Angle measured','Spheriod angle','Angle error', 'cx', 'cy', 'dx', 'dy', 'derr'])
            # Move through the frames one by one
            while self.frameno < self.maxframes:
                print(f'Frame {self.frameno+1} of {self.maxframes}')
                
                # Read each frame ret is True if there is a frame
                ret, self.frame = self.cap.read()
                
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
                    fitconfidencelist=self.runautotrack()
                    print(f'Mean fit confidence is {np.mean(fitconfidencelist):.2f} + std {np.std(fitconfidencelist):.2f}')
                    list_of_min_confidence.append(100*(np.mean(fitconfidencelist)-(2.5*np.std(fitconfidencelist))))
                    if len(list_of_min_confidence)>5:
                        list_of_min_confidence.pop(0)
                    
                    self.linear_conf_thresh=np.clip(np.mean(list_of_min_confidence), 35, 90)
                    print(f'Adjusting linear threshold to {self.linear_conf_thresh:.2f}')
                    
                    
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
                
                # Create a single affine matrix to rotate around self.center and translate
                M_zero = cv2.getRotationMatrix2D(self.center, 0, 1.0)
                M_zero[0, 2] += tx
                M_zero[1, 2] += ty
                
                self.M_values.append(M.copy())
                self.T_values.append(self.T)
                self.R_values.append(self.R)
                
                # Apply the transformation to the image
                self.rotated = cv2.warpAffine(self.frame.copy(), M, (self.width, self.height))
                
                
                
                # Apply the same transformation to the contour template and 
                self.contourtemplate = np.zeros(self.morath.shape, dtype=np.uint8)
                cv2.drawContours(self.contourtemplate, [self.hull], 0, 255, -1)
                self.contourtemp = cv2.warpAffine(self.contourtemplate, M, (self.width, self.height))
                self.rotatedframes.append(self.rotated.copy())
                self.ref_frames.append(self.frame.copy())
                self.ref_contour.append(self.contourtemp.copy())
                # Compare previous rotated-translated to the new frame.
               
                self.ref_warp.append(cv2.warpPolar(self.rotated.copy(), (self.warp_radius, 3600),self.center, self.width//2, self.flags))
                
                # Store only the last five frame currently only typically using -1 and -2, but could use more.
                if len(self.rotatedframes) > 3:
                    # self.rotatedframes.pop(1)
                    self.ref_frames.pop(1)
                    self.ref_contour.pop(1)
                    self.ref_warp.pop(1)

                # store the x, y and r shifts for anaylysis
                self.rotation.append(self.R_values[-1])
                self.x_shift.append(self.T[0, 2])
                self.y_shift.append(self.T[1, 2])
                self.frameshift.append(self.frameno)
                self.pcx,self.pcy=self.cx, self.cy
                # move to the next frame
                if len(self.rotatedframes)>1:
                    dst = cv2.addWeighted(self.rotatedframes[-1],0.5, self.rotatedframes[0], 0.5, 0)
                    resized = cv2.resize(dst, None, fx=0.25, fy=0.25)
                    cv2.imshow('First and last frame after rotation', resized)
                cv2.destroyWindow(f'Overlayed match update frame {self.frameno+1}')
                cv2.waitKey(1)
                

                self.frameno += 1
            
            # Release the video
            self.cap.release()
            
            stacked = np.vstack(self.rotatedframes)
            cv2.imshow('Stacked Rotated Frames', stacked)
            full_path = self.datapath(self.filename[:-4]+'stacked_rotated_frames.png')
            cv2.imwrite(full_path, stacked)
            stack = np.stack(self.rotatedframes, axis=3)
            max_proj = np.max(stack, axis=3).astype(np.uint8)
            cv2.imshow('Max Intensity Projection', max_proj)
            
            full_path = self.datapath(self.filename[:-4]+'max_projection.png')
            cv2.imwrite(full_path, max_proj)
            average_image = np.mean(stack, axis=3).astype(np.uint8)

            cv2.imshow('Average Frame', average_image)
            full_path = self.datapath(self.filename[:-4]+'average_frame.png')
            cv2.imwrite(full_path, average_image)
            
            
            cv2.waitKey(10)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            
    
           
            Rfilename1 = self.datapath(self.filename[:-4]+'Rvals.npy')
            Tfilename1 = self.datapath(self.filename[:-4]+'Tvals.npy')
            indexnumber=0
            # Store movement data
            if self.autotrack:
                while True:
                    RTfilename = self.datapath(f'RT_data_auto{indexnumber:03d}.xlsx')
                    if not os.path.isfile(RTfilename):
                        break
                    indexnumber += 1
            else:
                while True:
                    RTfilename = self.datapath(f'RT_data_man{indexnumber:03d}.xlsx')
                    if not os.path.isfile(RTfilename):
                        break
                    indexnumber += 1

        self.csvfile1.close()
        np.save(Rfilename1, self.R_values, allow_pickle=True)
        np.save(Tfilename1, self.T_values, allow_pickle=True)
        # save the shift data
        dRT = {'R': self.rotation, 'x': self.x_shift, 'y': self.y_shift}
        
        RTdataframe = pd.DataFrame(dRT)
        RTdataframe.to_excel(RTfilename)
        
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
        return self.cx, self.cy


    
    def runautotrack(self):
        #Initialise parameters
        
        self.lockin_counter=0
        best_derr = float('inf')
        rounds=0
        dx,dy=1,1
        self.angleclose=False
        if not hasattr(self, 'rotation_conf_all'):
            self.rotation_conf_all = []
        if not hasattr(self, 'confidence_history'):
            self.confidence_history = []
        
        # capture the assembliod without too much background
        cutsize=self.findcutsize()
        self.angle_test_list=[]
        self.tracking_run=[]
        if self.frameno == 1:
            self.calibration_derr = []
            self.calibration_anstde = []
        self.angleclose=False
        self.rotation_conf_thresh = 98.5
        self.rotation_lock=0
        self.stuckcounter=0
        self.restcount=0
        self.close_angle=self.R
        tolerance=self.tolerance
        fit_confidence_list=[]
        method=cv2.TM_CCORR_NORMED
        mask_meth=True
        good_match = True
        anglechange=10
        target_cal=10
        derr_tolorence=3.0
        derr=50
        best_conf=0
        
            
        if not hasattr(self, "derr_history"):
            self.derr_history = []
        
        if not hasattr(self, "skipR_counter"):
            self.skipR_counter = 0
        
        if not hasattr(self, "skipR_cooldown"):
            self.skipR_cooldown = 0
        
        if not hasattr(self, "baseline_keypoints"):
            self.baseline_keypoints = self.keypoints
            
        self.skipR=False
        self.divergent_count=0
        search_angle=self.search_angle
        while  rounds<self.maxrounds:
            
            
            
            if rounds<5:
                tolerance=0.0
            else:
                tolerance=self.tolerance
            tx = self.center[0] - self.cx
            ty = self.center[1] - self.cy
            M= cv2.getRotationMatrix2D(self.center, self.R, 1.0)
            M[0, 2] += tx
            M[1, 2] += ty

            # Load current frame (n)
            new_frame = cv2.warpAffine(self.frame.copy(), M, (self.width, self.height))
            # Load previous frames
            old_frame_1 = self.rotatedframes[-1].copy()
            old_warp_1 = self.ref_warp[-1].copy()
            old_contour_1=self.ref_contour[-1].copy()
            
            
            old_frame_2, old_warp_2 = None, None
            if len(self.rotatedframes) >= 2 and len(self.ref_warp) >= 2:
                old_frame_2 = self.rotatedframes[-2].copy()
                old_warp_2 = self.ref_warp[-2].copy()
                old_contour_2=self.ref_contour[-2].copy()
                
            
            if len(self.rotatedframes) < 3:
                dx, dy, derr, fitconfidence, conarray = self.estimate_T(rounds, cutsize, new_frame,old_frame_1, n_number=0)
            elif rounds%5==0 and rounds >5:
                dx, dy, derr, fitconfidence, conarray =self.estimate_T_dual(rounds, cutsize, new_frame,old_frame_1,old_frame_2)
            else: 
                dx, dy, derr, fitconfidence, conarray = self.estimate_T(rounds, cutsize, new_frame,old_frame_1,n_number=0)

            fit_confidence_list= fit_confidence_list+list(conarray)
            
            alpha1 = self.derr_angle_to_alpha(derr,anglechange)
            
            # Apply damped to translation
            
            self.cx+=dx*alpha1
            self.cy+=dy*alpha1
            tx = self.center[0] - self.cx
            ty = self.center[1] - self.cy
            M= cv2.getRotationMatrix2D(self.center, 0, 1.0) #change to 0
            M[0, 2] += tx
            M[1, 2] += ty

            # Test the combined transformation
            trans_frame=cv2.warpAffine(self.frame.copy(), M, (self.width, self.height))
          
            # Lock rotation if linear alignment is very strong
            if self.angleclose and self.lockin_counter==0:
                self.rotation_lock = 4
                print(f'Locking rotation for {self.rotation_lock} rounds due to strong alignment.')
                
            # Handle rotation lock countdown
            if self.rotation_lock > 0:
                self.rotation_lock -= 1
                print(f'Rotation locked: {self.rotation_lock} rounds remaining')
                anstde = 10
                an = self.R
                ag = self.R
                self.Rdiff = 1
                conf_list = []
                good_match = True
                self.angleclose=False
            
            elif self.skipR:
                print('Skip R calculation')
                conf_list=[]
            
            elif len(self.rotatedframes) >= 2 and rounds%5==0 and rounds >50:
                # Estimate rotation from both n-1 and n-2 -currently not activated
                print('Check against last frame and one before.')
                anstde_1, an_1, ag_1, conf_list_1, good_1 = self.estimate_R(rounds, trans_frame, old_warp_1,old_contour_1,good_match, method, mask_meth)
                anstde_2, an_2, ag_2, conf_list_2, good_2 = self.estimate_R(rounds, trans_frame, old_warp_2,old_contour_2,good_match, method, mask_meth)
            
               
                
                an=self.average_angles_deg([an_1,an_2], [1,1])
                an = (an_1  + an_2 ) /2
                anstde = (anstde_1 + anstde_2) / 2
            
                # Best guess angle (used as fallback)
                ag = ag_1
            
                # Merge confidence lists
                conf_list = conf_list_1 + conf_list_2
                good_match = good_1 or good_2
            
            else:
                # Fallback to just n-1 if n-2 is unavailable
                anstde, an, ag, conf_list, good_match = self.estimate_R(rounds, trans_frame, old_warp_1,old_contour_1, good_match, method, mask_meth)
            
        

                
            if len(conf_list) > 0:
                avg_conf = np.mean(fit_confidence_list)
                print(f'Rotation mean confidence {avg_conf:.2f}')
                self.confidence_history.append(avg_conf)
                self.confidence_history=[x for x in self.confidence_history if x<1]
            
                # Keep only last 5 frames
                if len(self.confidence_history) > 5:
                    self.confidence_history.pop(0)
            
                
                
            anglechange=self.angle_diff_deg(an, self.R)
            print(f'Angle change between interation = {anglechange:.1f}')
            
            # Creating some logic to prevent being stuck between two positions of just multiple fails to fit.
            #  There are limits to this, e.g. lower time points between frames is a big advantage. Low rotation helps.
           
        
            
            # Skip logic with a cooldown.
            
            if rounds<1:
                self.derr_history=[]
                self.angle_history=[]
                self.position_history=[]
            
            self.derr_history.append(derr)
            self.angle_history.append(self.R)
            self.position_history.append([self.cx, self.cy])
            
            good_recent = [d < 2.5 for d in self.derr_history[-3:]]
            
            if rounds>7:
                alpha=1.0
                if self.skipR_cooldown > 0 and derr < 5:
                    self.skipR_cooldown -= 1
                    self.skipR = True
                    print(f"Skipping rotation (cooldown: {self.skipR_cooldown})")
                    
                else:
                    
                    # Only try to enable skipping based on derr history if not in cooldown
                    if derr > 5:
                        self.skipR = False
                        self.skipR_counter = 0
                        self.keypoints = int(self.baseline_keypoints * 3)
                    
                    elif sum(good_recent) >= 2:
                        print('Initiating skip for next 3 rounds')
                        self.skipR = True
                        self.skipR_cooldown = 5
                        self.keypoints = int(self.baseline_keypoints * 0.9)
                        self.keypoints = max(10, min(self.keypoints, 150))
            
                        
                
                    else:
                        self.skipR = False
                        self.keypoints = int(self.keypoints * 0.8 + self.baseline_keypoints * 0.2)
                        self.keypoints = max(10, min(self.keypoints, 150))
            
                
            else:
                alpha=1.0
            
                
                # Clamp keypoints to valid range
            self.keypoints = max(10, min(self.keypoints, 100))
            
            if self.skipR:
                print('Skipping rotation')
            
            else:
                if anglechange<search_angle:
                    self.R=an
                    self.R = self.normalise_angle_deg(self.R)
            
            if anstde>4 and rounds>=self.maxrounds:
                self.manadjustRT()

            
            if rounds> 8:
                
                if self.is_diverging():
                    self.lockin_counter=0
                    
                    print('Attempt to correcting non-divergent behavior')
                    self.R=self.last_good_R
                    self.skipR=True
                    self.skipR_cooldown = max(self.skipR_cooldown, 5)
                    self.keypoints = 100
                    self.keypoints = max(10, min(self.keypoints, 150))
                    
                    self.divergent_count += 1
                else:
                    self.divergent_count = max(0, self.divergent_count - 1)        
                               
            if self.divergent_count > 2 or self.is_stuck():
                self.stuckcounter+=1
                if self.stuckcounter>1:
                    best_derr, self.R =self.bruteforce(rounds, cutsize, old_frame_1,self.search_angle/10, self.R,60, filterspikes=True)
                    best_derr, self.R =self.bruteforce(rounds, cutsize, old_frame_1,10, self.R, 20, filterspikes=False)
                    self.skipR=True
                    self.angle_history = [self.R]
                    self.derr_history = [best_derr]
                    #self.position_history = [[self.cx, self.cy]]
                    self.confidence_history = []
                    self.rotation_lock =5
                    self.fail_counter = 0
                    self.divergent_count = 0
                    self.stuckcounter = 0
                    self.skipR_cooldown = 5
                    self.fail_counter=0
                    search_angle=10
                    
                else:
                    print("Locking rotation at last stable R")
                    self.R = self.last_good_R
                    self.rotation_lock = 5
                    self.skipR = True
                    self.keypoints = int(self.baseline_keypoints * 0.7)
                    self.keypoints = max(10, min(self.keypoints, 150))
                    self.divergent_count = 0
            
            

            self.angle_test_list.append([self.R,anstde])
            if dx>150 or dy>150 or derr>100:
                print(f'might be a tranlation problem the error the {derr:.1f}')
            
            # Failure tracking
            if not hasattr(self, "fail_counter"):
                self.fail_counter = 0
            
            # Check for bad match (e.g., high error or big angle jump)
            if derr > 6 or abs(self.Rdiff) > 3 and rounds>5:
                self.fail_counter += 1
            else:
                self.fail_counter = max(0, self.fail_counter - 1)  # decay over time
            
            # Save best good state
            if derr < best_derr and abs(anglechange) < 3 and anstde < 3:
                best_derr = derr
                self.last_good_cx = self.cx
                self.last_good_cy = self.cy
                self.last_good_R = self.R
            
            # Reset if too many failures
            if self.fail_counter >= 8 and derr > 5:
                print("Too many failures — resetting to last known good alignment")
                if self.restcount<1:
                    self.cx = self.last_good_cx
                    self.cy = self.last_good_cy
                    self.R = self.last_good_R
                    self.restcount+=1
                else:
                    
                    best_derr, self.R =self.bruteforce(rounds, cutsize, old_frame_1,self.search_angle/10, self.R, 60, filterspikes=True)
                    best_derr, self.R =self.bruteforce(rounds, cutsize, old_frame_1,10, self.R, 20, filterspikes=False)
                    self.restcount+=1
                    self.skipR=True
                    self.angle_history = [self.R]
                    self.derr_history = [best_derr]
                    #self.position_history = [[self.cx, self.cy]]
                    self.confidence_history = []
                    self.fail_counter = 0
                    self.divergent_count = 0
                    self.stuckcounter = 0
                    self.skipR_cooldown = 6
                    self.rotation_lock =5
                    self.keypoints = int(self.baseline_keypoints * 2)
                    self.keypoints = max(10, min(self.keypoints, 150))
                
            angle=0
            
            self.Rdiff=self.R-self.Rprevious
            self.Rprevious = self.R
            
            self.tracking_run.append([self.frameno,rounds,self.R,an,angle,anstde, self.cx, self.cy, dx, dy, derr])
           
            print(f'file: {self.file1} ,frame {self.frameno+1}, iteration={rounds+1} , dx = {dx:.1f}, dy = {dy:.1f}, Angle = {self.R:.1f}, derr ={derr:.1f}, alpha1={alpha1:.2f} and alpha={alpha:.2f}')
            
            target_cal=np.sqrt(dx**2+dy**2+(self.Rdiff)**2)
            
            if target_cal < tolerance and derr < derr_tolorence:
                print("Converged: alignment and match quality both acceptable")
                break
            # interate until target is meet of maxrounds is reached.
            rounds+=1
            if rounds == 25 and self.cooldown==0:
                print(" Forcing brute force due to slow convergence")
                best_derr, self.R =self.bruteforce(rounds, cutsize, old_frame_1,self.search_angle/10, self.R, 60, filterspikes=True)
                best_derr, self.R =self.bruteforce(rounds, cutsize, old_frame_1,10, self.R,20, filterspikes=False)
                self.skipR=True
                self.angle_history = [self.R]
                self.derr_history = [best_derr]
                self.position_history = [[self.cx, self.cy]]
                self.confidence_history = []
                self.fail_counter = 0
                self.divergent_count = 0
                self.stuckcounter = 0
                self.skipR_cooldown = 6
                self.rotation_lock =5
                self.keypoints = 100
        if rounds >= self.maxrounds and best_derr < derr:
            print(f"Max rounds reached with poor match (derr={derr:.2f}) — reverting to best known state (derr={best_derr:.2f})")
            self.cx = self.last_good_cx
            self.cy = self.last_good_cy
            self.R = self.last_good_R
            
            self.keypoints = int(self.baseline_keypoints * 2)

        
        return fit_confidence_list
    
    def smooth_curve(self,y, window_size=5):
        window = np.ones(window_size) / window_size
        return np.convolve(y, window, mode='same')
    
    def bruteforce(self,rounds, cutsize, old_frame_1, search_angle, centre, no_points, filterspikes):
        print('Searching for the correct angle by brute force')
        derr_test=[]
        # Unwrap angle history in radians to avoid crossing discontinuity
        center = self.normalise_angle_deg(centre)
        delta = search_angle
        
        startR = center - delta
        endR   = center + delta
        
        # Handle wraparound cleanly
        if endR > 180:
            angletestrange = np.linspace(startR, endR - 360, no_points)
        elif startR < -180:
            angletestrange = np.linspace(startR + 360, endR, no_points)
        else:
            angletestrange = np.linspace(startR, endR, no_points)
        
        
        
        angletestrange = np.array([self.normalise_angle_deg(a) for a in angletestrange])
        
        
        for R in angletestrange:
            
            M= cv2.getRotationMatrix2D(self.center, R, 1.0)
            new_frame_test = cv2.warpAffine(self.frame.copy(), M, (self.width, self.height))
            dx, dy, derr, fitconfidence, conarray = self.estimate_T(rounds, cutsize, new_frame_test,old_frame_1, n_number=0)
            
            derr_test.append(derr)
        if filterspikes:
            derr_test=self.smooth_curve(derr_test)
            # Remove first and last 2 points due to smoothing artifacts
            derr_test=derr_test[2:-2]
            angletestrange=angletestrange[2:-2]
        interp_func = interp1d(angletestrange, derr_test, kind='quadratic', fill_value="extrapolate")
        min_angle = min(angletestrange)
        max_angle = max(angletestrange)
        angle_fine = np.linspace(min_angle, max_angle, no_points*5)
        derr_interp = interp_func(angle_fine)
        plt.plot(angletestrange,derr_test,'o')
        plt.plot( angle_fine,derr_interp)
        plt.xlabel('Angle')
        plt.ylabel('d-error')
        plt.show()
        res = minimize_scalar(interp_func, bounds=(min_angle, max_angle), method='bounded')

        if res.success:
            best_angle = res.x
            best_derr = res.fun
            print(f"Interpolated best angle: {best_angle:.2f}°, derr: {best_derr:.2f}")
            # Apply sanity check before accepting
            if abs(best_angle - self.R) < search_angle:
                new_R = self.normalise_angle_deg(best_angle)
                
                self.last_good_R = new_R
                self.angle_history.append(new_R)
                self.skipR = True
                self.skipR_cooldown = 5
                
            else:
                print("Interpolated angle jump too large — ignoring")
                new_R=centre
        else:
            print("Interpolation failed — fallback to direct min")
            best_angle = angletestrange[np.argmin(derr_test)]
            new_R = self.normalise_angle_deg(best_angle)
        return best_derr, new_R
    
    def is_stuck(self):
        
        if len(self.derr_history) < 5:
            return False
        recent_derr = self.derr_history[-5:]
        if np.std(recent_derr) < 1 and np.mean(recent_derr) > 2.5:
            print("Stuck")
            return True
        return False
    
    def angle_diff_deg(self,a, b):
        """Return the smallest signed difference between two angles (in degrees)."""
        d = a - b
        return (d + 180) % 360 - 180
    
    def normalise_angle_deg(self,angle):
        return (angle + 180) % 360 - 180

       
    def estimate_T_dual(self, rounds, cutsize, new_frame, old_frame_1, old_frame_2):
        # Get translation relative to n-1
        dx1, dy1, err1, conf1, conarray1 = self.estimate_T(rounds, cutsize, new_frame, old_frame_1, n_number=0)
        
        # Get translation relative to n-2
        dx2, dy2, err2, conf2, conarray2 = self.estimate_T(rounds, cutsize, new_frame, old_frame_2, n_number=-1)
        
        # Confidence-weighted average
        w1 = conf1 / (conf1 + conf2 + 1e-6)
        w2 = conf2 / (conf1 + conf2 + 1e-6)
    
        dx_combined = w1 * dx1 + w2 * dx2
        dy_combined = w1 * dy1 + w2 * dy2
        err_combined = (w1 * err1 + w2 * err2)
        conf_combined = (conf1 + conf2) / 2
    
        return dx_combined, dy_combined, err_combined, conf_combined, conarray1+conarray2

    def findcutsize(self):
        # Use the hull to determine the radius
        circle=cv2.minEnclosingCircle(self.hull)
        R=circle[1]
        cutsize=int((self.width//2)-(0.9*(R)))
        print(f'Frame Cut size is {cutsize} and Radius {int(R)}')
        if cutsize<0:
            cutsize=0
        return cutsize
    
    def estimate_T(self,rounds, cutsize, new_frame,old_frame, n_number):
        
       
        window_shift=650
        
        rot_draw=new_frame.copy() #changed from rotated
        confidenceth=self.linear_conf_thresh
        notlongenough=True
        panels=2
        stdxy=100
        solvefailcount=0
        outsideonly=True
        dxlonglist=[]
        dylonglist=[]
        delconlonglist=[]
        panth_level=7
        while notlongenough:
            print(f'Matching {panels} x {panels} panels')
            rot_draw, delxlist,delylist ,delconarray ,outcoord=self.linear_shift(cutsize, rot_draw, confidenceth,outsideonly,panels)
            
            if len(delxlist)>0:
                dxlonglist=dxlonglist+delxlist
                dylonglist=dylonglist+delylist
                delconlonglist=delconlonglist+delconarray

            if len(dxlonglist)>=5:
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
              
            if np.sqrt(xstd**2+ystd**2)<2.5 and len(delxlist)>6:
                print(f'Locked-on with a translation sd of {np.sqrt(xstd**2+ystd**2):2f}')
                self.angleclose = True
                self.close_angle = self.R

            else:
                self.angleclose = False
            
           
            cv2.circle(rot_draw,(int(self.width//2+delatxtm),int(self.height//2+delatytm)),20,(0,255,0),2)
            # cv2.circle(rot_draw,(int(self.width//2+delatxtm),int(self.height//2+delatytm)),int(self.width//2-cutsize),(0,255,0),2)
            if len(delconarray)>0:
                minconfindence=100*np.min(delconarray)
                print(f'Min condidence in panel match :{ minconfindence:.3f}')
            else:
                print('no confidence in fit')
                minconfindence=0.0
                dx1,dy1,stdxy,minconfindence,delconarray=10,0,10,[1],[1]
            dst2 = cv2.addWeighted(old_frame,0.3, rot_draw, 0.8, 0)
            dst3=cv2.addWeighted(self.rotatedframes[0],0.3, self.rot_raw, 0.7, 0)
            resized = cv2.resize(dst3, None, fx=0.25, fy=0.25)
            
            cv2.imshow('Overlay of frames 1 and current',resized)
            cv2.imshow(f'Overlayed match update frame {self.frameno+1}',dst2)
            cv2.moveWindow(f'Overlayed match update frame {self.frameno+1}',window_shift,0)
            cv2.waitKey(1)    
            r_prime =np.sqrt(delatxtm**2+delatytm**2)
            a_prime = 180*np.arctan2(delatytm, delatxtm)/pi
            
            sigma = (self.R-a_prime)*pi/180
            dx1 = r_prime*cos(sigma)
            dy1 = r_prime*sin(sigma)
            
            return dx1,dy1, stdxy,  minconfindence, delconarray
            
            
    
    def linear_shift(self, cutsize, rot_draw, confidenceth, outsideonly, panels=3):
        
        
        testimage=cv2.cvtColor(rot_draw,cv2.COLOR_BGR2GRAY)
        templateimage=cv2.cvtColor(self.rotatedframes[-1],cv2.COLOR_BGR2GRAY)
        method= cv2.TM_CCOEFF_NORMED
        methods = [
            cv2.TM_SQDIFF_NORMED,
            cv2.TM_CCORR_NORMED,
            cv2.TM_CCOEFF_NORMED,
        ]
        if panels==2:
            self.rot_raw=rot_draw.copy()
        if panels>5:
            method_index=self.change_method(method)
            print(f'Changing template method to {methods[method_index[0]]}')
            method=methods[method_index[0]]
            confidence=confidenceth/2
        else:
            confidence=confidenceth

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
                        
                        
                        
                        if max_val*100>=confidence:
                            cl=(255,0,0)
                        else:
                            cl=(0,0,255)
                        if max_val*100>confidence and not np.isinf(max_val):
                            delxlist.append(bottom_right[0]-(i+xt_step))
                            delylist.append(bottom_right[1]-(k+yt_step))
                            delconfidence.append(max_val)
                            cv2.rectangle(rot_draw, top_left, bottom_right, cl, 2)
                            panel_cx = i + xt_step // 2
                            panel_cy = k + yt_step // 2
                            outcoord.append([bottom_right[0], bottom_right[1]])

                        else:
                            cv2.rectangle(rot_draw, top_left, bottom_right, cl, 2)
        
                       
            
            delconarray=np.array(delconfidence)
            
            delxlistarray=np.array(delxlist)
            delylistarray=np.array(delylist)
            
            delxlist=list(delxlistarray)
            delylist=list(delylistarray)
            delconarraylist=list(delconarray)
      
        cv2.circle(rot_draw,(self.width//2,self.height//2),18,(0,0,255),2)
        
        return rot_draw, delxlist,delylist ,delconarraylist ,outcoord
    
    
    def removeoutliers(self, input_list):
        if len(input_list) == 0:
            return []
        sorted_list = np.sort(input_list)
        lower_q = np.percentile(sorted_list, 25)
        upper_q = np.percentile(sorted_list, 75)
        iqr = upper_q - lower_q
        lower_bound = lower_q - 1.5 * iqr
        upper_bound = upper_q + 1.5 * iqr
        return [x for x in sorted_list if lower_bound <= x <= upper_bound]
    

    def confidence_to_alpha(self, derr):
        """
        Map derr (expected to be good near 0–1, bad above 10) to alpha in [0.2, 0.95].
        Lower derr → higher alpha.
        """
        derr = max(0, min(derr, 10))  # clamp to [0, 10]
        alpha = 0.95 - (derr / 10) * (0.95 - 0.2)  # inverse linear map
        return alpha
    
    def derr_angle_to_alpha(self, derr, delta_angle):
        """
        Compute alpha blending weight based on:
        - derr: template match error, good if <1, bad if >10
        - delta_angle: change in angle in degrees, good if <2.5°, risky if >5°
        """
        # Clamp inputs
        derr = max(0, min(derr, 10))
        delta_angle = abs(delta_angle) % 360
        if delta_angle > 180:
            delta_angle = 360 - delta_angle  # ensure smallest angular difference
    
        # Scale derr: 0 → 0.0, 10 → 1.0
        derr_score = derr / 10
    
        # Scale angle: 0 → 0.0, ≥5° → 1.0
        angle_score = min(delta_angle / 5.0, 1.0)
    
        # Weighted risk score: adjust weights if needed
        risk_score = 0.4 * derr_score + 0.6 * angle_score
    
        # Map risk → alpha: 0.95 (low risk) → 0.2 (high risk)
        alpha = 0.95 - risk_score * (0.95 - 0.5)
        return alpha
    
    def change_method(self, current_method):
        """Given a current OpenCV template matching method, return the next one and whether it supports a mask."""
        methods = [
            (cv2.TM_SQDIFF_NORMED, False),
            (cv2.TM_CCORR_NORMED, True),
            (cv2.TM_CCOEFF_NORMED, False),
        ]
        
        method_list = [m[0] for m in methods]
        if current_method not in method_list:
            
            
            return methods[0]
    
        index = method_list.index(current_method)
        next_index = (index + 1) % len(methods)
        print(('Method change to methods[next_index][0]'))
        return methods[next_index]

    
    def estimate_R(self, rounds, trans_frame, old_warp, old_contour, good_fit, template_meth=cv2.TM_CCORR_NORMED, mask_meth=True):
        ''' Estimates rotation using template matching on polar-warped transformed frame images. '''
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        warp = cv2.warpPolar(trans_frame, (self.warp_radius, 3600), self.center, self.width // 2, self.flags)
        testimage = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        templateimage = cv2.cvtColor(old_warp, cv2.COLOR_BGR2GRAY)
        
        dst=cv2.addWeighted(testimage, 0.5, templateimage, 0.5,0)
        cv2.imshow('overlay of warps', dst)
        displayimage = cv2.hconcat([old_warp, warp])
        font = cv2.FONT_HERSHEY_SIMPLEX
        for o in range(0, 360, 45):
            cv2.putText(displayimage, str(o), (0, o * 10), font, 3, (200, 50, 0), 5, cv2.LINE_AA, False)
    
        eroded = cv2.erode(old_contour, kernel, iterations=5)
        contour_warp = cv2.warpPolar(eroded, (self.warp_radius, 3600), self.center, self.width // 2, self.flags)
        upperlim = 50
        lowerlim = int(self.warp_radius * 0.2)
    
        if mask_meth:
            white_cols = np.where(contour_warp.any(axis=0))[0]
            if len(white_cols) > 0:
                rightmost_x = white_cols[-1]
                upperlim = contour_warp.shape[1] - rightmost_x
    
        method = template_meth
        if rounds % 7 == 0 and rounds < 10 and rounds != 0:
            uniform = False
        else:
            uniform = True
    
        self.templateranges = self.findpointsofinterest(templateimage, self.keypoints, uniform=uniform)
        if len(self.templateranges) < 5:
            self.templateranges = self.findpointsofinterest(templateimage, self.keypoints * 3, uniform=True)
    
        conf_list = []
        con_y_shift_list = []
        MASK_COMPATIBLE = [cv2.TM_SQDIFF, cv2.TM_CCORR_NORMED]
    
        for n, rang_val in enumerate(self.templateranges):
            top_left1=(lowerlim,rang_val)
            bottom_right1=(self.warp_radius-upperlim,rang_val+self.windowsize)
            cv2.rectangle(displayimage,top_left1,bottom_right1,(255,255,0),2)
            template = templateimage[rang_val:rang_val + self.windowsize, lowerlim:-upperlim]
            mask_roi = contour_warp[rang_val:rang_val + self.windowsize, lowerlim:-upperlim].copy()
    
            starty = int(self.R_values[-1] * 10 + rang_val - self.search_angle)
            endy = int(self.R_values[-1] * 10 + rang_val + self.search_angle)
            res_shape = (testimage.shape[0] - template.shape[0] + 1, testimage.shape[1] - template.shape[1] + 1)
            search_template = np.zeros(res_shape, dtype=np.uint8)
    
            if starty < 0:
                cv2.rectangle(search_template, (lowerlim, 3600 + starty), (self.warp_radius - upperlim, 3600), 255, -1)
                cv2.rectangle(search_template, (lowerlim, 0), (self.warp_radius - upperlim, endy), 255, -1)
            elif endy > 3600:
                cv2.rectangle(search_template, (lowerlim, starty), (self.warp_radius - upperlim, 3600), 255, -1)
                cv2.rectangle(search_template, (lowerlim, 0), (self.warp_radius - upperlim, endy - 3600), 255, -1)
            else:
                cv2.rectangle(search_template, (lowerlim, starty), (self.warp_radius - upperlim, endy), 255, -1)
    
            mask_arg = mask_roi if mask_meth and method in MASK_COMPATIBLE else None
    
            try:
                res = cv2.matchTemplate(
                    testimage,
                    template,
                    method,
                    None,
                    mask=mask_arg
                )
            except cv2.error:
                continue
    
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res, mask=search_template)
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
                max_val = 1 - min_val
            else:
                top_left = max_loc
    
            rx = top_left[0]
            dr = abs(rx - lowerlim)
            if dr > 40 and rounds > 5:
                continue
    
            top_left2 = (top_left[0] + self.warp_radius, top_left[1])
            bottom_right2 = (top_left2[0] + template.shape[1], top_left2[1] + template.shape[0])
            clr = (0, 0, 255) if max_val * 100 < self.rotation_conf_thresh else (255, 0, 0)
            cv2.rectangle(displayimage, top_left2, bottom_right2, clr, 2)
            cv2.line(displayimage, (lowerlim, rang_val + self.windowsize), (top_left2[0], top_left2[1] + self.windowsize), (0, 255, 0), 2)
    
            conf_list.append(max_val)
            con_y_shift_list.append([max_val, top_left[1] - self.templateranges[n], rx])
    
        filtered_list = []
        counting = 0
        confidence_thresh = self.rotation_conf_thresh / 100
        condec = 0.01
    
        while counting < 10:
            y_shift_list = [x[1] for x in con_y_shift_list if x[0] >= confidence_thresh and abs(x[2] - lowerlim) < 45]
            y_shift_list = [(ys + 3600 if ys < -1800 else ys - 3600 if ys > 1800 else ys) for ys in y_shift_list]
            f_a_list = self.removeoutliers(y_shift_list)
            if len(f_a_list) > 4:
                filtered_list = f_a_list
                break
            confidence_thresh -= condec
            counting += 1
    
        if not filtered_list:
            filtered_list = [x[1] for x in con_y_shift_list if x[0] >= max(conf_list) - condec]
    
        if self.frameno == 0:
            an, ag = 0.0, 0.0
        elif self.frameno == 1:
            an, ag = self.R_values[-1], 0.0
        else:
            an, ag = self.R_values[-1], self.R_values[-2]
    
        if filtered_list:
            an, anstde = self.calculatetanangles(filtered_list)
            good_match = anstde <= 3.5 and len(filtered_list) >= 6
        else:
            anstde, an, good_match = 10, self.R_values[-1], False
    
        self.Rdiff = abs(an - self.Rprevious)
    
        if filtered_list:
            f_l = np.array(filtered_list)
            binx = np.argmin(abs(f_l - self.R_values[-1] * 10))
            ag = f_l[binx] / 10
            ag = (ag - 360 if ag > 180 else ag + 360 if ag < -180 else ag)
        else:
            ag = self.R_values[-1]
    
        cv2.imshow('Polar maps of frames n and n+1', displayimage)
        cv2.waitKey(1)
    
        return anstde, an, ag, conf_list, good_match

    def smart_alpha(self,anstde, Rdiff, softness=4.0, min_alpha=0.2, max_alpha=0.95, Rboost=15):
        """
        Calculates alpha based on both angular standard deviation (anstde)
        and how far off the current angle is (Rdiff).
        """
        # Confidence term (from anstde)
        confidence_weight = 1 / (1 + (anstde / softness)**2)
    
        # Boost term: if Rdiff is large, allow more responsiveness
        boost = min(1.0, abs(Rdiff) / Rboost)  # caps at 1.0
        weight = max(confidence_weight, boost)
    
        return min_alpha + (max_alpha - min_alpha) * weight
    
    def alpha_from_anstde(self,anstde, min_alpha=0.2, max_alpha=0.9, scale=1.5):
        """Returns alpha based on anstde: lower stddev → higher alpha."""
        confidence = np.exp(-scale * anstde)
        return min_alpha + (max_alpha - min_alpha) * confidence

    def average_angles_deg(self, angles, weights):
        angles_rad = np.radians(angles)
        sin_sum = np.sum(np.sin(angles_rad) * weights)
        cos_sum = np.sum(np.cos(angles_rad) * weights)
        avg_angle_rad = np.arctan2(sin_sum, cos_sum)
        return np.degrees(avg_angle_rad)   
    
    def findpointsofinterest(self, img, keypoints, uniform=True):
        # The orb detection seems to be ineffective for videos tested
        if uniform:
            self.halfwindowsize = self.windowsize // 2
        else:
            self.halfwindowsize=250
            
    
        templateranges = []


        if uniform:
            numberofpositions = keypoints
            angle_centers = np.linspace(0, 3600, numberofpositions, endpoint=False, dtype=int)
        
            for center in angle_centers:
                start = center - self.halfwindowsize
                # Handle wraparound explicitly
                if start < 0:
                    start += 3600
                templateranges.append(start)
        
    
            print(f'Uniform sampling: {len(templateranges)} templates')
    
        else:
            numberofpositions = 12
            angle_step_range = 3600 // numberofpositions
            if len(img.shape) == 3:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray_img=img.copy()
                
    
            for a in range(numberofpositions):
                if a == 0:
                    for start in [3150, 0]:
                        end = start + angle_step_range // 2
                        orb = cv2.ORB_create(nfeatures=keypoints//numberofpositions)
                        kp, _ = orb.detectAndCompute(gray_img[start:end, :], None)
                        templateranges += [
                            (int(x.pt[1]) - self.halfwindowsize) + start
                            for x in kp if x.pt[1] >= self.halfwindowsize
                        ]
                else:
                    start = a * angle_step_range - 300
                    end = start + angle_step_range
                    orb = cv2.ORB_create(nfeatures=keypoints//numberofpositions)
                    kp, _ = orb.detectAndCompute(gray_img[start:end, :], None)
                    templateranges += [
                        (int(x.pt[1]) - self.halfwindowsize) + start
                        for x in kp if x.pt[1] >= self.halfwindowsize
                    ]
    
            print(f'ORB-based sampling: {len(templateranges)} templates')
        return templateranges
    
    
    def calculatetanangles(self, fl):
        """
        Estimate average angle from a list (in tenths of degrees) using proximity filtering
        to the last known rotation angle.
        """
        # Convert to degrees
        angles_deg = np.array(fl) / 10.0
        reference = self.R_values[-1]  # Last known angle in degrees
    
        # Wrap angle differences to [-180, 180]
        def angle_diff(a, b):
            return ((a - b + 180) % 360) - 180
    
        # Filter angles close to the reference (within ±30°)
        filtered = [a for a in angles_deg if abs(angle_diff(a, reference)) < 30]
    
        if not filtered:
            print("Warning: No angles within ±30° of reference")
            return reference, 20.0  # fallback
    
        # Convert to radians
        angles_rad = np.deg2rad(filtered)
    
        # Compute mean angle using circular averaging
        mean_rad = np.arctan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad)))
        std_rad = np.std(np.angle(np.exp(1j * (angles_rad - mean_rad))))
    
        mean_deg = np.rad2deg(mean_rad)
        std_deg = np.rad2deg(std_rad)
        if len(filtered)>0:
            print(f"Angle {mean_deg:.1f} ± {std_deg/len(filtered):.1f} degrees using {len(filtered)} values")
        return mean_deg, std_deg

    def is_diverging(self):
        anglearray = np.array(self.angle_history[-6:])
        if len(anglearray) < 6:
            return False  # not enough data
    
        # Step differences
        diffs = np.diff(anglearray)
        mean_abs_diff = np.mean(np.abs(diffs))
        net_change = np.abs(anglearray[-1] - anglearray[0])
    
        print(f'Mean angle = {mean_abs_diff:.2f}, Net change = {net_change:.2f}')
        
        # Low mean diff + low net change = stuck
        if mean_abs_diff < 2.5 and net_change < 2.0:
            print("Detected bounded oscillation — stuck in local minimum")
            return True  
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
            dst = cv2.addWeighted(self.rotatedframes[-1], 0.5, self.rotated, 0.7, 0)
            dst1 = cv2.addWeighted(self.rotatedframes[0], 0.5, self.rotated, 0.7, 0)
    
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

                        # Create filename and path for the cutout image
                        cell_index = len(self.cells)
                        cutoutfile = f'img{self.frameno}_{cell_index}.jpg'
                        cutoutpath = os.path.join(self.imagepath, cutoutfile)
                        
                        # Extract the image region and save it
                        cutoutimg = self.img[sy:ey, sx:ex]
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

                                    # Create filename and path for the cutout image
                                    cell_index = len(self.cells)
                                    cutoutfile = f'img{self.frameno}_{cell_index}.jpg'
                                    cutoutpath = os.path.join(self.imagepath, cutoutfile)
                                    
                                    # Extract and save the cutout image
                                    cutoutimg = self.img[sy:ey, sx:ex]
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
    def __init__(self, file, path, scale, tpf, maxdistance, addnumbers,search_angle,keypoints,dpi,autotrack, w1, w2):
        super().__init__(file, path, scale, tpf, maxdistance, addnumbers,search_angle,keypoints,dpi,autotrack, w1, w2)
        

class ThreeDvideoAnalysis(Process_cells):
    def __init__(self, file, path, scale, tpf, maxdistance, addnumbers,search_angle, keypoints, dpi,autotrack,w1,w2):
        super().__init__(file, path, scale, tpf, maxdistance, addnumbers, search_angle,keypoints, dpi,autotrack, w1, w2)

    def start(self, anglestep, windowsize, allowedcentreshift, maxrounds, tolerance, panel_conf_thresh):
        
        self.transrotate(anglestep, windowsize, allowedcentreshift, maxrounds, tolerance,panel_conf_thresh) # adjusts the position of the frames to centre and orientate the assembloid
        self.run()
        self.calculate_traj()
        self.analyse_traj()
        self.vector_analysis()
        print('Complete and files saved')
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
        
        path='/Users/phykc/Library/CloudStorage/OneDrive-UniversityofLeeds/organiod/DyPheT/rdgtracker/data'
        #  scale is in units of µm per pixel
        # tpf is time per frame 3600s is 60 minutes
        # Maxdistance limits the distance in pixels that a cell in frame n can be associated to a cell in frame n+1.
        #  Search angle (in degrees limits the +/-)
        # Autotrack = True automates the tracking of rotation and translation.  False give manual control.
        # Most of the time angle shifts are a few degrees but in one case there is ~90 degree shift between frames hence below.
        # This means that if the video only shows small angle shifts you can reduce the search angle (delta angle).
        # If the polar and linear matching fails a brute force method rotates the image and finds the best linear translation match and returns that angle. This is much slower, but breaks an endless loop.
        # If linear fit panels are not matching tru relaxing the panel_conf_thresh to a lower number. It seems some videos require a lower TH compared to others.
        rgb_analysis = ThreeDvideoAnalysis(file, path, scale=1.0, tpf=3600, maxdistance=100, addnumbers=True, search_angle=60, keypoints=50, autotrack=True, dpi=300, w1=1,w2=2)
        rgb_analysis.start(anglestep=30, windowsize=400, allowedcentreshift=100, maxrounds=45, tolerance=1.1, panel_conf_thresh=62  )
        # try:
            
        # except Exception as e:
        #     print(f' Likely to be wrong path if the video does not start. Exception is {e}')
