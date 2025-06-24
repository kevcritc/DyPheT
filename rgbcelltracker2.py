#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 07:55:36 2022

@author: phykc
"""
import cv2 as cv2
import os
import numpy as np
from math import sqrt, pi
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import ast

class FileManager:
    def __init__(self, file, path):
        self.file=file
        self.path=path
        if file is not None:
            self.filename=os.path.join(self.path, self.file)
            self.newdirectoryname=self.file[:-4]+'cellimages'
            self.imagepath = os.path.join(self.path, self.newdirectoryname)
        try:
            os.makedirs(self.imagepath, exist_ok=True)
        except OSError as error:
                print(error, 'May overwrite previous images')
        self.datafolder = os.path.join(self.path, 'data')
        os.makedirs(self.datafolder, exist_ok=True)
    def datapath(self, fname):
        """Helper to return full path to file in the data folder"""
        return os.path.join(self.datafolder, fname)
        
        
        
class OpenVideo(FileManager):
    """Opens the video given of the path. Contains methods for runing the video, processcing the frames, and collecting cell information"""
    def __init__(self, file, path, search_angle,keypoints=30, autotrack=True):
        # self.autotrack = autotrack
        # self.search_angle=search_angle*10
        super().__init__(file, path)
        self.frames=[]
        if self.file is not None:
            try:
                os.mkdir(self.imagepath)
            except OSError as error: 
                print(error, 'May overwrite previous images')  
    def run(self):
        self.cap =cv2.VideoCapture(self.filename)
        self.maxframes=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
        print('File: ',self.filename)
        print('Size: ',self.width,' x ',self.height)
        print('Frames: ',self.maxframes)
        print('Runnning...')
        self.framenumber=0
        while self.framenumber<self.maxframes:
            print('frame ', self.framenumber+1,'of ', self.maxframes)
            #Red each frame ret id True if there is a frame
            ret, self.frame = self.cap.read()
            #This ends the movie when no frame is present or q is pressed
            if not ret:
                print('End of frames')
                cv2.waitKey(1)
                break
            key = cv2.waitKey(200) & 0xFF
            if key == ord('q'):
                cv2.waitKey(1)
                break
            self.frames.append(Frame(self.file, self.path, self.frame, f_no=self.framenumber))
            self.frames[-1].analyse()
            self.framenumber+=1
        cv2.destroyAllWindows()
        cv2.waitKey(10) 
        #Release the video 
        self.cap.release()
        cv2.waitKey(1)
           
    def rawdata(self):
        return self.frames

class Cell:
    def __init__(self, xpos, ypos, hue, area, rgb, frame_no, cell_number, cell_angle, aspectratio):
        self.xpos=xpos
        self.ypos=ypos
        self.hue=hue
        self.area=area
        self.rgb=rgb
        self.frame_no=frame_no
        self.cell_number=cell_number 
        self.cell_angle=cell_angle
        self.aspectratio=aspectratio

class Frame(FileManager):
    def __init__(self,file, path,img, f_no,ath=(31,-1), avepix=51, distort=50, area_th=120, timedisplay=(94,1406,142,57), scalebar=(1176,1443,1512,1521)):
        super().__init__(file,path)
        self.img=img
        self.frameno=f_no
        self.avepix=avepix
        self.distort=distort
        self.kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        self.area_th=area_th
        self.timedisplay=timedisplay
        self.cells=[]
        self.ath=ath
        self.scalebar=scalebar
        self.width=self.img.shape[1]
        self.height=self.img.shape[0]
    def analyse(self):
        def not_colliding(x,y, collisionlist):
            collide=False
            for obj in collisionlist:
                if ((x>obj[0] and x<obj[0]+obj[2]) and (y>obj[1] and y<obj[1]+obj[3])):
                    collide=True
            if collide:
                return False
            else:
                return True
        self.RGB=self.img.copy()
        self.display=self.img.copy()
        self.gray=cv2.cvtColor(self.RGB, cv2.COLOR_BGR2GRAY)
        self.back=cv2.blur(self.gray,(91,91),cv2.BORDER_DEFAULT)
        self.gray=cv2.subtract(self.gray, self.back)
        self.filtergray=cv2.bilateralFilter(self.gray,self.avepix,self.distort,self.distort)
        
        ret,self.th=cv2.threshold(self.filtergray,15,255, cv2.THRESH_BINARY)
        
        self.erode = cv2.erode(self.th,self.kernel,iterations = 2)
        self.contours, hierarchy = cv2.findContours(self.erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.collisionlist=[self.timedisplay, self.scalebar]
       
        for cnt in self.contours:  
            area=cv2.contourArea(cnt)
            if area>self.area_th:
                x,y,w,h = cv2.boundingRect(cnt)
                # ignore the timedisplay
                if not_colliding(x,y, self.collisionlist):
                    offset=int(w/2)
                    if x-offset<0:
                        sx=0
                    else:
                        sx=x-offset
                    if y-offset<0:
                        sy=0
                    else:
                        sy=y-offset
                    if x+w+offset>self.width:
                        ex=self.width
                    else:
                        ex=x+w+offset
                    if y+h+offset>self.height:
                        ey=self.height
                    else:
                        ey=y+h+offset
                        
                    cutoutimg=self.img[sy:ey,sx:ex]
                    cutoutfile='img'+str(self.frameno)+'_'+str(len(self.cells))+'.jpg'
                    cutoutpath=os.path.join(self.imagepath, cutoutfile)
                    cv2.imwrite(cutoutpath, cutoutimg)
                    ellipse = cv2.fitEllipse(cnt)
                    cell_angle=ellipse[2]
                    aspectratio=ellipse[1][1]/ellipse[1][0]
                    if aspectratio<1:
                        aspectratio=1/aspectratio
                    cv2.rectangle(self.display,(x,y),(x+w,y+h),(255,0,0),2)
                    (cx,cy),radius = cv2.minEnclosingCircle(cnt) 
                    mask = np.zeros(self.gray.shape,np.uint8)
                    cv2.drawContours(mask,[cnt],0,255,-1)
                    mean_val = cv2.mean(self.RGB,mask = mask)
                    mean_rgb=((mean_val[2])/255,(mean_val[1])/255,(mean_val[0])/255)
                    HSV=cv2.cvtColor(self.RGB, cv2.COLOR_BGR2HSV)
                    mean_HSV = cv2.mean(HSV,mask = mask)
                    
                    self.cells.append(Cell(cx,cy,mean_HSV[0],area, mean_rgb,self.frameno,[self.frameno,len(self.cells)], cell_angle, aspectratio))
        cv2.imshow('cells found',self.display)
        cv2.waitKey(1)

class Process_cells(OpenVideo):
    def __init__(self, file, path, scale,tpf, maxdistance, addnumbers,search_angle,keypoints, dpi=800,autotrack=True,w1=1, w2=2):
       
        super().__init__(file, path,search_angle,keypoints, autotrack)
        self.maxdistance=maxdistance
        self.w1=w1
        self.w2=w2
        self.tpf=tpf
        self.scale=scale
        self.addnumbers=addnumbers
        self.dpi=dpi
        self.trajectories=[]
        self.keypoints=keypoints
        
    def create_data_lists(self):
        self.frame_no_list=[]
        self.x_list=[]
        self.y_list=[]
        self.area_list=[]
        self.RGB_list=[]
        self.h_list=[]
        self.cell_number_list=[]
        self.cell_angle_list=[]
        self.aspectratio_list=[]
        
        for no in range(len(self.frames)):
            for cellinfo in self.frames[no].cells:
                self.frame_no_list.append(cellinfo.frame_no)
                self.x_list.append(cellinfo.xpos)
                self.y_list.append(cellinfo.ypos)
                self.area_list.append(cellinfo.area)
                self.RGB_list.append(cellinfo.rgb)
                self.h_list.append(cellinfo.hue)
                self.cell_number_list.append(cellinfo.cell_number)
                self.cell_angle_list.append(cellinfo.cell_angle)
                self.aspectratio_list.append(cellinfo.aspectratio)
        
    def pairup(self):
        framesl=np.arange(0,max(self.frame_no_list)+1)
        #Establish list of the indicies for cells tracked in each frame
        frameindexs=[]
        for frame in framesl:
            indexingframe=[]
            for i in range(len(self.frame_no_list)):
                if self.frame_no_list[i]==frame:
                    indexingframe.append(i)
            frameindexs.append(indexingframe)
        #Compare frame n with frame n+1 make smallest separations match, unless greater than 
        #a maxium to create a link
        self.frame1index=[]
        self.frame2index=[]
    
        for a in range(len(frameindexs)-1):
            #b and c become the IDs from the CSV file
            if len(frameindexs[a])>0 and len(frameindexs[a+1])>0:
             #Create an array to store the seprations from from fram n and n+1
                 separr=np.zeros((len(frameindexs[a]), len(frameindexs[a+1])),dtype=float)
                 for b1,b in enumerate(frameindexs[a]):
                 
                    for c1,c in enumerate(frameindexs[a+1]):
                        #find the separation for each particle and place them in an array
                        #Add that to colour difference
                        x1=self.x_list[b]
                        y1=self.y_list[b]
                        x2=self.x_list[c]
                        y2=self.y_list[c]
                        seperation=sqrt((x2-x1)**2+(y2-y1)**2)
                        huediff=abs(self.h_list[c]-self.h_list[b])
                        separr[b1,c1]=self.w1*seperation+self.w2*huediff
                 noresult=True

                 for i in range(b1+1):
                    rep=0
                    noresult=True
                    while noresult==True and rep<2:
                        minval=np.amin(separr[i,:])
                        result=(np.where(separr[i,:] == np.amin(separr[i,:])))
                     
                        if minval==np.amin(separr[i:,result]) and minval<=self.maxdistance:
                            self.frame1index.append(frameindexs[a][i])
                            self.frame2index.append(frameindexs[a+1][result[0][0]])
                            noresult=False
                        else:
                            separr[i,result]=500
                            
                        rep+=1
    def listup(self):    
        #Create a list of the linking indicies between frame na dn n+1
        linklist=[]
        linkedup=[]
        for d in range(len(self.frame_no_list)):
            if d in self.frame1index:
                indy=self.frame1index.index(d)
                linklist.append(self.frame2index[indy])
            else:
                linklist.append('none')
            linkedup.append(False)
                
        #Create a set of trijectories index lists
        self.particlelist=[]
        
        for q in range(len(self.frame_no_list)):
            train=[]
            
            if linkedup[q]==False:
                train.append(q)
                
                linkedup[q]=True
                z=linklist[q]
                while z!='none' and linkedup[z]==False:
                    
                    train.append(z)
                    linkedup[z]=True
                    z=linklist[z]
                self.particlelist.append(train)
    def cellpathways(self):    
        self.cellpaths=[]
        for cell in self.particlelist:
            numbersteps=len(cell)
            path=[]
            if numbersteps>1:
                for ID in cell:
                    #time in hours frame 1 in 0.
                    time=self.frame_no_list[ID]
                    x=self.x_list[ID]
                    y=self.y_list[ID]
                    area=self.area_list[ID]
                    rgb=self.RGB_list[ID]
                    h=self.h_list[ID]
                    cn=self.cell_number_list[ID]
                    ca=self.cell_angle_list[ID]
                    ar=self.aspectratio_list[ID]
                    path.append([x,y,time,area,rgb, h, cn, ca, ar])
                self.cellpaths.append(path)
    def createdatapack(self): 
        self.datapack=[]        
        for track in self.cellpaths:
            steps=len(track)
            tpos=[]
            xpos=[]
            ypos=[]
            area=[]
            rgb_point=[]
            h_point=[]
            c_number=[]
            c_angle=[]
            aspect_ratio=[]
            #loop through the cell paths 'track' and get the postions from frame n and n+1 for the calculations
            #possibly missing the very last points from x,y,t,r,inten data??
            for n in range(steps):
                
                x1=track[n][0]
                y1=track[n][1]
                t1=track[n][2]
                area.append(track[n][3])
                rgb_point.append(track[n][4])
                h_point.append(track[n][5])
                c_number.append(track[n][6])
                c_angle.append(track[n][7])
                aspect_ratio.append(track[n][8])
                tpos.append(t1)
                xpos.append(x1)
                ypos.append(y1)
            meanhue=[sum(h_point)/len(h_point)]*steps
            #Put all the data in this list such that each path is an object in the list   
            self.datapack.append([xpos, ypos, tpos, area,rgb_point, h_point, c_number, c_angle, aspect_ratio])    
        
        #Look for case of one frame missing and join  them up if hue and sepearation are below a threshold.
        maxvalueallowed=self.maxdistance*1.5
        
        i=0
        while i<len(self.datapack)-1:
            #check the last time point and then look to see if a new time point starts +2 which is in the same region and has same colour.
            endt=self.datapack[i][2][-1]
            endx=self.datapack[i][0][-1]
            endy=self.datapack[i][1][-1]
            endh=self.datapack[i][5][-1]
            matse=[]
            svals=[]
            for j in range(i+1,len(self.datapack)):
                if self.datapack[j][2][0]==endt+2:
                    
                    matse.append(j)
            
            if len(matse)>0:
            
                
                for v in matse:
                    sx=self.datapack[v][0][0]
                    sy=self.datapack[v][1][0]
                    sh=self.datapack[v][5][0]
                    sep=sqrt((sx-endx)**2+(sy-endy)**2)
                    deltah=abs(sh-endh)
                    svals.append(sep*self.w1+deltah*self.w2)
                   
                min_value=min(svals)
                min_index=svals.index(min_value)
                # This patches the 'the before and after' missed frame together
                if min_value<maxvalueallowed:
                    
                    for q in range(9):
                        a=self.datapack[i][q]
                        b=self.datapack[matse[min_index]][q]
                            
                        c=a+b
                        self.datapack[i][q]=c
                    self.datapack.pop(matse[min_index])
            i+=1
      
        
    def plotalltracks(self):
        for data in self.datapack:
            plt.figure(num=1, dpi=self.dpi)
            plt.axis('equal') 
            plt.ylim(self.height,0)
            plt.xlim(0,self.width)
            coldata=np.array(data[4])
            col=coldata+0.1
            col=col*1.2
            if np.amax(col)>1.0:
                col=col/np.amax(col)
            plt.scatter(data[0],data[1], c=col) 
            plt.plot(data[0],data[1], color='gray', linewidth=1.0)
            plt.xlabel('x /pixel')
            plt.ylabel('y /pixel')
        figfilename = self.datapath(self.filename[:-4]+'alltracks1.png')
        plt.savefig(figfilename)
        plt.show()
        for data in self.datapack:
            plt.figure(num=1, figsize=(7, 7), dpi=self.dpi)
            plt.axis('equal') 
            plt.ylim(self.height,0)
            plt.xlim(0,self.width)
            coldata=np.array(data[5])
            meanhue=np.mean(coldata)
            
            if meanhue<21:
                cellcolour='red'
            if meanhue>=21 and meanhue<30:
                cellcolour='yellow'
            if meanhue>=30:
                cellcolour='green'
            colourarray=np.full(coldata.shape,cellcolour)
            plt.scatter(data[0],data[1], c=colourarray) 
            plt.plot(data[0],data[1], color='black', linewidth=0.5)
            plt.xlabel('x /pixel')
            plt.ylabel('y /pixel')
        figfilename = self.datapath(self.filename[:-4]+'alltracks2.png')
        plt.savefig(figfilename)
        plt.show()
        
        for i,data in enumerate(self.datapack):
            plt.figure(num=1, figsize=(7, 7),dpi=self.dpi)
            plt.axis('equal') 
            plt.ylim(self.height,0)
            plt.xlim(0,self.width)
            plt.plot(data[0],data[1], label=str(i)) 
            if self.addnumbers:
                plt.annotate(str(i), # this is the text
                     (data[0][0],data[1][0]), # these are the coordinates to position at the first point
                     textcoords="offset points", # how to position the text
                     xytext=(20,20), # distance from text to points (x,y)
                     ha='center', arrowprops=dict(arrowstyle="->", color='black'))
            plt.xlabel('x /pixel')
            plt.ylabel('y /pixel')
        figfilename = self.datapath(self.filename[:-4]+'alltracks3.png')
        plt.savefig(figfilename)
        
        plt.show()
            
            
    def applyscales(self):
        def area_cover_cal(df):
            x_size=int(self.width/self.scale)
            y_size=int(self.height/self.scale)
            canvas=np.zeros((y_size,x_size),dtype=np.uint8)
            x_vals=df['x'].tolist()
            y_vals=df['y'].tolist()
            x_vals = [int(i) for i in x_vals]
            y_vals = [int(i) for i in y_vals]
            point_list=list(zip(x_vals,y_vals))
            mean_area=df['area'].mean()
            eff_diameter=int(sqrt(mean_area/pi)*2)
            for n in range(len(point_list)-1):
                canvas = cv2.line(canvas, point_list[n],point_list[n+1], 255, eff_diameter)
            contours, hierarchy = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull = cv2.convexHull(contours[0])
            cover_area=cv2.contourArea(contours[0])
            hull_area=cv2.contourArea(hull)
            
            return cover_area, hull_area
            
        def is_unique(s):
            a = s.to_numpy()
            return not (a[0] == a).all()
        def angleout(x1,y1,x2,y2):
            
            angle=180*np.arctan2((y2-y1),(x2-x1))/pi
            if angle<0:
                angle+=360
            return angle

        def gammaout(x1,y1,x2,y2,x3,y3,x4,y4):
           
            v1=x2-x1
            v2=y2-y1
            w1=x4-x3
            w2=y4-y3
            
            gamma=180*np.arctan2((w2*v1)-(w1*v2),(w1*v1)+(w2*v2))/pi
            if gamma<0:
                gamma+=360
            return gamma
        #Unpack datapack into a full dataframe for one file at a time.
        # data pack : [xpos, ypos, tpos]
        df1s=[]
        for i,dat in enumerate(self.datapack):
            pathlength=len(dat[0])
            c_data={'Cell number':[i]*pathlength,'x':dat[0],'y':dat[1],'t':dat[2], 'area':dat[3],'rgb_mean':dat[4], 'hue':dat[5],'Img ID':dat[6],'Cell Angle Fit':dat[7],'Cell aspect ratio':dat[8]}
            df1s.append(pd.DataFrame(data=c_data))
            df1s[-1]['deltax']=df1s[-1]['x'].diff()
            df1s[-1]['deltay']=df1s[-1]['y'].diff()
            df1s[-1]['deltat']=df1s[-1]['t'].diff()
            df1s[-1]['di']=np.sqrt(df1s[-1]['deltax']**2+df1s[-1]['deltay']**2)
            df1s[-1]['vi']=df1s[-1]['di']/df1s[-1]['deltat']
            df1s[-1]['delta_vx']=df1s[-1]['deltax']/df1s[-1]['deltat']
            df1s[-1]['delta_vy']=df1s[-1]['deltay']/df1s[-1]['deltat']
            df1s[-1]['angle']=180*np.arctan2(df1s[-1]['deltay'],df1s[-1]['deltax'])/pi
            df1s[-1]['angle'] = np.where(df1s[-1]['angle'] < 0,  df1s[-1]['angle']+360, df1s[-1]['angle'])
            df1s[-1]['gamma']=df1s[-1]['angle'].diff()
            conditions=[(df1s[-1]['hue']<=20), (df1s[-1]['hue']>20) &(df1s[-1]['hue']<30), (df1s[-1]['hue']>=30)]
            c_values=['red', 'yellow', 'green']
            df1s[-1]['Cell Colour']=np.select(conditions, c_values)
            df1s[-1]['Total Time']=dat[2][-1]-dat[2][0]
            
            df1s[-1]['Net Distance']=np.sqrt((dat[0][-1]-dat[0][0])**2+(dat[1][-1]-dat[1][0])**2)
            df1s[-1]['Total Distance']=df1s[-1]['di'].sum()
            df1s[-1]['Confinement ratio']=df1s[-1]['Net Distance']/df1s[-1]['Total Distance']
            df1s[-1]['Max Step']= df1s[-1]['di'].max()
            
            dmax=0.0
            for n in range(1,pathlength):
                d=np.sqrt((dat[0][n]-dat[0][0])**2+(dat[1][n]-dat[1][0])**2)
                if d>dmax:
                    dmax=d
            df1s[-1]['dmax']=dmax
            df1s[-1]['Mean curvilinear speed']=df1s[-1]['vi'].mean()
            df1s[-1]['Mean straight-line speed']=df1s[-1]['Net Distance']/df1s[-1]['Total Time']
            df1s[-1]['Linear Forward Progression']= df1s[-1]['Mean straight-line speed']/ df1s[-1]['Mean curvilinear speed']
            df1s[-1]['Net delta x']=dat[0][-1]-dat[0][0]
            df1s[-1]['Net delta y']=dat[1][-1]-dat[1][0]
            df1s[-1]['Net delta angle']=np.arctan2(df1s[-1]['Net delta y'],df1s[-1]['Net delta x'])*180/pi
            
            df1s[-1]['Net x velocity']=df1s[-1]['Net delta x']/df1s[-1]['Total Time']
            df1s[-1]['Net y velocity']=df1s[-1]['Net delta y']/df1s[-1]['Total Time']
            df1s[-1]['Net velocity']=np.sqrt(df1s[-1]['Net x velocity']**2+df1s[-1]['Net y velocity']**2)*180/pi
            df1s[-1]['Net velocity angle']=np.arctan2(df1s[-1]['Net y velocity'],df1s[-1]['Net x velocity'])
            df1s[-1]['mean hue']=df1s[-1]['hue'].mean()
            conditions=[(df1s[-1]['mean hue']<=20), (df1s[-1]['mean hue']>20) &(df1s[-1]['mean hue']<30), (df1s[-1]['mean hue']>=30)]
            c_values=['red', 'yellow', 'green']
            df1s[-1]['Average Colour']=np.select(conditions, c_values)
            df1s[-1]['Transition']=is_unique(df1s[-1]['Cell Colour'])
            area_covered, hull_area=area_cover_cal(df1s[-1])
            df1s[-1]['area_covered']=area_covered
            df1s[-1]['hull_area_covered']=hull_area
            self.trajectories.append(Trajectory(df1s[-1]))
            
             
        df1=pd.concat(df1s,ignore_index=True)    
        timepoints=df1['t'].tolist()
        for time in timepoints:
            timedf=df1.loc[df1['t']==time]
            if len(timedf)>25:
                meanvx=timedf['delta_vx'].mean()
                meanvy=timedf['delta_vy'].mean()
                
                
        
        df1['x']=df1['x']/self.scale
        df1['y']=df1['y']/self.scale
        df1['t']=df1['t']*self.tpf
        df1['deltat']=df1['deltat']*self.tpf
        df1['deltax']=df1['deltax']/self.scale
        df1['deltay']=df1['deltay']/self.scale
        df1['Net delta x']=df1['Net delta x']/self.scale
        df1['Net delta y']=df1['Net delta y']/self.scale
        df1['area']=df1['area']/self.scale**2
        df1['di']=df1['di']/self.scale
        df1['vi']=df1['vi']/(self.scale*self.tpf)
        df1['Total Distance']=df1['Total Distance']/self.scale
        df1['Net Distance']=df1['Net Distance']/self.scale
        df1['Max Step']=df1['Max Step']/self.scale
        df1['dmax']=df1['dmax']/self.scale
        df1['Total Time']=df1['Total Time']*self.tpf
        df1['Mean curvilinear speed']=df1['Mean curvilinear speed']/(self.scale*self.tpf)
        df1['Mean straight-line speed']=df1['Mean straight-line speed']/(self.scale*self.tpf)
        df1['Net x velocity']=df1['Net x velocity']/(self.scale*self.tpf)
        df1['Net y velocity']=df1['Net y velocity']/(self.scale*self.tpf)
        df1['Net velocity']=df1['Net velocity']/(self.scale*self.tpf)
        df1['area covered microns square']=df1['area_covered']/self.scale**2
        df1['hull area covered microns square']=df1['hull_area_covered']/self.scale**2
        
        self.dataframe=df1
        self.dataframe['MSD alpha']=np.nan
        del df1
    def write_csv(self):
        csvfilename = self.datapath(self.filename[:-4]+'all_final.csv')
        self.dataframe.to_csv(csvfilename)
    
    def MSDanalysis(self):
        def dMSD(t,k,alpha):
            return k*t**alpha
        
        alphacelldata=[]
        amplitudedata=[]
        cellnumber=[]
        alphahuncertainty=[]
        amplitudeuncertainty=[]
        plt.figure(dpi=self.dpi)
        plt.xlabel('Lag time/s')
        plt.ylabel('MSD')
        for traj in self.trajectories:
            if len(traj.msdlag)>6:
                plt.plot(traj.msdarray[:,0]*self.tpf,traj.msdarray[:,1]/self.scale,'o')
                try:
                    popt, pcov=curve_fit(dMSD, traj.msdarray[:,0]*self.tpf, traj.msdarray[:,1]/self.scale, p0=[1,1])
                    # print('Cell', b,' has an Alpha value of ', popt[1])
                    plt.plot(traj.msdarray[:,0]*self.tpf,dMSD(traj.msdarray[:,0]*self.tpf,popt[0],popt[1]))
                    alphacelldata.append(popt[1])
                    amplitudedata.append(popt[0])
                    traj.alpha=popt[1]
                    traj.amp=popt[0]
                    cellnumber.append(traj.cell_number)
                    sigma0 = np.sqrt(pcov[0,0]) 
                    sigma1 = np.sqrt(pcov[1,1])
                    alphahuncertainty.append(sigma1)
                    amplitudeuncertainty.append(sigma0)
                    inx=self.dataframe.loc[self.dataframe['Cell number']==traj.cell_number].index
                    self.dataframe.loc[inx,'MSD alpha']=popt[1]
    
                except:
                    print('Fitting of MSD data failed skipping cell')
        data={'Cell number':cellnumber, 'alpha':alphacelldata, 'alpha uncertainty':alphahuncertainty, 'amplitude':amplitudedata}
        msddf=pd.DataFrame(data=data)
        xlsfilename= self.datapath(self.filename[:-4]+'MSD.xlsx')
        msddf.to_excel(xlsfilename)
        figfilename=self.datapath(self.filename[:-4]+'MSD.png')
        plt.savefig(figfilename)
        plt.show()
        
        data1={'Cell number':cellnumber,'Alpha':alphacelldata,'alpha+/- sigma':alphahuncertainty, 'Amplitude':amplitudedata, 'amplitude +/- sigma':amplitudeuncertainty}
        datafit=pd.DataFrame(data=data1)
        plt.figure(dpi=self.dpi)
        if self.file is not None:
            newfilename=self.datapath(self.filename[:-4]+'MSDfitresults.csv')
            datafit.to_csv(newfilename)
        plt.hist(alphacelldata, bins=[0,0.25,.5,.75,1.0,1.25,1.5,1.75,2,2.25])
        plt.xlabel('Alpha')
        plt.ylabel('Frequency')
        figfilename=self.datapath(self.filename[:-4]+'MSDhist.png')
        plt.savefig(figfilename)
        plt.show()
    def vectorplot(self, subset='none',colour=True):
        self.subset=subset
        self.colour=colour
        cellnumbers=self.dataframe['Cell number'].tolist()
        cells= list(dict.fromkeys(cellnumbers))
        Umax=0.0
        Vmax=0.0
        plt.figure(dpi=self.dpi)
        for a in cells:
            
            selection=self.dataframe[self.dataframe['Cell number']==a]
            
            if len(selection)>3:
                times=selection['t'].tolist()
                endtime=max(times)
                lastline=selection[selection['t']==endtime]
                colorlist=selection['Average Colour'].tolist()
                color=colorlist[0]
                X = 0.0
                Y= 0.0
                U = float(lastline['Net delta x'])
                V = float(lastline['Net delta y'])
                
                if self.colour!=True:
                    color='black'
                
                if abs(U)>=Umax:
                    if U>=0.0:
                        Umax, Umin=U, -U
                    else:
                        Umax, Umin=-U, U
                        
                if abs(V)>=Vmax:
                    if V>=0.0:
                        Vmax, Vmin=V, -V
                    else:
                        Vmax, Vmin=V, -V
                if self.subset=='red'and color=='red':
                    plt.arrow(X,Y,U,V, head_width=3, head_length=5, fc=color, ec=color)
                if self.subset=='green' and color=='green':
                    plt.arrow(X,Y,U,V, head_width=3, head_length=5, fc=color, ec=color)
                if self.subset=='none': 
                    plt.arrow(X,Y,U,V, head_width=3, head_length=5, fc=color, ec=color)
        plt.xlabel('Net x separation / µm')
        plt.ylabel('Net y separation / µm')
        plt.xlim(Umin,Umax)
        plt.ylim(Vmax,Vmin)
        plt.axis('equal') 
        figfilename=self.datapath(self.filename[:-4]+'vect1.png')
        plt.savefig(figfilename)
        plt.show()
        
    def distance_histogram(self):
        plt.figure(dpi=self.dpi)
        di_values=self.dataframe['di'].tolist()
        newlist = [x for x in di_values if np.isnan(x) == False]
        # hist=np.histogram(newlist, bins=40)   
        plt.hist(newlist, bins='auto')
        plt.xlabel('Single Step Distance /pixels')
        plt.ylabel('Frequency')
        
        figfilename=self.datapath(self.filename[:-4]+'dishist.png')
        plt.savefig(figfilename)
        plt.show()
            
    def velocityvectorplot(self, subset='none',colour=True):
        plt.figure(dpi=self.dpi)
        cellnumbers=self.dataframe['Cell number'].tolist()
        cells= list(dict.fromkeys(cellnumbers))
        Umax=0.0
        Vmax=0.0
        Umin=0.0
        Vmin=0.0
        for a in cells:
            
            selection=self.dataframe[self.dataframe['Cell number']==a]
            
            if len(selection)>3:
                times=selection['t'].tolist()
                endtime=max(times)
                lastline=selection[selection['t']==endtime]
                X = 0.0
                Y= 0.0
                U = float(lastline['Net x velocity'])*1000
                V = float(lastline['Net y velocity'])*1000
                colorlist=selection['Average Colour'].tolist()
                color=colorlist[0]
                if self.colour!=True:
                    color='black'
                
                if abs(U)>=Umax:
                    if U>=0.0:
                        Umax, Umin=U, -U
                    else:
                        Umax, Umin=-U, U
                        
                if abs(V)>=Vmax:
                    if V>=0.0:
                        Vmax, Vmin=V, -V
                    else:
                        Vmax, Vmin=V, -V
                if subset=='red'and color=='red':
                    plt.arrow(X,Y,U,V, head_width=0.3, head_length=0.2, fc=color, ec=color)
                if subset=='green' and color=='green':
                    plt.arrow(X,Y,U,V, head_width=0.3, head_length=0.2, fc=color, ec=color)
                if subset!='green' and subset!='red': 
                    plt.arrow(X,Y,U,V, head_width=0.3, head_length=0.2, fc=color, ec=color)
            
                    
                        
        plt.xlabel('$v_{x}$ / $nms^{-1}$')
        plt.ylabel('$v_{y}$ / $nms^{-1}$')
        plt.xlim(Umin,Umax)
        plt.ylim(Vmax,Vmin)
        plt.axis('equal') 
        figfilename=self.datapath(self.filename[:-4]+'vect2.png')
        plt.savefig(figfilename)
        plt.show()    
    
        
    def plot_area_covered(self):
        plt.figure(dpi=self.dpi)
        data_select=self.dataframe[~pd.isnull(self.dataframe['area_covered'])]
        reddate=data_select[data_select['mean hue']<=20]
        redlist=reddate['area_covered'].tolist()
        yellowdate=data_select[(data_select['mean hue']>20) & (data_select['mean hue']<30)]
        yellowlist=yellowdate['area_covered'].tolist()
        greendate=data_select[data_select['mean hue']>=30]
        greenlist=greendate['area_covered'].tolist()
        plt.hist(redlist, bins='auto', color='red', alpha=0.5)
        plt.hist(greenlist, bins='auto', color='green', alpha=0.5)
        plt.hist(yellowlist, bins='auto', color='yellow', alpha=0.5)
        plt.title('Trajectory Area')
        plt.xlabel('Area of contained by the perimeter/ pixel squared')
        plt.ylabel('frequency')
        
        figfilename=self.datapath(self.filename[:-4]+'areacovered.png')
        plt.savefig(figfilename)
        plt.show()
    def plot_hull_area_covered(self):
        data_select=self.dataframe[~pd.isnull(self.dataframe['hull_area_covered'])]
        reddate=data_select[data_select['mean hue']<=20]
        redlist=reddate['hull_area_covered'].tolist()
        yellowdate=data_select[(data_select['mean hue']>20) & (data_select['mean hue']<30)]
        yellowlist=yellowdate['area_covered'].tolist()
        greendate=data_select[data_select['mean hue']>=30]
        greenlist=greendate['hull_area_covered'].tolist()
        plt.hist(redlist, bins='auto', color='red', alpha=0.5)
        plt.hist(greenlist, bins='auto', color='green', alpha=0.5)
        plt.hist(yellowlist, bins='auto', color='yellow', alpha=0.5)
        plt.title('Hull Area')
        plt.xlabel('Area of contained by the hull perimeter /pixel squared')
        plt.ylabel('frequency')
        
        figfilename=self.datapath(self.filename[:-4]+'hullareacovered.png')
        plt.savefig(figfilename)
        plt.show()

class OfflineAnalysis(Process_cells):
    def __init__(self, addnumbers,scale=1, imgsize=(1526,1526)):
        super().__init__(file=None, path=None, scale=1, tpf=None, maxdistance=500, w1=1, w2=2, addnumbers=True)
        self.height=imgsize[0]
        self.width=imgsize[1]
    def load_csv(self,csvfile):
        self.csvfile=csvfile
        self.dataframe=pd.read_csv(self.csvfile)
        dict_cells=self.dataframe['Cell number'].to_dict()
        self.celllist=list(dict_cells.keys())
    def load_multi_csv(self, listoffiles):
        self.listoffiles=listoffiles
        dflists=[]
        for file in self.listoffiles:
            dflists.append(pd.read_csv(file))
        numberoffiles=len(dflists)
        offsetcellnumber=0
        for a in range(numberoffiles):
            dflists[a]['Cell number']=dflists[a]['Cell number']+int(offsetcellnumber)
            offsetcellnumber+=dflists[a]['Cell number'].max()
            offsetcellnumber+=100
        self.dataframe=pd.DataFrame().append(dflists)
        dict_cells=self.dataframe['Cell number'].to_dict()
        self.celllist=list(dict_cells.keys())
            

    def plotalltracks(self):
        for cell in self.celllist:
            celldf=self.dataframe[self.dataframe['Cell number']==cell]
            plt.figure(num=1, figsize=(7, 7))
            plt.axis('equal') 
            plt.ylim(self.width,0)
            plt.xlim(0,self.height)
            col=celldf['rgb_mean'].to_list()
            col_list = [ast.literal_eval(i) for i in col]
            plt.scatter(celldf['x'],celldf['y'], c=col_list) 
            plt.plot(celldf['x'],celldf['y'], color='gray', linewidth=1.0)
            plt.xlabel('x /pixel')
            plt.ylabel('y /pixel')
        plt.show()
        for cell in self.celllist:
            celldf=self.dataframe[self.dataframe['Cell number']==cell]
            plt.figure(num=1, figsize=(7, 7))
            plt.axis('equal') 
            plt.ylim(self.width,0)
            plt.xlim(0,self.height)
            colours=celldf['Colour'].to_list()
            plt.scatter(celldf['x'],celldf['y'], c=colours) 
            plt.plot(celldf['x'],celldf['y'], color='black', linewidth=1.0)
            plt.xlabel('x /pixel')
            plt.ylabel('y /pixel')
        plt.show()
        for cell in self.celllist:
            celldf=self.dataframe[self.dataframe['Cell number']==cell]
            plt.figure(num=1, figsize=(7, 7))
            plt.axis('equal') 
            plt.ylim(self.height,0)
            plt.xlim(0,self.width)
            plt.plot(celldf['x'],celldf['y'], label=str(cell)) 
            indl=celldf.index.to_list()
            if len(indl)>0:
                start_ind=min(indl)
                if self.addnumbers:
                    plt.annotate(str(cell), # this is the text
                          (celldf.loc[start_ind, 'x'],celldf.loc[start_ind,'y']), # these are the coordinates to position at the first point
                          textcoords="offset points", # how to position the text
                          xytext=(15,15), # distance from text to points (x,y)
                          ha='center', arrowprops=dict(arrowstyle="->", color='black'))
            plt.xlabel('x /pixel')
            plt.ylabel('y /pixel') 
        plt.show()
    def plot_area_covered(self):
        areacoveredred=[]
        areacoveredgreen=[]
        areacoveredyellow=[]
        hullareacoveredred=[]
        hullareacoveredgreen=[]
        hullareacoveredyellow=[]
        for cell in self.celllist:
            celldf=self.dataframe[self.dataframe['Cell number']==cell]
            maxtime=celldf['t'].max()
            lastline=celldf[celldf['t']==maxtime]
            if len(lastline)>0:
                if (lastline['Colour']=='green').any():
                    areacoveredgreen.append(lastline['area_covered'].item())
                    hullareacoveredgreen.append(lastline['hull_area_covered'].item())
                if (lastline['Colour']=='red').any():
                    areacoveredred.append(lastline['area_covered'].item())
                    hullareacoveredred.append(lastline['hull_area_covered'].item())  
                if (lastline['Colour']=='yellow').any():
                    areacoveredyellow.append(lastline['area_covered'].item())
                    hullareacoveredyellow.append(lastline['hull_area_covered'].item())
        plt.hist(areacoveredred, bins='auto', color='red', alpha=0.5)
        plt.hist(areacoveredgreen, bins='auto', color='green', alpha=0.5)
        plt.hist(areacoveredyellow, bins='auto', color='yellow', alpha=0.5)
        plt.title('Trajectory Area')
        plt.xlabel('Area of contained by the perimeter/ pixel squared')
        plt.ylabel('frequency')
        plt.ylim(0,500)
        plt.show()
        plt.hist(hullareacoveredred, bins='auto', color='red', alpha=0.5)
        plt.hist(hullareacoveredgreen, bins='auto', color='green', alpha=0.5)
        plt.hist(hullareacoveredyellow, bins='auto', color='yellow', alpha=0.5)
        plt.title('Hull Area')
        plt.xlabel('Area of contained by the hull perimeter /pixel squared')
        plt.ylabel('frequency')
        plt.show()

    def plottracks(self, listofcells):
        self.listofcells=listofcells
        for cell in self.listofcells:
            celldf=self.dataframe[self.dataframe['Cell number']==cell]
            plt.figure(num=1, figsize=(7, 7))
            plt.axis('equal') 
            col=celldf['rgb_mean'].to_list()
            col_list = [ast.literal_eval(i) for i in col]
            plt.scatter(celldf['x'],celldf['y'], c=col_list) 
            plt.plot(celldf['x'],celldf['y'], color='gray', linewidth=1.0)
            plt.xlabel('x /pixel')
            plt.ylabel('y /pixel')
            ax = plt.gca()
            ax.invert_yaxis()

        plt.show()
        
    def process_all(self):
        # self.MSDanalysis()
        self.distance_histogram()
        self.plot_area_covered()
        
        self.vectorplot(colour=False,subset='none')
        self.velocityvectorplot(colour=False,subset='none')
        self.vectorplot(colour=True, subset='red')
        self.velocityvectorplot(colour=True, subset='red')
        self.vectorplot(colour=True, subset='green')
        self.velocityvectorplot(colour=True, subset='green')
        self.vectorplot(colour=True, subset='none')
        self.velocityvectorplot(colour=True, subset='none')
    
class Trajectory:
    def __init__(self, df):
        self.df=df.copy()
        self.cell_number=df.loc[0,'Cell number']
        self.msdlist=[]
        self.msdlag=[]
        if len(self.df)>8:
            for lag in range(1,15):
                self.df[f'x lag {lag}']= df['x'].diff(lag)
                self.df[f'y lag {lag}']= df['y'].diff(lag)
                self.df[f'r lag {lag}']=np.sqrt(self.df[f'x lag {lag}']**2+self.df[f'y lag {lag}']**2)
                self.df[f'sd lag {lag}']=self.df[f'r lag {lag}']**2
                if self.df[f'sd lag {lag}'].mean()>0:
                    self.msdlist.append(self.df[f'sd lag {lag}'].mean())
                    self.msdlag.append(lag)
                    self.msdarray=np.array(list(zip(self.msdlag,self.msdlist)))
                    
           
class VideoAnalysis(Process_cells):
    """ A wrapper for proceesing RG Cells motion"""
    def __init__(self,file,path, scale, tpf, maxdistance,search_angle, w1=1, w2=3, addnumbers=False):
    
        super().__init__(file, path,scale, tpf, maxdistance,search_angle, w1, w2, addnumbers)
        
    def start(self):
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
        self.vectorplot(colour=False,subset='none')
        self.velocityvectorplot(colour=False,subset='none')
        self.vectorplot(colour=True, subset='red')
        self.velocityvectorplot(colour=True, subset='red')
        self.vectorplot(colour=True, subset='green')
        self.velocityvectorplot(colour=True, subset='green')
        self.vectorplot(colour=True, subset='none')
        self.velocityvectorplot(colour=True, subset='none')

class Run_video:
    def __init__(self,file, path, search_angle,scale, tpf, maxdistance, addnumbers):
        try:
            rgb_analysis=VideoAnalysis(file, path, search_angle, scale, tpf, maxdistance, addnumbers)
            rgb_analysis.start()
        except ValueError:
            print("Stopping")
            print(ValueError)
            print('The filename is not a video file that is recognised. Check the name and path of the file')
        

if __name__=='__main__':        
    # this is the file and directory of the movie file
    path='/Users/phykc/Documents/Work/organiod/static_video_RGB_analyser/static videos/recentdata/'
    file='A1 FUCCI_GBM1 Composite-1.avi'
    Run_video(file, path,search_angle=None, scale=1, tpf=1800, maxdistance=150, addnumbers=True)
    # scale is in units of µm per pixel
    # tpf is time per frame 1800s is 30 minutes
    # Maxdistance limits the distance in pixels that a cell in frame n can be associated to a cell in frame n+1.
    
    
    # Offline anaylsis can be performed by commenting out the above 7 lines and uncommenting below.  The offline anaylsis is the same for both 3D and 2D so either csv files can be passed.
    #ana=OfflineAnalysis(addnumbers=True,imgsize=(1526,1526))        
    #ana.load_csv('/Users/phykc/Documents/Work/organiod/static_video_RGB_analyser/static videos/recentdata/A1 FUCCI_GBM1 Composite-1all_final.csv')
    #ana.load_multi_csv(['/Users/phykc/Documents/Work/organiod/static_video_RGB_analyser/static videos/recentdata/A1 FUCCI_GBM1 Composite-1all_final.csv','/Users/phykc/Downloads/drive-download-20220423T104519Z-001/C4_Overlay_0_160all_final.csv'])
    #ana.plotalltracks()
    # ana.process_all()
    #ana.plottracks([82])
    












        