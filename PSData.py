# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:19:06 2021

@author: Stephan
"""
from types import SimpleNamespace
import simplejson as json

class Unit:
    __slots__ = ['type', 's', 'c', 't']
    
    def __init__(self):
        self.type = ''
        self.s = ''
        self.q = ''
        self.a = ''

class Dataset:
    __slots__ = ['type', 'values']
    
    def __init__(self):
        self.type = ''
        self.values = []
    
class Appearance:
    __slots__ = ['type', 'autoassigncolor', 'color', 'linewidth', 'symbolsize', 'symboltype', 'noline']
    
    def __init__(self):
        self.type = ''
        self.autoassigncolor = False
        self.color = ''
        self.linewidth = 0
        self.symbolsize = 0
        self.symboltype = 0
        self.symbolfill = False
        self.noline = False
    
class Xaxisdataarray:
    __slots__ = ['type', 'arraytype', 'description', 'unit', 'datavalues', 'datavaluetype']
    
    def __init__(self):
        self.type = ''
        self.arraytype = 0
        self.description = ''
        self.unit = Unit()
        self.datavalues = []
        self.datavaluetype = ''
     
class Yaxisdataarray:
    __slots__ = ['type', 'arraytype', 'description', 'unit', 'datavalues', 'datavaluetype']
    
    def __init__(self):
        self.type = ''
        self.arraytype = 0
        self.description = ''
        self.unit = Unit()
        self.datavalues = []
        self.datavaluetype = ''

class Curve:
    __slots__ = ['appearance', 'title', 'hash', 'type', 'xaxis', 'xaxisdataarray', 'yaxisdataarray', 'meastype', 'peaklist', 'corrosionbutlervolmer', 'corrosiontafel']
    
    def __init__(self):
        self.appearance = Appearance()
        self.title = ''
        self.hash = []
        self.type = ''
        self.xaxis = 0
        self.yaxis = 0
        self.xaxisdataarray = Xaxisdataarray()
        self.yaxisdataarray = Yaxisdataarray()
        self.meastype = 0
        self.peaklist = []
        self.corrosionbutlervolmer = []
        self.corrosiontafel = []
    
class Measurement:
    __slots__ = ['title','timestamp','utctimestamp','deviceused','deviceserial','devicefw','type','dataset','method','curves','eisdatalist']
    
    def __init__(self):
        self.title = ''
        self.timestamp = object()
        self.utctimestamp = object()
        self.deviceused = 0
        self.deviceserial = ''
        self.devicefw = ''
        self.type = ''
        self.dataset = Dataset()
        self.method = ''
        self.curves = []
        self.eisdatalist = []
        
class Data:
    __slots__ = ['type','coreversion','methodformeasurement','measurements']
    
    def __init__(self):
        self.type = ''
        self.coreversion = ''
        self.methodformeasurement = ''
        self.measurements = []

class MethodType:
    __slots__ = ['CV','SWV','EIS','AD']
    
    def __init__(self):    
        self.CV = 'CV'
        self.SWV = 'SWV'
        self.AD = 'AD'
        self.EIS = 'EIS'

class EISMeasurement:
    __slots__ = ['freq','zdash','potential','zdashneg','Z','phase','current','npoints','tint','ymean','debugtext','Y','YRe','YIm','scale','Cdash','Cdashdash']
    
    def __init__(self):
        self.freq = []
        self.zdash = []
        self.potential = []
        self.zdashneg = []
        self.Z = []
        self.phase = []
        self.current = []
        self.npoints = []
        self.tint = []
        self.ymean = []
        self.debugtext = []
        self.Y = []
        self.YRe = []
        self.YIm = []
        self.Cdash = []
        self.Cdashdash = []
        self.scale = 100000 # standard set to mega ohms

class jparse:    
    @property
    def experimentList(self):
        return self._experimentList
    
    @property
    def parsedData(self):
        return self._parsedData
    
    @property
    def data(self):
        return self._data
    
    def __init__(self, filename):
        self._methodType = MethodType()
        self._experimentList = []
        self.file = None
        self._parsedData = self._parse(filename)
        self.experimentIndex = 0
        self.experimentToFileMap = {}
        self._data = self._simplify()
        
    # modified to accpet the .pssession file from the ByteIO
    def _parse(self, filename):
        # takes in the file 
        # parses the raw data to an object
        # simplifies the values and adds it to the 'data' object
        
        name = filename.name.replace('.pssession','')
        self.file = name
        readData = {}
        readData[self.file] = filename.read().decode('utf-16').replace('\r\n',' ').replace(':true',r':"True"').replace(':false',r':"False"')
           
        try:
            data2 = readData[self.file][0:(len(readData[self.file]) - 1)] # has a weird character at the end
            parsedData = json.loads(data2, object_hook=lambda d: SimpleNamespace(**d))
        except:
            return 'Failed to parse string to JSON'
        
        try:
            for measurement in parsedData.measurements:
                currentMethod = self._getMethodType(measurement.method).upper()
                index = len([i for i, s in enumerate(self._experimentList) if currentMethod in s])
                self._experimentList.append(currentMethod + ' ' + str(index + 1))
        except:
            return 'Failed to generate property: experimentList'

        return parsedData
    
    def _simplify(self):
        simplifiedData = {}
        experimentIndex = 0
        rawData = self._parsedData
        for measurement in rawData.measurements:
            currentMethod = self._getMethodType(measurement.method).upper()
            if currentMethod in self._methodType.SWV or currentMethod in self._methodType.CV or currentMethod in self._methodType.AD:
                simplifiedData[self._experimentList[experimentIndex] + ' xdata'] = self._getXYDataPoints(measurement)[0]
                simplifiedData[self._experimentList[experimentIndex] + ' ydata'] = self._getXYDataPoints(measurement)[1]
                simplifiedData[self._experimentList[experimentIndex] + ' total_scans'] = self._getXYDataPoints(measurement)[2]
                simplifiedData[self._experimentList[experimentIndex] + ' titles'] = self._getXYDataPoints(measurement)[3]
                simplifiedData[self._experimentList[experimentIndex] + ' Details'] = self._getXYUnits(measurement)
            if currentMethod in self._methodType.EIS:
                simplifiedData[self._experimentList[experimentIndex]] = self._getEISDataPoints(measurement)
            self.experimentToFileMap[self._experimentList[experimentIndex]] = self.file
            experimentIndex = experimentIndex + 1
        
        return simplifiedData
    
    def _getXYUnits(self, measurement):
        unit = {}
        try:
            xtext = measurement.curves[0].xaxisdataarray.unit.type
            ytext = measurement.curves[0].yaxisdataarray.unit.type
            if xtext is not None:
                unit['x'] = self._unitTextToScale(xtext)
                unit['y'] = self._unitTextToScale(ytext)
            unit['title'] = measurement.title
        except:
            print('Exception when processing units for SWV or CV.')
            unit = {}
        return unit
        
    
    def _unitTextToScale(self, text):
        unit = {}
        unit['scale'] = 1
        unit['unit'] = ''
        
        if 'Milli' in text:
            unit['scale'] = 1000
            unit['unit'] = 'm'
        if 'Micro' in text:
            unit['scale'] = 1000000
            unit['unit'] = "\u03BC"
        if 'Nano' in text:
            unit['scale'] = 1000000000
            unit['unit'] = 'n'
        if 'Pico' in text:
            unit['scale'] = 1000000000000
            unit['unit'] = 'p'
            
        return unit

     
    # modified to return xvalue, yvale, num_of_scan and scan_titles
    def _getXYDataPoints(self, measurement):
        xvalue = {}
        yvalue = {}
        titles = {}
        scans = 0 
        for curve in measurement.curves:
            xvalue[scans] = {}
            yvalue[scans] = {}
            titles[scans] = curve.title
            pos = 0
            for y in curve.yaxisdataarray.datavalues:
                xvalue[scans][pos] = curve.xaxisdataarray.datavalues[pos].v 
                yvalue[scans][pos] = y.v
                pos = pos + 1
            scans += 1

        return xvalue, yvalue, scans, titles 

    
    def _getEISDataPoints(self, measurement):
        eisdata = EISMeasurement()                   
        for eis in measurement.eisdatalist:
            for value in eis.dataset.values:
                if value.unit.q is not None:
                    v = []
                    for c in value.datavalues:
                        v.append(c.v)
                    if value.unit.q == "Frequency":
                        eisdata.freq = v
                    if value.unit.q == "Z'":
                        eisdata.zdash = v
                    if value.unit.q == "Potential'":
                        eisdata.potential = v
                    if value.unit.q == "-Z''":
                        eisdata.zdashneg = v
                    if value.unit.q == "Z":
                        eisdata.Z = v
                    if value.unit.q == "-Phase":
                        eisdata.phase = v
                    if value.unit.q == "npoints":
                        eisdata.npoints = v
                    if value.unit.q == "tint":
                        eisdata.tint = v
                    if value.unit.q == "ymean":
                        eisdata.ymean = v
                    if value.unit.q == "debugtext":
                        eisdata.debugtext = v
                    if value.unit.q == "Y":
                        eisdata.Y = v
                    if value.unit.q == "Y'":
                        eisdata.YRe = v
                    if value.unit.q == "Y''":
                        eisdata.YIm = v
                        
        pos = 0
        cd = []
        cdd = []
        for zdd in eisdata.zdashneg:
            denom = 2*3.141592653589793*eisdata.freq[pos]*(eisdata.zdash[pos]*eisdata.zdash[pos] + eisdata.zdashneg[pos]*eisdata.zdashneg[pos])
            cdd.append(zdd/(denom))
            cd.append(eisdata.zdash[pos]/(denom))
            pos += 1
        
        eisdata.Cdash = cd
        eisdata.Cdashdash = cdd
        
        return eisdata
                
    def _getMethodType(self, method):
        methodName = ''
        splitted = method.split("\r\n")
        for line in splitted:
            if "METHOD_ID" in line:
                methodName = line.split("=")[1]
        return methodName
    
    def inFile(self, experimentLabel):
        if experimentLabel in self.experimentToFileMap:
            print(experimentLabel + ' is in ' +  self.experimentToFileMap[experimentLabel] + ".pssession")
