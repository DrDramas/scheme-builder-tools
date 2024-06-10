#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wednesday April 5, 2023
Last updated on Tuesday, April 11, 2023

@author: mcarpenter, tbudner, dlenz
"""

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from functools import cached_property

import numpy as np
import numpy.random as rnd
import pandas as pd
import networkx as nx
from textwrap import dedent
import copy

# ###################################################### #
class LSGraph(object):

    def __init__(self, graph_dict=None):
        #""" initializes a graph object 
        #    If no dictionary or None is given, 
        #    an empty dictionary will be used
        #"""
        if graph_dict == None:
            graph_dict = {}
        self._graph_dict = graph_dict

    def edges(self, vertice):
        #""" returns a list of all the edges of a vertice"""
        return self._graph_dict[vertice]
        
    def all_vertices(self):
        #""" returns the vertices of a graph as a set """
        return set(self._graph_dict.keys())

    def all_edges(self):
        #""" returns the edges of a graph """
        return self.__generate_edges()

    def add_vertex(self, vertex):
        #""" If the vertex is not in 
        #    self._graph_dict, a key vertex with an empty
        #    list as a value is added to the dictionary. 
        #    Otherwise nothing has to be done. 
        #"""
        if vertex not in self._graph_dict:
            self._graph_dict[vertex] = []

    def add_edge(self, edge):
        #""" assumes that edge is of type set, tuple or list; 
        #    between two vertices can be multiple edges! 
        #"""
        edge = set(edge)
        vertex1, vertex2 = tuple(edge)
        for x, y in [(vertex1, vertex2), (vertex2, vertex1)]:
            if x in self._graph_dict:
                self._graph_dict[x].add(y)
            else:
                self._graph_dict[x] = [y]

    def __generate_edges(self):
        #""" A static method generating the edges of the 
        #    graph. Edges are represented as sets 
        #    with one (a loop back to the vertex) or two 
        #    vertices 
        #"""
        edges = []
        for vertex in self._graph_dict:
            for neighbour in self._graph_dict[vertex]:
                if {neighbour, vertex} not in edges:
                    edges.append({vertex, neighbour})
        return edges
    
    def __iter__(self):
        self._iter_obj = iter(self._graph_dict)
        return self._iter_obj
    
    def __next__(self):
        #""" allows us to iterate over the vertices """
        return next(self._iter_obj)

    def __str__(self):
        res = "vertices: "
        for k in self._graph_dict:
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.__generate_edges():
            res += str(edge) + " "
        return res
    
    def find_path(self, start_vertex, end_vertex, path=None):
        #""" find a path from start_vertex to end_vertex 
        #in graph """
        if path == None:
            path = []
        graph = self._graph_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return path
        if start_vertex not in graph:
            return None
        for vertex in graph[start_vertex]:
            if vertex not in path:
                extended_path = self.find_path(vertex,end_vertex,path)
            if extended_path: 
                return extended_path
        return None


    def find_all_paths(self, start_vertex, end_vertex, path=[]):
        #""" find all paths from start_vertex to 
        #    end_vertex in graph """
        graph = self._graph_dict 
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return [path]
        if start_vertex not in graph:
            return []
        paths = []
        for vertex in graph[start_vertex]:
            if vertex not in path:
                extended_paths = self.find_all_paths(vertex,end_vertex,path)
                for p in extended_paths: 
                    paths.append(p)
        return paths

# ###################################################### # 
class Gamma:
    
    def __init__(self, *args):

        line = args[0]

        # first decode the gamma energy
    
        geline = line[0:16]
        tmp = geline.split()
        self.isGam = True
        try:
            self.gE = float(tmp[0])
        except:
            self.gE = 0.0
            self.isGam = False
        self.gE_err = 0
        if len(tmp) == 2:
            try:
                self.gE_err = int(tmp[1])
            except:
                self.gE_err = 0

        # decode relative intensity
        RIline = line[17:32]
        tmp = RIline.split()
        try:
            self.RI = float(tmp[0])
        except:
            self.RI = -1.0
        self.RI_err = 0
        if len(tmp) == 2:
            try:
                self.RI_err = int(tmp[1])
            except:
                self.RI_err = 0
        
        # Multipolarity info - not sure how to decode this yet
    
        self.Multline = line[33:59].strip()
    
        # conversion coefficents
        CCline = line[60:71]
        tmp = CCline.split()
        try:
            self.CC = float(tmp[0])
        except:
            self.CC = 0.0
        self.CC_err = 0
        if len(tmp) == 2:
            try:
                self.CC_err = int(tmp[1])
            except:
                self.C_err =0
    
        # Initial Excited level
        ExTline = line[72:87]
        tmp = ExTline.split()
        try:
            self.ExT = float(tmp[0])
        except:
            self.ExT = 0.0
        self.SpnT = "none"
        if len(tmp) == 2:
            try:
                self.SpnT = tmp[1].strip()
            except:
                self.SpnT = "none"
    
        # Final Excited level
        ExBline = line[88:104]
        tmp = ExBline.split()
        try:
            self.ExB = float(tmp[0])
        except:
            self.ExB = 0.0
        self.SpnB = "none"
        if len(tmp) == 2:
            try:
                self.SpnB = tmp[1].strip()
            except:
                self.SpnB = "none"
        
        # Branching ratio (to be determined for all gammas at each level)
        self.BR = 0
        
    def list_values(self):

        print("Gamma Ray Transition Info")
        print("Gamma Energy, Error:",self.gE,self.gE_err)
        print("Relative Int., Error:",self.RI,self.RI_err)
        print("Multiploarity, Mixing Ratio:",self.Multline)
        print("Conversion Coeff., Error:",self.CC,self.CC_err)
        print("Excitation Energy and Spin of Initial Level:",self.ExT,self.SpnT)
        print("Excitation Energy and Spin of Final Level:",self.ExB,self.SpnB)
        print(" ")

# ###################################################### #
class Level:

    def __init__(self, *args):

        self.ExE = args[0]
        self.outGammas = []   # List of gamma rays emitted from this Level
        self.inGammas=[] # List of transitions that directly populate this state
        
    def update_excitation_energy(self,exEn):
    
        self.ExE = exEn

    def add_outgoing_gamma(self,gamma):

        self.outGammas.append(gamma)
        #print(self.ExE,gamma.gE)
    
    def add_incoming_gamma(self,gamma):

        self.inGammas.append(gamma)
        #print(self.ExE,gamma.gE)

    def compute_BRs(self):

        total = 0
        for gamma in self.outGammas:
            total += gamma.RI
        for gamma in self.outGammas:
            gamma.BR = gamma.RI/total

    def list_gammas(self):

        for gamma in self.outGammas:
            print("Initial Level:",self.ExE,"Gamma: ",gamma.gE,"Branching Ratio: ",gamma.BR)
            
# ###################################################### #
class LevelScheme:
    
    def __init__(self):
        
        self.nl = 0
        self.nt = 0
        self.g = nx.DiGraph()    # Directed graph object that represents the LevelScheme
        self.emax = 10000        # This is set for drawing
        self.levels = {}         # Dictionary {Level object : Node #} for LevelScheme
        self.placed = []         # List of gamma-rays that have already been placed in LevelScheme
        self.redundantNodes = [] # List of nodes that have been placed unnecessarily and have been labeled redundant
        self.srcToDst = {}       # Dictionary with the form {sourceNode:[destination1,destination2,destination3,...]}
        self.cascade = {}        # Dict {(source,destination):[pathway1,pathway2,pathway3,...]}; pathways are lists of gammas
        self.pathLengths = {}    # Dict {(source,destination):[length1,length2,length3,...]}; lengths are summed gamma energies
        self.savedNodes = []
        self.redundantEdges = []
        self.leafNodes = []
        self.rootNodes = []
        
    def __str__(self):

        return dedent(f"""
            Level Scheme with {self.nl} levels and {self.nt} transitions.
            Energies: {self.get_energies()}""")

    def get_energies(self):

        return [en for i, en in self.g.nodes.data('energy')]

    # def to_transition_scheme(self):
    #     return TransitionScheme.from_level_scheme(self)
    
    def add_level(self,lvl):
        
        nl=self.g.number_of_nodes()
        self.g.add_node(nl,energy=lvl.ExE)
        self.levels.update({nl:lvl})
    
    def map_from_transition_space(self,TS):
        
        for edge in TS.g.edges():
            # Scenario #1
            if edge[0] not in self.placed and edge[1] not in self.placed: # Make three new levels
                #print('-----Scenario 1------')
                # Add lowest level
                newLvl=Level(0.1)
                #newLvl.inGammas.append(edge[1])
                newLvl.inGammas.append(TS.g.nodes[edge[1]]['Gamma'])
                self.add_level(newLvl)
                # Add middle level
                newLvl=Level(0.1)
                #newLvl.inGammas.append(edge[0])
                newLvl.inGammas.append(TS.g.nodes[edge[0]]['Gamma'])
                #newLvl.outGammas.append(edge[1])
                newLvl.outGammas.append(TS.g.nodes[edge[1]]['Gamma'])
                self.add_level(newLvl)
                # Add highest level
                newLvl=Level(0.1)
                #newLvl.outGammas.append(edge[0])
                newLvl.outGammas.append(TS.g.nodes[edge[0]]['Gamma'])
                self.add_level(newLvl)
                # Add both transitions' energies to list of gammas placed in level scheme
                self.placed.append(edge[0])
                self.placed.append(edge[1])
            # Scenario #2
            elif edge[0] in self.placed and edge[1] not in self.placed:
                #print('-----Scenario 2------')
                for lvl in self.levels:
                    if TS.g.nodes[edge[0]]['Gamma'] in self.levels[lvl].inGammas:
                        #lvl.outGammas.append(edge[1]) # Add to existing list of outgoing gamma rays
                        self.levels[lvl].outGammas.append(TS.g.nodes[edge[1]]['Gamma'])
                        newLvl=Level(0.1) # Add new level to LevelScheme
                        #newLvl.inGammas.append(edge[1]) # Add previously unseen transition to list of incoming gamma rays
                        newLvl.inGammas.append(TS.g.nodes[edge[1]]['Gamma'])
                        self.add_level(newLvl)
                        self.placed.append(edge[1]) # Add transition energy to list of placed gamma-rays
                        break
                        
            # Scenario #3
            elif edge[1] in self.placed and edge[0] not in self.placed:
                #print('-----Scenario 3------')
                for lvl in self.levels:
                    if TS.g.nodes[edge[1]]['Gamma'] in self.levels[lvl].outGammas:
                        #lvl.inGammas.append(edge[0]) # Add to existing list of incoming gamma rays
                        self.levels[lvl].inGammas.append(TS.g.nodes[edge[0]]['Gamma'])
                        newLvl=Level(0.1) # Add new level to LevelScheme
                        #newLvl.outGammas.append(edge[0]) # Add previously unseen transition to list of outgoing gamma rays
                        newLvl.outGammas.append(TS.g.nodes[edge[0]]['Gamma'])
                        self.add_level(newLvl)
                        self.placed.append(edge[0]) # Add transition to list of placed gamma-rays
                        break
                        
            # Scenario #4
            else: # i.e. both gamma rays (edge[0] and edge[1]) have already been placed in the level scheme
                #print('-----Scenario 4------')
                #print('Gammas placed already: ',edge[0],edge[1])
                for lvl in self.levels:
                    #if edge[0] in lvl.inGammas: # Find level which the edge[0] gamma populates
                    if TS.g.nodes[edge[0]]['Gamma'] in self.levels[lvl].inGammas:
                        e0=lvl
                        #print('Level ',e0,' has incoming gamma ',edge[0])
                        break
                        
                for lvl in self.levels:
                    #if edge[1] in lvl.outGammas: # Find level that emits the edge[1] gamma
                    if TS.g.nodes[edge[1]]['Gamma'] in self.levels[lvl].outGammas:
                        e1=lvl
                        #print('Level ',e1,' has outgoing gamma ',edge[1])
                        break
                        
        #print('e0 level: ',e0)
        #print('e1 level: ',e1)
                if e0==e1:  # If these levels are the same, good. Move on to the next edge
                    continue
                else:  # Levels have been improperly duplicated. Remove the one more recently placed
                    for gamma in self.levels[e1].inGammas: # Copy incoming gammes for redundant node/level
                        if gamma not in self.levels[e0].inGammas:
                            self.levels[e0].inGammas.append(gamma) # Add them to existing node/level
                    self.levels[e1].inGammas.clear() # Delete this list so we don't accidentally reference redundant node/level
                    for gamma in self.levels[e1].outGammas: # Copy outgoing gammes from redundant node/level
                        if gamma not in self.levels[e0].outGammas: 
                            self.levels[e0].outGammas.append(gamma) # Add them to existing node/level
                    self.levels[e1].outGammas.clear() # Delete this list so we don't accidentally reference redundant node/level
                    if e1 not in self.redundantNodes:
                        self.redundantNodes.append(e1) # Add to list of redundant nodes
                        
    
    def connect_nodes_with_edges(self):
        
        for loLvl in self.levels:
            for gamma in self.levels[loLvl].inGammas:
                for hiLvl in self.levels:
                    if gamma in self.levels[hiLvl].outGammas:
                        self.g.add_edge(hiLvl,loLvl,energy=gamma.gE,weight=gamma.BR)
        
    def delete_edges_and_nodes(self):
        
        for edge in self.redundantEdges:
            self.g.remove_edge(edge[0],edge[1]) # Delete edge from graph
        self.redundantEdges.clear() # Once redundancies have been remove, empty the list of edges/nodes
        for node in self.redundantNodes:
            self.g.remove_node(node) # Delete node from graph
            self.levels.pop(node) # Remove this Level from dictionaroy
        self.redundantNodes.clear()
            
    def build_gamma_cascades(self,TS):
    
        for path in TS.allPaths:                 # Loop over all possible gamma-ray transition sequences
            for lvl in self.levels:              # Loop over levels/nodes in the level scheme space
                if TS.g.nodes[path[0]]['Gamma'] in self.levels[lvl].outGammas: # If first transition in path is an outgoing gamma of Level 
                    start_lvl=lvl             # This is the starting level
                    #print('Start level: ',start_lvl)
                    break
            for lvl in self.levels:
                if TS.g.nodes[path[len(path)-1]]['Gamma'] in self.levels[lvl].inGammas: # If last transition in path is incoming gamma...
                    stop_lvl=lvl                       # This is the stopping level
                    #print('Stop level: ',stop_lvl)
                    break
        
            if start_lvl not in self.srcToDst:            # If the source level hasn't been added yet
                self.srcToDst.update({start_lvl:[]})      # Update the dictionary
                self.srcToDst[start_lvl].append(stop_lvl) # Add final level to list of possible destination levels
            else:                                    
                self.srcToDst[start_lvl].append(stop_lvl)
        
            start_stop=(start_lvl,stop_lvl) # Declare tuple that specifies endpoints of transition pathway
            if start_stop not in self.cascade:   # If these end points have not been added to the cascade yet
                self.cascade.update({start_stop:[]})
                self.cascade[start_stop].append(path)
            else: # This tuple already exists in dictionary
                self.cascade[start_stop].append(path) # New possible pathway between source and destination levels
            
    def compute_path_lengths(self,energy_threshold=1):
        
        for start_stop in self.cascade: # Loop over all pairs of starting/stopping levels in cascade dictionary
            self.pathLengths.update({start_stop:[]})
            for pathway in self.cascade[start_stop]: # Loop over all pathways for a given pair of endpoints
                sumEnergy=0
                for gammaEnergy in pathway: # Loop over all gamma-rays within a given pathway
                    sumEnergy += gammaEnergy # Sum up the total energy between the source and destination levels
                self.pathLengths[start_stop].append(sumEnergy) # Add sum of gamma energies to the distance between two Levels
                
        for start_stop in self.pathLengths:
            for i in range(len(self.pathLengths[start_stop])-1):
                for j in range(i+1,len(self.pathLengths[start_stop])):
                    if abs(self.pathLengths[start_stop][i]-self.pathLengths[start_stop][j])>energy_threshold:
                        print('ERROR: path lenghts differ by more than energy threshold')
            sumEnergy=0
            for length in self.pathLengths[start_stop]:
                sumEnergy += length
            meanEnergy=sumEnergy/len(self.pathLengths[start_stop])
            self.pathLengths[start_stop].clear()
            self.pathLengths.update({start_stop:meanEnergy})        

    def merge_redundant_leaves(self,energy_threshold=1):
        
        updatedEdges = [] # List containing updated edges after nodes have been labeled as redundant
        gEnergies = []    # List of gamma energies that should be assigned to the new edges
        gWeights = [] # List of gamma intensities that should be assigned to the new edges
        
        for src in self.srcToDst: # Loop over all starting Levels
        
            if len(self.srcToDst[src])>1: # If the number of possible destination Levels is greater than one...
                for i in range(len(self.srcToDst[src])-1): 
                    #print('Node i: ',srcToDst[src][i])
                    if self.srcToDst[src][i] in self.redundantNodes: # If ith node has been labeled redundant, skip it
                        continue
                    for j in range(i+1,len(self.srcToDst[src])):
                        #print('Node j: ',srcToDst[src][j])
                        if self.srcToDst[src][i] in self.redundantNodes: # If jth node has been labeled redundant, skip it
                            continue
                        
            #if len(srcToDst[src])==2:
                        dsti=self.srcToDst[src][i] # ith leaf node
                        dstj=self.srcToDst[src][j] # jth leaf node
                        if dsti==dstj: # Both paths lead to the same Level
                            #if lvlEnergies[(src,dsti)]!=lvlEnergies[(src,dstj)]: # If sum of gamma energies differ
                            if abs(self.pathLengths[(src,dsti)]-self.pathLengths[(src,dstj)])>energy_threshold:
                                print('ERROR: Gamma energies do not sum to same level energy!')
                            else: # If they're the same, nothing to see here. What you'd expect
                                continue
                        else: # Both paths lead to different nodes
                            #if lvlEnergies[(src,dsti)]!=lvlEnergies[(src,dstj)]: # If sum of gamma energies differ
                            #if abs(lvlEnergies[(src,dsti)]-lvlEnergies[(src,dstj)])>energy_threshold:
                            if abs(self.pathLengths[(src,dsti)]-self.pathLengths[(src,dstj)])>energy_threshold:
                                continue # This is what you'd expect. One of these levels isn't the ground state
                            else: # Nodes have the same sum of gamma energies
                                self.savedNodes.append(dsti) # List of nodes that duplicates but should be preserved
                                #if dstj not in redundantNodes and dstj not in savedNodes:
                                if dstj not in self.redundantNodes:
                                    self.redundantNodes.append(dstj) # List of redundant nodes to be deleted
                                #for gamma in inGammas[dstj]:
                                for gamma in self.levels[dstj].inGammas:
                                    #if gamma not in inGammas[dsti]:
                                    if gamma not in self.levels[dsti].inGammas:
                                        #inGammas[dsti].append(gamma)
                                        self.levels[dsti].inGammas.append(gamma)
                                #else: # Already marked as a redundant node
                                #    continue
                                # Should be no outgoing gammas  in this leaf node
                                #for gamma in outGammas[e1]:
                                #    if gamma not in outGammas[e0]:
                                #        outGammas[e0].append(gamma)
                                #for edge in ls.edges:
                                for edge in self.g.edges:
                                    #if edge[1]==dstj: # and dstj not in savedNodes: # If edge contains redundant leaf node...
                                    if edge[1]==dstj:
                                        if edge not in self.redundantEdges:
                                            self.redundantEdges.append(edge)
                                            #gamEn=ls[edge[0]][dstj]
                                            gamEn=self.g.edges[edge]['energy']
                                            #gamEn=gamEn['energy']
                                            gEnergies.append(gamEn)
                                            gamBR=self.g.edges[edge]['weight']
                                            gWeights.append(gamBR)
                                            newEdge=(edge[0],dsti)
                                            updatedEdges.append(newEdge)
                                #ls.add_edge(edge[0],dst0) # Make new edge connecting existing leaf node
                                            print('New edge: ',newEdge)
                                        else: # Already marked as redundant edge
                                            continue
                        #ls.remove_node(dst1) # Delete redundant node
                        #print('Redundant node removed: ',dst1) 
        print('Updated edges: ',updatedEdges)
        g=0 # Index counter for gamma-ray energies
        for e in updatedEdges:
            self.g.add_edge(e[0],e[1],energy=gEnergies[g],weight=gWeights[g])
            g+=1

                        
    def find_leaf_nodes(self):
        
        self.leafNodes.clear()
        for lvl in self.levels: # Loop over all Levels in the LevelScheme
            if len(self.levels[lvl].outGammas)<1: # If the number of outgoing gammas from a level is zero...
                self.leafNodes.append(lvl) # Find its node number and add to the list of leaf nodes
                
    
    def leaf_node_deexcitation_energies(self):
        
        deexcitationEnergies = {}
        for node in self.leafNodes: # Loop over all potential ground states (i.e. leaf nodes)
            maxEnergy=0
            for start_stop in self.pathLengths: 
                if node==start_stop[1]: # If the stop node is a leaf node...
                    if self.pathLengths[start_stop]>maxEnergy: # Check if this is the largest deexcitation energy
                        maxEnergy=self.pathLengths[start_stop]
            deexcitationEnergies.update({start_stop:maxEnergy})
        print('Maximum energy lost when populating leaf nodes: ',deexcitationEnergies)
        
    def leaf_node_incoming_intensity(self,gammas,S):
        
        incomingIntensities = {}
        for node in self.leafNodes:
            gFlow=0
            for edge in self.g.edges:
                if edge[1]==node:
                    #gFlow+=self.g.edges[edge]['intensity']
                    gammaEnergy=self.g.edges[edge]['energy']
                    for gam in gammas: # Loop over all gammas
                        if gam.gE==gammaEnergy: # If the gamma energy corresponds to that of the edge connecting the leaf node
                            index=gammas.index(gam) # Get the index of this gamma-ray in the list
                            break
                    intensity=S[index] # Use the index of the gamma-ray of interest to get the intensity from Singles matrix
                    gFlow+=intensity # Add to the total gamma-ray flow into this level
            incomingIntensities.update({node:gFlow})
        print('Total number of gammas going into each leaf node: ',incomingIntensities)
    
    
    def find_ground_state(self,gammas,S):
        
        gamFlows={}
        for node in self.leafNodes:
            gFlow=0
            for edge in self.g.edges:
                if edge[1]==node:
                    #gFlow+=self.g.edges[edge]['intensity']
                    gammaEnergy=self.g.edges[edge]['energy']
                    for gam in gammas: # Loop over all gammas
                        if gam.gE==gammaEnergy: # If the gamma energy corresponds to that of the edge connecting the leaf node
                            index=gammas.index(gam) # Get the index of this gamma-ray in the list
                            break
                    intensity=S[index] # Use the index of the gamma-ray of interest to get the intensity from Singles matrix
                    gFlow+=intensity # Add to the total gamma-ray flow into this level
            gamFlows.update({node:gFlow})
        print(gamFlows)
        gs=max(gamFlows,key=gamFlows.get)
        print(gs)
        
    def find_root_nodes(self):
        
        self.rootNodes.clear()
        for lvl in self.levels: # Loop over all Levels in the LevelScheme
            if len(self.levels[lvl].inGammas)<1: # If the number of incoming gammas to this Level is zero...
                self.rootNodes.append(lvl) # Find its node number and add to the list of root nodes
                
    def compute_level_energies(self,gsNode):
        
        self.g.nodes[gsNode]['energy']=0.0 # Set energy of the ground state node to 0.0 keV 
        self.levels[gsNode].update_excitation_energy(0.0) # Update ground-state Level's excitation energy to 0.0 keV
              
        eAssigned = [] # List of nodes that have been assigned energies
        eAssigned.append(gsNode) # Add ground-state node to the list
        iteration=0 # Counts the number of iterations in while loop
        maxIter=10  # Used as break condition to avoid infinite loop
        
        while len(eAssigned)<self.g.number_of_nodes():
            for edge in self.g.edges:
                if edge[1] in eAssigned and edge[0] not in eAssigned: # If lower level has assigned energy...
                    gamEn=self.g[edge[0]][edge[1]]['energy']    # Get gamma energy of associated edge
                    self.g.nodes[edge[0]]['energy']=self.g.nodes[edge[1]]['energy']+gamEn # Add gamma energy to lower level
                    self.levels[edge[0]].update_excitation_energy(self.g.nodes[edge[0]]['energy']) # Update Level energy
                    eAssigned.append(edge[0]) # Add to list of assigned Level energies
                elif edge[0] in eAssigned and edge[1] not in eAssigned: # If lower level has assigned energy...
                    gamEn=self.g[edge[0]][edge[1]]['energy']
                    self.g.nodes[edge[1]]['energy']=self.g.nodes[edge[0]]['energy']-gamEn # Subtract gamma energy from upper
                    self.levels[edge[1]].update_excitation_energy(self.g.nodes[edge[1]]['energy'])
                    eAssigned.append(edge[1]) # Add to list of assigned Level energies
                else: # Either both level energies have been assigned already or neither have
                    continue
            iteration+=1
            if iteration>maxIter:
                print('ERROR: Total number of iterations over graph has exceeded max allowed.')
                print('Total number of nodes: ',self.g.number_of_nodes())
                print('Number of assigned level energies: ',len(eAssigned))
                break
                
    def merge_redundant_roots(self,energy_threshold):
        
        updatedEdges = [] # List containing updated edges after nodes have been labeled as redundant
        gEnergies = []    # List of gamma energies that should be assigned to the new edges
        gWeights = [] # List of gamma intensities that should be assigned to the new edges
        
        for i in range(len(self.rootNodes)):         
    
            if self.rootNodes[i] in self.redundantNodes:
                continue
            else:
                for j in range(i+1,len(self.rootNodes)):
                    if self.rootNodes[j] in self.redundantNodes:
                        continue
                    elif abs(self.levels[self.rootNodes[i]].ExE-self.levels[self.rootNodes[j]].ExE)<energy_threshold:
                    #elif abs(ls.nodes[rootNodes[i]]['energy']-ls.nodes[rootNodes[j]]['energy'])<energy_threshold:
                        #savedNodes.append(rootNodes[i])
                        self.redundantNodes.append(rootNodes[j])
                        for gamma in self.levels[self.rootNodes[j]].outGammas:
                            self.levels[self.rootNotes[i]].outGammas.append(gamma)
                            #for gamma in outGammas[rootNodes[j]]:
                            #outGammas[rootNodes[i]].append(gamma)
                        for edge in self.g.edges:
                            if edge[0]==rootNotes[j]:
                                self.redundantEdges.append(edge)
                                gamEn=self.g.edges[edge]['energy']
                                gEnergies.append(gamEn)
                                gamBR=self.g.edges[edge]['weight']
                                gWeights.append(gamBR)
                                newEdge=(rootNodes[i],edge[1])
                                updatedEdges.append(newEdge)
        
        g=0 # Index counter for gamma-ray energies/branching ratios
        for e in updatedEdges:
            self.g.add_edge(e[0],e[1],energy=gEnergies[g],weight=gWeights[g])
            g+=1
    
    # TODO write a cached_property resetter to update this whenever g is modified
    @cached_property
    def adj(self):
        A = np.zeros((self.nl, self.nl))
        for n in self.g:
            for nbr, datadict in self.g.adj[n].items():
                A[n, nbr] = datadict['weight']

        return A    

    def _add_out_transitions(self):

        # Add one transition from every level (except ground state)
        for i in reversed(range(1, self.nl)):
            nd = rnd.randint(i)

            # If energy levels of multiple nodes are equal,
            # we don't want to add a transition. Instead, we add
            # a transition down to the next highest energy
            while self.g.nodes[nd]['energy'] == self.g.nodes[i]['energy']:
                nd -= 1

            # Add an edge (transition) from high energy to low.
            # If this edge already exists, does nothing.
            self.g.add_edge(i, nd)

    def _add_in_transitions(self):

        # For every energy level that doesn't have a transition
        # into it from a higher energy, add a transition (except
        # for highest energy)
        for i in range(self.nl-1):
            if self.g.in_degree(i) == 0:
                nd = rnd.randint(i, high=self.nl)

                # Increase edge source until the source energy 
                # is greater than the energy at node 'i'
                while self.g.nodes[nd]['energy'] == self.g.nodes[i]['energy']:
                    nd += 1

                # Add edge from nd to i. If this edge already exists, 
                # this does nothing.
                self.g.add_edge(nd, i)
            
    def _add_rand_transitions(self):

        while self.g.number_of_edges() < self.nt:
            w = rnd.uniform(low=0.1, high=0.9)
            n1 = rnd.randint(self.nl-1)
            n2 = rnd.randint(self.nl-1)
            while n1 == n2:
                n2 = rnd.randint(self.nl-1)

            # Use elif statement so that no edge is added 
            # for equal energies
            if (self.g.nodes[n1]['energy'] > self.g.nodes[n2]['energy']):
                self.g.add_edge(n1, n2, weight=w)
            elif (self.g.nodes[n1]['energy'] < self.g.nodes[n2]['energy']):
                self.g.add_edge(n2, n1, weight=w)

    def _add_branch_probs(self):

        # Assign branching probabilities
        for n in self.g.nodes:
            probs_unnormalized = rnd.uniform(low=0.1, high=0.9, size=self.g.out_degree(n))
            probs = probs_unnormalized / np.sum(probs_unnormalized)

            for e, w in zip(self.g.out_edges(n), probs):
                self.g[e[0]][e[1]]['weight'] = w

    def make_random(self, num_levels, num_trans, EMax):

        self.nl = num_levels
        self.nt = num_trans
        self.emax = EMax

        # Create Nl different energy levels, with one guaranteed
        # to be 0. Then sort in ascencing order.
        energies = rnd.randint(0.01*EMax, high=EMax, size=self.nl-1)
        energies = np.append(energies, 0)
        energies = np.sort(energies)

        # Force the highest energy level to be unique
        if energies[-1] == energies[-2]:
            energies[-1] *= 1.2

        # Add each energy level as a node to the level graph
        for i, en in zip(range(self.nl), energies):
            self.g.add_node(i, energy = en)
            
        self._add_in_transitions()
        self._add_out_transitions()
        self._add_rand_transitions()
        self._add_branch_probs()
        
        return self.g


    def add_transition(self, initial, final, br):

        #if (self.g.nodes[n1]['energy'] > self.g.nodes[n2]['energy']):
        self.g.add_edge(initial,final, weight=br)
        self.nt=self.g.number_of_edges()
        #print('Initial: ',initial,'; Final: ',final)
        
        #maxE = max(outGammas)
        #self.g.add_node(self.nl,maxE)
        
# ###################################################### # 
class TransitionScheme:
    
    def __init__(self):

        self.nt = 0           # Total number of transitions in scheme
        self.g = nx.DiGraph() # Directed graph object used to represent TransitionScheme
        self.adjDict={}       # Dictionary of adjacent gamma transitions
        self.nodeDict={}      # Dictionary where key is Node # and definition is Gamma object 
        self.leafNodes=[]     # List of all leaf nodes in the TransitionScheme
        self.branchNodes=[]   # List of all branch nodes in the TransitionScheme
        self.gamEnergies=[]   # List of all gamma-ray energies; might replace this with list of Gamma objects
        self.allPaths=[]      # List of all possible transition pathways between two nodes in gamma cascade

    def __str__(self):

        return dedent(f"""
            Transition Scheme with {self.nt} transitions.""")
    
    def build_from_adjacency_matrix(self,adjMatrix,intensity_threshold,gammas):
        
        self.gamEnergies=gammas
        for i in range(len(adjMatrix[0])): # Loop over rows in adjacency matrix
            leaf=True # Is this a leaf node?
            for j in range(len(adjMatrix[0])): # Loop over columns in adjacency matrix
                
                if adjMatrix[i][j]<=intensity_threshold: # Matrix element not above threshold
                    continue # Keep looping
                else: # Adjacent transitions found!
                    leaf=False
                    if gammas[i].gE not in self.adjDict: # Gamma not placed in transition scheme yet
                        self.branchNodes.append(gammas[i].gE)  # Add to list of branch nodes
                        self.adjDict.update({gammas[i].gE:[]}) # Add gamma energy to dictionary of adjacent transitions
                    self.adjDict[gammas[i].gE].append(gammas[j].gE) # Add jth gamma to ith gamma's list of adjacent transitions
                    self.g.add_node(gammas[i].gE,Gamma=gammas[i]) # Add ith node to directed graph of TransitionScheme
                    self.g.add_node(gammas[j].gE,Gamma=gammas[j]) # Add jth node to directed graph of TransitionScheme
                    self.g.add_edge(gammas[i].gE,gammas[j].gE,weight=adjMatrix[i][j])
            if leaf==True: # If after looping over all columns leaf is still true...
                self.leafNodes.append(gammas[i].gE) # Add to list of leaf nodes
                self.adjDict.update({gammas[i].gE:[]}) # Add gamma to dictionary
                self.g.add_node(gammas[i].gE,Gamma=gammas[i]) # Add ith node to directed graph of TransitionScheme
        print(self.adjDict)
        
    def find_all_paths(self,source,destination):
        # Clear previously stored paths
        path = []
        path.append(source)
        #print("Source : " + str(src) + " Destination : " +  str(dst))

        # Use depth first search (with backtracking) to find all the paths in the graph
        self.depth_first_search(source,destination,path)

        # Print all paths
        #self.Print ()
        
    def print_paths(self):
        # print (self.allpaths)
        for path in self.allPaths:
            print("Path : " + str(path))
        #self.allpaths.clear()
        
    # This function uses DFS at its core to find all the paths in a graph
    #def DFS (self, adjlist : Dict[int, List[int]], src : int, dst : int, path : List[int]):
    def depth_first_search(self,source,destination,path):
        if source==destination:
            self.allPaths.append(copy.deepcopy(path))
        else:
            for adjNode in self.adjDict[source]:
                path.append(adjNode)
                self.depth_first_search(adjNode,destination,path)
                path.pop()  

    @cached_property
    def adj(self):
        A = np.zeros((self.nt, self.nt))
        for n in self.g:
            for nbr, datadict in self.g.adj[n].items():
                A[self.g.nodes[n]['id'], self.g.nodes[nbr]['id']] = datadict['weight']

        return A   


    @classmethod
    def from_level_scheme(cls, lsc):
        new_tsc = TransitionScheme()
        for n in lsc.g:
            for nbr, datadict in lsc.g.adj[n].items():
                new_tsc.g.add_node((n, nbr), id=new_tsc.nt)
                new_tsc.nt += 1

        for u,v in new_tsc.g:
            for src, dest in lsc.g.out_edges(v):
                w = lsc.g[src][dest]['weight']
                new_tsc.g.add_edge((u,v), (src, dest), weight=w)
        
        return new_tsc