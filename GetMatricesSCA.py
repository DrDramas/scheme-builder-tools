#! /bin/env python

#from Gamma import Gamma
#from Gamma import Level
#from Graph import Graph
from NuclearObjects import LSGraph
from NuclearObjects import Gamma
from NuclearObjects import Level

# at the moment this:
#   (1) reads in gamma-ray file
#   (2) creates gamma-ray objects (Gamma class)
#   (3) creates level-objects (Level class)
#   (4) determines which gammas are in coincidene and makes coincidence table

gammas = []
levels = []
gammaIDs={}
levelIDs={}
vertices = {}

# ###################################################### #
def MakeLevelsAndVertices(fname):
# ###################################################### #

# setup st.cmd for writing

    #fname = "Ta182_beta.gam"   
    #fname = "../Am241.gam"
    content = readin_ascii(fname)
 
  # Create Gamma objects

    for line in content:
        data = line.split()
        if len(data) > 0:
            try:
                tmp = float(data[0])
                gammas.append(Gamma(line))
                gamma.list_values()
                print(gamma.list_values())
            except:
                x = 0.0    
    for i in range(len(gammas)):gammaIDs.update({gammas[i].gE:i})

    

  # Create Level objects

    Make_Levels()
    Add_Ghost_Levels()
    for i in range(len(levels)):levelIDs.update({levels[i].ExE:i})


  # Find which gamma-rays are in coincidence
  
  # step 1: create a vertex dictionary and use it to create a Graph Object
    
    Make_Vertices() 
    #Glevel = Graph(vertices)
    Glevel = LSGraph(vertices)
  # add groundstate to vertex library.
    Glevel.add_vertex(0.0) 
    #print(Glevel)
    
    return Glevel

# ###################################################### #
def GetGammaEnergies():
# ###################################################### #

    gammaEnergies=[]
    for gamma in gammas:
        gammaEnergies.append(gamma.gE)
    return gammaEnergies

# ###################################################### #
def GetGammaObjects():
# ###################################################### #

    return gammas

# ###################################################### #
def GetSingles(nc=1): # Takes normalization constant (nc) as an argument to scale total number of counts
# ###################################################### #
    
    # Singles spectrum array
    S = [] #[len(gammaIDs)]
    for gamma in gammas:
        S.append(gamma.RI*nc)
        
    return S

# ###################################################### #
def GetCoincidences(Glevel,nc=1):
# ###################################################### #

   # Now define the coincidence 2D and make all elemets = 0
    
    gam_coinc_arr = [[0 for i in range(len(gammas))] for j in range(len(gammas))]

  # step 2: for each level, extract all possible paths
  # step 3: turn level list into gamma transition list for each path
  # step 4: take gamma list and mark 1 in coincidence matrix for each γ-γ

    ggc=0
    endpoints={}
    for level in levels:
    # step 2
        #print('In levels loop... ',level.ExE)
        paths = Glevel.find_all_paths(level.ExE,0.0) # Find all paths from ExE to gs.
    #print("# of Paths for level",level.ExE,"=",len(paths))
        for path in paths:
      # Step 3
            #print('In paths loop... ')
            if(len(path) > 2): # need at least 3 levels to define γ-γ
                glst = []
                genergies = []
                for i in range(1,len(path)):
                    id = levelIDs[path[i-1]]
                    for gam in levels[id].outGammas:
                        if gam.ExB == path[i]: #glst.append(gam.gE)
                            glst.append(gam)
                            genergies.append(gam.gE)
        # Step 4
                #print('Gamma list(len): ',genergies,len(genergies))
                for i in range(len(glst)-1):
                    for j in range(i+1,len(glst)):
                        if j==i+1: # If i, j are adjacent in the pathway...
                            ggc=glst[i].RI*glst[j].BR # Multiply relative intensity of ith gamma by branching ratio of jth
                        else:
                            ggc=ggc*glst[j].BR # Continue to multiply branching ratios of subsequent jth gammas
                        # Check to see if this matrix element C[i][j] has been populated before
                        start_stop=(glst[i].gE,glst[j].gE) # Tuple of including the initial, final gammas in sequence
                        #subpath=glst[i:j+1] # This doesn't work because elements of glst are Gamma objects, not float(energies)             
                        subpath=[] # Gamma-ray pathway from start to stop gammas
                        for k in range(i,j+1):
                            subpath.append(glst[k].gE)
                        if start_stop not in endpoints: # If these are new start/stop points
                            endpoints.update({start_stop:[]}) # Update the dictionary
                            endpoints[start_stop].append(subpath) # Add subpath to list of possible pathways
                        else: # These two end points have already been considered
                            if subpath in endpoints[start_stop]: # If this subpath is already recorded
                                continue # Redundant pathway; go to next gamma in sequence
                            else: # New subpath that leads from start gamma to stop gamma
                                endpoints[start_stop].append(subpath)
                        # Get index of coincidence matrix based on gamma energies
                        k=gammaIDs[glst[i].gE]
                        l=gammaIDs[glst[j].gE]
                        gam_coinc_arr[k][l]+=ggc*nc # Add coincidences from this unique path to matrix element
                        gam_coinc_arr[l][k]=gam_coinc_arr[k][l]


            
  # Now output the coincidence matrix with rows and columnds marked by Egam    

    #print(gammaIDs.keys())
    #print('gammaIDs:')
    #for i in range(0,len(gammaIDs)):
    #    print(gammas[i].gE,gam_coinc_arr[0:len(gammaIDs)][i])
    #for i in range(0,len(gammas)):
    #    for j in range(0,len(gammas)):
    #        print(gam_coinc_arr[i][j])
    
    return gam_coinc_arr

# ###################################################### #
def GetAdjacency():
# ###################################################### #

    A = [[0 for i in range(len(gammas))] for j in range(len(gammas))] # Populate adjacency matrix with all zeros

    for level in levels:
        for inGamma in level.outGammas: # Loop over all gammas emitted from initial level
            i = gammaIDs[inGamma.gE] # Get row index of incoming gamma based on energy
            #print('i, inGamma.gE: ',i,inGamma.gE)
            #print('Initial level energy, Level ID: ',inGamma.ExT,levelIDs[inGamma.ExT])
            #print('Final level energy, Level ID: ',inGamma.ExB,levelIDs[inGamma.ExB])
            fl = levelIDs[inGamma.ExB] # Get index of final level after transition of initial level
            #print('fl: ',fl)
            for outGamma in levels[fl].outGammas: # Loop over all gammas emitted from final level
                j = gammaIDs[outGamma.gE] # Get column index of outgoing gamma based on energy
                A[i][j] = outGamma.BR # Set A matrix element equation to branching ratio of adjacent gamma
                
    return A

# ###################################################### #
def readin_ascii(dbname):
# ###################################################### #
  try:
    dbfile = open(dbname,"r")
    #print dbname
  except StandardError:
    print('no file to open')
    exit()


  content = dbfile.readlines()
  return content


#comment="""

# ###################################################### #
def Make_Levels():
# ###################################################### #

    levels.append(Level(0.0))
    for gamma in gammas:
        ExE = gamma.ExT
        i=0
        for level in levels:
            if ExE == level.ExE: i=1
      
        if (i==0): levels.append(Level(ExE))
      
        for level in levels: 
            if ExE == level.ExE: 
                level.add_outgoing_gamma(gamma)
  
    # Now computer branching ratios for all levels 
    for level in levels:
        if level.ExE>0:
            level.compute_BRs()
    
    return

#"""


# ###################################################### #
def Add_Ghost_Levels():
# ###################################################### #

    for gamma in gammas:
        ghost=True                     # Assume there might be a level populated by gamma decay that does not emit a gamma
        for level in levels:
            if gamma.ExB == level.ExE: # Found this gamma-emmitting level!
                ghost=False           
                break                       # Break out of loop over levels
        if ghost == True:                   # If this level isn't found among those placed...
            print('BOO!')
            levels.append(Level(gamma.ExB)) # Add a new level to the scheme
    
    return 

# ###################################################### #
def Make_Vertices():
# ###################################################### #
# This makes dictionary which levels are conected by gammas
# Stored in dictionary as iinput to the Graph class. 

  for level in levels:
    
    vlist = []
    ExE = level.ExE
    for gamma in level.outGammas:
      vlist.append(gamma.ExB)
      vertices.update({ExE:vlist}) 

  #print(vertices)    
  return

# ###################################################### #
def Print_Level_Scheme():
# ###################################################### # 

    for level in levels:
        print('Level: ',level.ExE)
        for gamma in level.outGammas:
            print('Gamma: ',gamma.gE,' Branching ratio: ',gamma.BR,' Final state: ',gamma.ExB)

    return