# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import time
import networkx as nx
import joblib

def probabilitiesMoves(a,s,s_norm,i_c,j_e,g_c,g_min,g_max,tabu,tau, alpha, beta, dist_matrix):
    # Bepaal naar welke knooppunten een ant kan bewegen en wat de kans is voor elk knooppunt
    # i_c: huidige node van ant
    # j_e: eindpunt
    # g_c: afstand tot nu toe
    # g_min/gmax: minimale en maximale afstand
    # tabu: punten die al bezocht zijn
    
    possibleMoves = np.where(a[:,i_c]<1000)[0]

    probabilityMoves = np.zeros(len(possibleMoves))
    
    for i in range(len(possibleMoves)):
        j = possibleMoves[i]
        if j in tabu: # al bezocht 
            probabilityMoves[i] = 0
        elif dist_matrix[j] + g_c + a[i_c,j] > g_max: # te lange route tot eindpunt
            probabilityMoves[i] = 0
        elif j == j_e and g_c + a[i_c,j] < g_min: # te kort om te stoppen
            probabilityMoves[i] = 0
        else:
            probabilityMoves[i] = tau[i_c,j]**alpha * ((s_norm[i_c,j]+0.1)/a[i_c,j])**beta

    if sum(probabilityMoves) > 0:
        probabilityMoves /= sum(probabilityMoves)

    return possibleMoves, probabilityMoves


def selectMove(probabilities):
    # trek random item aan de hand van kansverdeling
    
    rand = np.random.rand(1)
    
    for i in range(len(probabilities)):
        if sum(probabilities[:(i+1)]) > rand:
            index = i 
            break
    
    return index

def distanceScoreRoute(a,s,solution):
    # Bereken de adstand en score gegeven een route
    
    distance = 0
    score = 0 
    for i in range(len(solution)-1):
        distance += a[solution[i], solution[i+1]]
        score += s[solution[i], solution[i+1]]
        
    if distance >0:
        score /= distance

    return distance, score
	
def looproutes_ant_colony_optimization(a,s,s_norm,i_s,j_e,g_min,g_max,n_ants = 200,n_iter = 12,alpha = 0.5,beta =0.5,rho = 0.5,Q3 = 1,paths = []):
    # ant colony optimization om looproutes te bepalen
    n = a.shape[0]
    
    start_time = time.time()
    # Kortste path van elke route naar eindpunt
    dist_matrix = shortest_path(csgraph=csr_matrix(a), directed=False, indices=j_e)

    # initialisatie tau
    # Voor alle i,j zet tau_ij(0) op 1

    tau = np.ones((n,n))
    #"""
    for i in range(len(paths)):
        _,score = distanceScoreRoute(a,s,paths[i])
        for j in range(len(paths[i])-1):
            tau[paths[i][j],paths[i][j+1]] += Q3*score
    #"""

    # beste route bewaren
    opt_route = []
    opt_value = 0
    opt_distance = 0

    # algoritme
    for i in range(n_iter):
        # Zorg dat voor alle ants bijgehouden wordt bij welke nodes ze al zijn geweest.
        # Ze starten allemaal in het startpunt.
        tabu = []

        current_length = np.zeros(n_ants)
        current_value = np.zeros(n_ants)
        value_ant = np.zeros(n_ants)

        for k in range(n_ants):
            tabu.append([i_s])

        # aantal ants dat eindpunt hebben bereikt
        end = 0

        # alle mieren bewegen over netwerk tot ze eindpunt bereiken of niet meer verder kunnen.
        while end < n_ants:  
            for k in range(n_ants):
                if tabu[k][-1] != j_e:
                    possibleMoves, probabilityMoves = probabilitiesMoves(a,s,s_norm,tabu[k][-1],j_e,current_length[k],g_min,g_max,tabu[k],tau, alpha, beta,dist_matrix)
                    if sum(probabilityMoves) > 0:
                        move = possibleMoves[selectMove(probabilityMoves)]
                        current_length[k] += a[tabu[k][-1],move]
                        current_value[k] += s[tabu[k][-1],move]
                        tabu[k].append(move)
                        if move == j_e:
                            value_ant[k] = current_value[k]/current_length[k]
                            end += 1
                    else:
                        tabu[k].append(j_e)
                        value_ant[k] = 0
                        end += 1
        # update tau   
        tau *= rho
        for k in range(n_ants):
            for i in range(len(tabu[k])-1):
                tau[tabu[k][i],tabu[k][i+1]] += Q3*value_ant[k]

        # beste route bewaren:
        if max(value_ant) > opt_value:
            opt_route = tabu[np.argmax(value_ant)]
            opt_value = max(value_ant)
            opt_distance = current_length[np.argmax(value_ant)]

    runtime = time.time()-start_time
    
    return opt_route, opt_distance, opt_value, runtime
