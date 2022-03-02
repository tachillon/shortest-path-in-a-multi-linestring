#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# License: © 2022 Achille-Tâm GUILCHARD All Rights Reserved
# Author: Achille-Tâm GUILCHARD
# Usage: python3 dikjstra.py

import os
import sys
import json
import uuid
import time
import folium
import random
from termcolor import colored
from geojson import MultiLineString, Point
from math import radians, cos, sin, asin, sqrt, atan2

class Graph(object):
    def __init__(self, nodes, init_graph):
        self.nodes = nodes
        self.graph = self.construct_graph(nodes, init_graph)
        
    def construct_graph(self, nodes, init_graph):
        '''
        This method makes sure that the graph is symmetrical. In other words, if there's a path from node A to B with a value V, there needs to be a path from node B to node A with a value V.
        '''
        graph = {}
        for node in nodes:
            graph[node] = {}
        
        graph.update(init_graph)
        
        for node, edges in graph.items():
            for adjacent_node, value in edges.items():
                if graph[adjacent_node].get(node, False) == False:
                    graph[adjacent_node][node] = value
                    
        return graph
    
    def get_nodes(self):
        "Returns the nodes of the graph."
        return self.nodes
    
    def get_outgoing_edges(self, node):
        "Returns the neighbors of a node."
        connections = []
        for out_node in self.nodes:
            if self.graph[node].get(out_node, False) != False:
                connections.append(out_node)
        return connections
    
    def value(self, node1, node2):
        "Returns the value of an edge between two nodes."
        return self.graph[node1][node2]

def dijkstra_algorithm(graph, start_node):
    unvisited_nodes = list(graph.get_nodes())
 
    # We'll use this dict to save the cost of visiting each node and update it as we move along the graph   
    shortest_path = {}
 
    # We'll use this dict to save the shortest known path to a node found so far
    previous_nodes = {}
 
    # We'll use max_value to initialize the "infinity" value of the unvisited nodes   
    max_value = sys.maxsize
    for node in unvisited_nodes:
        shortest_path[node] = max_value
    # However, we initialize the starting node's value with 0   
    shortest_path[start_node] = 0
    
    # The algorithm executes until we visit all nodes
    while unvisited_nodes:
        # The code block below finds the node with the lowest score
        current_min_node = None
        for node in unvisited_nodes: # Iterate over the nodes
            if current_min_node == None:
                current_min_node = node
            elif shortest_path[node] < shortest_path[current_min_node]:
                current_min_node = node
                
        # The code block below retrieves the current node's neighbors and updates their distances
        neighbors = graph.get_outgoing_edges(current_min_node)
        for neighbor in neighbors:
            tentative_value = shortest_path[current_min_node] + graph.value(current_min_node, neighbor)
            if tentative_value < shortest_path[neighbor]:
                shortest_path[neighbor] = tentative_value
                # We also update the best path to the current node
                previous_nodes[neighbor] = current_min_node
 
        # After visiting its neighbors, we mark the node as "visited"
        unvisited_nodes.remove(current_min_node)
    
    return previous_nodes, shortest_path

def print_result(previous_nodes, shortest_path, start_node, target_node):
    path       = []
    node       = target_node
    color_list = ['grey', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']

    while node != start_node:
        path.append(node)
        node = previous_nodes[node]
 
    # Add the start node manually
    path.append(start_node)
    
    print("We found the following best path with a value of {} kms.".format(shortest_path[target_node]/1000.0))
    print(" -> ".join(reversed(path)))
    return reversed(path)

def read_json(path):
    """Read json and store data in a dictionnary"""

    # Check if file ends with .json
    ext = os.path.splitext(path)[-1].lower()
    if ext != ".json":
        raise TypeError("The file is not a .json file!")

    with open(path) as json_file:
        data = json.load(json_file)
        return data

# Calculates distance between 2 GPS coordinates
def haversine(lat1, lon1, lat2, lon2):
    R = 6373.0 # Radius of Earth in km
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance * 1000 # Return distance in meters

def findExtremPointsOfMyWayIdentifier(listOfLinesStrings):
    print(colored("Finding extrem GPS points of the multi linestrings (may take some times because O(n2) and not optimized at all but feel free to do it yourself...)", 'red'))
    list_of_gps_points = list()

    maxDist = -10000000

    for linestring in listOfLinesStrings:
        for coordinate in linestring:
            longitude = coordinate[0]
            latitude  = coordinate[1]
            list_of_gps_points.append([latitude, longitude])

    for x in list_of_gps_points:
        for y in list_of_gps_points:
            if x!=y:
                latitude   = x[0]
                longitude  = x[1]

                latitude2  = y[0]
                longitude2 = y[1]

                dist = haversine(latitude, longitude, latitude2, longitude2)

                if dist > maxDist:
                    mostDistantPoints = [x, y]
                    maxDist = dist

    return mostDistantPoints

def findLinestringUUIDOfPoint(givenPoint, matchPointToLinestrings):
    for element in matchPointToLinestrings:
        latitude  = givenPoint[0]
        longitude = givenPoint[1]

        if latitude == element["latitude"] and longitude == element["longitude"]:
            return element["uuid_of_linestring"]

def findClosestPointsToAnotherPointAndReturnItsUuidOfLinestrings(point, matchPointToLinestrings, selfUUID):
    latitude          = point[0]
    longitude         = point[1]
    minimumDistance   = 50 # in meters
    UUID_to_return    = list()

    for element in matchPointToLinestrings:
        latitude2  = element["latitude"]
        longitude2 = element["longitude"]
        linestring_uuid = element["uuid_of_linestring"]
        if linestring_uuid != selfUUID:
            dist = haversine(latitude, longitude, latitude2, longitude2)
            if dist < minimumDistance:
                toAdd = (element["uuid_of_linestring"], dist)
                UUID_to_return.append(toAdd)

    listeUnique = list(set(UUID_to_return))

    # Just keep the 3 closest linestring...
    if len(listeUnique) > 3:
        listeUnique.sort(key = lambda x: x[1])
        toReturn = list()
        for element in listeUnique[0:3]:
            toReturn.append(str(element[0]))
        return toReturn
    elif len(listeUnique) > 0 and len(listeUnique) < 3:
        return [str(listeUnique[0][0])]
    else:
        return []

def findClosestLinestrings(uuid_of_linestring, matchPointToLinestrings, linestringsWithCorrespondingPoints):
    # On prend les points terminaux de la linestring d'UUID uuid_of_linestring
    pt1     = linestringsWithCorrespondingPoints[uuid_of_linestring][0]
    pt2     = linestringsWithCorrespondingPoints[uuid_of_linestring][-1]

    result1 = findClosestPointsToAnotherPointAndReturnItsUuidOfLinestrings(pt1, matchPointToLinestrings, uuid_of_linestring)
    result2 = findClosestPointsToAnotherPointAndReturnItsUuidOfLinestrings(pt2, matchPointToLinestrings, uuid_of_linestring)

    if len(result1) + len(result2) == 0:
        print("Caramba...") # TODO Peut-être que ça se passe mal dans ce cas là... Vérifier les conséquences.
        return []

    result = list()
    for element in result1:
        result.append(element)
    for element in result2:
        result.append(element)
            
    result = list(set(result))
    
    return result

def drawPathOnMapAndGenerateMultiLineStringResult(folium_initial_location, path, matchPointToLinestrings, linestringsWithCorrespondingPoints):
    m  = folium.Map(location=folium_initial_location, zoom_start=15)
    m2 = folium.Map(location=folium_initial_location, zoom_start=15)
    final_linestring = list()
    geosjon_multi_linestrings = list()
    for p in path:
        randomHexadecimalColor   = "%06x" % random.randint(0, 0xFFFFFF)
        lineString = list()
        geojson_linestring = []
        for point in linestringsWithCorrespondingPoints[p]:
            latitude  = point[0]
            longitude = point[1]
            folium.CircleMarker(
                        location=[latitude, longitude], radius=3.0,
                        color="#" + str(randomHexadecimalColor),
                        fill=True,
                        fill_color="#" + str(randomHexadecimalColor),
                    ).add_to(m)
            lineString.append([latitude, longitude])
            final_linestring.append([latitude, longitude])

            # Create output .geojson
            a_tuple = (longitude,latitude)
            geojson_linestring.append(Point(a_tuple))
        geosjon_multi_linestrings.append(geojson_linestring)

        folium.vector_layers.PolyLine(lineString, color="#" + str(randomHexadecimalColor), weight=5).add_to(m)
    folium.vector_layers.PolyLine(final_linestring, color="red", weight=5).add_to(m2)

    current_path = os.getcwd()

    print(colored("\nThe output .geojson file is located here: " + current_path + "/result.geojson.", 'green'))
    with open('./result.geojson', 'w') as outfile:
        json.dump(MultiLineString(geosjon_multi_linestrings), outfile, indent=4, sort_keys=True)

    print(colored("To visualize the path generated, click on path_multilinestring.html, located in " + current_path + ".", 'green'))
    m.save("path_multilinestring" + ".html")

    print(colored("To visualize the path generated as a long linestring, click on path_monolinestring.html, located in " + current_path + ".", 'green'))
    m2.save("path_monolinestring" + ".html")
    
def main():
    t_start = time.perf_counter()
    wayjsonpath                        = "./multilinestring.json"
    wayjson_info                       = read_json(wayjsonpath)
    listOfLinesStrings                 = wayjson_info["coordinates"]
    matchPointToLinestrings            = list()
    matchLenghtToLinestrings           = {}
    linestringsWithCorrespondingPoints = {}
    DEBUG                              = False
    folium_initial_location            = [listOfLinesStrings[0][0][1], listOfLinesStrings[0][0][0]]
    m = folium.Map(location=folium_initial_location, zoom_start=15)
    count    = 0
    for ls in listOfLinesStrings:
        lineString = []
        points = []
        lineStringLength = 0
        lineString_uuid = str(count)
        randomHexadecimalColor = "%06x" % random.randint(0, 0xFFFFFF)
        for c in ls:
            longitude = c[0]
            latitude  = c[1]
            lineString.append([latitude, longitude])
            folium.CircleMarker(
                            location=[latitude, longitude], radius=3.0,
                            color="#" + str(randomHexadecimalColor),
                            fill=True,
                            fill_color="#" + str(randomHexadecimalColor),
                            popup=str(count),
                        ).add_to(m)

            matchPointToLinestrings.append({"uuid_of_point": str(uuid.uuid4()), "latitude": latitude, "longitude": longitude, "uuid_of_linestring": lineString_uuid, "lenght_of_linestring":0})

        linestringsWithCorrespondingPoints[lineString_uuid] = lineString

        for i in range(0, len(lineString)-1):
            point1 = lineString[i]
            point2 = lineString[i + 1]
            lineStringLength += haversine(point1[1], point1[0], point2[1], point2[0])

        matchLenghtToLinestrings[lineString_uuid] = lineStringLength

        # folium.vector_layers.PolyLine(lineString, color="#" + str(randomHexadecimalColor), weight=5).add_to(m)
        count = count + 1

    for element in matchPointToLinestrings:
        element["lenght_of_linestring"] = matchLenghtToLinestrings[element["uuid_of_linestring"]]

    # On trouve les 2 points GPS de la multi-linestring qui sont les plus éloignés, et on regarde à quelle linestring ils appartiennent
    mostDistantPoints = findExtremPointsOfMyWayIdentifier(listOfLinesStrings)

    folium.CircleMarker(location=[mostDistantPoints[0][0], mostDistantPoints[0][1]], radius=10,
                        color="red",
                        fill=True,
                        fill_color="red",
                        ).add_to(m)
    folium.CircleMarker(location=[mostDistantPoints[1][0], mostDistantPoints[1][1]], radius=10,
                        color="red",
                        fill=True,
                        fill_color="red",
                        ).add_to(m)

    m.save("res" + ".html")

    if DEBUG:
        with open("AllPoints.json", 'w') as outfile:
            json.dump(matchPointToLinestrings, outfile, indent=4, sort_keys=True)   

        with open("linestringDistance.json", 'w') as outfile:
            json.dump(matchLenghtToLinestrings, outfile, indent=4, sort_keys=True) 

    # On remplit les vertex du graphe
    nodes = list()

    for element in matchLenghtToLinestrings: # Pour toutes les linestrings
        nodes.append(element)

    # On intialise le dictionnaire
    init_graph = {}
    for node in nodes:
        init_graph[node] = {}

    # On connecte les vertex du graphe entre eux
    for node in nodes:
        # Pour chaque vertex (ici linestring) on cherche sa linestring la plus proche
        result = findClosestLinestrings(node, matchPointToLinestrings, linestringsWithCorrespondingPoints)
        alreadyAddedList = list()  
        for r in result:
            if node != r:
                if node not in alreadyAddedList:
                    if r not in alreadyAddedList:
                        # On va de la linestring 1 à la linestring 2 par exemple, et la distance c'est la somme de longueur de la linestring 1 + la linestring 2
                        init_graph[node][r] = matchLenghtToLinestrings[node] + matchLenghtToLinestrings[r]
                        # init_graph[r][node] = matchLenghtToLinestrings[node] + matchLenghtToLinestrings[r]
                        alreadyAddedList.append(r)
                        if DEBUG:
                            print("{} <-> {} ({})".format(node, r, matchLenghtToLinestrings[node] + matchLenghtToLinestrings[r]))    
        if DEBUG:
            print("")

    graph                         = Graph(nodes, init_graph)
    previous_nodes, shortest_path = dijkstra_algorithm(graph=graph, start_node=findLinestringUUIDOfPoint(mostDistantPoints[0], matchPointToLinestrings))
    path                          = print_result(previous_nodes, shortest_path, start_node=findLinestringUUIDOfPoint(mostDistantPoints[0], matchPointToLinestrings), target_node=findLinestringUUIDOfPoint(mostDistantPoints[1], matchPointToLinestrings))
    drawPathOnMapAndGenerateMultiLineStringResult(folium_initial_location, path, matchPointToLinestrings, linestringsWithCorrespondingPoints)
    t_end = time.perf_counter()
    elapsed = (t_end - t_start)
    print(colored("Elapsed time = {:.3f} seconds".format(elapsed), 'blue'))

if __name__ == "__main__":
    main()