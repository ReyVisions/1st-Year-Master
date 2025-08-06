import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import heapq
import time
import pandas as pd

"""
Heuristique de l'ensemble des 20 villes a partir de chaque ville. (Pour l'algorithme A*)
Genere par IA pour des soucis de simplification.
"""

heuristics = {
    "Paris": {
        "Paris": 0.0, "Lille": 225.0, "Le Havre": 197.0, "Brest": 590.0, "Rennes": 350.0, "Nantes": 385.0,
        "Angers": 295.0, "Bordeaux": 585.0, "Toulouse": 680.0, "Montpellier": 750.0, "Marseille": 775.0,
        "Nice": 930.0, "Toulon": 870.0, "Lyon": 465.0, "Saint-Etienne": 510.0, "Grenoble": 560.0,
        "Dijon": 315.0, "Strasbourg": 490.0, "Metz": 330.0, "Reims": 145.0
    },
    "Lille": {
        "Paris": 225.0, "Lille": 0.0, "Le Havre": 280.0, "Brest": 680.0, "Rennes": 480.0, "Nantes": 550.0,
        "Angers": 500.0, "Bordeaux": 790.0, "Toulouse": 870.0, "Montpellier": 930.0, "Marseille": 970.0,
        "Nice": 980.0, "Toulon": 1020.0, "Lyon": 650.0, "Saint-Etienne": 700.0, "Grenoble": 750.0,
        "Dijon": 490.0, "Strasbourg": 460.0, "Metz": 350.0, "Reims": 170.0
    },
    "Le Havre": {
        "Paris": 197.0, "Lille": 280.0, "Le Havre": 0.0, "Brest": 450.0, "Rennes": 290.0, "Nantes": 370.0,
        "Angers": 320.0, "Bordeaux": 620.0, "Toulouse": 780.0, "Montpellier": 850.0, "Marseille": 890.0,
        "Nice": 910.0, "Toulon": 950.0, "Lyon": 570.0, "Saint-Etienne": 620.0, "Grenoble": 680.0,
        "Dijon": 420.0, "Strasbourg": 540.0, "Metz": 480.0, "Reims": 280.0
    },
    "Brest": {
        "Paris": 590.0, "Lille": 680.0, "Le Havre": 450.0, "Brest": 0.0, "Rennes": 240.0, "Nantes": 300.0,
        "Angers": 340.0, "Bordeaux": 590.0, "Toulouse": 800.0, "Montpellier": 890.0, "Marseille": 950.0,
        "Nice": 1260.0, "Toulon": 1030.0, "Lyon": 820.0, "Saint-Etienne": 870.0, "Grenoble": 930.0,
        "Dijon": 750.0, "Strasbourg": 950.0, "Metz": 890.0, "Reims": 650.0
    },
    "Rennes": {
        "Paris": 350.0, "Lille": 480.0, "Le Havre": 290.0, "Brest": 240.0, "Rennes": 0.0, "Nantes": 100.0,
        "Angers": 130.0, "Bordeaux": 450.0, "Toulouse": 600.0, "Montpellier": 720.0, "Marseille": 780.0,
        "Nice": 1100.0, "Toulon": 920.0, "Lyon": 650.0, "Saint-Etienne": 690.0, "Grenoble": 740.0,
        "Dijon": 570.0, "Strasbourg": 750.0, "Metz": 690.0, "Reims": 480.0
    },
    "Nantes": {
        "Paris": 385.0, "Lille": 550.0, "Le Havre": 370.0, "Brest": 300.0, "Rennes": 100.0, "Nantes": 0.0,
        "Angers": 80.0, "Bordeaux": 340.0, "Toulouse": 540.0, "Montpellier": 690.0, "Marseille": 780.0,
        "Nice": 950.0, "Toulon": 970.0, "Lyon": 630.0, "Saint-Etienne": 660.0, "Grenoble": 720.0,
        "Dijon": 600.0, "Strasbourg": 850.0, "Metz": 790.0, "Reims": 540.0
    },
    "Angers": {
        "Paris": 295.0, "Lille": 500.0, "Le Havre": 320.0, "Brest": 340.0, "Rennes": 130.0, "Nantes": 80.0,
        "Angers": 0.0, "Bordeaux": 500.0, "Toulouse": 650.0, "Montpellier": 800.0, "Marseille": 880.0,
        "Nice": 1000.0, "Toulon": 1020.0, "Lyon": 560.0, "Saint-Etienne": 590.0, "Grenoble": 650.0,
        "Dijon": 530.0, "Strasbourg": 780.0, "Metz": 720.0, "Reims": 560.0
    },
    "Toulon": {
        "Paris": 870.0, "Lille": 1020.0, "Le Havre": 950.0, "Brest": 1030.0, "Rennes": 920.0, "Nantes": 970.0,
        "Angers": 1020.0, "Bordeaux": 700.0, "Toulouse": 280.0, "Montpellier": 120.0, "Marseille": 40.0,
        "Nice": 110.0, "Toulon": 0.0, "Lyon": 430.0, "Saint-Etienne": 480.0, "Grenoble": 420.0,
        "Dijon": 590.0, "Strasbourg": 720.0, "Metz": 780.0, "Reims": 890.0
    },
    "Marseille": {
        "Paris": 775.0, "Lille": 970.0, "Le Havre": 890.0, "Brest": 950.0, "Rennes": 780.0, "Nantes": 780.0,
        "Angers": 880.0, "Bordeaux": 600.0, "Toulouse": 230.0, "Montpellier": 130.0, "Marseille": 0.0,
        "Nice": 160.0, "Toulon": 40.0, "Lyon": 320.0, "Saint-Etienne": 360.0, "Grenoble": 300.0,
        "Dijon": 530.0, "Strasbourg": 630.0, "Metz": 710.0, "Reims": 760.0
    },
    "Saint-Etienne": {
        "Paris": 510.0, "Lille": 700.0, "Le Havre": 620.0, "Brest": 870.0, "Rennes": 690.0, "Nantes": 660.0,
        "Angers": 590.0, "Bordeaux": 580.0, "Toulouse": 560.0, "Montpellier": 480.0, "Marseille": 360.0,
        "Nice": 480.0, "Toulon": 480.0, "Lyon": 60.0, "Saint-Etienne": 0.0, "Grenoble": 100.0,
        "Dijon": 240.0, "Strasbourg": 400.0, "Metz": 440.0, "Reims": 500.0
    },
    "Grenoble": {
        "Paris": 560.0, "Lille": 750.0, "Le Havre": 680.0, "Brest": 930.0, "Rennes": 740.0, "Nantes": 720.0,
        "Angers": 650.0, "Bordeaux": 620.0, "Toulouse": 500.0, "Montpellier": 360.0, "Marseille": 300.0,
        "Nice": 340.0, "Toulon": 420.0, "Lyon": 110.0, "Saint-Etienne": 100.0, "Grenoble": 0.0,
        "Dijon": 300.0, "Strasbourg": 470.0, "Metz": 530.0, "Reims": 560.0
    },
    "Montpellier": {
        "Paris": 750.0, "Lille": 930.0, "Le Havre": 850.0, "Brest": 890.0, "Rennes": 720.0, "Nantes": 690.0,
        "Angers": 800.0, "Bordeaux": 470.0, "Toulouse": 250.0, "Montpellier": 0.0, "Marseille": 130.0,
        "Nice": 330.0, "Toulon": 120.0, "Lyon": 160.0, "Saint-Etienne": 480.0, "Grenoble": 360.0,
        "Dijon": 460.0, "Strasbourg": 530.0, "Metz": 590.0, "Reims": 650.0
    },
    "Dijon": {
        "Paris": 315.0, "Lille": 490.0, "Le Havre": 420.0, "Brest": 750.0, "Rennes": 570.0, "Nantes": 600.0,
        "Angers": 530.0, "Bordeaux": 680.0, "Toulouse": 730.0, "Montpellier": 800.0, "Marseille": 850.0,
        "Nice": 580.0, "Toulon": 590.0, "Lyon": 195.0, "Saint-Etienne": 240.0, "Grenoble": 300.0,
        "Strasbourg": 230.0, "Metz": 330.0, "Reims": 170.0, "Dijon": 0.0
    },
    "Strasbourg": {
        "Paris": 490.0, "Lille": 460.0, "Le Havre": 540.0, "Brest": 950.0, "Rennes": 750.0, "Nantes": 850.0,
        "Angers": 780.0, "Bordeaux": 950.0, "Toulouse": 880.0, "Montpellier": 1010.0, "Marseille": 1040.0,
        "Nice": 700.0, "Toulon": 720.0, "Lyon": 400.0, "Saint-Etienne": 440.0, "Grenoble": 500.0,
        "Dijon": 230.0, "Metz": 60.0, "Reims": 320.0, "Strasbourg": 0.0,
    },
    "Metz": {
        "Paris": 330.0, "Lille": 350.0, "Le Havre": 480.0, "Brest": 890.0, "Rennes": 690.0, "Nantes": 790.0,
        "Angers": 720.0, "Bordeaux": 880.0, "Toulouse": 940.0, "Montpellier": 1010.0, "Marseille": 1050.0,
        "Nice": 780.0, "Toulon": 820.0, "Lyon": 450.0, "Saint-Etienne": 480.0, "Grenoble": 540.0,
        "Dijon": 330.0, "Strasbourg": 60.0, "Reims": 240.0, "Metz": 0.0,
    },
    "Reims": {
        "Paris": 145.0, "Lille": 170.0, "Le Havre": 280.0, "Brest": 650.0, "Rennes": 480.0, "Nantes": 540.0,
        "Angers": 480.0, "Bordeaux": 750.0, "Toulouse": 860.0, "Montpellier": 930.0, "Marseille": 960.0,
        "Nice": 850.0, "Toulon": 890.0, "Lyon": 400.0, "Saint-Etienne": 440.0, "Grenoble": 500.0,
        "Dijon": 170.0, "Strasbourg": 320.0, "Metz": 240.0, "Reims": 0.0
    },
    "Toulouse": {
        "Paris": 680.0, "Lille": 870.0, "Le Havre": 780.0, "Brest": 800.0, "Rennes": 600.0, "Nantes": 540.0,
        "Angers": 650.0, "Bordeaux": 240.0, "Toulouse": 0.0, "Montpellier": 250.0, "Marseille": 230.0,
        "Nice": 430.0, "Toulon": 280.0, "Lyon": 320.0, "Saint-Etienne": 560.0, "Grenoble": 500.0,
        "Dijon": 730.0, "Strasbourg": 880.0, "Metz": 940.0, "Reims": 860.0
    },
    "Bordeaux": {
        "Paris": 585.0, "Lille": 790.0, "Le Havre": 620.0, "Brest": 590.0, "Rennes": 450.0, "Nantes": 340.0,
        "Angers": 500.0, "Bordeaux": 0.0, "Toulouse": 240.0, "Montpellier": 470.0, "Marseille": 600.0,
        "Nice": 720.0, "Toulon": 700.0, "Lyon": 580.0, "Saint-Etienne": 580.0, "Grenoble": 620.0,
        "Dijon": 680.0, "Strasbourg": 850.0, "Metz": 880.0, "Reims": 750.0
    },
    "Lyon": {
        "Paris": 465.0, "Lille": 650.0, "Le Havre": 570.0, "Brest": 820.0, "Rennes": 650.0, "Nantes": 630.0,
        "Angers": 560.0, "Bordeaux": 580.0, "Toulouse": 320.0, "Montpellier": 160.0, "Marseille": 320.0,
        "Nice": 430.0, "Toulon": 430.0, "Lyon": 0.0, "Saint-Etienne": 60.0, "Grenoble": 110.0,
        "Dijon": 195.0, "Strasbourg": 400.0, "Metz": 450.0, "Reims": 400.0
    },
    "Nice": {
        "Paris": 930.0, "Lille": 980.0, "Le Havre": 910.0, "Brest": 1260.0, "Rennes": 1100.0, "Nantes": 950.0,
        "Angers": 1000.0, "Bordeaux": 720.0, "Toulouse": 430.0, "Montpellier": 330.0, "Marseille": 160.0,
        "Nice": 0.0, "Toulon": 110.0, "Lyon": 430.0, "Saint-Etienne": 480.0, "Grenoble": 340.0,
        "Dijon": 580.0, "Strasbourg": 700.0, "Metz": 780.0, "Reims": 850.0
    }
}



def dijkstra(graph, start, end, *args):
    """
    Implémente Dijkstra avec un graphe NetworkX, avec des heapq pour gérer les priorités sur les passages des noeuds à plus faible distance.
    
    :param graph: Graphe NetworkX (Graph)
    :param start: Nœud de départ
    :return: Dictionnaire des distances et des prédécesseurs
    """
    # Initialisation
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0
    predecessors = {node: None for node in graph.nodes}
    heap = [(0, start)]

    while heap:
        current_dist, current_node = heapq.heappop(heap)

        # Si la distance actuelle est déjà plus grande que celle enregistrée, on ignore ce nœud
        if current_dist > distances[current_node]:
            continue

        # Exploration des voisins
        for neighbor in graph.neighbors(current_node):
            weight = graph[current_node][neighbor]["weight"]
            new_dist = current_dist + weight

            # Si un chemin plus court est trouvé, on met à jour les distances et les prédécesseurs
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                predecessors[neighbor] = current_node
                heapq.heappush(heap, (new_dist, neighbor))

    return distances, predecessors

def reconstruct_path(predecessors, start, end):
    """
    Reconstruit le chemin optimal avec les prédécesseurs.
    """
    path = []
    current = end

    # Reconstruire le chemin en remontant les prédécesseurs
    while current is not None:
        path.append(current)
        current = predecessors[current]
    
    # Si le premier élément du chemin n'est pas le start, cela signifie qu'il n'y a pas de chemin
    if path[-1] == start:
        path.reverse()
        return path
    else:
        return []  # Pas de chemin trouvé

    


def a_star(graph, start, end, heuristics):
    """
    Implemente l'algorithme A* sur un graphe NetworkX.
    
    :param graph: Graphe NetworkX (Graph)
    :param start: Noeud de depart
    :param end: Noeud d'arrivee
    :param heuristics: Dictionnaire des heuristiques {noeud: {destination: heuristique}}
    :return: Chemin optimal et coût total
    """
    # Initialisation
    open_set = [(heuristics[start][end], start)]  # (f(n), node) - Min-Heap
    g_costs = {node: float('inf') for node in graph.nodes}
    g_costs[start] = 0
    predecessors = {node: None for node in graph.nodes}

    while open_set:
        _, current_node = heapq.heappop(open_set)

        # Si on atteint le but, reconstruire le chemin
        if current_node == end:
            path = []
            while current_node:
                path.append(current_node)
                current_node = predecessors[current_node]
            return path[::-1], g_costs[end]

        # Explorer les voisins
        for neighbor in graph.neighbors(current_node):
            weight = graph[current_node][neighbor]["weight"]
            new_g_cost = g_costs[current_node] + weight

            if new_g_cost < g_costs[neighbor]:  # Meilleur chemin trouvé ?
                g_costs[neighbor] = new_g_cost
                predecessors[neighbor] = current_node
                f_cost=0
                if neighbor in heuristics and end in heuristics[neighbor]:
                    f_cost = new_g_cost + heuristics[neighbor][end]/1.5
                else:
                    f_cost = new_g_cost  # On considère une heuristique nulle (équivalent à Dijkstra)
                heapq.heappush(open_set, (f_cost, neighbor))

    return None, float('inf')  # Pas de chemin trouvé


def bellman_ford(graph, start, end, *args):
    """
    Implémente l'algorithme de Bellman-Ford pour trouver le plus court chemin depuis un nœud de départ.
    
    :param graph: Graphe NetworkX (DiGraph)
    :param start: Nœud de départ
    :les autres paramètres sont là pour simplifier le code de evaluate_algorithm
    :les autres paramètres sont là pour simplifier le code de evaluate_algorithm
    :return: Dictionnaire des distances et des prédécesseurs
    """
    # Initialisation des distances et des prédécesseurs
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0
    predecessors = {node: None for node in graph.nodes}
    
    # Relaxation des arêtes (V-1) fois
    for _ in range(len(graph.nodes) - 1):
        for u, v, data in graph.edges(data=True):
            weight = data.get('weight', 1)  # Poids par défaut égal à 1 si non spécifié
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                predecessors[v] = u

    # Vérification des cycles négatifs
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1)
        if distances[u] + weight < distances[v]:
            print("Cycle négatif détecté")
            return None, None
    
    return distances, predecessors

def evaluate_algorithm(graph, algorithm, *args):
    """
    Calcule le temps moyen d'exécution pour l'algorithme spécifié.
    :param graph: Graphe NetworkX
    :param algorithm: Fonction d'algorithme (dijkstra, bellman_ford, a_star)
    :param args: Arguments supplémentaires pour l'algorithme
    :return: Temps moyen d'exécution
    """
    nodes = list(graph.nodes)
    total_time = 0
    num_pairs = 0

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            start = nodes[i]
            end = nodes[j]

            start_time = time.time()

            algorithm(graph, start, end, *args)

            end_time = time.time()

            pair_time = end_time - start_time

            total_time += pair_time
            num_pairs += 1

    average_time = total_time / num_pairs if num_pairs > 0 else 0
    return average_time



def main():
    temps_tot=time.time()
    # Graphe non oriente avec les temps de trajet en minutes
    G1 = nx.Graph()

    # Ajout des aretes avec leurs poids
    edges = [
        ("Paris", "Dijon", 194), ("Paris", "Reims", 93), ("Paris", "Lille", 152),
        ("Paris", "Le Havre", 198), ("Paris", "Angers", 181), ("Paris", "Toulouse", 396),
        ("Paris", "Rennes", 214), ("Lille", "Le Havre", 194), ("Lille", "Metz", 225),
        ("Le Havre", "Brest", 300), ("Le Havre", "Rennes", 181), ("Brest", "Rennes", 225),
        ("Rennes", "Nantes", 95), ("Nantes", "Angers", 62), ("Nantes", "Bordeaux", 204),
        ("Bordeaux", "Lyon", 323), ("Bordeaux", "Toulouse", 215), ("Toulouse", "Montpellier", 156),
        ("Montpellier", "Lyon", 187), ("Montpellier", "Marseille", 114), ("Lyon", "Saint-Etienne", 83),
        ("Lyon", "Grenoble", 89), ("Lyon", "Dijon", 195), ("Grenoble", "Marseille", 181),
        ("Marseille", "Nice", 138), ("Marseille", "Toulon", 51), ("Nice", "Toulon", 109),
        ("Dijon", "Strasbourg", 206), ("Strasbourg", "Metz", 117), ("Metz", "Reims", 116)
    ]

    for u, v, w in edges:
        G1.add_edge(u, v, weight=w)

    # Coordonnees des villes pour le dessin
    positions = {
        "Paris": (2.35, 48.85), "Lille": (3.06, 50.63), "Le Havre": (0.10, 49.49),
        "Brest": (-4.49, 48.39), "Rennes": (-1.68, 48.11), "Nantes": (-1.75, 47.22),
        "Angers": (-0.55, 47.47), "Bordeaux": (-0.58, 44.84), "Toulouse": (1.44, 43.60),
        "Montpellier": (3.88, 43.61), "Marseille": (5.2, 43.15),
        "Nice": (7.26, 43.71), "Toulon": (6.5, 43.0),
        "Lyon": (4.97, 46.00), "Saint-Etienne": (4.30, 45.44), "Grenoble": (5.72, 45.18),
        "Dijon": (5.04, 47.32), "Strasbourg": (7.75, 48.58), "Metz": (6.18, 49.12), "Reims": (4.03, 49.26)
    }

    # Ecrire ICI la ville de depart et d'arrivee
    start, end = "Lille", "Marseille"

 ####################################################################################################################

    # Utilisation de l'algorithme de Dijsktra
    debut_temps= time.time()
    distances, predecessors = dijkstra(G1, start,0,0)
    fin_temps1 = time.time()
    shortest_path = reconstruct_path(predecessors, start, end)
    shortest_distance = distances[end]

    print("\nResultats pour l'algorithme Dijkstra : \n")
    print(f"Chemin optimal de {start} à {end} : {shortest_path}")
    print(f"Temps minimal : {shortest_distance} minutes")


    # Calcul de la distance parcourue pour le plus court chemin
    total_distance_dijkstra = 0
    for i in range(len(shortest_path) - 1):
        node1, node2 = shortest_path[i], shortest_path[i + 1]
        total_distance_dijkstra += heuristics[node1][node2]

    print(f"Distance totale parcourue pour le plus court chemin (Dijkstra) : {total_distance_dijkstra} km")

    # Dessin du graphe
    plt.figure(figsize=(10, 10))
    nx.draw(G1, pos=positions, with_labels=True, node_color="lightblue", edge_color="gray",
        node_size=800, font_size=9, font_weight="bold", arrows=True, width=1.5)

    # Ajouter les labels des arêtes (poids)
    labels = nx.get_edge_attributes(G1, 'weight')
    nx.draw_networkx_edge_labels(G1, pos=positions, edge_labels=labels, font_size=8, label_pos=0.5)

    if shortest_path:
        path_edges = list(zip(shortest_path, shortest_path[1:]))
        nx.draw_networkx_edges(G1, pos=positions, edgelist=path_edges, edge_color="red", width=2)


    plt.title("Graphe des villes françaises avec temps de trajet", fontsize=12)
    plt.show()

############################################################################################################
    debut_temps2=time.time()
    # Execution de A*
    shortest_path, total_cost = a_star(G1, start, end, heuristics)

    fin_temps2 = time.time()
    print("\n\nResultats pour l'algorithme A* : \n")
    print(f"Chemin optimal de {start} à {end} : {shortest_path}")
    print(f"Cout total estimé : {total_cost} minutes")

    # Calcul de la distance parcourue pour le plus court chemin
    total_distance_astar = 0.0
    for i in range(len(shortest_path) - 1):
        node1, node2 = shortest_path[i], shortest_path[i + 1]
        total_distance_astar += heuristics[node1][node2]

    print(f"Distance totale parcourue pour le plus court chemin (Dijkstra) : {total_distance_astar} km")

    # Dessin du graphe avec le chemin A*
    plt.figure(figsize=(10, 10))
    nx.draw(G1, pos=positions, with_labels=True, node_color="lightblue", edge_color="gray",
        node_size=800, font_size=9, font_weight="bold", arrows=True, width=1.5)

    # Ajouter les labels des arêtes (poids)
    nx.draw_networkx_edge_labels(G1, pos=positions, edge_labels=labels, font_size=8, label_pos=0.5)

    if shortest_path:
        path_edges = list(zip(shortest_path, shortest_path[1:]))
        nx.draw_networkx_edges(G1, pos=positions, edgelist=path_edges, edge_color="red", width=2)


    plt.title(f"Chemin A* de {start} à {end}", fontsize=12)
    plt.show()

#######################################################################################################################


    # TEST DU GRAPHE ORIENTE avec poids négatif avec Bellman-Ford
    G2=nx.DiGraph()
    edges = [
        ("Paris", "Dijon", 194), ("Paris", "Reims", 93), ("Lille","Paris",  152),
        ("Le Havre","Paris", 198), ("Paris" ,"Angers", 181), ("Paris", "Toulouse", 396),
        ("Rennes","Paris",  214), ("Le Havre","Lille",  194), ("Lille", "Metz", 225),
        ("Le Havre", "Brest", 300), ("Le Havre", "Rennes", -181), ("Brest", "Rennes", 225),
        ("Rennes", "Nantes", 95), ("Angers","Nantes",  62), ("Nantes", "Bordeaux", 204),
        ("Bordeaux", "Lyon", 323),("Lyon","Bordeaux",  323), ("Bordeaux", "Toulouse", 215), ("Toulouse", "Montpellier", 156),
        ("Lyon","Montpellier",  187), ("Montpellier", "Marseille", 114), ("Lyon", "Saint-Etienne", 83),
        ("Lyon", "Grenoble", 89), ("Lyon", "Dijon", 195), ("Grenoble", "Marseille", 181),
        ("Marseille", "Nice", 138), ("Marseille", "Toulon", 51), ("Toulon","Nice",  109),
        ("Strasbourg","Dijon",  206), ("Metz","Strasbourg",  117), ("Reims","Metz",  116)
    ]
    for u, v, w in edges:
        G2.add_edge(u, v, weight=w)

    # MODIFIER ICI pour la source pour appliquer Bellman-ford! (Il faudra modifier le graph pour qu'il soit cohérent : On veut orienter les arêtes vers la destination !)
    source = "Le Havre"


    plt.figure(figsize=(10, 10))
    nx.draw(G2, pos=positions, with_labels=True, node_color="lightblue", edge_color="gray",
        node_size=800, font_size=9, font_weight="bold", arrows=True, width=1.5)

    # Ajouter les labels des arêtes (poids)
    labels = nx.get_edge_attributes(G2, 'weight')
    nx.draw_networkx_edge_labels(G2, pos=positions, edge_labels=labels, font_size=8, label_pos=0.5)



    plt.title("Graphe des villes françaises avec temps de trajet", fontsize=12)
    plt.show()

    debut_temps3 = time.time()
    distances,shortest_path = bellman_ford(G2, source,0,0)
    fin_temps3 = time.time()
    print("\n\nApplication de l'algorithme de Bellman-Ford\n")
    print(f"Distances depuis {source} : {distances}")


    # Initialisation de la figure et des axes
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))  # Deux sous-graphiques côte à côte

    # Dessin du graphe à gauche
    edge_labels = {(u, v): f"{data['weight']}" for u, v, data in G2.edges(data=True)}
    nx.draw(G2, pos=positions, with_labels=True, node_color="lightblue", edge_color="gray",
        node_size=800, font_size=9, font_weight="bold", width=1.5, ax=ax[0])  # Ajout de ax=ax[0]
    nx.draw_networkx_edge_labels(G2, pos=positions, edge_labels=edge_labels, font_size=8, ax=ax[0])  # Ajout de ax=ax[0]

    # Désactivation des axes du graphe à gauche
    ax[0].axis("off")

    # Création du tableau des distances à droite avec tabulation ajoutée

    predecessors, distances = nx.bellman_ford_predecessor_and_distance(G2, source)
    distance_data = [(f"{ville}", f"{distances.get(ville, 'Inconnue')}") for ville in G2.nodes]
    table = ax[1].table(cellText=distance_data, colLabels=["Ville", "Distance (min)"], loc='center', cellLoc='center')

    # Ajustements esthétiques
    ax[1].axis("off")  # Désactivation des axes à droite pour ne laisser que le tableau
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width([0, 1])

    fin_temps_tot=time.time()
    # Titre et affichage du graphique
    plt.suptitle(f"Bellman-Ford Algorithm de la ville {source}", fontsize=16)
    plt.tight_layout()  # Pour éviter le chevauchement
    plt.show()

#############################################################################################################
    # Exécution de l'algorithme de Floyd-Warshall
    debut_temps4 = time.time()
    predecessors, distances = nx.floyd_warshall_predecessor_and_distance(G2)
    fin_temps4 = time.time()

    # Conversion en DataFrame
    df_distances = pd.DataFrame(distances).fillna("∞")  # Matrice des distances
    df_predecessors = pd.DataFrame(predecessors).fillna("-")  # Matrice des prédécesseurs

    # Première fenêtre : Matrice des distances
    plt.figure(figsize=(10, 8))  # Taille de la figure agrandie
    plt.title("Matrice des distances (Floyd-Warshall)")
    plt.axis("off")
    table_distances = plt.table(cellText=df_distances.values, 
                            colLabels=df_distances.columns, 
                            rowLabels=df_distances.index, 
                            cellLoc="center", loc="center")
    table_distances.auto_set_font_size(False)
    table_distances.set_fontsize(7)  # Augmenter la taille de la police

    # Agrandir la taille des cases
    for (i, j), cell in table_distances.get_celld().items():
        cell.set_height(0.03)  # Augmenter la hauteur des cellules
        cell.set_width(0.05)   # Augmenter la largeur des cellules

    plt.show(block=False)  # Affichage sans bloquer l'exécution du programme

    # Deuxième fenêtre : Matrice des prédécesseurs
    plt.figure(figsize=(10, 8))  # Taille de la figure agrandie
    plt.title("Matrice des prédécesseurs (Floyd-Warshall)")
    plt.axis("off")
    table_predecessors = plt.table(cellText=df_predecessors.values, 
                               colLabels=df_predecessors.columns, 
                               rowLabels=df_predecessors.index, 
                               cellLoc="center", loc="center")
    table_predecessors.auto_set_font_size(False)
    table_predecessors.set_fontsize(6)  # Augmenter la taille de la police

    # Agrandir la taille des cases
    for (i, j), cell in table_predecessors.get_celld().items():
        cell.set_height(0.03)  # Augmenter la hauteur des cellules
        cell.set_width(0.065)   # Augmenter la largeur des cellules

    plt.show()  # Affichage de la deuxième fenêtre

    # Affichage des temps d'execution
    print("\n\nTest de temps d'exécution : \n")
    print(f"Temps d'exécution de l'algorithme Dijkstra : {fin_temps1 - debut_temps} secondes")
    print(f"Temps d'exécution de l'algorithme A* : {fin_temps2 - debut_temps2} secondes")
    print(f"Temps d'exécution de l'algorithme Bellman-Ford : {fin_temps3 - debut_temps3} secondes")
    print(f"Temps d'exécution de tout le programme : {fin_temps_tot - temps_tot} secondes")

#######################################################################################################################

    # PARTIE STATISTIQUE:
    
    # Evaluation du temps moyen d'exécution pour Dijkstra
    print("\nÉvaluation Dijkstra:")
    average_time_dijkstra = evaluate_algorithm(G1, dijkstra, None)
    print(f"\nTemps moyen d'exécution de Dijkstra : {average_time_dijkstra:.6f} secondes")

    # Evaluation du temps moyen d'execution pour A*
    print("\nÉvaluation A*:")
    average_time_astar = evaluate_algorithm(G1, a_star, heuristics)
    print(f"\nTemps moyen d'execution d'A* : {average_time_astar:.6f} secondes")

    # Évaluation du temps moyen d'execution pour Bellman-Ford
    print("\nÉvaluation Bellman-Ford pour le graphe non orienté sans aretes negatives:")
    average_time_bellman_ford = evaluate_algorithm(G1, bellman_ford, None)
    print(f"\nTemps moyen d'exécution de Bellman-Ford : {average_time_bellman_ford:.6f} secondes")

    # Évaluation du temps moyen d'execution pour Bellman-Ford
    print("\nÉvaluation Bellman-Ford pour le graphe oreinté avec aretes negatives:")
    average_time_bellman_ford = evaluate_algorithm(G2, bellman_ford, None)
    print(f"\nTemps moyen d'exécution de Bellman-Ford : {average_time_bellman_ford:.6f} secondes")

    #Evaluation du temps d'execution pour Floyd-Warshall
    print(f"\nTemps d'exécution de l'algorithme Floyd Warshall : {fin_temps4 - debut_temps4} secondes")


if __name__ == "__main__":
    main()