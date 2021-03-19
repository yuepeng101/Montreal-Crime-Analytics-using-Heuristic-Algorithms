

import shapefile
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from matplotlib.path import Path
import matplotlib.patches as patches

import math
import sys
from itertools import compress
import re
from random import choice
import time

# Global variables
grid_vertices = [(-73.59, 45.49), (-73.55, 45.49), (-73.55, 45.53), (-73.59, 45.53)]
grid_width = 0.0  # grid width
accessible_list = []  # all grid areas with crime rate < threshold


def assign_cost(lst):
    """
    assign True or False and cost value based on parameter lst
    :param lst: list of boolean elements
    :return: list with two elements, first is a bool, second is cost value
    """
    count = lst.count(True)
    if count == 1:
        return [True, 1.3]
    elif count == 2:
        return [True, 1]
    else:
        return [False, None]


def create_grid_path(point):
    """
    :param point: tuple of coordinates(x, y)
    :return: path of the block area based on four vertices
    """
    x, y = point
    area = [(x, y), (x + grid_width, y), (x + grid_width, y + grid_width), (x, y + grid_width)]
    area = list(map(new_round, area))
    return Path(area)


def new_round(t):
    """
    keep three digits
    :param t: tuple of a coordinate
    :return: tuple of a coordinate
    """
    x, y = t
    return round(x, 3), round(y, 3)


def check_in_scope(node):
    """
    check whether the point within the scope
    :param node: tuple of a coordinate
    :return: true if it is in scope
    """
    x, y = node
    return -73.59 <= x <= -73.55 and 45.49 <= y <= 45.53


def neighbors(node):
    """
    find out the accessible neighbors
    :param node: tuple of a coordinate
    :return: list of tuples that contains coordinates and cost of each neighbor
    """
    x, y = node
    w = grid_width

    # Get the node's surrounding blocks representing by left, leftBottom, and bottom coordinates
    surr_blocks = [(x, y), (x - w, y), (x - w, y - w), (x, y - w)]
    surr_blocks = list(map(new_round, surr_blocks))  # keep 3 digits

    # Get the True or False value of blocks, True means the crime rate of this block > threshold
    tf = [i in accessible_list for i in surr_blocks]

    # Get boolean value and cost value along x, y axis
    axis_dire = [[tf[0], tf[1]], [tf[1], tf[2]], [tf[2], tf[3]], [tf[3], tf[0]]]

    # Assign isAccessible and cost to the four directions along axis
    axis_lst = list(map(assign_cost, axis_dire))

    # Result of the 8 neighbors with [isAccessible, cost] attribute
    result = [[i, 1.5] for i in tf] + axis_lst
    tf_lst = [i[0] for i in result]  # T F list of eight directions
    cost_lst = [i[1] for i in result]  # cost list of eight directions

    # all possible neighbor coordinates with same order mapping to result
    candidates = [(x + w, y + w), (x - w, y + w), (x - w, y - w), (x + w, y - w),
                  (x, y + w), (x - w, y), (x, y - w), (x + w, y)]
    candidates = list(map(new_round, candidates))  # keep 3 digits

    # ensure neighbors are accessible
    neighbor_lst = list(zip(candidates, cost_lst))
    neighbor_lst = list(compress(neighbor_lst, tf_lst))

    # ensure neighbors are valid
    if y == 45.49 or y == 45.53:
        # remove those out of scope
        neighbor_lst = [i for i in neighbor_lst if check_in_scope(i[0])]
        # remove path along the most outer edges
        neighbor_lst = [i for i in neighbor_lst if i[0][1] != y]

    if x == -73.59 or x == -73.55:
        neighbor_lst = [i for i in neighbor_lst if check_in_scope(i[0])]
        neighbor_lst = [i for i in neighbor_lst if i[0][0] != x]

    if len(neighbor_lst) == 0:
        print('Due to blocks, no path is found. Please change the map and try again')
        print('\n---- The program has terminated. Thanks for using! ----')
        sys.exit(0)

    return neighbor_lst


def get_distance(pt_1, pt_2):
    """
    get the direct distance between two points
    :param pt_1: 1st point
    :param pt_2: 2nd point
    :return: float of the distance
    """
    p1 = np.array(pt_1)
    p2 = np.array(pt_2)
    v = p2 - p1
    return math.hypot(v[0], v[1])


def a_star_search(start, goal):
    """
    a star search implementation, using Manhattan distance as Heuristic evaluation func
    :param start: start point
    :param goal: goal point
    :return: a list of points from start to goal
    """
    open_list = []  # open list
    close_list = {}  # close list, key saves this node and value saves parent node
    g = {}  # record cost so far, key saves this node and value saves its cost so far

    # Initialize these sets
    close_list.update({start: None})
    g.update({start: 0})

    d = get_distance(start, goal)
    open_list.append([d, start])

    # Keep looping until solution found
    while True:
        # No solution
        if len(open_list) == 0:
            print('Due to blocks, no path is found. Please change the map and try again')
            return None

        open_list = sorted(open_list, key=lambda x: x[0])
        current = open_list.pop(0)[1]  # Remove and return the 1st coordinate of open list

        # Shortest path found
        if current == goal:
            break

        # Follow current node to explore its accessible neighbors
        for neighbor in neighbors(current):
            new_g = g[current] + neighbor[1]  # total cost g from start to the neighbor
            # If the neighbor is not in open list, or it is but its cost so far is smaller
            # then add it to open list, close list, and g set
            if neighbor[0] not in g or new_g < g[neighbor[0]]:
                for ele in open_list:
                    if ele[1] == neighbor[0]:
                        open_list.remove(ele)
                open_list.append([new_g + get_distance(neighbor[0], goal), neighbor[0]])
                close_list.update({neighbor[0]: current})
                g.update({neighbor[0]: new_g})

    # Return the path
    path = []
    tmp = goal
    path.append(goal)
    while close_list[tmp] is not None:
        path.append(close_list[tmp])
        tmp = close_list[tmp]

    path.reverse()

    # Calculate total cost of the path
    total_cost = [g[i] for i in path]
    print('\nTotal cost of the path: ', sum(total_cost))
    return path


def refine_float(str_node):
    """
    extract float from user inputs
    :param str_node: string
    :return: tuple of floats
    """
    tup = re.findall(r'-?\d+\.?\d*e?-?\d*?', str_node)
    try:
        tup = tuple([float(i) for i in tup])
    except ValueError:
        print('Invalid coordinate!')
        return None
    return tup


def run_main():
    """
        main function
    """
    # Step 1. ---- Read the file data and get the coordinates ----
    file_path = './crime_dt.shp'
    shape = shapefile.Reader(file_path, encoding='ISO-8859-1')
    shapeRecords = shape.shapeRecords()

    # Get the x and y coordinates of each crime case
    x_coordinates = []
    y_coordinates = []

    for i in range(len(shapeRecords)):
        x_coordinates.append(shapeRecords[i].shape.__geo_interface__["coordinates"][0])
        y_coordinates.append(shapeRecords[i].shape.__geo_interface__["coordinates"][1])

    # Generate 2d np.array representing the coordinates of each crime case
    crime_points = np.column_stack((x_coordinates, y_coordinates))

    # Step 2. ---- Create the grid ----
    # Prompt user to enter the size of the grid
    while True:
        grid_size = input('\nPlease enter the size \'n\' of the grid (n = 20 by default): ')
        if grid_size == '':
            grid_size = 20
        else:
            try:
                grid_size = int(grid_size)
            except ValueError:
                print('Invalid input!')
                continue

        if grid_size <= 0:
            print('Input invalid, please input again!')
        else:
            break

    global grid_width
    grid_width = round(0.04 / grid_size, 6)

    # Generate n x n intervals linearly along x axis and y axis
    linear_x = np.linspace(-73.59, -73.55, grid_size + 1, endpoint=True)
    linear_y = np.linspace(45.49, 45.53, grid_size + 1, endpoint=True)

    # Matrix of x and y coordinates
    grid_x, grid_y = np.meshgrid(linear_x, linear_y)

    # Generate one dimension array of x and y coordinates
    flat_x = grid_x.flatten()
    flat_y = grid_y.flatten()

    # Get the grid's coordinate of each point
    grid_coordinates = [i for i in zip(flat_x, flat_y)]  # [(x, y), ...] len=441
    grid_coordinates = list(map(new_round, grid_coordinates))

    # Generate blocks list representing by its left bottom coordinate
    blocks = [i for i in grid_coordinates if not (i.__contains__(-73.55) or i.__contains__(45.53))]

    # Generate the list of paths of each grid
    block_paths_lst = list(map(create_grid_path, blocks))

    # Step 3. ---- Compute the number of total crimes of each block area ----
    count_lst = []

    for bp in block_paths_lst:
        lst = bp.contains_points(crime_points)
        count_lst.append(Counter(lst)[True])

    count_lst_nd = np.array(count_lst)  # translate to a ndarray

    # Calculate threshold
    while True:
        threshold = input('\nPlease enter a threshold (0 ~ 1) of crime rate (0.5 by default): ')
        if threshold == '':
            threshold = 0.5
        else:
            try:
                threshold = float(threshold)
            except ValueError:
                print('Invalid input!')
                continue
            if threshold >= 1 or threshold <= 0:
                print('Invalid threshold value, please input again!')
                continue
        break

    index = int(grid_size ** 2 * (1 - threshold))
    threshold_value = sorted(count_lst, reverse=True)[index - 1]
    print('\nThe threshold value = ', threshold_value)
    print('The average of all grids = ', np.average(count_lst_nd))
    print('The standard deviation of all grids = ', np.std(count_lst_nd, ddof=1))

    # Reshape the array to (n * n)
    total_count = count_lst_nd.reshape(grid_size, grid_size)
    # Change the order of the array from left bottom to right up
    total_count = total_count[::-1]
    print('\nDisplay the number of total crimes in each grid: \n', total_count)

    # A coordinates list contains all blocks < threshold
    global accessible_list
    t_tmp = [i < threshold_value for i in count_lst]  # accessible vertices
    f_tmp = [i >= threshold_value for i in count_lst]  # inaccessible vertices
    accessible_list = list(compress(blocks, t_tmp))

    # blocks labelled in purple for less risky, yellow for high risky
    purple_blocks = list(compress(block_paths_lst, t_tmp))
    yellow_blocks = list(compress(block_paths_lst, f_tmp))

    # Step 4. ---- Display the graph and draw the optimal path on the map ----
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for ele in purple_blocks:
        patch = patches.PathPatch(ele, facecolor='purple', lw=1, zorder=10)
        ax.add_patch(patch)

    for ele in yellow_blocks:
        patch = patches.PathPatch(ele, facecolor='yellow', lw=1, zorder=10)
        ax.add_patch(patch)

    ax.set_xlim(-73.59, -73.55)
    ax.set_ylim(45.49, 45.53)

    # Find the optimal path from two coordinates by A* algorithm
    # Input a start node
    while True:
        start_node = input('\nPlease input a start node (x, y): \n'
                           '(Press enter will choose a start and goal point randomly)\n')
        # Chose start and goal randomly
        if start_node == '':
            start_node = choice(grid_coordinates)
            while True:
                goal_node = choice(grid_coordinates)
                if goal_node != start_node:
                    break
            break

        start_node = refine_float(start_node)
        if start_node is None:
            continue

        if not check_in_scope(start_node):
            print(start_node, ' is invalid start coordinates, please input again!')
            continue

        if start_node not in grid_coordinates:
            for i in block_paths_lst:
                if i.contains_point(start_node):
                    start_node = blocks[block_paths_lst.index(i)]
                    break

        # Input a goal coordinate
        goal_node = input('\nPlease input a goal node (x, y): \n')
        goal_node = refine_float(goal_node)
        if goal_node is None:
            continue

        if not check_in_scope(goal_node):
            print(goal_node, ' is invalid goal coordinates, please input again!')
            continue

        if goal_node not in grid_coordinates:
            for i in block_paths_lst:
                if i.contains_point(goal_node):
                    goal_node = blocks[block_paths_lst.index(i)]
                    break
        break

    # Found the solution path
    # Record the execution time
    start_time = time.perf_counter()
    verts = a_star_search(start_node, goal_node)
    # verts = a_star_search((-73.59, 45.49), (-73.55, 45.53))
    end_time = time.perf_counter()
    exe_time = end_time - start_time

    print('Running time: %s Seconds' % exe_time)
    if exe_time > 10:
        print('Time is up. The optimal path is not found.')
        print('---- The program has terminated. Thanks for using! ----')
        sys.exit(0)

    if verts is None:
        print('No path solution found, please try other coordinates!')
        print('---- The program has terminated. Thanks for using! ----')
        sys.exit(0)

    xs, ys = zip(*verts)
    xs = list(xs)
    ys = list(ys)
    ax.scatter(start_node[0], start_node[1], c='r', marker='*', zorder=30)
    ax.scatter(goal_node[0], goal_node[1], c='g', marker='*', zorder=30)
    plt.plot(xs, ys, lw=2, zorder=20)
    plt.show()

    print('\n---- The program has terminated. Thanks for using! ----')


if __name__ == '__main__':
    run_main()
