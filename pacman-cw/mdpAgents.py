# mapAgent.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api


# Creates a grid that can be used as a map: 2D array
#
# The map itself is implemented as a nested list, and the interface
# allows it to be accessed by specifying x, y locations.
#
# This class is used for visualization and debugging purpose
#
# This class 'Grid' is copied from Practical 05: sample solution ---- mapAgents.py
class Grid:

    # Constructor
    # width: number of columns
    # height: number of rows
    def __init__(self, width, height):
        self.width = width
        self.height = height
        subgrid = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(0)
            subgrid.append(row)

        self.grid = subgrid

    # Print the grid out.
    def display(self):
        for i in range(self.height):
            for j in range(self.width):
                # print grid elements with no newline
                print self.grid[i][j],
            # A new line after each line of the grid
            print
            # A line after the grid
        print

    # The display function prints the grid out upside down. This
    # prints the grid out so that it matches the view we see when we
    # look at Pacman.
    def prettyDisplay(self):
        for i in range(self.height):
            for j in range(self.width):
                # print grid elements with no newline
                print self.grid[self.height - (i + 1)][j],
            # A new line after each line of the grid
            print
            # A line after the grid
        print

    # Set and get the values of specific elements in the grid.
    # Here x and y are indices.
    def setValue(self, x, y, value):
        self.grid[y][x] = value

    def getValue(self, x, y):
        return self.grid[y][x]

    # Return width and height to support functions that manipulate the
    # values stored in the grid.
    def getHeight(self):
        return self.height

    def getWidth(self):
        return self.width


class MDPAgent(Agent):
    """
        README:
        To get the best performance, initial rewards are fine tuned for small grid and medium grid.
        During the game, rewards will dynamically change based on the game state:
            a. Food reward suppression: If a piece of food is surrounded by 2 or 3 walls, the reward of
                this piece will be suppressed; food at a crossing will get a higher reward because it gives pacman
                more options to move.
            b. Terminal state reward boosting: The reward of the last piece of food will be boosted so that pacman
                will try to get to it and finish the game as quick as possible, so that no time will be wasted.
            c. Ghost reward radiation: In order to encourage Pacman to avoid ghost and improve win rate, negative
                reward of ghost will be radiated to its neighbours to a certain distance. The effect of radiation
                will decrease as distance increases. Similar things happen when ghost is scared.
            d. Chasing ghost: When Pacman eats a capsule, the reward of ghost will be boosted to attract Pacman.
                Pacman will calculate the distance between scaring ghost and itself, as well as their remaining
                scaring time. If distance/time is less than a threshold, the reward (+) of scared ghost will be
                boosted and radiated to neighbour cells. The threshold indicates the willingness of Pacman to boost
                reward. It is also fine tuned.
    """

    # Constructor: this gets run when we first invoke pacman.py
    def __init__(self):
        """
            All maps (game map, states, reward map and utility map), parameters (reward), hyper-parameters (gamma,
            factors, errors, breadth, etc.) and  are declared here.

            Notice, here we only declare variables without assigning values to them. They will be initialized
            and updated dynamically when the game starts running.
        """

        print "Starting up MDPAgent!"

        # Maps:
        self.map = None                               # Grid: stores symbols of each cell, for visualization
        self.rewardMap = None                         # Grid: stores the reward of each state, for visualization
        self.utilityMap = None                        # Grid: stores utility of each state, for visualization
        self.grid = set()                             # a set of all possible states in the game
        self.mappedStates = {}                        # a dict mapping all possible states to 4 directions
        self.reward = {}                              # a dict which stores reward of each cell
        self.utils = {}                               # a dict which stores utility of each cell

        # Hyper-parameters:
        self.width = 0                                # width of the game map
        self.height = 0                               # height of the game map
        self.error = 0.001                            # error for convergence of value iteration
        self.gamma = 0.9                              # discount factor
        self.catchable_threshold = 2                  # threshold: pacman-ghost distance / scaring time
        self.actionProb = api.directionProb           # non-deterministic probability of policy: 0.8
        self.otherActionProb = (1-self.actionProb)/2  # 0.1
        self.breadth = 0                              # reward radiation distance, i.e., breadth in BFS
        self.scaringGhost_reward_factor = 0           # scaring-ghost-reward boosting factor

        # Parameters / Reward:
        self.food_reward = 0                          # Food reward
        self.ghost_reward = 0                         # Ghost reward
        self.scaringGhost_reward = 0                  # Scaring ghost reward
        self.capsule_reward = 0                       # Capsule reward
        self.empty_reward = 0                         # Empty cell reward

    # Gets run after an MDPAgent object is created and once there is
    # game state to access.
    def registerInitialState(self, state):
        print "Running registerInitialState for MDPAgent!"
        # 1. Create a set of all states of the game                         -> self.grid
        self.makeGrid(state)
        # 2. Initialize rewards values based on the size of the map         -> self.xxx_reward and hyper-parameters
        self.initializeReward(state)
        # 3. Initialize rewards of each state (except for walls)            -> self.reward (dict)
        self.updateReward(state)
        # 4. Initialize all grids' utility to 0                             -> self.utility (dict)
        self.resetUtility(state)
        # 5. Map Pacman's all possible states to its 4 adjacent directions  -> self.mappedStates
        self.mapState(state)
        # 6. Create Grid instances for printing and visualization           -> self.map/self.rewardMap/self.utilityMap
        self.makeMap(state)

    # This is what gets run in between multiple games
    def final(self, state):
        print "Looks like the game just ended!"

    def getAction(self, state):
        # 1. Update rewards for each cell at the beginning of every round
        self.updateReward(state)
        # 2. Reset utility for each cell to 0
        self.resetUtility(state)
        # 3. Value iteration
        self.valueIteration(state)
        # 4. Update pacman position, reward map and utility map only for printing and visualization
        self.updateMap(state)

        # Print out game map, reward map and utility map for debugging
        # print "Map: "
        # self.map.prettyDisplay()
        # print "Reward Map: "
        # self.rewardMap.prettyDisplay()
        # print "Utility Map: "
        # self.utilityMap.prettyDisplay()

        # 5. Choose a policy that maximize the expected utility
        policy = self.choosePolicy(state)
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        return api.makeMove(policy, legal)

    def valueIteration(self, state):
        """ Do value iteration for each cell until delta is less than self.error, i.e. until converge """

        states = self.mappedStates      # transition model/dict
        reward = self.reward            # reward dictionary
        utils = self.utils              # utility dictionary
        prevUtils = dict(utils)         # keep a record of previous utility dictionary

        # each loop is one round of value iteration
        while True:
            delta = 0
            # for every cell in the game map, calculate the maximum expected utility
            # coord: utility's map coordinate
            # utility: utility value
            for coord, utility in prevUtils.items():
                # A temporary utility list for 4 directions
                tempUtils = []
                currentGridReward = reward[coord]
                # for each coordinate/state, there are 4 possible move directions
                # potentialGrids: 3 potential grids that one direction can lead to,
                # potentialGrids[0] is the main direction; potentialGrids[1] and potentialGrids[2] are left and right...
                for _, potentialGrids in states[coord].items():
                    tempUtils.append(currentGridReward + self.gamma * (
                            self.actionProb * prevUtils[potentialGrids[0]] + self.otherActionProb *
                            prevUtils[potentialGrids[1]] + self.otherActionProb * prevUtils[potentialGrids[2]]))
                utils[coord] = max(tempUtils)
                delta = max(delta, abs(utils[coord] - utility))
            prevUtils = dict(utils)
            if delta < self.error:
                # if the biggest delta is smaller than error, then break
                self.utils = dict(utils)
                break

    def checkLastFood(self, state):
        """ check if there is only one piece of food """

        foods = api.food(state)
        if len(foods) == 1:
            return True
        else:
            return False

    def choosePolicy(self, state):
        # choose a direction which returns a maximum expected utility
        pacman = api.whereAmI(state)
        (x, y) = pacman
        walls = api.walls(state)
        north = (x, y+1)
        south = (x, y-1)
        east = (x+1, y)
        west = (x-1, y)

        # check if the next location is a wall, if its a wall, stop; otherwise, go
        if north in walls:
            north = pacman
        if south in walls:
            south = pacman
        if east in walls:
            east = pacman
        if west in walls:
            west = pacman

        # calculate expected utility without reward, since reward is the same
        North_EU = self.actionProb * self.utils[north] + \
                   self.otherActionProb * self.utils[west] + \
                   self.otherActionProb * self.utils[east]
        South_EU = self.actionProb * self.utils[south] + \
                   self.otherActionProb * self.utils[west] + \
                   self.otherActionProb * self.utils[east]
        East_EU = self.actionProb * self.utils[east] + \
                  self.otherActionProb * self.utils[north] + \
                  self.otherActionProb * self.utils[south]
        West_EU = self.actionProb * self.utils[west] + \
                  self.otherActionProb * self.utils[north] + \
                  self.otherActionProb * self.utils[south]

        # get the index of max expected utility
        list = [North_EU, South_EU, East_EU, West_EU]
        maxIndex = 0
        for i in range(len(list)):
            if list[i] > list[maxIndex]:
                maxIndex = i

        # return direction: 0-north  1-south  2-east  3-west
        if maxIndex == 0:
            return Directions.NORTH
        if maxIndex == 1:
            return Directions.SOUTH
        if maxIndex == 2:
            return Directions.EAST
        if maxIndex == 3:
            return Directions.WEST

    def initializeReward(self, state):
        """ Initialize reward for every state. Reward are fine tuned according to the size of the map """

        if self.width + self.height < 20:
            # smallGrid:
            self.food_reward = 11
            self.ghost_reward = -25
            self.scaringGhost_reward = 8
            self.capsule_reward = 9
            self.empty_reward = -1
            self.breadth = 17
        else:
            # mediumClassic
            self.food_reward = 5
            self.ghost_reward = -25
            self.scaringGhost_reward = 4
            self.scaringGhost_reward_factor = 3
            self.capsule_reward = 8
            self.empty_reward = -2
            self.breadth = 5

    def updateReward(self, state):
        """
            Update reward for every cell that is not a wall.
            Food reward: food reward will be set differently based on the number of surrounded walls, e.g. food in the
                corners will assigned lower value than those at a crossing. Additionally, we will boost the reward of
                the last piece of food since it's the terminal state of the game.
            Ghost reward: scared ghost has positive reward while normal/brave ghost has positive reward. When Pacman eats
                a capsule, ghost will be scared and has positive reward. If Pacman thinks ghost can be caught within
                ghost's scaring time, it will boost their reward. Otherwise just normal scared ghost reward.
            Reward of ghost will be radiated to neighbour cells.
        """

        walls = api.walls(state)
        foods = api.food(state)
        capsules = api.capsules(state)
        ghostsStates = api.ghostStatesWithTimes(state)

        # Empty reward
        self.reward = {state: -1 for state in self.grid if state not in walls}
        # Food reward
        for food in foods:
            count = self.countWalls(state, food)
            if count <= 2:
                # If the food is only surrounded by 2 or less than 2 walls, just set the reward for this food
                self.reward[food] = self.food_reward
            else:
                # If the food is only surrounded by greater than 2 walls, suppress the reward for this food
                self.reward[food] = self.food_reward / (1 + count * count)
        if self.checkLastFood(state):
            self.reward[foods[0]] += self.food_reward
        # Capsule reward
        self.reward.update({state: self.capsule_reward for state in self.reward.keys() if state in capsules})

        # Firstly set the reward for scared ghost, then set the reward for brave ghost, so that positive reward won't
        # overlap negative reward
        scaredGhost = []
        braveGhost = []
        for ghost in ghostsStates:
            if ghost[1] > 1:
                scaredGhost.append(ghost)
            else:
                braveGhost.append(ghost)

        # Set reward for scared ghost
        for ghost in scaredGhost:
            # cast ghost coordinate to integers
            ghost_coord = (int(round(ghost[0][0])), int(round(ghost[0][1])))
            if ghost_coord in self.reward.keys():
                if ghost[1] > 0:
                    distance = self.distance(state, api.whereAmI(state), ghost[0])  # distance from pacman to ghost
                    time = ghost[1]     # remaining scared time
                    if self.reachable(state, distance, time):
                        # if pacman thinks he can catch ghost before the ghost turns to normal,
                        # we will boost the reward of scaring ghost
                        self.reward[ghost_coord] = self.scaringGhost_reward * self.scaringGhost_reward_factor
                        # radiate boosted reward to ghost's neighbours to encourage pacman to catch ghost
                        self.radiate_reward(state, ghost_coord, self.breadth)
                    else:
                        self.reward[ghost_coord] = self.scaringGhost_reward

        # Set reward for brave ghost
        for ghost in braveGhost:
            # cast ghost coordinate to integers
            ghost_coord = (int(round(ghost[0][0])), int(round(ghost[0][1])))
            if ghost_coord in self.reward.keys():
                if ghost[1] == 0:
                    self.reward[ghost_coord] = self.ghost_reward
                    # radiate this reward to keep pacman away from it
                    self.radiate_reward(state, ghost_coord, self.breadth)

    def countWalls(self, state, food):
        """ Count the number of walls that near a piece of food """

        walls = api.walls(state)
        north = (food[0], food[1] + 1)
        south = (food[0], food[1] - 1)
        west = (food[0] + 1, food[1])
        east = (food[0] - 1, food[1])
        count = 0
        if north in walls:
            count += 1
        if south in walls:
            count += 1
        if west in walls:
            count += 1
        if east in walls:
            count += 1
        return count

    def bfs_neighbours(self, state, origin, breadth):
        """
            Based of BFS, categorise neighbour cells based on their distance to the origin.
            A list is used to store neighbour cells' coordinate, and index is their distance to the origin
        """

        walls = api.walls(state)
        states = self.reward.keys()
        neighbours = [[]]
        queue = []
        visit = set()
        for i in range(breadth):
            neighbours.append([])
        queue.insert(0, origin)
        visit.add(origin)
        for i in range(1, breadth+1, 1):
            size = len(queue)
            for j in range(size):
                # expand nodes in 4 directions
                state = queue.pop()
                north = (state[0], state[1] + 1)
                south = (state[0], state[1] - 1)
                west = (state[0] - 1, state[1])
                east = (state[0] + 1, state[1])
                if north in states and north not in walls and north not in visit:
                    neighbours[i].append(north)
                    queue.insert(0, north)
                    visit.add(north)
                if south in states and south not in walls and south not in visit:
                    neighbours[i].append(south)
                    queue.insert(0, south)
                    visit.add(south)
                if west in states and west not in walls and west not in visit:
                    neighbours[i].append(west)
                    queue.insert(0, west)
                    visit.add(west)
                if east in states and east not in walls and east not in visit:
                    neighbours[i].append(east)
                    queue.insert(0, east)
                    visit.add(east)
        return neighbours

    def radiate_reward(self, state, origin, distance):
        """ Radiate a cell's reward to certain distance. The reward will decrease along the way. """

        # Cast coordinate to integer numbers
        origin = (int(round(origin[0])), int(round(origin[1])))
        # Get all neighbours within a certain distance
        neighbours = self.bfs_neighbours(state, origin, distance)
        # Decrease delta is based on the rediate distance
        delta = self.reward[origin] * 1.0 / distance
        for i in range(1, distance+1, 1):
            for state in neighbours[i]:
                # Reward will decrease 1 'delta' as distance increase 1 cell
                new_reward = self.reward[origin] - i * delta
                self.reward[state] += new_reward

    def distance(self, state, origin, target):
        """
            Based on BFS searching algorithm, calculate the distance between origin and target.
            If origin or target is wall, return -1; If there is no path from origin to target, return -1.
        """

        walls = api.walls(state)
        if origin in walls or target in walls:
            return -1
        states = self.reward.keys()
        queue = []
        visit = set()
        distance = 0
        queue.insert(0, origin)
        visit.add(origin)
        while len(queue) > 0:
            distance += 1
            size = len(queue)
            for i in range(size):
                # expand
                state = queue.pop()
                north = (state[0], state[1] + 1)
                south = (state[0], state[1] - 1)
                west = (state[0] - 1, state[1])
                east = (state[0] + 1, state[1])
                if north == target or south == target or west == target or east == target:
                    return distance
                if north in states and north not in walls and north not in visit:
                    queue.insert(0, north)
                    visit.add(north)
                if south in states and south not in walls and south not in visit:
                    queue.insert(0, south)
                    visit.add(south)
                if west in states and west not in walls and west not in visit:
                    queue.insert(0, west)
                    visit.add(west)
                if east in states and east not in walls and east not in visit:
                    queue.insert(0, east)
                    visit.add(east)
        return -1

    def reachable(self, state, distance, time):
        """
            If the quotient of distance and time is less than a threshold, return true; otherwise return false.
            This threshold will determine whether to boost the reward of scared ghost, and therefore affect the
            willingness of pacman to chase the scared ghost.
        """

        if (distance * 1.0) / time < self.catchable_threshold:
            return True
        else:
            return False

    def resetUtility(self, state):
        """ Initialize all cells' utility to 0 """

        walls = api.walls(state)
        self.utils = {state: 0 for state in self.grid if state not in walls}

    def mapState(self, state):
        """ # Map reachable states for 4 directions of all states """

        walls = api.walls(state)
        states = dict.fromkeys(self.reward.keys())

        # Iterate all cells in the map and map their potential states in 4 directions.
        for cell in states.keys():
            neighbours = self.fourNeighbours(cell)
            states[cell] = {'north': [neighbours[3], neighbours[0], neighbours[2]],
                            'south': [neighbours[1], neighbours[0], neighbours[2]],
                            'east': [neighbours[0], neighbours[3], neighbours[1]],
                            'west': [neighbours[2], neighbours[3], neighbours[1]]}

            # Iterate all 4 directions and iterate all 3 potential states in one direction if any potential state
            # is wall, set it the original cell
            for _, possibleStates in states[cell].items():
                for state in possibleStates:
                    if state in walls:
                        possibleStates[possibleStates.index(state)] = cell
        self.mappedStates = states

    def fourNeighbours(self, cell):
        """ Get 4 adjacent neighbours of a cell """

        (x, y) = cell
        east = (x + 1, y)
        south = (x, y - 1)
        west = (x - 1, y)
        north = (x, y + 1)
        return [east, south, west, north]

    def makeGrid(self, state):
        """
            Create a grid of the whole game map.
            Notice the data structure for grid is set because it can be quickly accessed.
        """

        corners = api.corners(state)
        BL = corners[0]         # Bottom left corner
        TR = corners[3]         # Top right corner
        self.width = TR[0]+1    # store the width of the game map
        self.height = TR[1]+1   # store the height of the game map
        width = range(BL[0], TR[0]+1)
        height = range(BL[1], TR[1]+1)
        self.grid = set((x, y) for x in width for y in height)  # build up the game grid in a python set

    def makeMap(self, state):
        """
            Create 'Grid' instances called map. These maps are created for printing, visualization and debugging.
            Notice in value iteration, we don't use values from these maps.
        """

        corners = api.corners(state)
        height = self.getLayoutHeight(corners)
        width = self.getLayoutWidth(corners)
        self.map = Grid(width, height)          # pass real width and height (not index)
        self.rewardMap = Grid(width, height)
        self.utilityMap = Grid(width, height)
        self.addWallsToMap(state)
        self.updateMap(state)

    def updateMap(self, state):
        """ Update map value for printing """

        # First, make all grid elements that aren't walls blank.
        for i in range(self.map.getWidth()):
            for j in range(self.map.getHeight()):
                if self.map.getValue(i, j) != '%':
                    # Update grid (printing) map
                    self.map.setValue(i, j, ' ')
                    # Update reward and utility map
                    reward = int(round(self.reward[(i, j)]))
                    ''' Here we set utility round to 1 digit for easy printing '''
                    utility = (round(self.utils[(i, j)], 1))
                    self.rewardMap.setValue(i, j, reward)
                    self.utilityMap.setValue(i, j, utility)

        food = api.food(state)
        for i in range(len(food)):
            self.map.setValue(food[i][0], food[i][1], '.')

        cap = api.capsules(state)
        for i in range(len(cap)):
            self.map.setValue(cap[i][0], cap[i][1], 'o')

        ghosts = api.ghosts(state)
        for i in range(len(ghosts)):
            ghostX = int(round(ghosts[i][0]))
            ghostY = int(round(ghosts[i][1]))
            self.map.setValue(ghostX, ghostY, 'X')
        pacman = api.whereAmI(state)
        self.map.setValue(pacman[0], pacman[1], 'P')

    def getLayoutHeight(self, corners):
        height = -1
        for i in range(len(corners)):
            if corners[i][1] > height:
                height = corners[i][1]
        return height + 1

    def getLayoutWidth(self, corners):
        width = -1
        for i in range(len(corners)):
            if corners[i][0] > width:
                width = corners[i][0]
        return width + 1

    def addWallsToMap(self, state):
        walls = api.walls(state)
        for i in range(len(walls)):
            self.map.setValue(walls[i][0], walls[i][1], '%')
            self.rewardMap.setValue(walls[i][0], walls[i][1], '%')
            self.utilityMap.setValue(walls[i][0], walls[i][1], '%')



