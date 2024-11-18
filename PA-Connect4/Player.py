#Modified 10.3.2023 by Chris Archibald to
#  - incorporate MCTS with other code
#  - pass command line param string to each AI

import numpy as np
import time


class AIPlayer:
    def __init__(self, player_number, name, ptype, param):
        self.player_number = player_number
        self.name = name
        self.type = ptype
        self.player_string = 'Player {}: '.format(player_number)+self.name
        self.other_player_number = 1 if player_number == 2 else 2

        #Parameters for the different agents
        
        self.depth_limit = 3 #default depth-limit - change if you desire
        #Alpha-beta
        # Example of using command line param to overwrite depth limit
        if self.type == 'ab' and param:
            self.depth_limit = int(param)

        #Expectimax
        # Example of using command line param to overwrite depth limit
        if self.type == 'expmax' and param:
            self.depth_limit = int(param)

        #MCTS
        self.max_iterations = 1000 #Default max-iterations for MCTS - change if you desire
        # Example of using command line param to overwrite max-iterations for MCTS
        if self.type == 'mcts' and param:
            self.max_iterations = int(param)

    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        start_time = time.perf_counter()

        moves = self.get_working_valid_moves(board)
        best_move = np.random.choice(moves)
        
        depth = 0

        #YOUR ALPHA-BETA CODE GOES HERE
        minmax = [-1 * np.inf, np.inf]
        for move in moves:
            #don't check above what we are currently deciding
            #make new board and execute the move
            newBoard = [[0] * 7 for _ in range(6)]
            #create a new board
            #copy board values
            for row in range(0, 6):
                for col in range(0, 7):
                    newBoard[row][col] = board[row][col]
            make_move(newBoard, move, self.player_number)

            value = self.get_recursive_alpha_beta_move(newBoard, self.other_player_number, depth + 1, minmax)
            
            if (self.player_number == 1 and value > minmax[0]):
                minmax[0] = value
                best_move = move
            elif (self.player_number == 2 and value < minmax[1]):
                minmax[1] = value
                best_move = move
        

        print("-----------------------------------------------------------------------")
        print(" end of alpha beta move")
        print("best_move: " + str(best_move))
        end_time = time.perf_counter()

        print("time taken to find move: " + str(end_time - start_time))
        return best_move
    
    def get_working_valid_moves(self, board):
        valid_moves = []
        #columns
        for currCol in range(0, 7):
            #Look at highest row (row 0)
            if board[0][currCol] == 0:
                #add to valid moves
                valid_moves.append(currCol)
        return valid_moves
        
    
    #return a move
    def get_recursive_alpha_beta_move(self, board, player_num, depth, parent_range):
        # returns a tuple with (action, value associated).
        # Actions that are closer to a goal will return a higher absolute value.
        # Goal states will have the highest absolute value of any state

        #possible combinations:
        # Horizontal: 4 different ways per row (6 rows) = 24 goal states
        # Vertical: 3 different ways per column (7 col) = 21 goal states
        # Diagonal: Requires a 4x4 grid, 2 times per grid.
        #           3 different grids vertically, 4 different grids horizontally.
        #           (2 * 3 * 4) = 24 goal states
        # Max number of goal states = 24 + 21 + 24 = 69 (nice!)
        # Note: can also get two goal states at the same time (diagonal and vertical for example)

        #base case: one player wins
        if (is_winning_state(board, player_num)):
            #return the max value for the current player
            if (player_num == 1):
                #hi there
                return 300
            elif (player_num == 2):
                return -300

        #base case: tie
        if (get_valid_moves(board) == None):
            #return the tie value (not negative or positive)
            return 0
        
        if (depth >= self.depth_limit):
            #if we are stopping give our best guess for the current state of the board
            return self.evaluation_function(board)
        
        moves = get_valid_moves(board)
        minmax = [-1 * np.inf, np.inf]
        for move in moves:

            if (minmax[0] != -1 * np.inf and player_num == 1 and minmax[0] > parent_range[1]):
                #skip if the ranges do not coorespond
                continue
            elif (minmax[1] != np.inf and player_num == 2 and minmax[1] < parent_range[0]):
                #skip if the ranges do not coorespond
                continue

            #make new board and execute the move
            newBoard = [[0] * 7 for _ in range(6)]
            #copy board values
            for row in range(0, 6):
                for col in range(0, 7):
                    newBoard[row][col] = board[row][col]
            make_move(newBoard, move, player_num)

            #get the value of the new move
            otherPlayer = None
            if (player_num == 1):
                otherPlayer = 2
            if (player_num == 2):
                otherPlayer = 1
            value = self.get_recursive_alpha_beta_move(newBoard, otherPlayer, depth + 1, minmax)

            if (player_num == 1 and value > minmax[0]):
                minmax[0] = value
            elif (player_num == 2 and value < minmax[1]):
                minmax[1] = value
        
        if (player_num == 1):
            return minmax[0]
        elif (player_num == 2):
            return minmax[1]
            

            
            




    def get_mcts_move(self, board):
        """
        Use MCTS to get the next move
        """

        #How many iterations of MCTS will we do?
        max_iterations = 1000 #Modify to work for you

        #Make the MCTS root node from the current board state
        root = MCTSNode(board, self.player_number, None)

        #Run our MCTS iterations
        for i in range(max_iterations):

            #Select + Expand
            cur_node = root.select()
            
            #Simulate + backpropate
            cur_node.simulate()

        #Print out the info from the root node
        root.print_node()
        print('MCTS chooses action', root.max_child())
        return root.max_child()

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        
        start_time = time.perf_counter()

        moves = self.get_working_valid_moves(board)
        best_move = np.random.choice(moves)
        
        depth = 0

        #YOUR ALPHA-BETA CODE GOES HERE
        minmax = [-1 * np.inf, np.inf]
        for move in moves:
            #don't check above what we are currently deciding
            #make new board and execute the move
            newBoard = [[0] * 7 for _ in range(6)]
            #create a new board
            #copy board values
            for row in range(0, 6):
                for col in range(0, 7):
                    newBoard[row][col] = board[row][col]
            make_move(newBoard, move, self.player_number)

            value = self.get_recursive_expectimax_move(newBoard, self.other_player_number, depth + 1, minmax)
            
            if (self.player_number == 1 and value > minmax[0]):
                minmax[0] = value
                best_move = move
            elif (self.player_number == 2 and value < minmax[1]):
                minmax[1] = value
                best_move = move
        

        print("-----------------------------------------------------------------------")
        print(" end of expectimax move")
        print("Depth limit: " + str(self.depth_limit))
        print("best_move: " + str(best_move))
        end_time = time.perf_counter()

        print("time taken to find move: " + str(end_time - start_time))
        return best_move

    def get_recursive_expectimax_move(self, board, player_num, depth, parent_range):
        # returns a tuple with (action, value associated).
        # Actions that are closer to a goal will return a higher absolute value.
        # Goal states will have the highest absolute value of any state

        #possible combinations:
        # Horizontal: 4 different ways per row (6 rows) = 24 goal states
        # Vertical: 3 different ways per column (7 col) = 21 goal states
        # Diagonal: Requires a 4x4 grid, 2 times per grid.
        #           3 different grids vertically, 4 different grids horizontally.
        #           (2 * 3 * 4) = 24 goal states
        # Max number of goal states = 24 + 21 + 24 = 69 (nice!)
        # Note: can also get two goal states at the same time (diagonal and vertical for example)

        highestVal = 300
        lowestVal = -300

        #-------------- base cases (don't change) --------------------#

        otherPlayer = None
        if (player_num == 1):
            otherPlayer = 2
        if (player_num == 2):
            otherPlayer = 1

        #base case: one player wins
        if (is_winning_state(board, player_num)):
            #return the max value for the current player
            if (player_num == 1):
                #hi there
                return highestVal
            elif (player_num == 2):
                return lowestVal
        #prevent unnecessary depth with a check for the other player winning
        if (is_winning_state(board, otherPlayer)):
            if (otherPlayer == 1):
                return highestVal
            elif (otherPlayer == 2):
                return lowestVal

        #base case: tie
        if (get_valid_moves(board) == None):
            #return the tie value (not negative or positive)
            return 0
        
        #base case: hit depth limit
        if (depth >= self.depth_limit):
            #if we are stopping give our best guess for the current state of the board
            return self.evaluation_function(board)

        #----------------------------------------------------#
        
        moves = get_valid_moves(board)
        minmax = [-1 * np.inf, np.inf]
        #get the values and associated probabilities for each decision
        probs = [1.0 / len(moves)] * len(board[0])
        highVals = [np.inf] * len(board[0])
        lowVals = [-np.inf] * len(board[0])
        exploredMoves = []
        highValAvg = 0
        lowValAvg = 0
        for move in moves:

            #skip if the ranges do not coorespond
            if (minmax[0] != -1 * np.inf and minmax[0] > parent_range[1]):
                continue
            elif (minmax[1] != np.inf and minmax[1] < parent_range[0]):
                continue

            #mark moves that we are exploring
            exploredMoves.append(move)

            #make new board and execute the move
            newBoard = [[0] * 7 for _ in range(6)]
            #copy board values
            for row in range(0, 6):
                for col in range(0, 7):
                    newBoard[row][col] = board[row][col]
            #update board
            make_move(newBoard, move, player_num)

            #get the value of the new move
            value = self.get_recursive_expectimax_move(newBoard, otherPlayer, depth + 1, minmax)

            #update the expectimax value
            if (player_num == self.player_number):
                #This AI player uses alpha beta, model it as such
                if (player_num == 1):
                    #Maximize minimax[0]
                    if (value > minmax[0]):
                        minmax[0] = value
                if (player_num == 2):
                    #Minimize minmax[1]
                    if (value < minmax[1]):
                        minmax[1] = value
            elif (player_num == self.other_player_number):
                #Model the other player as random choice
                #store the current value for this move inside the high and low vals arrays
                highVals[move - 1] = value
                lowVals[move - 1] = value

                #update max
                #for every move, multiply the moves probability by it's value
                highValAvg = 0
                for movInd in range(len(moves)):
                    #if have a high value, use the high value
                    if highVals[movInd] != np.inf:
                        highValAvg += highVals[movInd] * probs[movInd]
                    else:
                        highValAvg += highestVal * probs[movInd]

                #update min
                lowValAvg = 0
                for movInd in range(len(moves)):
                    #if have a low value, use the low value
                    if lowVals[movInd] != -np.inf:
                        lowValAvg += lowVals[movInd] * probs[movInd]
                    else:
                        lowValAvg += lowestVal * probs[movInd]
        
        if (player_num == self.player_number):
            #use alpha beta
            if (player_num == 1):
                return minmax[0]
            elif (player_num == 2):
                return minmax[1]
        elif (player_num == self.other_player_number):
            #use random, if children are pruned this value should not affect the parent node
            return highValAvg


    def evaluation_function(self, board):
        """
        Given the current stat of the board, return the scalar value that 
        represents the evaluation function for the current player
       
        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """

        #YOUR EVALUATION FUNCTION GOES HERE

        # Player 1 is maximizing values
        # Player 2 is minimizing values

        #Evaluation function will count the number of available goal states still available,
        #   for the current player, (the more possibilities the better) minus the number of available
        #   goal states for the other player

        #For partial goals the closer the goal is to being complete the better, thus for every goal
        # state that is considered, multiply the initial value of the goal state being possible by...
        # (1 + num of tokens in goal state). Because the function should stop on a solution,
        # the max number we could get to is 69 * (1 + 3) = 69 * 4 = 276. Thus lets make a goal nodes value
        # equal to a little above that at 300 lets say.

        #two tokens in a row may count for two goal states, one counted with a multiplier of 2 while the other
        # has a multiplier of 3. (not so if there is a token blocking the end of the board or not)

        #heuristics possibilities:
        # most possible win's
        # most filled goal positions
        # prefer goal positions that cannot be immediatley blocked (there is not a coin under it)
        # prefer positions closer to the middle of the board (more possible combinations)


        #return max values for a winning state
        if is_winning_state(board, 1):
            #return max evaluation value
            return 300
        elif is_winning_state(board, 2):
            return -300
        
        #Implementing most possible wins evaluation
        numGoalsFor1 = 0
        numGoalsFor2 = 0
        #columns
        for currCol in range(0, 7):
            #possible start rows
            for startR in range(0, 3):
                #for every goal list

                filled1Positions = 0
                filled2Positions = 0
                isGoal1 = True #assume this goalList is all 0's
                isGoal2 = True
                for goalOffsetR in range(3, -1, -1):
                    #go backwards, if we encounter a 0, the rest are also 0
                    goalPosR = startR + goalOffsetR
                    posVal = board[goalPosR][currCol]
                    if (posVal == 1):
                        #filled1Positions += 1
                        isGoal2 = False
                    elif (posVal == 2):
                        #filled2Positions += 1
                        isGoal1 = False
                    else:
                        #value of 0
                        break
                    if isGoal2 == False and isGoal1 == False:
                        #no valid goal, go to next goal list
                        break

                #add the valid goals to the total
                if isGoal1 == True:
                    numGoalsFor1 += 1
                if isGoal2 == True:
                    numGoalsFor2 += 1
        
        #rows
        for currRow in range(0, 6):
            #possible start cols
            for startC in range(0, 4):
                #for every goal list

                filled1Positions = 0
                filled2Positions = 0
                isGoal1 = True #assume this goalList is all 0's
                isGoal2 = True
                for goalOffsetC in range(0, 4):
                    goalPosC = startC + goalOffsetC
                    posVal = board[currRow][goalPosC]
                    #if invalid position for a player
                    if (posVal == 1):
                        isGoal2 = False
                    if (posVal == 2):
                        isGoal1 = False
                    
                    if isGoal1 == False and isGoal2 == False:
                        break
                
                #add the valid goals to the total
                if isGoal1 == True:
                    numGoalsFor1 += 1
                if isGoal2 == True:
                    numGoalsFor2 += 1
        
        #diagonals (top left to bottom right)
        for startRow in range(0, 3):
            for startCol in range(0, 4):
                #if invalid position for a player mark it as so
                filled1Positions = 0
                filled2Positions = 0
                isGoal1 = True #assume this goalList is all 0's
                isGoal2 = True
                #for each top left to down right goal list
                for tokenNum in range(0, 4):
                    goalPosR = startRow + tokenNum
                    goalPosC = startCol + tokenNum
                    posVal = board[goalPosR][goalPosC]
                    #if invalid position for a player
                    if (posVal == 1):
                        isGoal2 = False
                    if (posVal == 2):
                        isGoal1 = False
                    
                    if isGoal1 == False and isGoal2 == False:
                        break
                #add the valid goals to the total
                if isGoal1 == True:
                    numGoalsFor1 += 1
                if isGoal2 == True:
                    numGoalsFor2 += 1

        #diagonals (bottom left to top right)
        for startRow in range(0, 3):
            for startCol in range(0, 4):
                #if invalid position for a player mark it as so
                filled1Positions = 0
                filled2Positions = 0
                isGoal1 = True #assume this goalList is all 0's
                isGoal2 = True

                #for each down left to top right goal list
                oppositeRow = 5 - startRow
                for tokenNum in range(0, 4):
                    goalPosR = oppositeRow - tokenNum
                    goalPosC = startCol + tokenNum
                    posVal = board[goalPosR][goalPosC]
                    #if invalid position for a player
                    if (posVal == 1):
                        isGoal2 = False
                    if (posVal == 2):
                        isGoal1 = False
                    
                    if isGoal1 == False and isGoal2 == False:
                        break
                #add the valid goals to the total
                if isGoal1 == True:
                    numGoalsFor1 += 1
                if isGoal2 == True:
                    numGoalsFor2 += 1

        return numGoalsFor1 - numGoalsFor2


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.name = 'random'
        self.player_string = 'Player {}: random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:,col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)

class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.name = 'human'
        self.player_string = 'Player {}: human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move, Human: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move


#CODE FOR MCTS 
class MCTSNode:
    def __init__(self, board, player_number, parent):
        self.board = board
        self.player_number = player_number
        self.other_player_number = 1 if player_number == 2 else 2
        self.parent = parent
        self.moves = get_valid_moves(board)
        self.terminal = (len(self.moves) == 0) or is_winning_state(board, player_number) or is_winning_state(board, self.other_player_number)
        self.children = dict()
        for m in self.moves:
            self.children[m] = None

        #Set up stats for MCTS
        #Number of visits to this node
        self.n = 0 

        #Total number of wins from this node (win = +1, loss = -1, tie = +0)
        # Note: these wins are from the perspective of the PARENT node of this node
        #       So, if self.player_number wins, that is -1, while if self.other_player_number wins
        #       that is a +1.  (Since parent will be using our UCB value to make choice)
        self.w = 0 

        #c value to be used in the UCB calculation
        self.c = np.sqrt(2) 
    

    def print_tree(self):
        #Debugging utility that will print the whole subtree starting at this node
        print("****")
        print_node(self)
        for m in self.moves:
            if self.children[m]:
                self.children[m].print_tree()
        print("****")

    def print_node(self):
        #Debugging utility that will print this node's information
        print('Total Node visits and wins: ', self.n, self.w)
        print('Children: ')
        for m in self.moves:
            if self.children[m] is None:
                print('   ', m, ' is None')
            else:
                print('   ', m, ':', self.children[m].n, self.children[m].w, 'UB: ', self.children[m].upper_bound(self.n))

    def max_child(self):
        #Return the most visited child
        #This is used at the root node to make a final decision
        max_n = 0
        max_m = None

        for m in self.moves:
            if self.children[m].n > max_n:
                max_n = self.children[m].n
                max_m = m
        return max_m

    def upper_bound(self, N):
        #This function returns the UCB for this node
        #N is the number of samples for the parent node, to be used in UCB calculation

        # YOUR MCTS TASK 1 CODE GOES HERE

        #To do: return the UCB for this node (look in __init__ to see the values you can use)

        return 0

    def select(self):
        #This recursive function combines the selection and expansion steps of the MCTS algorithm
        #It will return either: 
        # A terminal node, if this is the node selected
        # The new node added to the tree, if a leaf node is selected

        max_ub = -np.inf  #Track the best upper bound found so far
        max_child = None  #Track the best child found so far

        if self.terminal:
            #If this is a terminal node, then return it (the game is over)
            return self

        #For all of the children of this node
        for m in self.moves:
            if self.children[m] is None:
                #If this child doesn't exist, then create it and return it
                new_board = np.copy(self.board) #Copy board/state for the new child
                make_move(new_board,m,self.player_number) #Make the move in the state

                self.children[m] = MCTSNode(new_board, self.other_player_number, self) #Create the child node
                return self.children[m] #Return it

            #Child already exists, get it's UCB value
            current_ub = self.children[m].upper_bound(self.n)

            #Compare to previous best UCB
            if current_ub > max_ub:
                max_ub = current_ub
                max_child = m

        #Recursively return the select result for the best child 
        return self.children[max_child].select()


    def simulate(self):
        #This function will simulate a random game from this node's state and then call back on its 
        #parent with the result

        # YOUR MCTS TASK 2 CODE GOES HERE

        # Pseudocode in comments:
        #################################
        # If this state is terminal (meaning the game is over) AND it is a winning state for self.other_player_number
        #   Then we are done and the result is 1 (since this is from parent's perspective)
        #
        # Else-if this state is terminal AND is a winning state for self.player_number
        #   Then we are done and the result is -1 (since this is from parent's perspective)
        #
        # Else-if this is not a terminal state (if it is terminal and a tie (no-one won, then result is 0))
        #   Then we need to perform the random rollout
        #      1. Make a copy of the board to modify
        #      2. Keep track of which player's turn it is (first turn is current nodes self.player_number)
        #      3. Until the game is over: 
        #            3.1  Make a random move for the player who's turn it is
        #            3.2  Check to see if someone won or the game ended in a tie 
        #                 (Hint: you can check for a tie if there are no more valid moves)
        #            3.3  If the game is over, store the result
        #            3.4  If game is not over, change the player and continue the loop
        #
        # Update this node's total reward (self.w) and visit count (self.n) values to reflect this visit and result


        # Back-propagate this result
        # You do this by calling back on the parent of this node with the result of this simulation
        #    This should look like: self.parent.back(result)
        # Tip: you need to negate the result to account for the fact that the other player
        #    is the actor in the parent node, and so the scores will be from the opposite perspective
        print("not implemented")

    def back(self, score):
        #This updates the stats for this node, then backpropagates things 
        #to the parent (note the inverted score)
        self.n += 1
        self.w += score
        if self.parent is not None:
            self.parent.back(-score) #Score inverted before passing along


#UTILITY FUNCTIONS

#This function will modify the board according to 
#player_number moving into move column
def make_move(board,move,player_number):
    emptyrow = -1
    for row in range(5, -1, -1):
        if (board[row][move] == 0):
            emptyrow = row
            break
    if emptyrow != -1:
        board[emptyrow][move] = player_number

#This function will return a list of valid moves for the given board
def get_valid_moves(board):
    valid_moves = []
    for c in range(7):
        for r in range(6):
            if 0 == board[r][c]:
                valid_moves.append(c)
                break
    return valid_moves

#This function returns true if player_num is winning on board
def is_winning_state(board, player_num):
    #Check if winning for...

    #columns
    for currCol in range(0, 7):
        #possible start rows
        for startR in range(0, 3):
            #for every goal list

            isGoal1 = True #assume this goalList is all 0's
            isGoal2 = True
            for goalOffsetR in range(3, -1, -1):
                #go backwards, if we encounter a 0, the rest are also 0
                goalPosR = startR + goalOffsetR
                posVal = board[goalPosR][currCol]
                if (posVal == 1):
                    #filled1Positions += 1
                    isGoal2 = False
                elif (posVal == 2):
                    #filled2Positions += 1
                    isGoal1 = False
                else:
                    isGoal1 = False
                    isGoal2 = False

                if isGoal1 == False and isGoal2 == False:
                    break

            #add the valid goals to the total
            if isGoal1 == True and player_num == 1:
                return True
            if isGoal2 == True and player_num == 2:
                return True
    
    #rows
    for currRow in range(0, 6):
        #possible start cols
        for startC in range(0, 4):
            #for every goal list

            isGoal1 = True #assume this goalList is all 0's
            isGoal2 = True
            for goalOffsetC in range(0, 4):
                goalPosC = startC + goalOffsetC
                posVal = board[currRow][goalPosC]
                #if invalid position for a player
                if (posVal == 1):
                    isGoal2 = False
                elif (posVal == 2):
                    isGoal1 = False
                else:
                    isGoal1 = False
                    isGoal2 = False

                if isGoal1 == False and isGoal2 == False:
                    break
            
            #add the valid goals to the total
            if isGoal1 == True and player_num == 1:
                return True
            if isGoal2 == True and player_num == 2:
                return True
    
    #diagonals (top left to bottom right)
    for startRow in range(0, 3):
        for startCol in range(0, 4):
            #if invalid position for a player mark it as so
            isGoal1 = True #assume this goalList is all 0's
            isGoal2 = True
            #for each top left to down right goal list
            for tokenNum in range(0, 4):
                goalPosR = startRow + tokenNum
                goalPosC = startCol + tokenNum
                posVal = board[goalPosR][goalPosC]
                #if invalid position for a player
                if (posVal == 1):
                    isGoal2 = False
                elif (posVal == 2):
                    isGoal1 = False
                else:
                    isGoal1 = False
                    isGoal2 = False
                
                if isGoal1 == False and isGoal2 == False:
                    break
            #add the valid goals to the total
            if isGoal1 == True and player_num == 1:
                return True
            if isGoal2 == True and player_num == 2:
                return True

    #diagonals (bottom left to top right)
    for startRow in range(0, 3):
        for startCol in range(0, 4):
            #if invalid position for a player mark it as so
            isGoal1 = True #assume this goalList is all 0's
            isGoal2 = True

            #for each down left to top right goal list
            oppositeRow = 5 - startRow
            for tokenNum in range(0, 4):
                goalPosR = oppositeRow - tokenNum
                goalPosC = startCol + tokenNum
                posVal = board[goalPosR][goalPosC]
                #if invalid position for a player
                if (posVal == 1):
                    isGoal2 = False
                elif (posVal == 2):
                    isGoal1 = False
                else:
                    isGoal1 = False
                    isGoal2 = False
                
                if isGoal1 == False and isGoal2 == False:
                    break
            #add the valid goals to the total
            if isGoal1 == True and player_num == 1:
                return True
            if isGoal2 == True and player_num == 2:
                return True






    # player_win_str = '{0}{0}{0}{0}'.format(player_num)
    # to_str = lambda a: ''.join(a.astype(str))

    # def check_horizontal(b):
    #     for row in b:
    #         if player_win_str in to_str(row):
    #             return True
    #     return False

    # def check_verticle(b):
    #     return check_horizontal(b.T)

    # def check_diagonal(b):
    #     for op in [None, np.fliplr]:
    #         op_board = op(b) if op else b
            
    #         root_diag = np.diagonal(op_board, offset=0).astype(int)
    #         if player_win_str in to_str(root_diag):
    #             return True

    #         for i in range(1, b.shape[1]-3):
    #             for offset in [i, -i]:
    #                 diag = np.diagonal(op_board, offset=offset)
    #                 diag = to_str(diag.astype(int))
    #                 if player_win_str in diag:
    #                     return True

    #     return False

    # return (check_horizontal(board) or
    #         check_verticle(board) or
    #         check_diagonal(board))

