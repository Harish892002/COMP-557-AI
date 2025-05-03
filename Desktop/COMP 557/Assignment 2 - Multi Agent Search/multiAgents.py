# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    """

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.
        Returns a Directions.X for X in {North, South, West, East, Stop}
        """
        legalMoves = gameState.getLegalActions()
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [i for i in range(len(scores)) if scores[i] == bestScore]
        return legalMoves[random.choice(bestIndices)]

    def evaluationFunction(self, currentGameState, action):
        """
        Evaluation function that considers food proximity, ghost positions, and capsules
        to determine the best action for Pacman.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        
        # Food strategy: reward proximity to closest food
        foodList = newFood.asList()
        foodBonus = 100 if not foodList else 1.0 / (min([manhattanDistance(newPos, food) for food in foodList]) + 1)
        
        # Ghost strategy: avoid active ghosts, chase scared ones
        ghostScore = 0
        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            dist = manhattanDistance(newPos, ghostPos)
            
            if ghost.scaredTimer > 0:  # Chase scared ghosts
                ghostScore += 1.0 / (dist + 1)
            elif dist < 2:  # Avoid close active ghosts
                return -float('inf')
            else:  # Maintain distance from active ghosts
                ghostScore -= 3.0 / dist
        
        # Capsule strategy: prioritize power pellets
        capsules = currentGameState.getCapsules()
        capsuleBonus = 0 if not capsules else 1.0 / (min([manhattanDistance(newPos, cap) for cap in capsules]) + 1)
        
        # Penalize remaining food
        foodLeftPenalty = len(foodList) * 2.0
        
        # Combine all features
        return successorGameState.getScore() + foodBonus + ghostScore + capsuleBonus - foodLeftPenalty

def scoreEvaluationFunction(currentGameState):
    """
    Returns the score of the state as displayed in the Pacman GUI.
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    Abstract class for multi-agent search algorithms.
    """
    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent implementation.
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using depth-limited search.
        """
        def minimax(state, agentIndex, depth):
            # Terminal state check
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            # Determine next agent and handle depth changes
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth + 1 if nextAgent == 0 else depth
            
            # Maximizing player (Pacman)
            if agentIndex == 0:
                return max((minimax(state.generateSuccessor(agentIndex, action), 1, depth) for action in state.getLegalActions(agentIndex)),default=-float('inf'))
            
            # Minimizing players (Ghosts)
            else:
                return min((minimax(state.generateSuccessor(agentIndex, action), nextAgent, nextDepth) for action in state.getLegalActions(agentIndex)),default=float('inf'))
        
        # Find best action for Pacman
        return max(gameState.getLegalActions(0),key=lambda action: minimax(gameState.generateSuccessor(0, action), 1, 0))

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Minimax agent with alpha-beta pruning.
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using alpha-beta pruning.
        """
        def alphaBeta(state, depth, agentIndex, alpha, beta):
            # Terminal state check
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            # Setup for next agent and depth
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth + 1 if nextAgent == 0 else depth
            
            # No legal actions check
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)
            
            # Maximizing agent (Pacman)
            if agentIndex == 0:
                value = -float('inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, alphaBeta(successor, nextDepth, nextAgent, alpha, beta))
                    if value > beta:  # Pruning
                        return value
                    alpha = max(alpha, value)
                return value
            # Minimizing agents (Ghosts)
            else:
                value = float('inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = min(value, alphaBeta(successor, nextDepth, nextAgent, alpha, beta))
                    if value < alpha:  # Pruning
                        return value
                    beta = min(beta, value)
                return value
        
        # Find best action using alpha-beta search
        bestAction = None
        bestValue = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        for action in gameState.getLegalActions(0):
            value = alphaBeta(gameState.generateSuccessor(0, action), 0, 1, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)
        
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent that models ghosts with random probability distributions.
    """
    def getAction(self, gameState):
        """
        Returns the expectimax action using depth-limited search.
        """
        def expectimax(state, agentIndex, depth):
            # Terminal state check
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            # Setup for next agent and depth
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth + 1 if nextAgent == 0 else depth
            
            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)
            
            # Maximizing agent (Pacman)
            if agentIndex == 0:
                return max(
                    expectimax(state.generateSuccessor(agentIndex, action), nextAgent, nextDepth)
                    for action in actions
                )
            # Chance nodes (Ghosts)
            else:
                # Average over all possible ghost actions (uniform probability)
                return sum(expectimax(state.generateSuccessor(agentIndex, action), nextAgent, nextDepth) for action in actions) / len(actions)
        
        # Find best action for Pacman using expectimax
        return max(gameState.getLegalActions(0),key=lambda action: expectimax(gameState.generateSuccessor(0, action), 1, 0))

def betterEvaluationFunction(currentGameState):
    """
    DESCRIPTION: This function evaluates states by considering:
    1) Current score as baseline
    2) Food proximity (closer is better)
    3) Ghost behavior (avoid active ghosts, chase scared ones)
    4) Power pellet strategy
    5) Remaining food and capsules (fewer is better)
    """
    # Handle terminal states
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return -float('inf')
    
    # Extract game state information
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()
    
    # Food proximity reward
    if food:
        minFoodDistance = min(manhattanDistance(pacmanPos, f) for f in food)
        score += 10.0 / (minFoodDistance + 1)
    else:
        score += 100  # Big bonus for clearing all food
    
    # Ghost behavior management
    for ghostState in ghostStates:
        dist = manhattanDistance(pacmanPos, ghostState.getPosition())
        
        # Immediate loss check
        if dist == 0 and ghostState.scaredTimer == 0:
            return -float('inf')
        
        # Handle scared vs. active ghosts
        if ghostState.scaredTimer > 0:
            score += 5.0 / (dist + 1)  # Chase scared ghosts
        else:
            score -= 200 if dist < 2 else 2.0 / (dist + 1)  # Avoid active ghosts
    
    # Capsule strategy
    if capsules:
        minCapsuleDist = min(manhattanDistance(pacmanPos, c) for c in capsules)
        score += 5.0 / (minCapsuleDist + 1)
    score -= 20 * len(capsules)  # Encourage collecting capsules
    
    # Penalty for remaining food
    score -= 4 * len(food)
    
    return score

# Abbreviation
better = betterEvaluationFunction
