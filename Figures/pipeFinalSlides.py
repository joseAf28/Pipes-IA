# Grupo 026:
# 96546 José Filipe Bernardo Afonso
# 96554 Martim da Costa Graça Marques Ferreira


import sys
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)
import numpy as np


##! Initial Information
##? 1st - left, 2nd - right, 3rd - up, 4th - down
##? (x, y) - order


##! Globals variables as they are constant and they are used in the whole code
npType = np.int16
SquareVersors = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]], dtype=npType)

##! F points
arrFC = np.array([1, 0, 0, 0], dtype=npType)
arrFB = np.array([0, 1, 0, 0], dtype=npType)
arrFE = np.array([0, 0, 0, 1], dtype=npType)
arrFD = np.array([0, 0, 1, 0], dtype=npType)

##! B points
arrBC = np.array([1, 0, 1, 1], dtype=npType)
arrBB = np.array([0, 1, 1, 1], dtype=npType)
arrBE = np.array([1, 1, 0, 1], dtype=npType)
arrBD = np.array([1, 1, 1, 0], dtype=npType)

##! V points
arrVC = np.array([1, 0, 0, 1], dtype=npType)
arrVB = np.array([0, 1, 1, 0], dtype=npType)
arrVE = np.array([0, 1, 0, 1], dtype=npType)
arrVD = np.array([1, 0, 1, 0], dtype=npType)

###! L points
arrLH = np.array([0, 0, 1, 1], dtype=npType)
arrLV = np.array([1, 1, 0, 0], dtype=npType)

##! Total points in the square
vecTot = np.array([1, 1, 1, 1], dtype=npType)

##! Dictionary to convert the string to the array of the absolute coordinates
dictStr2Arr = { 
    'FC': arrFC.reshape(-1,1)*SquareVersors, 'FB': arrFB.reshape(-1,1)*SquareVersors, 
    'FE': arrFE.reshape(-1,1)*SquareVersors, 'FD': arrFD.reshape(-1,1)*SquareVersors,
    'BC': arrBC.reshape(-1,1)*SquareVersors, 'BB': arrBB.reshape(-1,1)*SquareVersors,
    'BE': arrBE.reshape(-1,1)*SquareVersors, 'BD': arrBD.reshape(-1,1)*SquareVersors,
    'VC': arrVC.reshape(-1,1)*SquareVersors, 'VB': arrVB.reshape(-1,1)*SquareVersors,
    'VE': arrVE.reshape(-1,1)*SquareVersors, 'VD': arrVD.reshape(-1,1)*SquareVersors,
    'LH': arrLH.reshape(-1,1)*SquareVersors, 'LV': arrLV.reshape(-1,1)*SquareVersors
}
arrTot = vecTot.reshape(-1, 1)*SquareVersors


class Board:
    """Representação interna de um tabuleiro de PipeMania."""
    
    def __init__(self, tableInit, maskFrameCheckInit):
        self.table = tableInit
        self.maskFrameCheck = maskFrameCheckInit
    
    @staticmethod
    def parse_instance():
        
        linesData = []
        while True:
            line = sys.stdin.readline()
            if line == '':  
                break
            else:
                linesData.append(line.split())
                
        dataArr = np.array(linesData)
        maskFrameCheck = np.full_like(dataArr, False, dtype=np.bool_)
    
        return Board(dataArr, maskFrameCheck)
    
    
    @staticmethod
    def lookUpFunc(x):
        """Converts the string to the array of the absolute coordinates."""
        ## Default to x if not found in dict
        return dictStr2Arr.get(x,x) 
    
    
    @staticmethod
    def getPointsSquare(x, y, arr):
        """Convert the relative coordinates to the absolute coordinates."""
        pos0 = np.array([2*x+1, 2*y+1], dtype=npType)
        
        mask = np.all(arr == 0, axis=1)
        result = np.full_like(arr, -1)
        result[~mask] = pos0 + arr[~mask]

        return result
    
    
    @staticmethod
    def getAllPointsSquare(x, y):
        """Get the absolute cordiantes of all the points in the boundaries of the square where the point (x, y) is located."""
        pos0 = np.array([2*x+1, 2*y+1], dtype=npType)
        result = arrTot + pos0
        return result
    
    
    @staticmethod
    def getAdjacentPoints(x, y, maskFrameCheck):
        """Get the absolute cordiantes of  points in the boundaries of the square where the adjacent pieces were sucessfully placed before."""
        checkFrame2 = np.full_like(maskFrameCheck, False)
        xMod = x-1
        yMod = y-1
        if xMod < 0:
            xMod = 0
        if yMod < 0:
            yMod = 0
        checkFrame2[xMod:x+2, y] = True
        checkFrame2[x, yMod:y+2] = True
        
        checkFrame2 = np.logical_and(checkFrame2, maskFrameCheck)
        return np.where(checkFrame2)
    
    
    def strategyOne(self, x2, y2, maskFrameCheck, boolLayerExt=False):
        """Strategy One: Get the possible actions for a given point (x2, y2) in the board."""
        
        mSize = self.table.shape[0]
        idx2 = self.getAdjacentPoints(x2, y2, maskFrameCheck)
        
        detAdjPoints = np.array([self.getPointsSquare(x, y, self.lookUpFunc(self.table[x,y])) for x, y in zip(*idx2)], dtype=npType)
        
        setF = np.array(['FC', 'FB', 'FE', 'FD'])
        setB = np.array(['BC', 'BB', 'BE', 'BD'])
        setV = np.array(['VC', 'VB', 'VE', 'VD'])
        setL = np.array(['LH', 'LV'])
        
        idPoint = self.table[x2, y2]
    
        ## get the points of the boundary
        pointsXY = self.getAllPointsSquare(x2, y2)
        
        ## get the points and the coordinates of the deterministic external points
        matches = np.array([np.any((detAdjPointsAux[:, np.newaxis, :] == pointsXY).all(axis=2), axis=1) for detAdjPointsAux in detAdjPoints])
        idxMatch = np.where(matches)
        idxMatchMat = np.array(list(zip(*idxMatch)))
        
        ## get the proper Match points on the boundary
        if idxMatchMat.size != 0:
            properPointsBoundary = detAdjPoints[idxMatchMat[:, 0], idxMatchMat[:, 1]]
        else:
            properPointsBoundary = np.array([], dtype=npType)
        
        
        ##! hipothessis of rejecting pieces that are not possible
        pointsAdjXY = np.array([self.getAllPointsSquare(x, y) for x, y in zip(*idx2)], dtype=npType)
        matchesAllPossible = np.array([np.any((pointsAdjXYAux[:, np.newaxis, :] == pointsXY).all(axis=2), axis=1) for pointsAdjXYAux in pointsAdjXY])
        idxMatchAllPossible = np.where(matchesAllPossible)
        idxMatchMatAllPossible = np.array(list(zip(*idxMatchAllPossible)))
        
        maskNotPossible = np.logical_and(matchesAllPossible, np.logical_not(matches))
        idxNotPossible = np.where(maskNotPossible)
        idxMatchMatNotPossible = np.array(list(zip(*idxNotPossible)))
        
        if idxMatchMatNotPossible.size != 0:
            properPointsBoundaryNot = pointsAdjXY[idxMatchMatNotPossible[:, 0], idxMatchMatNotPossible[:, 1]]
        else:
            properPointsBoundaryNot = np.array([], dtype=npType)
        
        
        actions = []
        
        if idPoint in setF:
            
            ##! exclude the adjacent F pieces cannot be connected
            adjacentPointsF = np.array([[x2-1,y2], [x2+1,y2], [x2,y2+1], [x2,y2-1]])
            adjacentPiecesF = []
            for row in adjacentPointsF:
                x, y = row
                if x == -1 or y == -1 or x == mSize or y == mSize:
                    adjacentPiecesF.append("N")
                else:
                    adjacentPiecesF.append(self.table[x, y])
            
            piecesFAdj = np.isin(adjacentPiecesF, setF)
            setNotF = np.full_like(setF, True, dtype=np.bool_)

            if piecesFAdj[0]:
                setNotF[0] = False
            if piecesFAdj[1]:
                setNotF[1] = False
            if piecesFAdj[2]:
                setNotF[3] = False
            if piecesFAdj[3]:
                setNotF[2] = False
            
            ## in case of not having points on the boundary, the set is returned
            if properPointsBoundary.size == 0 and properPointsBoundaryNot.size == 0 and np.logical_not(boolLayerExt):
                return setF
            
            ## get the points of all the possible points available
            gridPointsSet = np.array([self.getPointsSquare(x2, y2, self.lookUpFunc(x)) for x in setF], dtype=npType)
            
            if properPointsBoundaryNot.size !=0:
                checkSimilarityNot = np.array([np.any(np.all(gridPointsSetAux[:, np.newaxis, :] == properPointsBoundaryNot, axis=(2)), axis=1) for gridPointsSetAux in gridPointsSet], dtype=npType)
                checkSimilarityNot = np.sum(checkSimilarityNot, axis=1) != 0
            else:
                checkSimilarityNot = np.full((gridPointsSet.shape[0],), False, dtype=np.bool_)
            
            ## in case of not having points on the boundary and we arent in the external layer, the only the subset of setF that is compatible with those conditions is returned
            if properPointsBoundary.size == 0 and np.logical_not(boolLayerExt):
                
                idxCheckBoundaries = np.logical_and(np.logical_not(checkSimilarityNot), setNotF)
                return setF[idxCheckBoundaries]
            
            ## the same as above, but we are in the external layer
            elif properPointsBoundary.size == 0 and boolLayerExt:
                
                checkBoundaries = np.any(np.logical_or(gridPointsSet == 0, gridPointsSet == 2*mSize), axis=(1,2))
                idxCheckBoundaries = np.logical_not(np.logical_or(checkBoundaries, checkSimilarityNot))
                
                idxCheckBoundaries = np.logical_and(idxCheckBoundaries, setNotF)
                return setF[idxCheckBoundaries]
            
            ## now in the case of having points on the boundary and being in the external layer
            ## it returns the susbset of setF that is compatible with the conditions
            elif boolLayerExt:
                
                checkBoundaries = np.any(np.logical_or(gridPointsSet == 0, gridPointsSet == 2*mSize), axis=(1,2))
                idxCheckBoundaries = np.logical_not(np.logical_or(checkBoundaries, checkSimilarityNot))
                idxCheckBoundaries = np.logical_and(idxCheckBoundaries, setNotF)
                
            else:
                idxCheckBoundaries = np.logical_not(checkSimilarityNot)
            
            ## we must reject the subset of setF that allows that adjacent F pieces can be connected
            idxCheckBoundaries = np.logical_and(idxCheckBoundaries, setNotF)
            
            ## compute the similarity between the external points and the boundary points
            similarityVec = np.array([np.any(np.all(gridPointsSetAux[:, np.newaxis, :] == properPointsBoundary, axis=(2)), axis=1) for gridPointsSetAux in gridPointsSet], dtype=npType)
            similarityVec = np.sum(similarityVec, axis=1)
            
            ##! returns the set of actions that are compatible with the conditions that are compatible with adjacent pieces and gives the maximum similarity
            actions = setF[np.where(np.logical_and(similarityVec == np.max(similarityVec), idxCheckBoundaries))]
            
        elif idPoint in setV:
            ##! for the V pieces, the same logic as the F pieces is applied
            ##! the only exception is that there are no restriction on connecting the adjacent V pieces
            
            if properPointsBoundary.size == 0 and properPointsBoundaryNot.size == 0:
                return setV
            
            gridPointsSet = np.array([self.getPointsSquare(x2, y2, self.lookUpFunc(x)) for x in setV], dtype=npType)
            
            if properPointsBoundaryNot.size !=0:
                checkSimilarityNot = np.array([np.any(np.all(gridPointsSetAux[:, np.newaxis, :] == properPointsBoundaryNot, axis=(2)), axis=1) for gridPointsSetAux in gridPointsSet], dtype=npType)
                checkSimilarityNot = np.sum(checkSimilarityNot, axis=1) != 0
            else:
                checkSimilarityNot = np.full((gridPointsSet.shape[0],), False, dtype=np.bool_)


            if properPointsBoundary.size == 0 and np.logical_not(boolLayerExt):
                return setV[np.logical_not(checkSimilarityNot)]
            
            elif properPointsBoundary.size == 0 and boolLayerExt:

                checkBoundaries = np.any(np.logical_or(gridPointsSet == 0, gridPointsSet == 2*mSize), axis=(1,2))
                idxCheckBoundaries = np.logical_not(np.logical_or(checkBoundaries, checkSimilarityNot))
                
                return setV[idxCheckBoundaries]
            elif boolLayerExt:
                
                checkBoundaries = np.any(np.logical_or(gridPointsSet == 0, gridPointsSet == 2*mSize), axis=(1,2))
                idxCheckBoundaries = np.logical_not(np.logical_or(checkBoundaries, checkSimilarityNot))
                
            else:
                idxCheckBoundaries = np.logical_not(checkSimilarityNot)
            
            similarityVec = np.array([np.any(np.all(gridPointsSetAux[:, np.newaxis, :] == properPointsBoundary, axis=(2)), axis=1) for gridPointsSetAux in gridPointsSet], dtype=npType)
            similarityVec = np.sum(similarityVec, axis=1)
            
            ## find the best match(es) : all the options with less than the maximum similarity are discarded
            actions = setV[np.where(np.logical_and(similarityVec == np.max(similarityVec), idxCheckBoundaries))]

        elif idPoint in setL:
            ##! for the L pieces, the same logic as the F points is applied
            
            if properPointsBoundary.size == 0 and properPointsBoundaryNot.size == 0:
                return setL
            
            gridPointsSet = np.array([self.getPointsSquare(x2, y2, self.lookUpFunc(x)) for x in setL], dtype=npType)

            if properPointsBoundaryNot.size !=0:
                checkSimilarityNot = np.array([np.any(np.all(gridPointsSetAux[:, np.newaxis, :] == properPointsBoundaryNot, axis=(2)), axis=1) for gridPointsSetAux in gridPointsSet], dtype=npType)
                checkSimilarityNot = np.sum(checkSimilarityNot, axis=1) != 0
            else:
                checkSimilarityNot = np.full((gridPointsSet.shape[0],), False, dtype=np.bool_)
            
            checkSimilarityNot = np.logical_not(checkSimilarityNot)
            
            if properPointsBoundary.size != 0:
                similarityVec = np.array([np.any(np.all(gridPointsSetAux[:, np.newaxis, :] == properPointsBoundary, axis=(2)), axis=1) for gridPointsSetAux in gridPointsSet], dtype=npType)
                similarityVec = np.sum(similarityVec, axis=1)
            else:
                similarityVec = np.zeros((gridPointsSet.shape[0],), dtype=npType)
            
            # ## find the best match(es) : all the options with less than the maximum similarity are discarded
            actions = setL[np.where(np.logical_and(similarityVec == np.max(similarityVec), checkSimilarityNot))]
            
        elif idPoint in setB:
            ##! for the B pieces, we apply the same logic as before
            
            ## in case of not having points on the boundary, the full set is returned
            if properPointsBoundary.size == 0 and properPointsBoundaryNot.size == 0:
                return setB
            
            # ## get the points of all the possible points available
            gridPointsSet = np.array([self.getPointsSquare(x2, y2, self.lookUpFunc(x)) for x in setB], dtype=npType)
            
            if properPointsBoundaryNot.size !=0:
                checkSimilarityNot = np.array([np.any(np.all(gridPointsSetAux[:, np.newaxis, :] == properPointsBoundaryNot, axis=(2)), axis=1) for gridPointsSetAux in gridPointsSet], dtype=npType)
                checkSimilarityNot = np.sum(checkSimilarityNot, axis=1) != 0
            else:
                checkSimilarityNot = np.full((gridPointsSet.shape[0],), False, dtype=np.bool_)
            
            checkSimilarityNot = np.logical_not(checkSimilarityNot)
            
            ## compute the similarity between the external points and the boundary points
            if properPointsBoundary.size != 0:
                similarityVec = np.array([np.any(np.all(gridPointsSetAux[:, np.newaxis, :] == properPointsBoundary, axis=(2)), axis=1) for gridPointsSetAux in gridPointsSet], dtype=npType)
                similarityVec = np.sum(similarityVec, axis=1)
            else:
                similarityVec = np.zeros((gridPointsSet.shape[0],), dtype=npType)
            
            ## find the best match(es) : all the options with less than the maximum similarity are discarded
            actions = setB[np.where(np.logical_and(similarityVec == np.max(similarityVec), checkSimilarityNot))]
    
        return actions
    
    
    @staticmethod
    def findTheBestPoint(maskFrame, maskFrameCheck, mSize):
        """Find the best point to put a piece in the board."""
        x2 = 0
        y2 = 0
        counterMax = 0
        idxMaskFrame = np.where(maskFrame)
        for x, y in zip(*idxMaskFrame):
            maskAuxUpper = np.full_like(maskFrameCheck, False)
            xMin = x-1
            yMax = y-1
            xMax = x+1
            yMin = y+1
            if xMin < 0:
                xMin = 0
            if yMin < 0:
                yMin = 0
            if xMax >= mSize:
                xMax = mSize-1
            if yMin >= mSize:
                yMin = mSize-1
            maskAuxUpper[xMin, y] = True
            maskAuxUpper[x, yMax] = True
            maskAuxUpper[xMax, y] = True
            maskAuxUpper[x, yMin] = True
            
            maskResult = np.logical_and(maskAuxUpper, maskFrameCheck)
            
            counter = np.sum(maskResult)
            if counter > counterMax:
                counterMax = counter
                x2 = x
                y2 = y
                
        return x2, y2, counterMax
    
    
    def deterministicInference(self):
        """Method to deterministically infer the proper place of the pieces in the board for the external layer."""
        
        if self.table.shape[0] == 0:
            return True
        mSize = self.table.shape[0]
        
        dataPointsGrid = np.array([[self.getPointsSquare(x, y, self.lookUpFunc(self.table[x,y])) for y in range(mSize)] for x in range(mSize)], dtype=npType)
        
        setF = np.array(['FC', 'FB', 'FE', 'FD'])
        setB = np.array(['BC', 'BB', 'BE', 'BD'])
        setV = np.array(['VC', 'VB', 'VE', 'VD'])
        setL = np.array(['LH', 'LV'])
        
        ##! Zero Layer Border
        maskBorder = np.full_like(self.table, False, dtype=np.bool_)
        maskBorder[0, :] = True
        maskBorder[-1, :] = True
        maskBorder[:, 0] = True
        maskBorder[:, -1] = True
        
        ##! Corner Border
        matCorner = np.full_like(self.table, False, dtype=np.bool_)
        matCorner[0, 0] = True
        matCorner[0, -1] = True
        matCorner[-1, 0] = True
        matCorner[-1, -1] = True
        
        
        ##! Putting the outter layer deterministic pices in the proper place
        ##! Obtain the position of the external B, L and corner V points
        idxMatB = np.isin(self.table, setB)
        idxMatL = np.isin(self.table, setL)
        idxMatV = np.isin(self.table, setV)
        idxMatV = np.logical_and(idxMatV, matCorner)
        
        idxMatVBL = np.logical_or(np.logical_or(idxMatB, idxMatL), idxMatV)
            
        matDetermWrong = np.logical_and(np.any(np.logical_or(dataPointsGrid == 0,dataPointsGrid == 2*mSize), axis=(2,3)), idxMatVBL)
        matDetermRight = np.logical_and(np.logical_and(idxMatVBL, np.logical_not(matDetermWrong)), maskBorder)
        
        pointsDeterm = np.where(np.logical_and(idxMatVBL, matDetermWrong))
        idGridDeterm = np.array([self.table[x, y] for x, y in zip(*pointsDeterm)])
        
        idxDeterm = np.array(list(zip(*pointsDeterm)))
        
        ##! Put the deterministic pieces in the proper place
        for i in range(idxDeterm.shape[0]):
            x, y = idxDeterm[i]
            idPoint = self.table[x, y]
            
            if idPoint in setB:
                
                gridPointsSet = np.array([self.getPointsSquare(x, y, self.lookUpFunc(piece)) for piece in setB], dtype=npType)
                checkBoundaries = np.any(np.logical_or(gridPointsSet == 0, gridPointsSet == 2*mSize), axis=(1,2))
                
                idxCheckBoundaries = np.where(~checkBoundaries)[0]
                
                if len(idxCheckBoundaries) > 1:
                    pass
                else:
                    self.table[x, y] = setB[idxCheckBoundaries[0]]
                
            elif idPoint in setL:
                
                gridPointsSet = np.array([self.getPointsSquare(x, y, self.lookUpFunc(piece)) for piece in setL], dtype=npType)
                checkBoundaries = np.any(np.logical_or(gridPointsSet == 0, gridPointsSet == 2*mSize), axis=(1,2))
                
                idxCheckBoundaries = np.where(~checkBoundaries)[0]
                
                if len(idxCheckBoundaries) > 1:
                    pass
                else:
                    self.table[x,y] = setL[idxCheckBoundaries[0]]

            elif idPoint in setV:
                
                gridPointsSet = np.array([self.getPointsSquare(x, y, self.lookUpFunc(piece)) for piece in setV], dtype=npType)
                checkBoundaries = np.any(np.logical_or(gridPointsSet == 0, gridPointsSet == 2*mSize), axis=(1,2))
                
                idxCheckBoundaries = np.where(~checkBoundaries)[0]
                
                if len(idxCheckBoundaries) > 1:
                    pass
                else:
                    self.table[x, y] = setV[idxCheckBoundaries[0]]
        
        ##! combones the wrong and right pieces
        matDetermBoth = np.logical_or(matDetermRight, matDetermWrong)
        idxDetermBoth = np.where(matDetermBoth)
        
        ##! the maskFrameCheck tells us which points are already determined
        ##! Each piece is marked in table in which True means that the piece is already determined and False otherwise
        self.maskFrameCheck = np.full_like(self.table, False, dtype=np.bool_)
        
        ##! Zero Iteration
        maskAux = np.full_like(self.table, False, dtype=np.bool_)
        for x, y in zip(*idxDetermBoth):
            self.maskFrameCheck[x, y] = True
            xMod = x-1
            yMod = y-1
            if xMod < 0:
                xMod = 0
            if yMod < 0:
                yMod = 0
                
            maskAux[xMod:x+2, y] = True
            maskAux[x, yMod:y+2] = True
            
        for x, y in zip(*idxDetermBoth):
            maskAux[x, y] = False
        
        maskBorderZero = np.logical_and(maskBorder, maskAux)
        
        idx2ChangeBorderZero = np.array(list(zip(*np.where(maskBorderZero))))
        idx2FrameCheck = np.array(list(zip(*np.where(self.maskFrameCheck))))
        
        
        for i in range(idx2ChangeBorderZero.shape[0]):
            x2, y2 = idx2ChangeBorderZero[i]
            actions = self.strategyOne(x2, y2, self.maskFrameCheck, True)
            
            if len(actions) == 1:
                self.table[x2, y2] = actions[0]
                self.maskFrameCheck[x2, y2] = True
            else:
                pass



class PipeManiaState:
    """Representação interna de um estado do PipeMania."""
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = PipeManiaState.state_id
        PipeManiaState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id



class PipeMania(Problem):
    
    def __init__(self, board: Board):
        self.initial = PipeManiaState(board)
    
    
    @staticmethod
    def updateAdjacentMultiple(matrix, entries):
        """Set the adjacent positions of the entries to True in the matrix."""
        maxRows, maxCols = matrix.shape

        # Create arrays for row and column indices for each entry
        rowIndices = np.array(entries)[:, 0]
        colIndices = np.array(entries)[:, 1]

        # Define relative positions for adjacent cells
        relativePositions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

        # Compute all adjacent positions
        allPositions = rowIndices[:, None] + relativePositions[:, 0], colIndices[:, None] + relativePositions[:, 1]

        # Filter positions that are inside the matrix bounds
        valid = (allPositions[0] >= 0) & (allPositions[0] < maxRows) & \
                (allPositions[1] >= 0) & (allPositions[1] < maxCols)

        # Flatten valid positions and make them unique to avoid repeated setting of the same cell
        validRows = allPositions[0][valid]
        validRols = allPositions[1][valid]
        uniquePositions = np.unique((validRows, validRols), axis=1)

        # Set the valid adjacent positions to True
        matrix[uniquePositions[0], uniquePositions[1]] = True
    
    
    @staticmethod
    def isArrayVectorized(x):
        """Check if the each element of the array is np.ndarray"""
        return np.vectorize(lambda x: isinstance(x, np.ndarray))(x)
    
    
    def actionsAux(self, state: PipeManiaState):
        """Auxiliar Function that returns the non infered actions from a state."""
        
        ###! every inferered action is directly implemented in the state
        ###! only the nearby pieces of the changed pieces are calculated inside the loop
        ###! Since this function changes the state, it cannot be defined as the "actions" method and it must be
        ###! called inside the "actions" method and to a copy of the state
        
        boolDeterm = True
        boolInit = True
        mSize = state.board.maskFrameCheck.shape[0]-1
        
        matMarked = np.full_like(state.board.maskFrameCheck, False, dtype=np.bool_)
        matActions = np.full_like(state.board.maskFrameCheck, 0, dtype=object)
        
        while boolDeterm: 
            if boolInit:
                idxActions = np.where(state.board.maskFrameCheck == False)
                boolInit = False
            else:
                # update just the near terms
                markedPoints = list(zip(*np.where(matMarked)))
                matrixAdjacents = np.full_like(state.board.maskFrameCheck, False, dtype=np.bool_)
                self.updateAdjacentMultiple(matrixAdjacents, markedPoints)
                
                idxActions = np.where(np.logical_and(state.board.maskFrameCheck == False, matrixAdjacents))
                
            sizeDeterm = 0
            for x, y in zip(*idxActions):
                
                if x == 0 or y == 0 or x == mSize or y == mSize:
                    actions = state.board.strategyOne(x, y, state.board.maskFrameCheck, True)
                else:
                    actions = state.board.strategyOne(x, y, state.board.maskFrameCheck)
                
                if len(actions) == 0:
                    matActions[x,y] = [-1]
                    
                elif len(actions) == 1:
                    state.board.table[x,y] = actions[0]
                    state.board.maskFrameCheck[x,y] = True
                    sizeDeterm += 1
                    
                    matMarked[x,y] = True
                    matActions[x,y] = 0
                else:
                    matActions[x,y] = actions
                
            if sizeDeterm == 0:
                boolDeterm = False
        
        actionNonDeterm = []
        
        for x, y in zip(*np.where(self.isArrayVectorized(matActions))):
            for action in matActions[x,y]:
                actionNonDeterm.append([x, y, action])
                
                if action == -1:
                    return np.array([], dtype=object)
        
        actionNonDeterm = np.array(actionNonDeterm, dtype=object)
        
        ##! We use an Heuristic for the non-deterministic actions:
        ##!  1. MRV heuristic: the pieces that have less options are put first
        ##!     1.1 Higher degree heuristic: the pieces that have more adjacent pieces are put first
        
        if actionNonDeterm.size == 0:
            return np.array([], dtype=object)
            
        else:
            countVec = np.sum(np.all(actionNonDeterm[:, np.newaxis, :2] == actionNonDeterm[:, :2], axis=2), axis=1)
            
            uniqueSize = np.unique(countVec)
            actionCounts = np.array([actionNonDeterm[countVec == size] for size in uniqueSize], dtype=object)
            
            setF = np.array(['FC', 'FB', 'FE', 'FD'])
            setB = np.array(['BC', 'BB', 'BE', 'BD'])
            setV = np.array(['VC', 'VB', 'VE', 'VD'])
            setL = np.array(['LH', 'LV'])
            
            for i in range(actionCounts.shape[0]):
                
                if actionCounts[i].shape[0] < 2:
                    continue
                else: 
                    refVec = np.zeros_like(actionCounts[i][:,2], dtype=np.int_)
                    
                    idxB = np.isin(actionCounts[i][:,2], setB)
                    idxL = np.isin(actionCounts[i][:,2], setL)
                    idxV = np.isin(actionCounts[i][:,2], setV)
                    idxF = np.isin(actionCounts[i][:,2], setF)
                    
                    ##! the order of the pieces is B, L, V, F
                    refVec[idxB] = 1
                    refVec[idxL] = 2
                    refVec[idxV] = 3
                    refVec[idxF] = 4
                    
                    idxSort = np.argsort(refVec)
                    actionCounts[i] = actionCounts[i][idxSort]
                    
            actionNonDeterm = np.concatenate(actionCounts)
            
            return actionNonDeterm
    
    
    def actions(self, state: PipeManiaState):
        """Returns the actions that can be taken in the current state."""
        
        newTable = state.board.table
        newMaskFrameCheck = state.board.maskFrameCheck
        newBoard = Board(newTable, newMaskFrameCheck)
        newPipeManiaState = PipeManiaState(newBoard)
        actions = self.actionsAux(newPipeManiaState)
        
        return actions
    
    
    def result(self, state: PipeManiaState, action):
        """Returns the state that results from taking the given action in the current state."""
        
        ##! Create a new state with the action applied
        x = action[0]
        y = action[1]
        action = action[2]
        
        newTable = state.board.table.copy()
        newMaskFrameCheck = state.board.maskFrameCheck.copy()
        
        newBoard = Board(newTable, newMaskFrameCheck)
        newPipeManiaState = PipeManiaState(newBoard)
        
        ##! Apply the action
        newPipeManiaState.board.table[x, y] = action
        newPipeManiaState.board.maskFrameCheck[x, y] = True
        
        return newPipeManiaState
    
    
    def goal_testAux(self, state: PipeManiaState):
        """Auxiliar Goal Test Function where we actually check if the state is a goal state"""
        
        tableArr = np.array([[state.board.lookUpFunc(x) for x in row] for row in state.board.table], dtype=npType)
        mSize = tableArr.shape[0]
        
        dataPointsGrid = np.array([[state.board.getPointsSquare(x, y, tableArr[x,y]) for y in range(mSize)] for x in range(mSize)], dtype=npType)
        
        dotsX2 = np.zeros((mSize-1, mSize, 2, 2), dtype=npType) ##! slices along x constant
        dotsY2 = np.zeros((mSize, mSize-1, 2, 2), dtype=npType) ##! slices along y constant
        
        dotsX2[:, :, 0, :] = dataPointsGrid[:-1, :, 1]
        dotsX2[:, :, 1, :] = dataPointsGrid[1:, :, 0]

        dotsY2[:, :, 0, :] = dataPointsGrid[:, :-1, 2]
        dotsY2[:, :, 1, :] = dataPointsGrid[:, 1:, 3]
        
        matLogicalX = (dotsX2[:, :, 0, :] == dotsX2[:, :, 1, :]) & (np.any(dotsX2[:, :, 1, :] > -1)) & (np.any(dotsX2[:, :, 0, :] > -1))
        matLogicalY = (dotsY2[:, :, 0, :] == dotsY2[:, :, 1, :]) & (np.any(dotsX2[:, :, 1, :] > -1)) & (np.any(dotsX2[:, :, 1, :] > -1))
        
        resLogicalX = np.all(matLogicalX, axis=2)
        resLogicalY = np.all(matLogicalY, axis=2)
        
        return np.all(resLogicalX) and np.all(resLogicalY)
    
    
    def goal_test(self, state: PipeManiaState):
        """The goal test function: Returns True if the state is a goal state, False otherwise.
        Before checking the goal, we apply forward checking to the state with the purpose of infering all the pieces possible as a result
        of applying the action selected by the search algorithm."""
        
        self.actionsAux(state)
        if self.goal_testAux(state):
            return True
        else:
            return False
    
    
    def h(self, node: Node):
        ##! Not used in the current implementation as we manipulate the order in which the actions are going to be 
        ##! executed in the actionsAux method by the search algorithm
        pass


if __name__ == "__main__":
    
    ##! Obtain the Board from the standard input
    board = Board.parse_instance()
    
    ##? Deterministic Inference
    ##! Start with the Board class and simplify the outer layer as much as possible.
    board.deterministicInference()
    
    
    ##! Create the problem and the initial state
    problem = PipeMania(board)
    s0 = PipeManiaState(board)
    
    
    ##! Create the node and apply the search algorithm
    node = depth_first_tree_search(problem)
    
    
    ##! Print the output
    table = node.state.board.table
    with open("output.txt", "w") as f:
        for i, row in enumerate(table):
            for j, x in enumerate(row):
                if j < len(row)-1:
                    f.write(x + "\t")
                else:
                    f.write(x + "\n")