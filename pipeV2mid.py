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


dictStr2Arr = { 
    'FC': arrFC, 'FB': arrFB, 'FE': arrFE, 'FD': arrFD,
    'BC': arrBC, 'BB': arrBB, 'BE': arrBE, 'BD': arrBD,
    'VC': arrVC, 'VB': arrVB, 'VE': arrVE, 'VD': arrVD,
    'LH': arrLH, 'LV': arrLV
}


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
        ## Default to x if not found in dict
        return dictStr2Arr.get(x,x) 
    
    
    @staticmethod
    def getPointsSquare(x, y, arr):
        pos0 = np.array([2*x+1, 2*y+1], dtype=npType)
        vec = arr.reshape(-1, 1)*SquareVersors
        
        mask = np.all(vec == 0, axis=1)
        result = np.full_like(vec, -1)
        result[~mask] = pos0 + vec[~mask]

        return result
    
    
    @staticmethod
    def getAllPointsSquare(x, y):
        pos0 = np.array([2*x+1, 2*y+1], dtype=npType)
        vec = vecTot.reshape(-1, 1)*SquareVersors
        
        result = vec + pos0
        return result
    
    
    @staticmethod
    def getAdjacentPoints(x, y, maskFrameCheck):
        
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
        
        mSize = self.table.shape[0]
        idx2 = self.getAdjacentPoints(x2, y2, maskFrameCheck)
        
        # detAdjPoints = np.array([self.getPointsSquare(x, y, self.tableArr[x,y]) for x, y in zip(*idx2)], dtype=npType)
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
            
            if properPointsBoundary.size == 0 and properPointsBoundaryNot.size == 0:
                return setF
            
            ## get the points of all the possible points available
            gridPointsSet = np.array([self.getPointsSquare(x2, y2, self.lookUpFunc(x)) for x in setF], dtype=npType)
            
            if properPointsBoundaryNot.size !=0:
                checkSimilarityNot = np.array([np.any(np.all(gridPointsSetAux[:, np.newaxis, :] == properPointsBoundaryNot, axis=(2)), axis=1) for gridPointsSetAux in gridPointsSet], dtype=npType)
                checkSimilarityNot = np.sum(checkSimilarityNot, axis=1) != 0
            else:
                checkSimilarityNot = np.full((gridPointsSet.shape[0],), False, dtype=np.bool_)
            
            # idxCheckBoundaries = np.full((gridPointsSet.shape[0],), True, dtype=np.bool_)
            
            ## in case of not having points on the boundary, the full set is returned and we arent in the external layer
            if properPointsBoundary.size == 0 and np.logical_not(boolLayerExt):
                return setF[np.logical_not(checkSimilarityNot)]
            
            elif properPointsBoundary.size == 0 and boolLayerExt:
                
                checkBoundaries = np.any(np.logical_or(gridPointsSet == 0, gridPointsSet == 2*mSize), axis=(1,2))
                idxCheckBoundaries = np.logical_not(np.logical_or(checkBoundaries, checkSimilarityNot))
                
                return setF[idxCheckBoundaries]
            elif boolLayerExt:
                
                checkBoundaries = np.any(np.logical_or(gridPointsSet == 0, gridPointsSet == 2*mSize), axis=(1,2))
                idxCheckBoundaries = np.logical_not(np.logical_or(checkBoundaries, checkSimilarityNot))
            
            else:
                idxCheckBoundaries = np.logical_not(checkSimilarityNot)
            
            ## compute the similarity between the external points and the boundary points
            similarityVec = np.array([np.any(np.all(gridPointsSetAux[:, np.newaxis, :] == properPointsBoundary, axis=(2)), axis=1) for gridPointsSetAux in gridPointsSet], dtype=npType)
            similarityVec = np.sum(similarityVec, axis=1)
        
            ## find the best match(es) : all the options with less than the maximum similarity are discarded
            actions = setF[*np.where(np.logical_and(similarityVec == np.max(similarityVec), idxCheckBoundaries))]
            
        elif idPoint in setV:
            
            ## in case of not having points on the boundary, the full set is returned
            if properPointsBoundary.size == 0 and properPointsBoundaryNot.size == 0:
                return setV
            
            ## get the points of all the possible points available
            gridPointsSet = np.array([self.getPointsSquare(x2, y2, self.lookUpFunc(x)) for x in setV], dtype=npType)
            
            if properPointsBoundaryNot.size !=0:
                checkSimilarityNot = np.array([np.any(np.all(gridPointsSetAux[:, np.newaxis, :] == properPointsBoundaryNot, axis=(2)), axis=1) for gridPointsSetAux in gridPointsSet], dtype=npType)
                checkSimilarityNot = np.sum(checkSimilarityNot, axis=1) != 0
            else:
                checkSimilarityNot = np.full((gridPointsSet.shape[0],), False, dtype=np.bool_)
            
            
            ## in case of not having points on the boundary, the full set is returned and we arent in the external layer
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
            
            
            ## compute the similarity between the external points and the boundary points
            similarityVec = np.array([np.any(np.all(gridPointsSetAux[:, np.newaxis, :] == properPointsBoundary, axis=(2)), axis=1) for gridPointsSetAux in gridPointsSet], dtype=npType)
            similarityVec = np.sum(similarityVec, axis=1)
            
            # ## find the best match(es) : all the options with less than the maximum similarity are discarded
            actions = setV[*np.where(np.logical_and(similarityVec == np.max(similarityVec), idxCheckBoundaries))]
            # actions = setV[*np.where(np.logical_and(similarityVec == np.max(similarityVec), np.logical_not(checkSimilarityNot)))]
            
        elif idPoint in setL:
            
            if properPointsBoundary.size == 0 and properPointsBoundaryNot.size == 0:
                return setL
            
            # ## get the points of all the possible points available
            gridPointsSet = np.array([self.getPointsSquare(x2, y2, self.lookUpFunc(x)) for x in setL], dtype=npType)

            if properPointsBoundaryNot.size !=0:
                checkSimilarityNot = np.array([np.any(np.all(gridPointsSetAux[:, np.newaxis, :] == properPointsBoundaryNot, axis=(2)), axis=1) for gridPointsSetAux in gridPointsSet], dtype=npType)
                checkSimilarityNot = np.sum(checkSimilarityNot, axis=1) != 0
            else:
                checkSimilarityNot = np.full((gridPointsSet.shape[0],), False, dtype=np.bool_)
            
            checkSimilarityNot = np.logical_not(checkSimilarityNot)
            
            # ## compute the similarity between the external points and the boundary points
            # if properPointsBoundary.size == 0 and np.logical_not(boolLayerExt):
            #     return setL[np.logical_not(checkSimilarityNot)]
            
            # elif properPointsBoundary.size == 0 and boolLayerExt:
                
            #     checkBoundaries = np.any(np.logical_or(gridPointsSet == 0, gridPointsSet == 2*mSize), axis=(1,2))
            #     idxCheckBoundaries = np.logical_not(np.logical_or(checkBoundaries, checkSimilarityNot))
                
            #     return setL[idxCheckBoundaries]
            # elif boolLayerExt:
                
            #     checkBoundaries = np.any(np.logical_or(gridPointsSet == 0, gridPointsSet == 2*mSize), axis=(1,2))
            #     idxCheckBoundaries = np.logical_not(np.logical_or(checkBoundaries, checkSimilarityNot))
            
            # else:
            #     idxCheckBoundaries = np.logical_not(checkSimilarityNot)
            
            # similarityVec = np.array([np.any(np.all(gridPointsSetAux[:, np.newaxis, :] == properPointsBoundary, axis=(2)), axis=1) for gridPointsSetAux in gridPointsSet], dtype=npType)
            # similarityVec = np.sum(similarityVec, axis=1)
            
            
            if properPointsBoundary.size != 0:
                similarityVec = np.array([np.any(np.all(gridPointsSetAux[:, np.newaxis, :] == properPointsBoundary, axis=(2)), axis=1) for gridPointsSetAux in gridPointsSet], dtype=npType)
                similarityVec = np.sum(similarityVec, axis=1)
            else:
                similarityVec = np.zeros((gridPointsSet.shape[0],), dtype=npType)
            
            # ## find the best match(es) : all the options with less than the maximum similarity are discarded
            actions = setL[*np.where(np.logical_and(similarityVec == np.max(similarityVec), checkSimilarityNot))]
            
        elif idPoint in setB:
            
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
            
            # ## find the best match(es) : all the options with less than the maximum similarity are discarded
            actions = setB[*np.where(np.logical_and(similarityVec == np.max(similarityVec), checkSimilarityNot))]
    
        return actions

    

    @staticmethod
    def findTheBestPoint(maskFrame, maskFrameCheck, mSize):
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
    
    
    
    def strategyDetermisticLoop(self, loopNumber):
        
        mSize = self.table.shape[0]
        
        maskLevelUpper = np.full_like(self.table, False, dtype=np.bool_)
        maskLevelUpper[loopNumber:-loopNumber, loopNumber] = True
        maskLevelUpper[loopNumber:-loopNumber, -loopNumber-1] = True
        maskLevelUpper[loopNumber, loopNumber:-loopNumber] = True
        maskLevelUpper[-loopNumber-1, loopNumber:-loopNumber] = True
        
        boolUpperLayer = np.any(np.logical_and(maskLevelUpper, np.logical_not(self.maskFrameCheck)))
        
        ##! Fill the upper layer: it ends up when there are no more deterministic points to fill
        while boolUpperLayer:
            x2, y2, counterMax = self.findTheBestPoint(maskLevelUpper, self.maskFrameCheck, mSize)
            
            if counterMax == 0:
                boolUpperLayer = False
            else:
                actions = self.strategyOne(x2, y2, self.maskFrameCheck, False)
                
                if len(actions) == 1:
                    self.table[x2, y2] = actions[0]
                    self.tableArr[x2, y2] = self.lookUpFunc(actions[0])
                    self.maskFrameCheck[x2, y2] = True
                    self.matActions[x2, y2] = 0
                else:
                    self.matActions[x2, y2] = actions
                
                maskLevelUpper[x2, y2] = False
        
        
        maskLevelUpper = np.full_like(self.table, False, dtype=np.bool_)
        maskLevelUpper[loopNumber:-loopNumber, loopNumber] = True
        maskLevelUpper[loopNumber:-loopNumber, -loopNumber-1] = True
        maskLevelUpper[loopNumber, loopNumber:-loopNumber] = True
        maskLevelUpper[-loopNumber-1, loopNumber:-loopNumber] = True
        
        maskFrameCheck2nd = np.logical_and(np.logical_not(self.maskFrameCheck), maskLevelUpper)
        
        for x2, y2 in zip(*np.where(maskFrameCheck2nd)):
            actions = self.strategyOne(x2, y2, self.maskFrameCheck)
            
            if len(actions) == 1:
                self.table[x2, y2] = actions[0]
                self.tableArr[x2, y2] = self.lookUpFunc(actions[0])
                self.maskFrameCheck[x2, y2] = True
                self.matActions[x2, y2] = 0
            else:
                self.matActions[x2, y2] = actions
    
    
    def deterministicInference(self):
        
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
                
                ##? future improvement: remove the wrong pieces from the set at trying to put them in the right place
                gridPointsSet = np.array([self.getPointsSquare(x, y, self.lookUpFunc(piece)) for piece in setB], dtype=npType)
                checkBoundaries = np.any(np.logical_or(gridPointsSet == 0, gridPointsSet == 2*mSize), axis=(1,2))
                
                idxCheckBoundaries = np.where(~checkBoundaries)[0]
                
                if len(idxCheckBoundaries) > 1:
                    print("error: more than one possible solution")
                else:
                    self.table[x, y] = setB[idxCheckBoundaries[0]]
                
            elif idPoint in setL:
                
                gridPointsSet = np.array([self.getPointsSquare(x, y, self.lookUpFunc(piece)) for piece in setL], dtype=npType)
                checkBoundaries = np.any(np.logical_or(gridPointsSet == 0, gridPointsSet == 2*mSize), axis=(1,2))
                
                idxCheckBoundaries = np.where(~checkBoundaries)[0]
                
                if len(idxCheckBoundaries) > 1:
                    print("error: more than one possible solution")
                else:
                    self.table[x,y] = setL[idxCheckBoundaries[0]]

            elif idPoint in setV:
                
                gridPointsSet = np.array([self.getPointsSquare(x, y, self.lookUpFunc(piece)) for piece in setV], dtype=npType)
                checkBoundaries = np.any(np.logical_or(gridPointsSet == 0, gridPointsSet == 2*mSize), axis=(1,2))
                
                idxCheckBoundaries = np.where(~checkBoundaries)[0]
                
                if len(idxCheckBoundaries) > 1:
                    print("error: more than one possible solution")
                else:
                    self.table[x, y] = setV[idxCheckBoundaries[0]]
        
        ##! Build the maskFrameCheck and matActions matrices
        matDetermBoth = np.logical_or(matDetermRight, matDetermWrong)
        idxDetermBoth = np.where(matDetermBoth)
        
        ##! the maskFrameCheck tells us which points are already determined
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
            actions = self.strategyOne(x2, y2, self.maskFrameCheck)
            
            if len(actions) == 1:
                self.table[x2, y2] = actions[0]
                self.maskFrameCheck[x2, y2] = True
            else:
                pass



    def is_goal(self):
        return np.all(self.maskFrameCheck)
    
    


class PipeManiaState:
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


    # old version: less efficient
    # def actions(self, state: PipeManiaState):
    #     """Devolve as ações não deterministicas possíveis a partir de um estado."""
        
    #     ### todas as acões deterministas são diretamente implementadas no estado
        
    #     ####? Aumentar a eficiência do código: apenas as ações das peças nearby das peças mudadas são calculadas
        
    #     boolDeterm = True
    #     sizeDeterm = 0
    #     mSize = state.board.maskFrameCheck.shape[0]-1
        
    #     while boolDeterm: 
            
            
    #         ## update just the near terms
    #         idxActions = np.where(state.board.maskFrameCheck == False)
            
    #         actionNonDeterm = []
    #         actionDeterm = []
    #         sizeDeterm = 0
            
            
    #         for x, y in zip(*idxActions):
                
    #             if x == 0 or y == 0 or x == mSize or y == mSize:
    #                 actions = board.strategyOne(x, y, state.board.maskFrameCheck, True)
    #             else:
    #                 actions = board.strategyOne(x, y, state.board.maskFrameCheck)
                
    #             if len(actions) == 1:
    #                 actionDeterm.append([x, y, actions[0]])
    #                 sizeDeterm += 1
                    
    #             else:
    #                 for action in actions:
    #                     actionNonDeterm.append([x, y, action])
                
    #         # print("size D:" , sizeDeterm)
                
    #         if sizeDeterm == 0:
    #             boolDeterm = False
    #         else:
    #             ##! update the deterministic pieces in the state
    #             for action in actionDeterm:
    #                 state.board.table[action[0],action[1]] = action[2]
    #                 state.board.maskFrameCheck[action[0],action[1]] = True
                        
    #         # print("Table: ", state.board.table)
    #         # print("Mask: ", state.board.maskFrameCheck)
        
    #     return  np.array(actionNonDeterm, dtype=object)
    
    
    
    @staticmethod
    def update_adjacent_multiple(matrix, entries):
        max_rows, max_cols = matrix.shape

        # Create arrays for row and column indices for each entry
        row_indices = np.array(entries)[:, 0]
        col_indices = np.array(entries)[:, 1]

        # Define relative positions for adjacent cells
        relative_positions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

        # Compute all adjacent positions
        all_positions = row_indices[:, None] + relative_positions[:, 0], col_indices[:, None] + relative_positions[:, 1]

        # Filter positions that are inside the matrix bounds
        valid = (all_positions[0] >= 0) & (all_positions[0] < max_rows) & \
                (all_positions[1] >= 0) & (all_positions[1] < max_cols)

        # Flatten valid positions and make them unique to avoid repeated setting of the same cell
        valid_rows = all_positions[0][valid]
        valid_cols = all_positions[1][valid]
        unique_positions = np.unique((valid_rows, valid_cols), axis=1)

        # Set the valid adjacent positions to True
        matrix[unique_positions[0], unique_positions[1]] = True
    
    
    @staticmethod
    def isArrayVectorized(x):
        
        return np.vectorize(lambda x: isinstance(x, np.ndarray))(x)
    
    
    def actions(self, state: PipeManiaState):
        """Devolve as ações não deterministicas possíveis a partir de um estado."""
        
        ### todas as acões deterministas são diretamente implementadas no estado
        ### apenas as peças nearby das peças mudadas são calculadas dentro do loop
        
        boolDeterm = True
        sizeDeterm = 0
        mSize = state.board.maskFrameCheck.shape[0]-1
        
        matMarked = np.full_like(state.board.maskFrameCheck, False, dtype=np.bool_)
        matActions = np.full_like(state.board.maskFrameCheck, 0, dtype=object)
        
        boolInit = True
        
        
        while boolDeterm: 
            
            if boolInit:
                idxActions = np.where(state.board.maskFrameCheck == False)
                boolInit = False
            else:
                ## update just the near terms
                markedPoints = list(zip(*np.where(matMarked)))
                
                matrixAdjacents = np.full_like(state.board.maskFrameCheck, False, dtype=np.bool_)
                self.update_adjacent_multiple(matrixAdjacents, markedPoints)
                
                idxActions = np.where(np.logical_and(state.board.maskFrameCheck == False, matrixAdjacents))
            
            sizeDeterm = 0
            for x, y in zip(*idxActions):
                
                if x == 0 or y == 0 or x == mSize or y == mSize:
                    actions = board.strategyOne(x, y, state.board.maskFrameCheck, True)
                else:
                    actions = board.strategyOne(x, y, state.board.maskFrameCheck)
                
                if len(actions) == 1:
                    
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
        
        # print("mat action")
        # print(matActions)
        
        return  np.array(actionNonDeterm, dtype=object)
    
    
    def result(self, state: PipeManiaState, action):
        
        x = action[0]
        y = action[1]
        action = action[2]
        
        newTable = state.board.table.copy()
        newTable[x, y] = action
        
        newMaskFrameCheck = state.board.maskFrameCheck.copy()
        newMaskFrameCheck[x, y] = True
        
        newBoard = Board(newTable, newMaskFrameCheck)
        return PipeManiaState(newBoard)
    
    
    def goal_test(self, state: PipeManiaState):
        
        tableArr = np.array([[state.board.lookUpFunc(x) for x in row] for row in state.board.table], dtype=npType)
        
        mSize = tableArr.shape[0]
        
        dataPointsGrid = np.array([[state.board.getPointsSquare(x, y, tableArr[x,y]) for y in range(mSize)] for x in range(mSize)], dtype=npType)
        
        
        # return True
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
    
    
    def h(self, node: Node):
        
        # print("Node: ")
        # print("Action: ", node.action)
        # print("State: ", node.state.board.table)
        
        # print("*"*50)
        # print(Node.actions)
        
        return 1


import time

if __name__ == "__main__":
    
    ##? Implementar procura gaanciosa ...
    
    timeInit = time.time()
    ##! Obtain the Board from the standard input
    board = Board.parse_instance()
    
    
    ##? Deterministic Inference
    ##! Start with the Board class and simplify the outer layer as much as possible.
    board.deterministicInference()
    
    
    timeInit = time.time()
    
    problem = PipeMania(board)
    s0 = PipeManiaState(board)
    
    # actions = problem.actions(s0)
    
    # print("action 0: ", actions.shape)
    # print(actions)
    
    
    result = greedy_search(problem)
    
    # result1 = problem.result(s0, actions[0])
    # actions2 = problem.actions(result1)
    
    # result2 = problem.result(result1, actions2[0])
    # actions3 = problem.actions(result2)
    
    # print("action 1: ", actions2.shape)
    
    # print("action 2: ", actions3.shape)
    
    # result = depth_first_tree_search(problem)
    
    print("Time: ", time.time()-timeInit)
    
    
    # print(actions)
    # print(actions2)
    # print(actions3)
    
    
    # print("Result: ", result)
    # actions = problem.actions(s0)
    # print("Actions: ", actions)
    
    # print(s0.board.maskFrameCheck)
    
    with open("output.txt", "w") as f:
        table = s0.board.table
        for row in table:
            f.write("\t".join(row) + "\n")
            
    with open("outputMask.txt", "w") as f:
        table = s0.board.maskFrameCheck.astype(np.str_)
        for row in table:
            f.write("\t".join(row) + "\n")
    
    # result = depth_first_tree_search(problem)
    # result = breadth_first_tree_search(problem)
    # result = greedy_search(problem)
    
    

    # actions = pipe.actions(board)
    # print(actions)
    
    # countVec = np.sum(np.all(actions[:, np.newaxis, :2] == actions[:, :2], axis=2), axis=1)
        
    # minValue = np.min(countVec)
    # minValuesVec = countVec == minValue
    
    
    # print(countVec)
    # print(minValuesVec)
    
    
    # if minValue == 1:
    #     newIdx = np.where(minValuesVec)[0]
    #     actions2 = actions[newIdx]
    #     print(newIdx)
    #     print(actions2)
    # else:
    #     newIdx = np.where(minValuesVec)
    #     print(newIdx)
    
        
        # if minValue == 1:
        #     newIdx = np.where(minValuesVec)[0]
        #     return actions[newIdx], newIdx.shape[0]
        # else:
        #     newIdx = np.where(minValuesVec)[0][0]
        #     return actions[newIdx], 1
    
    
    
    
    
    # ##! check if the board is already solved
    # if board.is_goal():
    #     print("Deterministically solved.")
        
    #     print("vars: ", vars(board))
    # else:
    #     ##! Create PipeMania to solve the unsolved domain of the board
    #     pipeOne = PipeMania(board)
        
    #     print("check variables before Search Algorithm \n")
    #     print(vars(pipeOne.initial))
    
    #     ##! Search Algorithms without Heuristic for now
    #     result = breadth_first_tree_search(pipeOne)
    #     # result = depth_first_tree_search(pipeOne)
        
    #     print("check variables after Search Algorithm \n")
    #     print(vars(result.state))
        
    #     print()
    #     print("Deterministic Inference + Search Algorithm")
    #     print("Reached the goal: ", pipeOne.goal_test(result.state))

    # print("Time: ", time.time()-timeInit)