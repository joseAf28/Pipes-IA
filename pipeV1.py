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
    
    def __init__(self, tableInit):
        self.table = tableInit
        self.maskFrameCheck = None
        self.matActions = None
    
    @staticmethod
    def parse_instance():
        
        linesData = []
        while True:
            line = sys.stdin.readline()
            if line == '':  
                break
            else:
                linesData.append(line.split()) 
    
        return Board(np.array(linesData))
    
    
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
        
        ##! watch out table must be declared and filled before calling this function
        ##! possible source of error
        
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
            
            ## get the points of all the possible points available
            gridPointsSet = np.array([self.getPointsSquare(x2, y2, self.lookUpFunc(x)) for x in setF], dtype=npType)
            
            if properPointsBoundaryNot.size !=0:
                checkSimilarityNot = np.array([np.any(np.all(gridPointsSetAux[:, np.newaxis, :] == properPointsBoundaryNot, axis=(2)), axis=1) for gridPointsSetAux in gridPointsSet], dtype=npType)
                checkSimilarityNot = np.sum(checkSimilarityNot, axis=1) != 0
            else:
                checkSimilarityNot = np.full((gridPointsSet.shape[0],), False, dtype=np.bool_)
            
            idxCheckBoundaries = np.full((gridPointsSet.shape[0],), True, dtype=np.bool_)
            
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
                pass
            
            ## compute the similarity between the external points and the boundary points
            similarityVec = np.array([np.any(np.all(gridPointsSetAux[:, np.newaxis, :] == properPointsBoundary, axis=(2)), axis=1) for gridPointsSetAux in gridPointsSet], dtype=npType)
            similarityVec = np.sum(similarityVec, axis=1)
        
            ## find the best match(es) : all the options with less than the maximum similarity are discarded
            actions = setF[*np.where(np.logical_and(similarityVec == np.max(similarityVec), idxCheckBoundaries))]
            
        elif idPoint in setV:
            
            ## get the points of all the possible points available
            gridPointsSet = np.array([self.getPointsSquare(x2, y2, self.lookUpFunc(x)) for x in setV], dtype=npType)
            
            if properPointsBoundaryNot.size !=0:
                checkSimilarityNot = np.array([np.any(np.all(gridPointsSetAux[:, np.newaxis, :] == properPointsBoundaryNot, axis=(2)), axis=1) for gridPointsSetAux in gridPointsSet], dtype=npType)
                checkSimilarityNot = np.sum(checkSimilarityNot, axis=1) != 0
            else:
                checkSimilarityNot = np.full((gridPointsSet.shape[0],), False, dtype=np.bool_)
            
            idxCheckBoundaries = np.full((gridPointsSet.shape[0],), True, dtype=np.bool_)
            
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
                pass
            
            ## compute the similarity between the external points and the boundary points
            similarityVec = np.array([np.any(np.all(gridPointsSetAux[:, np.newaxis, :] == properPointsBoundary, axis=(2)), axis=1) for gridPointsSetAux in gridPointsSet], dtype=npType)
            similarityVec = np.sum(similarityVec, axis=1)
            
            # ## find the best match(es) : all the options with less than the maximum similarity are discarded
            actions = setV[*np.where(np.logical_and(similarityVec == np.max(similarityVec), idxCheckBoundaries))]
            
        elif idPoint in setL:
            
            if properPointsBoundary.size == 0:
                return setL
            
            # ## get the points of all the possible points available
            gridPointsSet = np.array([self.getPointsSquare(x2, y2, self.lookUpFunc(x)) for x in setL], dtype=npType)

            if properPointsBoundaryNot.size !=0:
                checkSimilarityNot = np.array([np.any(np.all(gridPointsSetAux[:, np.newaxis, :] == properPointsBoundaryNot, axis=(2)), axis=1) for gridPointsSetAux in gridPointsSet], dtype=npType)
                checkSimilarityNot = np.sum(checkSimilarityNot, axis=1) != 0
            else:
                checkSimilarityNot = np.full((gridPointsSet.shape[0],), False, dtype=np.bool_)
            
            checkSimilarityNot = np.logical_not(checkSimilarityNot)
            
            ## compute the similarity between the external points and the boundary points
            similarityVec = np.array([np.any(np.all(gridPointsSetAux[:, np.newaxis, :] == properPointsBoundary, axis=(2)), axis=1) for gridPointsSetAux in gridPointsSet], dtype=npType)
            similarityVec = np.sum(similarityVec, axis=1)
            
            # ## find the best match(es) : all the options with less than the maximum similarity are discarded
            actions = setL[*np.where(np.logical_and(similarityVec == np.max(similarityVec), checkSimilarityNot))]
            
        elif idPoint in setB:
            
            ## in case of not having points on the boundary, the full set is returned
            if properPointsBoundary.size == 0:
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
            similarityVec = np.array([np.any(np.all(gridPointsSetAux[:, np.newaxis, :] == properPointsBoundary, axis=(2)), axis=1) for gridPointsSetAux in gridPointsSet], dtype=npType)
            similarityVec = np.sum(similarityVec, axis=1)
            
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
        
        self.tableArr = np.array([[self.lookUpFunc(x) for x in row] for row in self.table], dtype=npType)
        
        if self.tableArr.shape[0] == 0:
            return True
        mSize = self.tableArr.shape[0]
        
        dataPointsGrid = np.array([[self.getPointsSquare(x, y, self.tableArr[x,y]) for y in range(mSize)] for x in range(mSize)], dtype=npType)

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
        
        
        ##! Putting the deterministic pices in the proper place
            
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
                    self.tableArr[x, y] = self.lookUpFunc(setB[idxCheckBoundaries[0]])
                
            elif idPoint in setL:
                
                gridPointsSet = np.array([self.getPointsSquare(x, y, self.lookUpFunc(piece)) for piece in setL], dtype=npType)
                checkBoundaries = np.any(np.logical_or(gridPointsSet == 0, gridPointsSet == 2*mSize), axis=(1,2))
                
                idxCheckBoundaries = np.where(~checkBoundaries)[0]
                
                if len(idxCheckBoundaries) > 1:
                    print("error: more than one possible solution")
                else:
                    self.table[x,y] = setL[idxCheckBoundaries[0]]
                    self.tableArr[x, y] = self.lookUpFunc(setL[idxCheckBoundaries[0]])
                
            elif idPoint in setV:
                
                gridPointsSet = np.array([self.getPointsSquare(x, y, self.lookUpFunc(piece)) for piece in setV], dtype=npType)
                checkBoundaries = np.any(np.logical_or(gridPointsSet == 0, gridPointsSet == 2*mSize), axis=(1,2))
                
                idxCheckBoundaries = np.where(~checkBoundaries)[0]
                
                if len(idxCheckBoundaries) > 1:
                    print("error: more than one possible solution")
                else:
                    self.table[x, y] = setV[idxCheckBoundaries[0]]
                    self.tableArr[x, y] = self.lookUpFunc(setV[idxCheckBoundaries[0]])
                    
        
        ##! Build the maskFrameCheck and matActions matrices
        matDetermBoth = np.logical_or(matDetermRight, matDetermWrong)
        idxDetermBoth = np.where(matDetermBoth)
        
        ##! the maskFrameCheck tells us which points are already determined
        ##! matActions tells us which points have more than one possible action and the actions themselves
        self.matActions = np.zeros((mSize, mSize), dtype=object) 
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
                self.tableArr[x2, y2] = self.lookUpFunc(actions[0])
                self.maskFrameCheck[x2, y2] = True
            else:
                self.matActions[x2, y2] = actions
        
        
        ##! Recurrent Iterations
        boolLayerExt = np.logical_not(np.all(np.logical_and(self.maskFrameCheck, maskBorder)))
        maskLoopCheck = self.maskFrameCheck.copy()
        
        while boolLayerExt:
            
            idxDeterm = np.where(maskLoopCheck)
            
            ##! Define the mask for the new boundaries
            maskAux = np.full_like(self.table, False, dtype=np.bool_)
            for x, y in zip(*idxDeterm):
                maskAux[x-1:x+2, y] = True
                maskAux[x, y-1:y+2] = True
            
            for x, y in zip(*idxDeterm):
                maskAux[x, y] = False
            
            ##! ensure we do not leave the external layer
            maskBorderZero = np.logical_and(maskBorder, maskAux)
            
            idx2ChangeBorderZero = np.array(list(zip(*np.where(maskBorderZero))))
            
            ##! check if the iteration over the external layer is finished
            if idx2ChangeBorderZero.size == 0:
                boolLayerExt = False
            
            for i in range(idx2ChangeBorderZero.shape[0]):
                x2, y2 = idx2ChangeBorderZero[i]
                actions = self.strategyOne(x2, y2, self.maskFrameCheck, boolLayerExt)
            
                if len(actions) == 1:
                    self.table[x2, y2] = actions[0]
                    self.tableArr[x2, y2] = self.lookUpFunc(actions[0])
                    self.maskFrameCheck[x2, y2] = True
                    self.matActions[x2, y2] = 0
                else:
                    self.matActions[x2, y2] = actions
                    
                maskLoopCheck[x2, y2] = True
                
        
        ##! Deterministic Inference for the internal upper layers
        if mSize >= 3:
            mLoops = int(np.ceil(mSize/2))
            
            ##! We proced from the the second layer to the internal mLoops layer each one at a time
            for i in range(1, mLoops):
                self.strategyDetermisticLoop(i)
            
            ##! Final Deterministic Inference where the actions can be reduced in unsolved domain
            maskFrameCheck3rd = np.logical_not(self.maskFrameCheck)
            bool3rdLayer = np.any(maskFrameCheck3rd)
            
            while bool3rdLayer:
                x2, y2, counterMax = self.findTheBestPoint(maskFrameCheck3rd, self.maskFrameCheck, mSize)
                
                if counterMax == 0:
                    bool3rdLayer = False
                else:
                    boolLayerExt = False
                    if x2 == 0 or x2 == mSize-1 or y2 == 0 or y2 == mSize-1:
                        boolLayerExt = True
                    actions = self.strategyOne(x2, y2, self.maskFrameCheck, boolLayerExt)
                    
                    if len(actions) == 1:
                        self.table[x2, y2] = actions[0]
                        self.tableArr[x2, y2] = self.lookUpFunc(actions[0])
                        self.maskFrameCheck[x2, y2] = True
                        self.matActions[x2, y2] = 0
                    else:
                        self.matActions[x2, y2] = actions
                    
                    maskFrameCheck3rd[x2, y2] = False
        else:
            pass
        
        ##! Delete the tableArr as it is not needed anymore
        del self.tableArr


class PipeManiaState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = PipeManiaState.state_id
        PipeManiaState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id



class PipeMania(Problem):
    
    def __init__(self,state: Board):
        self.initState = board
    
    
    # # old 
    # def actions(self, state: PipeManiaState):
    #     """Devolve as ações possíveis a partir de um estado."""
            
    #     checkZeroString = np.vectorize(lambda x: isinstance(x, np.ndarray))
        
    #     ##! check if there are any actions available
    #     isZeroString = checkZeroString(state.matActions)
        
    #     ##? However is more efficient to compute again....
        
    #     actionsVec = []
        
    #     if np.any(isZeroString):
    #         idxActions = np.where(state.maskFrameCheck == False)
    #         matActions = state.matActions
            
    #         for x, y in zip(*idxActions):
    #             for action in matActions[x, y]:
    #                 actionsVec.append((x, y, action))
    #     else:
    #         idxActions = np.where(state.maskFrameCheck == False)
    #         for x, y in zip(*idxActions):
    #             actions = state.strategyOne(x, y, state.maskFrameCheck)
    #             for action in actions:
    #                 actionsVec.append((x, y, action))
            
    #     return actionsVec
    

    def actions(self, state: PipeManiaState):
        """Devolve as ações possíveis a partir de um estado."""
        
        idxActions = np.where(state.maskFrameCheck == False)
        actionsVec = []
        
        for x, y in zip(*idxActions):
            actions = state.strategyOne(x, y, state.maskFrameCheck)
            for action in actions:
                actionsVec.append([x, y, action])
        
        return np.array(actionsVec, dtype=object)
    
    
    def result(self, state: PipeManiaState, action):
        
        x = action[0]
        y = action[1]
        action = action[2]
        tableAux = state.table.copy()
        tableAux[x, y] = action
        
        newMaskFrameCheck = state.maskFrameCheck.copy()
        newMaskFrameCheck[x, y] = True
        
        newBoard = Board(tableAux)
        newBoard.maskFrameCheck = newMaskFrameCheck
        
        return newBoard
    
    
    def goal_test(self, state: PipeManiaState):
        return np.all(state.maskFrameCheck)
    
    
    # def h(self, node: Node):
    #     ##! heuristic function
    #     ## greedy: choose the one that has no other options first
    #     pass
    
    
    ##! wrong arguments for now ...
    def h(self, actions):
        
        countVec = np.sum(np.all(actions[:, np.newaxis, :2] == actions[:, :2], axis=2), axis=1)
        
        minValue = np.min(countVec)
        minValuesVec = countVec == minValue
        
        if minValue == 1:
            newIdx = np.where(minValuesVec)[0]
            return actions[newIdx], newIdx.shape[0]
        else:
            newIdx = np.where(minValuesVec)[0][0]
            return actions[newIdx], 1


if __name__ == "__main__":
    
    ##! Obtain the Board from the standard input
    board = Board.parse_instance()
    
    # print("Init Table:")
    # print(board.table)
    # print()
    
    ##? Deterministic Inference
    ##! Start with the Board class and simplify it the most we can.
    board.deterministicInference()
    
    
    print("check variables")
    print(vars(board))
    
    ##! PipeMania Declaration and Methods' Verification
    ## check bug: last action tensor shape

    board.matActions = np.full_like(board.table, 0, dtype=object)
    newPipes = PipeMania(board)
    actions = newPipes.actions(board)
    
    bestAction, size = newPipes.h(actions)
    for row in bestAction:
        board = newPipes.result(board, row)
    
    actions = newPipes.actions(board)
    actions, size = newPipes.h(actions)

    if size == 1:
        board = newPipes.result(board, actions)
    else:
        for row in actions:
            board = newPipes.result(board, row)
    
    actions = newPipes.actions(board)
    actions, size = newPipes.h(actions)
    
    for row in actions:
            board = newPipes.result(board, row)
    
    print("Final Table:")
    print(board.table)
    print("check goal: ", newPipes.goal_test(board))
    
    ##! Search Algotithms over the Unsolved Domain
    
    ## try the different search algorithms
    ## primeiro em largura e profundidade
    ## to be done
